"""
Step 22: Two remaining levers not yet tried in the log (model_variations_log.md,
E01-E80), after that log already showed a hard ceiling around AUROC 0.677-0.682
with every tested change moving the score by <=0.008:

Part 1  True stacked meta-learner. The existing ensemble (06_ensemble.py) blends
        CatBoost + XGBoost + LogReg with a fixed weight found by grid search on
        validation AUPRC. Here a logistic regression is instead *trained* on the
        three base models' validation-set predictions (as features) against
        y_val, then applied to their test-set predictions. Base models are the
        same architectures/hyperparameters as 06_ensemble.py; only the blending
        step changes.
Part 2  Wider CatBoost hyperparameter search. 05_catboost.py tried 3x3=9
        (depth, learning_rate) combinations. This tries a randomized search over
        depth, learning_rate, l2_leaf_reg and bagging_temperature (30 draws),
        selected on validation AUPRC exactly as before, test scored once.

Same patient-level split as every other script; paired bootstrap (patient-
grouped resamples) against the current best (M0 = CatBoost E59/E78, AUROC
0.677) for a fair comparison.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]

train = pd.read_csv("data/train.csv", keep_default_na=False)
val = pd.read_csv("data/val.csv", keep_default_na=False)
test = pd.read_csv("data/test.csv", keep_default_na=False)

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_val = val.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---- patient-grouped bootstrap resamples on the test set, shared by both parts ----
groups = test["patient_nbr"].to_numpy()
y_test_arr = y_test.to_numpy()
row_idx_by_patient = pd.Series(np.arange(len(y_test_arr))).groupby(groups).apply(lambda s: s.to_numpy()).tolist()
rng = np.random.default_rng(1)
resamples = [
    np.concatenate([row_idx_by_patient[i] for i in rng.integers(0, len(row_idx_by_patient), len(row_idx_by_patient))])
    for _ in range(1000)
]


def ci(p):
    a = [roc_auc_score(y_test_arr[r], p[r]) for r in resamples if y_test_arr[r].min() != y_test_arr[r].max()]
    return np.percentile(a, [2.5, 97.5])


def paired_diff_ci(p, base):
    d = [roc_auc_score(y_test_arr[r], p[r]) - roc_auc_score(y_test_arr[r], base[r])
         for r in resamples if y_test_arr[r].min() != y_test_arr[r].max()]
    return np.percentile(d, [2.5, 97.5])


def op_point(val_probs, test_probs):
    # threshold chosen on validation (max tpr - fpr, same convention as scripts 15/18/20)
    fpr, tpr, th = roc_curve(y_val, val_probs)
    t = th[np.argmax(tpr - fpr)]
    yp = test_probs >= t
    acc = np.mean(yp == y_test_arr)
    sens = np.mean(yp[y_test_arr == 1])
    spec = np.mean(~yp[y_test_arr == 0])
    return t, acc, sens, spec


rows = []
print(f"test n={len(y_test_arr):,}, positive rate {y_test_arr.mean():.3f}\n")

# =====================================================================================
# Part 0: M0 baseline -- CatBoost with the log's adopted params (depth 4, lr 0.1), for
# a same-run, directly comparable reference point (previous CI runs used seed 1 too;
# re-fit here rather than reloading so the paired bootstrap uses identical resamples).
# =====================================================================================
X_train_cb, X_val_cb, X_test_cb = X_train.copy(), X_val.copy(), X_test.copy()
for df_ in (X_train_cb, X_val_cb, X_test_cb):
    for c in cat_cols:
        df_[c] = df_[c].astype(str)
train_pool = Pool(X_train_cb, y_train, cat_features=cat_cols)
val_pool = Pool(X_val_cb, y_val, cat_features=cat_cols)
test_pool = Pool(X_test_cb, cat_features=cat_cols)

m0 = CatBoostClassifier(
    depth=4, learning_rate=0.1, iterations=1000, loss_function="Logloss", eval_metric="AUC",
    scale_pos_weight=pos_weight, random_seed=42, verbose=False, early_stopping_rounds=50,
)
m0.fit(train_pool, eval_set=val_pool)
m0_val = m0.predict_proba(val_pool)[:, 1]
m0_test = m0.predict_proba(test_pool)[:, 1]
base = m0_test
t, acc, sens, spec = op_point(m0_val, m0_test)
lo, hi = ci(m0_test)
auroc0 = roc_auc_score(y_test_arr, m0_test)
rows.append(dict(id="E81", model="M0 CatBoost (depth 4, lr 0.1) -- reference", auroc=auroc0, lo=lo, hi=hi,
                  auprc=average_precision_score(y_test_arr, m0_test), accuracy=acc, sensitivity=sens, specificity=spec,
                  diff=0.0, diff_lo=0.0, diff_hi=0.0))
print(f"M0 CatBoost reference: AUROC={auroc0:.3f} [{lo:.3f},{hi:.3f}]  Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}\n")

# =====================================================================================
# Part 1: true stacked meta-learner (base models trained once, same hyperparameters as
# 06_ensemble.py; meta-learner is a logistic regression on the three val-set predictions)
# =====================================================================================
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

logreg = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
logreg.fit(X_train, y_train)
logreg_val, logreg_test = logreg.predict_proba(X_val)[:, 1], logreg.predict_proba(X_test)[:, 1]

xgb = Pipeline([("prep", preprocess), ("clf", XGBClassifier(
    max_depth=5, n_estimators=400, learning_rate=0.03, eval_metric="aucpr",
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1,
))])
xgb.fit(X_train, y_train)
xgb_val, xgb_test = xgb.predict_proba(X_val)[:, 1], xgb.predict_proba(X_test)[:, 1]

cb = CatBoostClassifier(
    depth=6, learning_rate=0.03, iterations=791, loss_function="Logloss", eval_metric="AUC",
    scale_pos_weight=pos_weight, random_seed=42, verbose=False,
)
cb.fit(train_pool)
cb_val, cb_test = cb.predict_proba(val_pool)[:, 1], cb.predict_proba(test_pool)[:, 1]
print("Base models for stacking trained (same hyperparameters as 06_ensemble.py).\n")

meta_X_val = np.column_stack([logreg_val, xgb_val, cb_val])
meta_X_test = np.column_stack([logreg_test, xgb_test, cb_test])
meta = LogisticRegression(max_iter=2000)
meta.fit(meta_X_val, y_val)
stack_test = meta.predict_proba(meta_X_test)[:, 1]
stack_val = meta.predict_proba(meta_X_val)[:, 1]

t, acc, sens, spec = op_point(stack_val, stack_test)
lo, hi = ci(stack_test)
dlo, dhi = paired_diff_ci(stack_test, base)
auroc = roc_auc_score(y_test_arr, stack_test)
rows.append(dict(id="E82", model="Stacked meta-learner (LogReg on val predictions of CatBoost+XGBoost+LogReg)",
                  auroc=auroc, lo=lo, hi=hi, auprc=average_precision_score(y_test_arr, stack_test),
                  accuracy=acc, sensitivity=sens, specificity=spec,
                  diff=auroc - auroc0, diff_lo=dlo, diff_hi=dhi))
print(f"Stacked meta-learner: AUROC={auroc:.3f} [{lo:.3f},{hi:.3f}] | vs M0 {auroc - auroc0:+.4f} [{dlo:+.4f},{dhi:+.4f}] | "
      f"Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")
print(f"  meta-learner coefficients (logreg, xgb, catboost): {meta.coef_[0].round(3)}\n")

# =====================================================================================
# Part 2: wider CatBoost hyperparameter search (random search, 30 draws) vs the 9-combo
# grid in 05_catboost.py. Selection on validation AUPRC; test scored once for the winner.
# =====================================================================================
search_rng = np.random.default_rng(7)
N_DRAWS = 30
best_auprc, best_model, best_params = -1, None, None
for i in range(N_DRAWS):
    depth = int(search_rng.integers(3, 10))
    lr = float(10 ** search_rng.uniform(-2.3, -0.5))     # ~0.005 to ~0.3
    l2 = float(10 ** search_rng.uniform(-1, 1.5))         # ~0.1 to ~30
    bag_temp = float(search_rng.uniform(0, 2))
    model = CatBoostClassifier(
        depth=depth, learning_rate=lr, l2_leaf_reg=l2, bagging_temperature=bag_temp,
        iterations=1000, loss_function="Logloss", eval_metric="AUC",
        scale_pos_weight=pos_weight, random_seed=42, verbose=False, early_stopping_rounds=50,
    )
    model.fit(train_pool, eval_set=val_pool)
    val_prob = model.predict_proba(val_pool)[:, 1]
    val_auprc = average_precision_score(y_val, val_prob)
    if val_auprc > best_auprc:
        best_auprc, best_model = val_auprc, model
        best_params = dict(depth=depth, lr=round(lr, 4), l2_leaf_reg=round(l2, 2),
                            bagging_temperature=round(bag_temp, 2), best_iteration=model.get_best_iteration())
    print(f"  draw {i + 1:2d}/{N_DRAWS}: depth={depth} lr={lr:.4f} l2={l2:.2f} bag_temp={bag_temp:.2f} "
          f"-> val_AUPRC={val_auprc:.4f}")

print(f"\nBest of {N_DRAWS} random-search draws: {best_params}, val_AUPRC={best_auprc:.4f}\n")

wide_val = best_model.predict_proba(val_pool)[:, 1]
wide_test = best_model.predict_proba(test_pool)[:, 1]
t, acc, sens, spec = op_point(wide_val, wide_test)
lo, hi = ci(wide_test)
dlo, dhi = paired_diff_ci(wide_test, base)
auroc = roc_auc_score(y_test_arr, wide_test)
rows.append(dict(id="E83", model=f"CatBoost (wider random search, 30 draws): {best_params}",
                  auroc=auroc, lo=lo, hi=hi, auprc=average_precision_score(y_test_arr, wide_test),
                  accuracy=acc, sensitivity=sens, specificity=spec,
                  diff=auroc - auroc0, diff_lo=dlo, diff_hi=dhi))
print(f"CatBoost (wider search): AUROC={auroc:.3f} [{lo:.3f},{hi:.3f}] | vs M0 {auroc - auroc0:+.4f} [{dlo:+.4f},{dhi:+.4f}] | "
      f"Acc={acc:.3f} Sens={sens:.3f} Spec={spec:.3f}")

out = pd.DataFrame(rows)
out.to_csv("5_results/ablations/stacking_and_wider_search.csv", index=False)
print("\nSaved 5_results/ablations/stacking_and_wider_search.csv")
print("\n" + "=" * 90)
print("Summary vs M0 (CatBoost, depth 4 lr 0.1, AUROC {:.3f}):".format(auroc0))
for r in rows[1:]:
    sig = "SIGNIFICANT" if r["diff_lo"] > 0 or r["diff_hi"] < 0 else "not significant (CI crosses 0)"
    print(f"  {r['model'][:70]:70s} {r['diff']:+.4f} [{r['diff_lo']:+.4f},{r['diff_hi']:+.4f}]  {sig}")
