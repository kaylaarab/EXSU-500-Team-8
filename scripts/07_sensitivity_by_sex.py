"""
Sensitivity (and specificity) by sex/gender subgroup, for every main model.

SUPERSEDED (2026-11-09): this script does not follow the evaluation
protocol's patient-level bootstrap (it resamples individual rows, so
patients with multiple stays are double-counted), uses an arbitrary 0.5
threshold on raw/uncalibrated scores rather than any model's real operating
threshold, and uses freshly-retrained models rather than the deployed one.
It found the sex-sensitivity gap "not significant," which contradicts the
protocol-compliant analysis in scripts/18_calibration_subgroups.py
(patient-level bootstrap, the deployed calibrated model, its real 11.6%
operating threshold -> 5_results/evaluation/subgroup_performance.csv),
which finds the gap IS significant. See
2_midterm_report/material_for_final_report.md for the correction. Kept for
history; do not cite this script's output.

Trains each model exactly as in scripts/0X_*.py (same hyperparameters), then
breaks test-set performance out by `gender` (Female / Male; 'Unknown/Invalid'
has only 2 rows in test so it's reported but not bootstrapped).

Bootstrap CIs are computed WITHIN each subgroup (resampling only that
subgroup's rows), so the CI width reflects that subgroup's own sample size,
not the pooled test set.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

TARGET = "readmit_30d"
ID_COLS = ["patient_nbr"]
N_BOOT = 1000
RNG_SEED = 42

train = pd.read_csv("data/train.csv", keep_default_na=False)
val = pd.read_csv("data/val.csv", keep_default_na=False)
test = pd.read_csv("data/test.csv", keep_default_na=False)

gender_test = test["gender"].copy()

y_train, y_val, y_test = train[TARGET], val[TARGET], test[TARGET]
X_train = train.drop(columns=[TARGET] + ID_COLS)
X_val = val.drop(columns=[TARGET] + ID_COLS)
X_test = test.drop(columns=[TARGET] + ID_COLS)

cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---------------- Logistic Regression ----------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
])
logreg.fit(X_train, y_train)
logreg_test = logreg.predict_proba(X_test)[:, 1]
print("Logistic Regression trained.")

# ---------------- XGBoost ----------------
xgb = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        max_depth=5, n_estimators=400, learning_rate=0.03, eval_metric="aucpr",
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
        random_state=42, n_jobs=-1,
    )),
])
xgb.fit(X_train, y_train)
xgb_test = xgb.predict_proba(X_test)[:, 1]
print("XGBoost trained.")

# ---------------- CatBoost ----------------
X_train_cb, X_val_cb, X_test_cb = X_train.copy(), X_val.copy(), X_test.copy()
for df_ in (X_train_cb, X_val_cb, X_test_cb):
    for c in cat_cols:
        df_[c] = df_[c].astype(str)
train_pool = Pool(X_train_cb, y_train, cat_features=cat_cols)
test_pool = Pool(X_test_cb, cat_features=cat_cols)
catboost = CatBoostClassifier(
    depth=6, learning_rate=0.03, iterations=791, loss_function="Logloss",
    eval_metric="AUC", scale_pos_weight=pos_weight, random_seed=42, verbose=False,
)
catboost.fit(train_pool)
cb_test = catboost.predict_proba(test_pool)[:, 1]
print("CatBoost trained.\n")

# ---------------- Ensemble (best weights from 06_ensemble.py) ----------------
ens_test = 0.0 * logreg_test + 0.3 * xgb_test + 0.7 * cb_test

models = {
    "Logistic Regression": logreg_test,
    "XGBoost": xgb_test,
    "CatBoost": cb_test,
    "Ensemble": ens_test,
}


def subgroup_metrics(y_true, y_prob, mask, rng):
    y_true_sub = y_true.to_numpy()[mask]
    y_prob_sub = y_prob[mask]
    y_pred_sub = (y_prob_sub >= 0.5).astype(int)
    n = len(y_true_sub)
    n_pos = int(y_true_sub.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return dict(n=n, n_pos=n_pos, sens=float("nan"), spec=float("nan"),
                    sens_ci=(float("nan"), float("nan")), spec_ci=(float("nan"), float("nan")))
    tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred_sub, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)

    boot_sens, boot_spec = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yt, yp = y_true_sub[idx], y_pred_sub[idx]
        if yt.sum() == 0 or (yt == 0).sum() == 0:
            continue
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        if (tp_b + fn_b) > 0:
            boot_sens.append(tp_b / (tp_b + fn_b))
        if (tn_b + fp_b) > 0:
            boot_spec.append(tn_b / (tn_b + fp_b))
    sens_ci = tuple(np.percentile(boot_sens, [2.5, 97.5])) if boot_sens else (float("nan"), float("nan"))
    spec_ci = tuple(np.percentile(boot_spec, [2.5, 97.5])) if boot_spec else (float("nan"), float("nan"))
    return dict(n=n, n_pos=n_pos, sens=sens, spec=spec, sens_ci=sens_ci, spec_ci=spec_ci)


rows = []
for model_name, y_prob in models.items():
    rng = np.random.default_rng(RNG_SEED)
    print(f"\n=== {model_name} ===")
    for g in ["Female", "Male"]:
        mask = (gender_test == g).to_numpy()
        m = subgroup_metrics(y_test, y_prob, mask, rng)
        print(f"  {g:7s} n={m['n']:5d} n_pos={m['n_pos']:4d}  "
              f"Sensitivity={m['sens']:.3f} (95% CI [{m['sens_ci'][0]:.3f}, {m['sens_ci'][1]:.3f}])  "
              f"Specificity={m['spec']:.3f} (95% CI [{m['spec_ci'][0]:.3f}, {m['spec_ci'][1]:.3f}])")
        rows.append(dict(model=model_name, gender=g, **m, sens_ci_lo=m["sens_ci"][0], sens_ci_hi=m["sens_ci"][1],
                          spec_ci_lo=m["spec_ci"][0], spec_ci_hi=m["spec_ci"][1]))

    f_mask = (gender_test == "Female").to_numpy()
    m_mask = (gender_test == "Male").to_numpy()
    f_m = subgroup_metrics(y_test, y_prob, f_mask, np.random.default_rng(RNG_SEED))
    m_m = subgroup_metrics(y_test, y_prob, m_mask, np.random.default_rng(RNG_SEED))
    gap = f_m["sens"] - m_m["sens"]
    overlap = not (f_m["sens_ci"][1] < m_m["sens_ci"][0] or m_m["sens_ci"][1] < f_m["sens_ci"][0])
    print(f"  --> Sensitivity gap (Female - Male) = {gap:+.3f}  "
          f"[subgroup 95% CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'}]")

    # Unknown/Invalid: report raw count only, no stats (n=2)
    u_mask = (gender_test == "Unknown/Invalid").to_numpy()
    if u_mask.sum() > 0:
        print(f"  Unknown/Invalid: n={u_mask.sum()} (too few to analyze)")

out_df = pd.DataFrame(rows)
out_path = "5_results/fairness/sensitivity_by_sex.csv"
import os
os.makedirs("5_results/fairness", exist_ok=True)
out_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
