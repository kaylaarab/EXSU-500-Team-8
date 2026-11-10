"""
Builds the consolidated experiment log: every model variation we ran, in one table.

    python scripts/build_experiment_log.py

Writes 5_results/model_variations_log.csv and 5_results/model_variations_log.md.

How it stays correct: rows for the current pipeline are READ from the result CSVs in
5_results/, so they update automatically when a script is rerun. Historical rows (earlier
pipeline versions, quick exploratory checks) are entered by hand below because their
result files were overwritten; each carries a note saying where the numbers came from.
Accuracy that was not recorded at the time is derived exactly from sensitivity,
specificity and the test positive rate (marked "derived"). To add an experiment, append
a row to HAND_ROWS or load its CSV in the loaders section, then rerun this script.
"""
from pathlib import Path

import pandas as pd

R = Path("5_results")
P1, P2 = 0.1104, 0.1112          # test positive rate: first-pass split / cleaned split
rows = []


def add(stage, date, script, change, cohort, model, auroc=None, ci=None, auprc=None, acc=None,
        sens=None, spec=None, p=None, note="", src=""):
    acc_src = "measured"
    if acc is None and sens is not None and spec is not None and p is not None:
        acc, acc_src = sens * p + spec * (1 - p), "derived"
    if acc is None:
        acc_src = ""
    rows.append(dict(stage=stage, date=date, script=script, change_vs_previous=change, cohort_target=cohort,
                     model=model, auroc=auroc, auroc_ci=ci, auprc=auprc, accuracy=acc, accuracy_source=acc_src,
                     sensitivity=sens, specificity=spec, takeaway=note, source=src))


S0 = "0 First pass (pipeline v1)"
S1 = "1 Quick diagnostics (exploratory)"
S2 = "2 Cleaned pipeline v2 (payer/specialty dropped)"
S3 = "3 Final pipeline v3 (payer/specialty kept)"
S4 = "4 Ablations and sensitivity"
S5 = "5 Target definitions and combinations"
S6 = "6 SMOTE leakage demonstration"
S7 = "7 Planned or running"
S8 = "8 Stacked meta-learner and wider CatBoost search"

# ---------------------------------------------------------------- hand-entered history
V1 = "all 101,766 encounters; death/hospice kept; A1C feature bug; IDs numeric"
for m, a, ap, se, sp, nt in [
    ("Baseline (always no)", 0.500, 0.110, 0.0, 1.0, ""),
    ("Logistic regression", 0.643, 0.2035, 0.519, 0.679, ""),
    ("XGBoost (tuned)", 0.677, 0.2260, 0.587, 0.674, "Looked best; inflated by easy death/hospice rows"),
    ("MLP", 0.655, 0.208, 0.614, 0.617, ""),
    ("CatBoost + engineered features", 0.678, 0.2293, 0.580, 0.671, ""),
    ("Ensemble (CatBoost+XGBoost+LogReg)", 0.679, 0.2299, 0.582, 0.673, "Superseded: uncleaned data and a feature bug"),
]:
    add(S0, "2026-10-15/16", "02-06", "First complete pipeline", V1, m, a, None, ap, None, se, sp, P2 if False else P1, nt,
        "first-pass results file (overwritten); MLP from literature notes")

QD = "HistGradientBoosting quick test, 99k-row cohort"
for ch, a, ap, nt in [
    ("As committed (buggy A1C flag, IDs numeric)", 0.6756, 0.2305, "reference"),
    ("A1C flag fixed", 0.6751, 0.2294, "Bug did not change the score, but the feature was not what we described"),
    ("Admission/discharge IDs as categories", 0.6797, 0.2366, "Only real gain: +0.004"),
    ("Death/hospice discharges excluded", 0.6707, 0.2321, "Lower score: removes easy negatives; kept for validity"),
    ("All three fixes", 0.6738, 0.2376, "Adopted as pipeline v2"),
]:
    add(S1, "2026-10-23", "scratch (not in repo)", ch, QD, "HistGradientBoosting", a, None, ap, None, None, None, None,
        nt, "console output; not saved to a file")
add(S1, "2026-10-23", "scratch (not in repo)", "Pipeline control: no weights vs random oversampling", "logistic regression",
    "LogReg, no weights", 0.661, None, None, None, None, None, None, "Plain model", "console output")
add(S1, "2026-10-23", "scratch (not in repo)", "Pipeline control: no weights vs random oversampling", "logistic regression",
    "LogReg, random oversampling to 1:1", 0.663, None, None, None, None, None, None,
    "Matches class weights, so the pipeline is sound and SMOTE-NC itself hurts", "console output")

V2 = "99,343 encounters; fixes applied; payer/specialty dropped"
for m, a, ap, se, sp, nt in [
    ("Baseline (always no)", 0.500, 0.1112, 0.0, 1.0, ""),
    ("Logistic regression", 0.663, 0.212, 0.528, 0.689, ""),
    ("XGBoost (tuned)", 0.665, 0.222, 0.540, 0.695, ""),
    ("MLP", 0.659, 0.219, 0.539, 0.686, ""),
    ("CatBoost + engineered features", 0.669, 0.221, 0.568, 0.670, ""),
    ("Ensemble", 0.669, 0.224, 0.550, 0.683, "Honest numbers; models statistically tied"),
]:
    add(S2, "2026-10-23", "02-06", "Cleaned cohort, categorical IDs, A1C fixed", V2, m, a, None, ap, None, se, sp, P2, nt,
        "results file before payer/specialty change (overwritten)")

# ---------------------------------------------------------------- loaders: current results
mc = pd.read_csv(R / "main_models/model_comparison.csv")
V3 = "99,343 encounters; payer/specialty kept as categories"
for _, r in mc.iterrows():
    add(S3, "2026-10-23", "02-06", "Keep payer_code and medical_specialty ('Unknown' for missing)", V3, r.model,
        r.auroc, None, r.auprc, r.get("accuracy"), r.sensitivity, r.specificity, P2,
        "Current best: CatBoost" if "CatBoost" in r.model else "", "5_results/main_models/model_comparison.csv")

sa = pd.read_csv(R / "ablations/sensitivity_analysis.csv")
for _, r in sa.iterrows():
    add(S4, "2026-10-23", "07", "Payer/specialty and first-encounter checks (patient-level bootstrap)",
        f"CatBoost fixed params; {int(r.n_test):,} test rows; pos rate {r.pos_rate:.3f}", r.analysis,
        r.auroc, f"{r.auroc_lo:.3f}-{r.auroc_hi:.3f}", r.auprc, None, None, None, None,
        {"B": "+0.008 vs A (paired CI +0.005 to +0.011): adopted", "C": "Harder: no history for first stays"}.get(r.analysis[0], ""),
        "5_results/ablations/sensitivity_analysis.csv")

for tag, data in [("payer/specialty dropped (v2)", [("Base (no history)", .6703, .2213, ""), ("+ stay number", .6726, .2224, "CI -0.002 to +0.007"),
                                                  ("+ stay number + previous label", .6748, .2292, "CI 0.000 to +0.009"),
                                                  ("+ all four history features", .6757, .2308, "CI +0.001 to +0.010")])]:
    for m, a, ap, nt in data:
        add(S4, "2026-10-23", "08", "Patient-history features, " + tag, "HistGradientBoosting; 99,343 encounters", m, a, None, ap,
            None, None, None, None, nt, "results file before payer change (overwritten)")
tf = pd.read_csv(R / "ablations/trajectory_features.csv")
for _, r in tf.iterrows():
    add(S4, "2026-10-23", "08", "Patient-history features, payer/specialty kept (v3)", "HistGradientBoosting; 99,343 encounters",
        r.model, r.auroc, f"{r.auroc_lo:.3f}-{r.auroc_hi:.3f}", r.auprc, None, None, None, None,
        f"paired gain CI {r.diff_lo:+.3f} to {r.diff_hi:+.3f}" if r.diff_hi else "",
        "5_results/ablations/trajectory_features.csv")

sm = pd.read_csv(R / "ablations/smote_ablation.csv")
for _, r in sm.iterrows():
    add(S4, "2026-10-23", "09", "SMOTE-NC vs class weights (fitted on train only)", "payer/specialty dropped; fixed params",
        f"{r.model}, {r.arm}", r.auroc, None, r.auprc, r.accuracy, r.sensitivity, r.specificity, None,
        "" if r.arm == "weights" else f"vs weights CI {r.diff_vs_weights_lo:+.3f} to {r.diff_vs_weights_hi:+.3f}: worse",
        "5_results/ablations/smote_ablation.csv")

hd = pd.read_csv(R / "ablations/hospice_death_ablation.csv")
for _, r in hd.iterrows():
    add(S4, "2026-10-23", "10", "Death/hospice exclusion ablation (CatBoost, one patient split)",
        f"train {int(r.n_train):,}; scored on common test", r.cohort, r.auroc_common_test,
        f"{r.auroc_common_lo:.3f}-{r.auroc_common_hi:.3f}", r.auprc_common_test, None, None, None, None,
        {"all": "Same score as the cleaned cohort (paired diff -0.0003): exclusion is for validity, not score",
         "no_death": "", "no_death_hospice": "Our cohort"}[r.cohort], "5_results/ablations/hospice_death_ablation.csv")

td = pd.read_csv(R / "ablations/target_definitions.csv")
for _, r in td.iterrows():
    add(S5, "2026-10-23", "13", "Different definition of 'readmitted'", f"{int(r.n_test):,} test rows; prevalence {r.prevalence:.3f}",
        r.target, r.auroc, None, r.auprc, r.accuracy, r.sensitivity, r.specificity, None,
        {"T": ""}.get(r.target[0], "") or ("Easier question, not a better model" if r.target[1] in "23" else "Headline target"),
        "5_results/ablations/target_definitions.csv")

ct = pd.read_csv(R / "ablations/combined_targets.csv")
for _, r in ct.iterrows():
    add(S5, "2026-10-23", "14", "Use the '>30 days' label to help predict 30-day readmission (all encounters)",
        "HistGradientBoosting; 14,913 test rows; threshold from validation", r.method, r.auroc, None, r.auprc, r.accuracy,
        r.sensitivity, r.specificity, None,
        "baseline" if r.auroc_diff_vs_A == 0 else f"vs A {r.auroc_diff_vs_A:+.4f} (CI {r.diff_lo:+.4f} to {r.diff_hi:+.4f})",
        "5_results/ablations/combined_targets.csv")
for name, path in [("first-encounter scenario", R / "ablations/combined_targets_first_encounter.csv")]:
    if path.exists():
        for _, r in pd.read_csv(path).iterrows():
            add(S5, "2026-10-24", "14", "Same, first encounter per patient only", "HistGradientBoosting; 69,990 patients",
                r.method, r.auroc, None, r.auprc, r.accuracy, r.sensitivity, r.specificity, None,
                "baseline" if r.auroc_diff_vs_A == 0 else f"vs A {r.auroc_diff_vs_A:+.4f} (CI {r.diff_lo:+.4f} to {r.diff_hi:+.4f})", str(path))

lc = pd.read_csv(R / "ablations/lace_comorbidity.csv")
for _, r in lc.iterrows():
    add(S4, "2026-10-24", "15", "LACE clinical score and comorbidity features (CatBoost, fixed params; threshold from validation)",
        "99,343 encounters; 14,913 test rows", r.model, r.auroc, f"{r.lo:.3f}-{r.hi:.3f}", r.auprc, r.accuracy, r.sensitivity,
        r.specificity, None,
        "Clinical benchmark: our model beats it by 0.10 AUROC" if r.model.startswith("LACE") else
        ("baseline" if r.model.startswith("M0") else f"vs M0 {r['diff']:+.4f} (CI {r.diff_lo:+.4f} to {r.diff_hi:+.4f}): no gain"),
        "5_results/ablations/lace_comorbidity.csv")

td_ = R / "ablations/tabular_dl.csv"
if td_.exists():
    for _, r in pd.read_csv(td_).iterrows():
        ref = r.model.startswith("CatBoost")
        add(S4, "2026-10-25", "16", "Stronger tabular deep learning (3 seeds averaged; shortened training for CPU time)", "99,343 encounters; 14,913 test rows; threshold from validation",
            r.model, r.auroc_seed_ensemble, None, r.auprc, r.accuracy, r.sensitivity, r.specificity, None,
            "reference" if ref else f"vs CatBoost {r.diff_vs_catboost:+.4f} (CI {r.diff_lo:+.4f} to {r.diff_hi:+.4f})", str(td_))

path = R / "ablations/smote_leakage_demo.csv"
if path.exists():
    for _, r in pd.read_csv(path).iterrows():
        add(S6, "2026-10-24", "11", "SMOTE placement relative to 6-fold CV", "first encounter only; HistGradientBoosting",
            r.protocol, r.auroc, f"+/-{r.auroc_sd:.3f} (folds)", r.auprc, r.get("accuracy"), r.get("sensitivity"),
            r.get("specificity"), None,
            "Reproduces EmbPred30-sized numbers by protocol alone" if r.protocol.startswith("A") else "", str(path))
else:
    for prot, a, sd, ap, ac, nt in [
        ("A SMOTE before CV, scored on all held-out rows (real + synthetic)", .964, .001, None, .942, "Reproduces EmbPred30-sized numbers by protocol alone"),
        ("B SMOTE before CV, scored on real held-out rows only", .640, .004, None, None, ""),
        ("C SMOTE inside training folds only (correct)", .614, .006, .147, None, ""),
        ("D class weights, no SMOTE (our method)", .654, .009, .182, None, "")]:
        add(S6, "2026-10-23", "11", "SMOTE placement relative to 6-fold CV", "first encounter only; HistGradientBoosting; first run",
            prot, a, f"+/-{sd:.3f} (folds)", ap, ac, None, None, None, nt, "console output of first run; rerun with sensitivity queued")

path = R / "ablations/embeddings_tuning.csv"
if path.exists():
    for _, r in pd.read_csv(path).iterrows():
        add(S4, "2026-10-24", "12", "Entity embeddings / extra tuning", "99,343 encounters; fixed test", r.model, r.auroc_seed_ensemble, None,
            r.auprc_seed_ensemble, r.get("accuracy"), r.get("sensitivity"), r.get("specificity"), None, str(r.get("best_config", "")), str(path))
else:
    add(S7, "queued", "12", "Entity-embedding MLP vs tuned one-hot MLP; CatBoost wider search", "99,343 encounters", "pending", note="interrupted by a machine restart; rerunning")
path = R / "main_models/model_comparison_patient_ci.csv"
if path.exists():
    for _, r in pd.read_csv(path).iterrows():
        if r.model.startswith("Baseline"):
            continue
        add(S4, "2026-10-30", "21", "Patient-level confidence intervals; each model refit once with fixed settings (MLP checkpoint bug fixed)",
            "99,343 encounters; 14,913 test rows from 10,499 patients", r.model, r.auroc, f"{r.auroc_lo:.3f}-{r.auroc_hi:.3f} (patient-level)",
            r.auprc, r.accuracy, r.sensitivity, r.specificity, "measured", "Same conclusion as the main table", str(path))

path = R / "ablations/stacking_and_wider_search.csv"
if path.exists():
    sw = pd.read_csv(path)
    for _, r in sw.iterrows():
        if r["diff"] == 0.0 and r.model.startswith("M0"):
            nt = "reference (refit in this run for an identical paired bootstrap)"
        else:
            sig = "significant" if (r.diff_lo > 0 or r.diff_hi < 0) else "not significant (CI crosses 0)"
            nt = f"vs M0 {r['diff']:+.4f} (CI {r.diff_lo:+.4f} to {r.diff_hi:+.4f}): {sig}"
        add(S8, "2026-11-10", "22", "True stacked meta-learner (val-set predictions -> LogReg) and wider CatBoost "
            "random search (30 draws: depth/lr/l2_leaf_reg/bagging_temperature) vs the 9-combo grid in script 05. "
            "Rerun 2026-11-10 after discovering the 2026-11-09 run used a drifted, uncleaned local data/*.csv "
            "(15,219 test rows, missing payer_code/medical_specialty) -- see 8_final_deliverables/test_results.md.",
            "99,343 encounters; 14,913 test rows", r.model, r.auroc, f"{r.lo:.3f}-{r.hi:.3f}", r.auprc, r.accuracy,
            r.sensitivity, r.specificity, None, nt, str(path))

df = pd.DataFrame(rows)
df.insert(0, "id", [f"E{i + 1:02d}" for i in range(len(df))])
df.to_csv(R / "model_variations_log.csv", index=False)

# ---------------------------------------------------------------- markdown
def fmt(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def acc_cell(r):
    if pd.isna(r.accuracy):
        return "n/r"
    return fmt(r.accuracy) + ("*" if r.accuracy_source == "derived" else "")


candidates = df[df.stage.isin([S3, S8])]
best = candidates.loc[candidates.auroc.idxmax()]
best_is_significant_gain = not (best.model.startswith("M0") or pd.isna(best.get("takeaway")) or "not significant" in str(best.takeaway))
md = ["# Model variations log: every experiment we ran", "",
      "One table for all the work behind the final model, in the order we did it, so the path to the best combination is visible. "
      "Generated by `scripts/build_experiment_log.py` from the result files (spreadsheet version: `model_variations_log.csv`). "
      "**Update after each significant experiment** by rerunning that script.", "",
      f"**Current best (by point estimate):** {best.model}, AUROC {best.auroc:.3f}, accuracy {acc_cell(best)}, sensitivity {best.sensitivity:.3f}, "
      f"specificity {best.specificity:.3f} (always-no accuracy 0.889)."
      + ("" if best_is_significant_gain else " **Not a statistically significant improvement over plain CatBoost** "
         "(paired bootstrap CI on the difference crosses zero) -- effectively tied with the simpler model; see stage 8 below."),
      "",
      "Legend: `*` accuracy derived exactly from sensitivity, specificity and the test positive rate; `n/r` not recorded in that run. "
      "AUROC and AUPRC compare across rows only within the same target and cohort. Sensitivity is always shown with accuracy.", ""]
ncols = "| ID | Date | Script | What changed | Cohort / target | Model | AUROC | AUPRC | Acc | Sens | Spec | Take-away |"
for stage, g in df.groupby("stage", sort=True):
    md += [f"## {stage}", "", ncols, "|" + "---|" * 12]
    for _, r in g.iterrows():
        a = fmt(r.auroc) + (f" ({r.auroc_ci})" if isinstance(r.auroc_ci, str) and r.auroc_ci else "")
        md.append(f"| {r.id} | {r.date} | {r.script} | {r.change_vs_previous} | {r.cohort_target} | {r.model} | {a} | "
                  f"{fmt(r.auprc)} | {acc_cell(r)} | {fmt(r.sensitivity)} | {fmt(r.specificity)} | {r.takeaway} |")
    md.append("")

md += ["## Hyperparameter searches (selection on validation only; test scored once)", "",
       "| Model | Search space | Chosen (pipeline v3) | Validation AUPRC | Earlier choices |", "|---|---|---|---|---|",
       "| XGBoost | depth {3,5,7} x trees {200,400,600} x learning rate {0.03,0.05,0.1} (27 fits) | depth 5, 400 trees, lr 0.05 | 0.251 | v2: depth 5, 600, 0.03 (0.243); v1: depth 5, 400, 0.03 |",
       "| CatBoost | depth {4,6,8} x learning rate {0.03,0.05,0.1}, early stopping (9 fits) | depth 4, lr 0.1, 422 iterations | 0.247 | v2: depth 4, lr 0.1, 280 iterations (0.239) |",
       "| Ensemble blend | weights on a 0.1 grid, tuned on validation AUPRC | XGBoost 0.5 + CatBoost 0.5 | 0.250 | v2: 0.7 / 0.3 (0.242); LogReg weight 0 in both |",
       "| MLP | fixed architecture 128-64, dropout 0.3, Adam; early stopping on validation AUPRC | stopped at epoch 23 | n/a | wider search queued (Script 12) |",
       "| Logistic regression | `class_weight=balanced`, C=1 (not tuned) | n/a | n/a | |", ""]
md += ["## What the log shows", "",
       "1. **Cleaning mattered more than modelling.** Fixing the cohort and features moved scores down by about 0.01 (honest numbers), "
       "while every modelling change since has moved them by less than 0.01.",
       "2. **Real gains:** payer/specialty kept (+0.008), categorical IDs (+0.004), patient-history features (+0.007). A true stacked "
       "meta-learner over a fixed-weight blend and a wider CatBoost hyperparameter search were also tried (stage 8); on the corrected, "
       "cleaned cohort neither cleared statistical significance (see item 5).",
       "3. **No gain:** extra model families (all within about 0.015 of each other), SMOTE (worse), three ways of using the '>30 days' label, "
       "LACE/Charlson/3-digit-diagnosis features, and (after correction) the stacked meta-learner and wider CatBoost search.",
       "4. **High scores come from the evaluation protocol:** SMOTE before cross-validation gives 0.96 on the same data (E-rows, stage 6).",
       "5. **Ceiling confirmed, not broken:** stage 8's first run (2026-11-09) reported a significant +0.005 AUROC gain from stacking and "
       "wider tuning, but that run turned out to have used a drifted, uncleaned local data/*.csv (caught by `8_final_deliverables/"
       "test_pipeline_integrity.py`; see `test_results.md`). Rerun on the corrected cohort (2026-11-10): stacking +0.0016 "
       "(CI -0.0011 to +0.0042) and the wider search +0.0011 (CI -0.0024 to +0.0046) -- **neither is statistically significant**. "
       "After 80+ experiments including this dedicated attempt, the honest conclusion is that plain CatBoost (~0.676-0.677) is not "
       "reliably beaten by anything tried. Treat 0.65-0.70 AUROC as the practical ceiling for this dataset, not a plateau to push "
       "through with more tuning.", ""]
(R / "model_variations_log.md").write_text("\n".join(md))
print(f"{len(df)} rows -> {R / 'model_variations_log.md'}")
