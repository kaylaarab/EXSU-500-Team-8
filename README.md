# Predicting 30-Day Hospital Readmission in Diabetic Patients

**EXSU 500 — Fundamentals of Artificial Intelligence in Medicine, Fall 2026**
**Team 8 — Predictive analytics & clinical decision support**

Team members: Siham Khaldoun, Larissa Chiu, Magalie Achim, Kayla Arab

## Problem statement

Diabetic patients are at elevated risk of early hospital readmission, which is
both a recognized indicator of care quality and a significant driver of
healthcare cost. We predict whether a diabetic inpatient encounter will result
in a readmission within 30 days of discharge, using information available at
discharge: demographics, admission/discharge details, diagnoses, procedures,
lab results, and medication changes. A reliable risk flag at discharge could
support triage of patients for closer follow-up or discharge planning.

This is **not** intended as a deployable clinical tool. It is a course
exercise in building and honestly evaluating a clinical risk-prediction
pipeline.

## Task type

**Binary classification.** The source label `readmitted` has three levels
(`<30`, `>30`, `NO`); we collapse it to a binary target: readmitted within 30
days (`1`) vs. not (`0`, covering both `>30` and `NO`). This choice is made
because `<30` is the clinically actionable readmission window, and because
collapsing to binary keeps the evaluation protocol (AUROC/AUPRC, thresholded
sensitivity/specificity) simple and defensible. The resulting positive rate is
~11.2%, so the task is meaningfully imbalanced.

## Dataset

**Diabetes 130-US Hospitals for Years 1999-2008**

- Source: UCI Machine Learning Repository, [archive.ics.uci.edu/dataset/296](https://archive.ics.uci.edu/dataset/296)
- Original reference: Strack et al. (2014), "Impact of HbA1c Measurement on
  Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient
  Records," *BioMed Research International*.
- Licence: CC BY 4.0 (UCI ML Repository)
- Access tier: Open — no registration, no agreement required.
- Size: 101,766 inpatient encounters, 50 columns, covering 130 US hospitals
  over 1999–2008.
- **Grouping structure:** encounters are not independent — 101,766 encounters
  belong to only 71,518 unique patients (`patient_nbr`), with up to 40
  encounters for a single patient. All train/validation/test splits are made
  at the **patient level**, not the encounter level, to avoid leakage.
- **Class balance:** `NO` = 54,864 / `>30` = 35,545 / `<30` = 11,357 (~11.2%
  positive after binarization).
- **Known missingness:** `weight` is 96.9% missing (excluded as a feature);
  `medical_specialty` 49.1% missing and `payer_code` 39.6% missing (kept as
  an explicit "Unknown" category and kept in the main pipeline since 23 Oct); `race`, `diag_1/2/3`
  under 2.5% missing.
- `IDS_mapping.csv` (included with the download) decodes the numeric
  `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`
  columns into human-readable categories.

## How to obtain the data

Data is **not** committed to this repository. To reproduce:

```bash
# Download directly from UCI (no account required)
curl -L -o data/diabetic_data.zip \
  https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip
unzip data/diabetic_data.zip -d data/
```

This produces `data/diabetic_data.csv` (~19 MB) and `data/IDS_mapping.csv`.

## Evaluation protocol

- **Cohort:** 99,343 encounters from 69,990 patients after excluding 2,423
  death/hospice discharges (those patients cannot be readmitted).
- **Splits** (made before any modelling, seed 42): train 69,519 encounters /
  48,993 patients, validation 14,911 / 10,498, held-out test 14,913 / 10,499.
  Split at the **patient** level (`patient_nbr`): no patient's encounters appear
  in more than one split (asserted in `scripts/01_preprocess.py`). Positive rate
  is 11.4% / 11.7% / 11.1%.
- **Tune on validation, report on test.** The test set is scored once per model.
- **Metrics:** AUROC and AUPRC (primary, given class imbalance); sensitivity and
  specificity at a stated threshold of 0.5. Accuracy is always reported **together with** sensitivity and specificity, never alone: the always-"no" baseline scores 88.9% accuracy with 0% sensitivity.
- **Uncertainty:** 95% bootstrap confidence intervals on the test set. The main
  table uses encounter-level resampling; `scripts/07_sensitivity.py` adds
  patient-level (cluster) bootstrap intervals, which are slightly wider.
- **Interpretability:** SHAP feature importance (`5_results/main_models/shap_summary.png`).
  Top drivers: prior-year inpatient visits, then discharge destination.
- **Imbalance handling:** class weights. SMOTE-NC was tested as an ablation and was
  clearly worse on real test data (`scripts/09_smote_ablation.py`). No synthetic
  data is used for training or evaluation.
- Rationale for each decision and its trade-offs: `3_dataset/dataset_analysis_report.md`.

## Models and current results

1. **Baseline**: majority-class predictor.
2. **Classical ML**: regularised logistic regression (`class_weight=balanced`).
3. **Classical ML**: XGBoost, tuned on validation (depth, trees, learning rate).
4. **Classical ML**: CatBoost with native categorical handling and engineered features.
5. **Deep learning**: MLP on one-hot features.
6. **Ensemble**: validation-tuned blend of XGBoost and CatBoost.

All metrics below are at a fixed 0.5 probability threshold, from the patient-level-CI run
([`5_results/main_models/model_comparison_patient_ci.csv`](5_results/main_models/model_comparison_patient_ci.csv)),
matching the manuscript's Table I. (An earlier, slightly different XGBoost refit in
[`5_results/main_models/model_comparison.csv`](5_results/main_models/model_comparison.csv) gave
AUROC 0.672 / accuracy 0.682 / sensitivity 0.539 / specificity 0.700 — within normal run-to-run
variance for an independently retrained model, not a threshold difference, but superseded by the
numbers below for consistency with the manuscript.)

| Model | AUROC | AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Baseline (majority-class) | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic Regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost | 0.673 | 0.224 | 0.673 | 0.561 | 0.687 |
| MLP | 0.668 | 0.222 | 0.616 | 0.617 | 0.616 |
| CatBoost | 0.676 | 0.227 | 0.672 | 0.555 | 0.686 |
| Ensemble | 0.676 | 0.228 | 0.674 | 0.560 | 0.688 |

95% CI on the best models is roughly +/-0.015; the models are statistically close.
`medical_specialty` and `payer_code` are kept (they add about +0.008 AUROC, paired 95% CI
+0.005 to +0.011). Restricting to each patient's first
encounter gives about 0.652, and patient-history features add about +0.007 (paired 95% CI +0.003 to +0.011).

**Further experiments (all in [`5_results/model_variations_log.md`](5_results/model_variations_log.md), 80 variations; none beats CatBoost reliably):**
tuned MLP 0.674, entity embeddings 0.668, RealMLP 0.669, TabM 0.674, FT-Transformer 0.675 (AUROC); different target definitions and combined targets;
LACE clinical score alone 0.575 and as an added feature (no gain); SMOTE worse than class weights; adding history features +0.007.
A learning curve shows boosted trees still gain about +0.005 AUROC per doubling of data (0.640 at 5% to 0.677 at 100%), so the ceiling is soft, not a wall.
The final model (CatBoost with Platt calibration) is well calibrated (Brier 0.094, expected calibration error 0.007); at its validation-chosen threshold sensitivity is 0.574, specificity 0.666 and accuracy 0.655 on the test set, against 0.889 accuracy and 0 sensitivity for always answering "no". Sensitivity is lower for men (0.52) than women (0.62) and AUROC falls with age (0.72 under 50, 0.64 at 70+); see [`5_results/evaluation/`](5_results/evaluation/README.md).
These are the current numbers; the manuscript will use the final reruns.

## Repository structure

Numbered folders, one per topic. Open the folder you need; each has its own README.

```
README.md                 this file (project description for grading)
requirements.txt          pinned packages
1_course_materials/       course specification, dataset list, group assignments (PDFs)
2_midterm_report/         midterm drafts (Word) and draft text for editing (due 9 Nov)
3_dataset/                dataset_analysis_report.md: data quality, decisions, trade-offs, bias
4_benchmarking/           what other work on this dataset reports, and how we compare
                          (benchmarking_report.md, EmbPred30 review, sources table, downloaded papers)
5_results/                all tables and figures; start with model_variations_log.md (every model variation we ran)
7_interface/              demo web app (Streamlit): risk calculator + results, examples, run steps (local or GitHub Codespaces); private, not deployed, not for clinical use
8_final_deliverables/     manuscript outline and 7-minute presentation script (drafts)
6_project_progress/       PROJECT_STATUS.md (current status + to-dos), progress_log.md, statements.md, CONTRIBUTING.md
.devcontainer/           one-click GitHub Codespaces setup for the interface
scripts/                  all code, numbered in run order
data/                     raw CSV + generated splits (git-ignored; never committed)
```

Scripts: `00_profile` (profile raw data) · `01_preprocess` (clean, exclude death/hospice,
patient-level split) · `02_train_evaluate` (baseline, LogReg, XGBoost) · `03_interpretability`
(SHAP) · `04_deep_learning` (MLP) · `05_catboost` · `06_ensemble` · `07_sensitivity`
(specialty/payer, first-encounter, patient-level CIs) · `08_trajectory_features` (patient history) ·
`09_smote_ablation` · `10_hospice_death_ablation` · `11_smote_leakage_demo` ·
`12_embeddings_tuning` · `13_target_definitions` · `14_combined_targets` · `15_lace_and_comorbidity` · `16_tabular_dl` (RealMLP, TabM, FT-Transformer) · `17_build_demo_model` (final model for the interface) · `18_calibration_subgroups` · `19_learning_curve` · `20_sex_gap` · `21_patient_level_cis` (patient-level confidence intervals for every headline model, `5_results/main_models/README.md`).

Run order: `00` (optional) then `01` to `06`; `07` to `20` are analyses and the interface model (full list in `3_dataset/dataset_analysis_report.md` section 10). Run all
from the repo root after downloading the data per above and installing the pinned
`requirements.txt`.

**Required repository items:** model card ([`MODEL_CARD.md`](MODEL_CARD.md)); licence ([`LICENSE`](LICENSE), MIT, proposed: team to confirm); CRediT and Generative-AI statements (drafts in [`6_project_progress/statements.md`](6_project_progress/statements.md), to be completed by each member); calibration and subgroup analysis ([`5_results/evaluation/`](5_results/evaluation/README.md)).

## Use of Generative AI

Claude (Anthropic) was used throughout as a coding and analysis assistant, run from Siham Khaldoun's account: pipeline code, ablations, the interface, literature searches and drafts of the documentation. All results come from scripts in this repository, and every cited source is to be read by a team member before it is used. The draft statement, with each member's own part still to be completed, is in [`6_project_progress/statements.md`](6_project_progress/statements.md).
