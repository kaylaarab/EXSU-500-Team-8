# Model card: 30-day readmission risk for diabetic inpatients

**Status:** research prototype for EXSU 500 (Fall 2026, Team 8). **Not a medical device. Not for clinical use.**
**Last updated:** 2026-11-09 · Code: this repository · Licence: MIT (code); dataset CC BY 4.0, not included.

**Note:** this card describes the model actually deployed in `7_interface/` (CatBoost, AUROC 0.677). A stacked meta-learner / wider-tuned CatBoost reaches AUROC 0.679 (a small, statistically significant gain) but has not been deployed to the interface; see `5_results/model_variations_log.md` stage 8 for the full comparison.

## Model
- **Task:** binary classification: will a diabetic inpatient be readmitted to hospital within 30 days of discharge, using information known at discharge.
- **Model:** CatBoost gradient boosting (depth 4, learning rate 0.1, early stopping on validation), with class weighting for the 11% positive rate, followed by Platt calibration fitted on the validation set so the output is a probability. Built by `scripts/17_build_demo_model.py`; saved in `7_interface/model/`.
- **Inputs (48 features):** age band, sex, race, payer, admitting specialty, admission type and source, discharge destination, length of stay, counts of lab tests, procedures, medications and diagnoses, prior-year inpatient/outpatient/emergency visits, HbA1c and glucose results, 23 medication fields and change flags, three diagnosis groups, and three derived features.
- **Output:** a risk percentage, a band (low under 8.2%, moderate, high from 16.1%, set on validation quantiles), and SHAP contributions for one patient.

## Intended use
Teaching and research: to illustrate how a readmission model is built and evaluated, and to compare methods honestly. **Out of scope:** any real clinical, administrative or payment decision; any population other than the one described below.

## Data
- **Source:** UCI *Diabetes 130-US Hospitals for Years 1999-2008* (Strack et al., 2014), 101,766 inpatient encounters from 130 US hospitals, de-identified, CC BY 4.0.
- **Cohort used:** 99,343 encounters from 69,990 patients after excluding 2,423 stays that ended in death or hospice (those patients cannot be readmitted). 11.4% were readmitted within 30 days.
- **Split (made before modelling, by patient so no patient is in two sets):** train 69,519 stays / 48,993 patients; validation 14,911 / 10,498; test 14,913 / 10,499.
- **Not in the data:** dates, hospital identity, vitals, lab values, social factors, follow-up plans, notes.

## Performance (test set, scored once; always-"no" accuracy 0.889)
| Metric | Value |
|---|---|
| AUROC | 0.677 (95% CI about 0.66-0.69) |
| AUPRC | 0.228 (random baseline 0.111) |
| Accuracy at the operating threshold | 0.655 |
| Sensitivity | 0.574 |
| Specificity | 0.666 |
| Precision | about 0.18 |
| Brier score (calibrated) | 0.094 |

At the operating threshold (11.6% predicted risk) about 1 in 5 or 6 flagged patients is truly readmitted and about 4 in 10 readmissions are missed. The high band (top 20%) was readmitted 21.8% of the time, the low band (bottom 40%) 5.6%. Full comparison of models, ablations and benchmarks: `5_results/model_variations_log.md`, `3_dataset/dataset_analysis_report.md`, `4_benchmarking/benchmarking_report.md`.

## Fairness and subgroup results (`5_results/evaluation/`)
- **Sex:** sensitivity 0.62 for women vs 0.52 for men (intervals do not overlap); AUROC 0.694 vs 0.658. The model separates men's readmissions less well (it does not use sex directly); sex-specific thresholds would cut the gap to about 4-5 points at almost no accuracy cost but raise ethical questions (`5_results/evaluation/README.md`).
- **Race:** no reliable difference between Caucasian and African American patients; Hispanic, Asian and Other groups have fewer than 35 readmissions each, so no conclusion is possible.
- **Age:** AUROC falls from 0.72 (under 50) to 0.64 (70 and over); older patients are flagged more often with lower precision.
- **Calibration:** good overall (expected calibration error 0.007, slope 0.96), slightly over-confident at the highest risks.

## Limitations
1. **Old, narrow data:** 1999-2008, US, one coding system (ICD-9), no modern diabetes drugs; practice and coding have changed.
2. **Limited information:** the data lack the variables that most drive readmission, so performance is limited to roughly AUROC 0.68-0.70. A learning curve shows boosted trees still gain about 0.005 AUROC per doubling of training data (an estimated 0.69 with five times as many stays), while simpler models have plateaued; an independent audit on the same cohort found boosted trees within about 0.002 of a measurement ceiling at 0.669.
3. **Population:** about 76% recorded White, 45% aged 70+; small groups cannot be assessed.
4. **Label noise:** readmissions elsewhere, and deaths after discharge, appear as "not readmitted"; the ordering of encounters was assumed (no dates).
5. **No external validation** on another site or period.
6. **Proxy variables:** discharge destination, prior visits and payer reflect social circumstances and access to care.

## Populations on which it should not be used
Children and adolescents (very few in the data), patients who died or went to hospice (excluded from training), non-US or post-2008 practice settings, patients outside inpatient diabetes care, and any individual-level clinical decision.

## Ethical considerations
De-identified public data; no attempt at re-identification. Race is included because it is in the dataset and is used for auditing, not as a recommendation for use. Any deployment would need local validation, clinician oversight, a subgroup audit, and an assessment of whether acting on flags would help or harm.

## How to reproduce
See `README.md` and `3_dataset/dataset_analysis_report.md` section 10: download the data, `pip install -r requirements.txt`, run scripts 01 to 18.
