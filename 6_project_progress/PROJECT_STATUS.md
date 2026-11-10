# Where we stand: progress report

For the team, not for submission. Plain language, so everyone can follow the
state of the project without reading code.
**Last updated:** 2026-11-09 (see `progress_log.md` for the dated history). Details and reasoning behind every decision:
[`3_dataset/dataset_analysis_report.md`](../3_dataset/dataset_analysis_report.md). Literature and
numbers to compare against: [`4_benchmarking/benchmarking_report.md`](../4_benchmarking/benchmarking_report.md).

## Key dates

| Date | What | Weight |
|---|---|---|
| **Mon 9 Nov** | **Midterm report** (3 pages IEEE, 15+ references, PDF + Word/LaTeX source) | 20% |
| Thu 3 Dec | Course wrap-up (Q&A only) | n/a |
| Fri 11 Dec | Final deliverables: manuscript (6-8 pages), 7-minute video, repo URL, peer evaluation | 45% |
| Fri 11 Dec | Optional bonuses: public repo (+3%), public interface (+2%) | up to +5% |

## The problem in one sentence
Can we predict which diabetic patients will be readmitted within 30 days of
discharge, using only what is known at discharge? (Dataset: Diabetes 130-US
Hospitals, 99,343 usable hospital stays, 11.4% readmitted.)

## Where the results are now

| Model | AUROC* | What it is |
|---|---|---|
| Baseline | 0.500 | Always says "not readmitted" |
| Logistic regression | 0.668 | Simple classic model |
| XGBoost | 0.672 | Boosted decision trees |
| MLP (deep learning) | 0.668 | Small neural network |
| CatBoost | **0.677** | Boosted trees, handles categories natively |
| Ensemble | 0.676 | Blend of XGBoost + CatBoost |

*AUROC: 0.5 = coin flip, 1.0 = perfect. All models are **statistically tied**
(95% intervals roughly 0.65-0.69). Published studies on this dataset land in
the same 0.64-0.68 range, so we are where a correct analysis should be.

**Reporting rule (team decision, 23 Oct):** always report **sensitivity alongside accuracy** (with specificity and AUROC/AUPRC). Full table with accuracy, sensitivity and specificity: `5_results/main_models/model_comparison.csv`. The always-"no" baseline has 88.9% accuracy but 0% sensitivity, which is why accuracy alone is meaningless here.

**What that means in practice:** about 1 in 6 patients the model flags is
really readmitted, and about 45% of real readmissions are missed. Good enough
to rank risk and write a solid report; not good enough for clinical use. The
report must say so.

## What we have done (details and dates in `progress_log.md`)
- **Fixed real problems** in our first pass (a "None" parsing bug, death/hospice patients, ID codes treated as numbers) and re-ran everything; first-pass scores were slightly inflated.
- **Profiled the dataset** and wrote the decision log (`3_dataset/`); kept specialty and payer (+0.008 AUROC).
- **Tested many ideas, none beat CatBoost reliably:** SMOTE (worse), other target definitions and combined targets, patient-history features (+0.007), LACE and comorbidity scores, tuning and entity embeddings, stronger deep models (RealMLP, TabM, FT-Transformer). All 80 variations are in `5_results/model_variations_log.md`.
- **Showed why a published 0.974 AUROC is not credible:** SMOTE before cross-validation reproduces that scale (0.964) with an ordinary model while the same model scores 0.64 on real patients (`4_benchmarking/embpred30_review.md`).
- **Audited the final model:** well calibrated (Brier 0.094); sensitivity lower for men than women because the model separates their readmissions less well (0.519 vs 0.622, 95% CIs do not overlap — statistically significant); AUROC falls with age; learning curve says the ceiling is soft (+0.005 per doubling of data). See `5_results/evaluation/`. **Reconciled 2026-11-09:** three conflicting sex-gap analyses existed; the protocol-compliant one (Script 18, patient-level bootstrap on the deployed model at its real threshold) is authoritative and says the gap IS significant — this reverses the midterm report's "not significant" claim, which used a non-compliant analysis (Script 07, now marked superseded). See `progress_log.md` (2026-11-09) and `2_midterm_report/material_for_final_report.md` for the full correction and the sentence to use in the final manuscript.
- **New best model (2026-11-09):** a true stacked meta-learner / wider-search CatBoost, AUROC 0.679 (patient-level CI 0.663-0.695), a small but statistically significant gain over the earlier best (CatBoost alone, 0.674-0.677). See `5_results/model_variations_log.md` stage 8 (E81-E83).
- **Built the midterm report end-to-end:** `2_midterm_report/Midterm_Report_Claude_Draft.docx` covers all six required sections and 17 references, trimmed to exactly what the rubric scores (no results section — that's saved for the final manuscript in `2_midterm_report/material_for_final_report.md`). Data lead and modelling lead recorded as Siham Khaldoun; interface lead and writing lead still open.
- **Built the demo interface** (`7_interface/`): risk calculator with SHAP explanation, results tab, screenshots, low/high-risk examples, run steps for a local computer and for GitHub Codespaces (nothing to install). Private, not deployed.
- **Wrote the required repo items as drafts:** `MODEL_CARD.md`, `LICENSE` (MIT, proposed), `statements.md` (CRediT and Generative-AI), midterm draft text, presentation script and manuscript outline (`8_final_deliverables/`).
- **Patient-level confidence intervals for every headline model (Script 21):** CatBoost, XGBoost and the ensemble are tied; logistic regression and the MLP are reliably but only slightly lower. See `5_results/main_models/README.md`.
- **Checked reproducibility:** a fresh clone with only the UCI CSV added rebuilds byte-identical splits.
- **Pinned package versions** in `requirements.txt` (the interface has its own light `7_interface/requirements.txt`); the raw data and splits are git-ignored and never committed.

## Settled decisions (use these in the report)
- Predict **per stay**, split by **patient** (no patient in two sets).
- Binary label: readmitted within 30 days.
- Exclude death/hospice discharges.
- Class weights, **not** synthetic data.
- Tune on validation, score the test set once; report AUROC + AUPRC with
  patient-level confidence intervals.

## Still to do

**Code / analysis (Siham, with Claude):** the main analysis is done. Optional: intersectional subgroup checks (sex by age), ethical review before any sex-specific threshold, decide whether history features join the headline model (they assume `encounter_id` is chronological).

**Midterm report (everyone; biggest risk, due Mon 9 Nov):** a complete draft now exists — `2_midterm_report/Midterm_Report_Claude_Draft.docx`, IEEE two-column, all six required sections, 17 references. Still needed from the team:
- [ ] **15+ references, each read by at least one team member.** Still the biggest gap — no reference has been confirmed read by a human yet. Pools: Larissa's folder (`4_benchmarking/LARISSA/`), `pubmed_candidates.md`, Kayla's `references kayla`. Kayla's and Magalie's sections in `team_literature_log.md` are still empty. Once confirmed, remove the yellow reference flags in the draft.
- [ ] Open `Midterm_Report_Claude_Draft.docx` in Word and check the layout renders correctly (page count, table fit).
- [x] Data lead and modelling lead declared (Siham Khaldoun). Interface lead and writing lead still need the team to decide.
- [ ] Each member completes their own part of the CRediT and Generative-AI statements (draft in `statements.md` and `2_midterm_report/material_for_final_report.md`).

**Team / repo hygiene:**
- [ ] Confirm the MIT licence (proposed).
- [ ] Confirm `@AHooshiar` and the TAs have read access to the repo.
- [ ] Everyone commits under their own account (commit history is graded against the contribution statement). So far commits come from Siham, Kayla and one from Magalie; Larissa has none yet.
- [ ] Interface: stays private until the whole project is finished; the team decides then whether to publish it (bonus +2%). Teammates can try it now through Codespaces (`7_interface/TEAM_ACCESS.md`); that route is not yet tested on GitHub.

## Where to find things
```
README.md                                   formal project description (for grading)
1_course_materials/                         specification, dataset list, group assignments
2_midterm_report/                           midterm drafts and draft text for editing (due 9 Nov)
3_dataset/dataset_analysis_report.md        dataset, decisions, trade-offs, bias, results
4_benchmarking/benchmarking_report.md       what others report + our comparison + 21-source table
4_benchmarking/embpred30_review.md          why the 0.974 AUROC claim is not credible
4_benchmarking/LARISSA/                     Larissa's 22-paper reading folder; team_literature_log.md is still empty
5_results/model_variations_log.md          EVERY model variation we ran, in one table
5_results/main_models/                      headline model_comparison.csv, SHAP
5_results/ablations/                        sensitivity, SMOTE, targets, history, ... result tables
6_project_progress/PROJECT_STATUS.md        this file
6_project_progress/progress_log.md          dated log of finished work
6_project_progress/CONTRIBUTING.md          how to use git here
5_results/evaluation/                        calibration, decision curve, subgroup audit, learning curve, sex gap
7_interface/                                demo app, examples, run steps, TEAM_ACCESS.md (Codespaces)
8_final_deliverables/                       presentation script and manuscript outline (drafts)
MODEL_CARD.md, LICENSE                      required repo items (licence proposed)
scripts/00-20_*.py                          profile, preprocess, models, SHAP, ablations, calibration, interface model
```
Run order is in `3_dataset/dataset_analysis_report.md` section 10. The dataset itself is
not in the repo; download it from UCI (instructions in README).
