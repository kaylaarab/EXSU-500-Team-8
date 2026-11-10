# Material for the Final Manuscript (not in the midterm report)

Removed from `Midterm_Report_CLAUDEgenerated.docx` on 2026-10-30 because it is
not required by the midterm rubric (problem statement, literature synthesis,
dataset description, evaluation protocol, candidate models, work plan/roles/
risks — no results section is scored at midterm). Everything below is
finished work, kept here so it drops straight into the final manuscript
(due Dec 11; 6-8 pages IEEE, needs ≥2 models, calibration/interpretability
artifact, variability quantified, repo statements).

## Table I. Preliminary Test Results (14,913 stays; always-"no" accuracy = 0.889)

| Model | AUROC | AUPRC | Acc. | Sens. | Spec. |
|---|---|---|---|---|---|
| Baseline (majority) | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic Regression | 0.668 | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost | 0.672 | 0.226 | 0.682 | 0.539 | 0.700 |
| CatBoost | 0.677 | 0.228 | 0.671 | 0.555 | 0.685 |
| MLP (deep learning) | 0.668 | 0.222 | 0.616 | 0.617 | 0.616 |
| Ensemble (blend) | 0.676 | 0.228 | 0.674 | 0.560 | 0.688 |
| LACE score alone | 0.575 | 0.139 | 0.642 | 0.435 | 0.668 |

## Findings so far

1. All models are within about 0.015 AUROC of each other; pairwise differences
   fall inside bootstrap confidence intervals.
2. Machine learning adds roughly 0.10 AUROC over the LACE clinical score alone.
3. Real gains came from data handling — keeping payer code and medical
   specialty (+0.008 AUROC), encoding administrative ID fields as categories
   rather than numbers, and adding patient-history features (+0.007) — not
   from model choice.
4. Calibrated probabilities are reliable (Brier score 0.094, expected
   calibration error 0.007).
5. **Sensitivity-by-sex subgroup analysis — corrected 2026-11-09, see below.
   The midterm report's claim (not significant) was based on a weaker
   analysis and should NOT be reused in the final manuscript.**

## Sensitivity by sex — corrected analysis (supersedes the midterm claim)

**Correction (2026-11-09):** the midterm report states this gap is "not
statistically significant," based on `scripts/07_sensitivity_by_sex.py`.
That script does not follow the evaluation protocol stated in the midterm
report itself (patient-level bootstrap resampling): it bootstraps
individual test *rows* rather than patients, uses a fixed 0.5 threshold on
raw (uncalibrated) model scores rather than the model's actual operating
threshold, and retrains fresh models rather than using the calibrated
model shipped in `7_interface/model/`. The repo's protocol-compliant
analysis — `scripts/18_calibration_subgroups.py`, patient-level bootstrap
(500 resamples grouped by `patient_nbr`), on the deployed calibrated
CatBoost model, at its real operating threshold (11.6%) — gives a
different, authoritative answer, already saved in
`5_results/evaluation/subgroup_performance.csv` and correctly reflected in
`MODEL_CARD.md`:

| Group | n | Sensitivity | 95% CI (patient-level) |
|---|---|---|---|
| Female | 8,079 | 0.622 | [0.580, 0.662] |
| Male | 6,834 | 0.519 | [0.476, 0.555] |

**Conclusion: the CIs do not overlap — the gap IS statistically
significant** for the deployed model at its real operating threshold.
AUROC also differs: 0.694 (female) vs. 0.658 (male). This reverses the
midterm report's conclusion; `scripts/07_sensitivity_by_sex.py` should be
treated as superseded, not as a second valid estimate.

Suggested sentence for the final manuscript's Limitations/Fairness section:
> "At the model's operating threshold, sensitivity for predicting 30-day
> readmission was higher for women than men (0.622 [0.580, 0.662] vs.
> 0.519 [0.476, 0.555], patient-level bootstrap 95% CIs, non-overlapping),
> and AUROC was correspondingly higher for women (0.694 vs. 0.658). The
> model does not use sex as an input, so the gap likely reflects
> differences in how the available features associate with readmission by
> sex. A preliminary check of sex-specific thresholds (`scripts/
> 20_sex_gap.py`) narrows the sensitivity gap at a small cost in overall
> accuracy; whether to adopt sex-specific thresholds, and the fairness
> framework for doing so, is flagged for human/ethical review before any
> deployment, not resolved by this analysis."

## Fuller Risks paragraph (midterm version was trimmed)

(a) Limited information: performance is held to roughly 0.68-0.70 AUROC by
the recorded variables (a learning curve shows only about +0.005 AUROC per
doubling of training data for boosted trees, and none for simpler models),
so the project's contribution is methodological rigour, not raw accuracy.
(b) Data age and single-country origin limit generalisability; no external
validation set is available. (c) Small subgroups cannot be assessed
reliably. (d) Time: verified contributions and read references from every
member are needed before the deadlines. (e) Metric misreading: mitigated by
always reporting sensitivity alongside accuracy.

## Draft CRediT contribution statement

Conceptualization — S. Khaldoun, K. Arab; Data curation, Methodology,
Software, Validation, Formal analysis, Investigation — S. Khaldoun;
Investigation (literature) — K. Arab, L. Chiu (in progress), M. Achim
(TBD); Writing, original draft — S. Khaldoun, K. Arab; Writing, review and
editing — TBD; Project administration — S. Khaldoun, K. Arab.

**Each team member must review and correct their own line before
submission.** Full draft table also in `6_project_progress/statements.md`.

## Draft Use of Generative AI statement

Generative AI (Claude Code, Anthropic) was used, directed by team members,
for: drafting and debugging the data preprocessing, model training,
evaluation, ablation, calibration, and interface code; drafting analysis
documents and report text from computed results and retrieved literature;
and literature search/retrieval. AI-generated literature summaries are
screening aids only, not evidence of human reading.

**The team must verify this statement is accurate and complete before
submission.** Full draft also in `6_project_progress/statements.md`.
