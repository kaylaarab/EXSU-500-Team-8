# Manuscript outline (6-8 pages, IEEE two-column, references included; due Fri 11 Dec 2026)

**Status:** DRAFT outline mapping each required section to repository material, as of 2026-10-25. The midterm draft (`../2_midterm_report/midterm_draft_for_editing.md`) can be extended into the manuscript; overlap is permitted.

Required structure (spec section 7.1): title and authors; abstract (max 250 words); introduction and clinical motivation; related work; data; methods; results (must include the model comparison table **with uncertainty**, the interpretability or calibration artifact, and at least one figure that is our own work); discussion including limitations (what the model cannot do and what would be required before clinical use); conclusion; CRediT statement; Generative-AI statement; references.

| Section | Source material in this repo | Status |
|---|---|---|
| Abstract | write last from results | not started |
| Introduction, motivation | midterm draft section I | draft |
| Related work | midterm draft II; `../4_benchmarking/benchmarking_report.md`; EmbPred30 review | draft; references need human reading |
| Data | `../3_dataset/dataset_analysis_report.md` sections 1-2 | draft |
| Methods: cohort, splits, features | dataset report section 3-4; `../scripts/01_preprocess.py` | draft |
| Methods: models, tuning, calibration | `../5_results/model_variations_log.md` (hyperparameter table); `../MODEL_CARD.md` | draft |
| Results: model comparison with CIs | `../5_results/main_models/model_comparison.csv`, `../5_results/ablations/` (patient-level bootstrap CIs, paired differences) | needs a single final table with CIs for every model |
| Results: ablations | log stages 4-5: payer/specialty, history, SMOTE, targets, LACE, combined labels, tuning, tabular deep models | draft |
| Results: calibration, interpretability | `../5_results/evaluation/`, `../5_results/main_models/shap_summary.png` | ready |
| Results: subgroup audit | `../5_results/evaluation/subgroup_performance.csv` | ready (first audit) |
| Own figure (required) | calibration + decision curve; SHAP; leakage-demo bar chart (to make) | partly ready |
| Discussion and limitations | model card limitations; benchmark comparison; sex gap; ceiling argument | draft |
| Conclusion | write last | not started |
| CRediT and AI statements | `../6_project_progress/statements.md` | drafts; members to confirm |
| References | 15+ (the final list should be 20-30), all read | **not started by humans** |

## Open scientific items before the manuscript
1. ~~One final model-comparison table with patient-level CIs~~ done (Scripts 21, 22): see `../5_results/model_variations_log.md` stages 4 and 8. Current best: a true stacked meta-learner / wider-search CatBoost, AUROC 0.679 (patient-level CI 0.663-0.695), a small but statistically significant gain over the single-CatBoost baseline (0.674).
2. Decide whether patient-history features join the headline model (relies on unverifiable encounter ordering).
3. **Sex gap: corrected 2026-11-09, use this version, not the midterm report's.** The midterm report says the sex-sensitivity gap is not significant; that was based on a non-protocol-compliant analysis (Script 07: row-level bootstrap, arbitrary threshold, non-deployed model). The protocol-compliant analysis (Script 18: patient-level bootstrap, the deployed calibrated model, its real operating threshold) finds the gap IS significant (female sensitivity 0.622 [0.580,0.662] vs. male 0.519 [0.476,0.555]). See `../2_midterm_report/material_for_final_report.md` for the full correction and suggested manuscript sentence. Remaining: intersectional checks and ethical framing of the sex-specific-threshold mitigation (Script 20).
4. ~~Learning curve~~ done (Script 19): the ceiling is soft, +0.005 per doubling for trees; present it as a limitation, not a proven hard cap.
5. A clean-clone reproducibility run.
