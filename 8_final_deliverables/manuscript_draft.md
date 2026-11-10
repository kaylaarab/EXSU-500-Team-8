# Manuscript draft (6-8 pages IEEE two-column; due Fri 11 Dec 2026)

**Status: DRAFT, 2026-11-09.** Built from the midterm report plus everything unlocked since (final results table with patient-level CIs, the corrected sex-gap finding, calibration/interpretability artifacts, the benchmarking comparison). Numbers are sourced from `5_results/`, `MODEL_CARD.md` and `4_benchmarking/benchmarking_report.md` as of this date — **refresh before submission if any script is rerun.** Items still needing a human: confirm every reference has been read (`4_benchmarking/team_literature_log.md`), insert the three required figures (calibration/decision curve, SHAP summary, model-comparison bar chart — all already generated in `5_results/evaluation/` and `5_results/main_models/`), each member reviews their own CRediT line, word-count/page check once converted to the IEEE Word template, and the abstract should be the last thing re-read since it's written from the rest.

---

# Predicting 30-day hospital readmission in patients with diabetes: a comparative machine learning study

Kayla Arab, Siham Khaldoun, Magalie Achim, Larissa Chiu — Faculty of Medicine and Health Sciences, McGill University, Montreal, QC

## Abstract (target ≤250 words)

Hospital readmission within 30 days is a costly, common outcome for patients with diabetes. We built and compared statistical and machine-learning models to predict 30-day readmission from routine clinical data, emphasizing leak-free evaluation and honest uncertainty quantification — concerns under-addressed in several published benchmarks on this dataset. Using the UCI Diabetes 130-US Hospitals dataset (99,343 encounters, 69,990 patients, 11.4% readmitted within 30 days, after excluding death/hospice discharges), we split by patient to prevent leakage and compared a majority-class baseline, logistic regression, XGBoost, CatBoost, a multilayer perceptron, an ensemble, and — after the midterm stage — a stacked meta-learner and a more extensively tuned CatBoost. The best model (CatBoost) reached AUROC 0.677 (95% patient-level bootstrap CI 0.660–0.693), consistent with every credible published approach to this dataset; the post-midterm refinements did not significantly improve on it (paired CIs cross zero). Calibration was good (Brier score 0.094, ECE 0.007), and SHAP and decision-curve analyses support the model's use as a triage aid, not a diagnostic tool. A patient-level subgroup audit found sensitivity significantly higher for women than men (0.622 vs. 0.519, non-overlapping CIs), requiring ethical review before deployment. We also reproduce, as a cautionary case study, how oversampling applied before cross-validation can manufacture an AUROC of 0.96 on this data — nearly matching a 0.974 literature claim — showing that evaluation protocol, not model sophistication, is the dominant lever on reported performance here.

## I. Introduction and Clinical Significance

Hospital readmission within 30 days of discharge is an important healthcare quality and cost measure [1], [2]. Patients with diabetes are particularly vulnerable, with reported readmission rates of 14–20%, roughly twice those of patients without diabetes [1]. Factors associated with readmission include comorbidity burden, socioeconomic characteristics, and prior hospitalizations [1]. Discharge planning, timely follow-up, and medication management can reduce avoidable readmissions [1], [2], but targeting these interventions requires identifying at-risk patients before they leave hospital.

Machine-learning approaches to this problem have shown both promise and pitfalls [3], [4]. A central motivation of this project is that reported performance on this task varies enormously across the published literature — from AUROC near 0.58 to a claimed 0.974 [4]–[6], [20] — and that gap is, in our analysis, driven far more by evaluation methodology (how data leakage, class imbalance, and patient-level dependence are or are not handled) than by model choice. We therefore aim to (1) develop and honestly evaluate several models of increasing complexity for 30-day readmission prediction in diabetic inpatients, (2) quantify uncertainty and subgroup performance rather than reporting a single headline number, and (3) use this dataset as a case study in how evaluation protocol shapes reported results, with direct relevance to clinical AI reporting standards such as TRIPOD+AI [16].

## II. Literature Review and Related Work

Prior studies identify consistent predictors of 30-day readmission in diabetes: age, sex, race, prior inpatient/outpatient/emergency visits, comorbidity burden, admission type, length of stay, discharge disposition, procedure and lab counts, and diabetes-medication use and dosage changes [4]–[7]. Glycemic markers — HbA1c and serum glucose testing — have also been studied as indicators of diabetes management quality [5], [7].

Modeling approaches on the UCI Diabetes 130-US Hospitals dataset [7] span logistic regression, random forests, XGBoost, and deep neural networks [4]–[6]. Emi-Johnson and Nkrumah [4] report AUROC 0.667 (XGBoost), 0.642 (logistic regression) and 0.579 (DNN) on a stratified 80/20 split; we verified against their full text that their reported 0.84–0.89 "recall" in Table 1 does not reconcile with Table 2's class-1 recall of 0.002–0.143 (LogReg 0.016, random forest 0.002, XGBoost 0.017, DNN 0.143), and that their 0.89 accuracy equals the always-"no" baseline at this dataset's prevalence — a reminder that accuracy and headline "recall" figures can mask near-zero sensitivity to the minority (readmitted) class in imbalanced clinical prediction tasks [16]. Tan et al. [3] report an external AUROC of 0.633–0.671 for logistic regression in elderly Type 2 diabetes patients, decreasing from 0.769 measured internally — evidence that performance on this task degrades under distribution shift even within the same broad modeling approach.

An outlier in the literature is EmbPred30 [20], which reports AUROC 0.974 and accuracy 95.2% using categorical embeddings. We obtained and read the full text: its preprocessing pipeline lists SMOTE oversampling to a 1:1 class ratio as a step prior to 6-fold cross-validation, with no statement that oversampling was confined to training folds, and feature selection by importance after training, also undescribed as fold-restricted. Both are recognized sources of evaluation leakage [16]. We treat this as a cautionary result rather than a performance target (Section V-D).

Chowdhury et al. [21], using the identical cleaned cohort we use (99,343 encounters, 69,990 patients, prevalence 0.1139), report gradient boosting AUROC 0.669 and argue this is close to the ceiling set by the variables recorded in this dataset rather than by model capacity — a claim we test directly with a learning-curve analysis (Section V-C). Gupta and Choudhary [22], applying SMOTE correctly (training folds only), report a best ROC-AUC of only 0.608, consistent with our own finding that synthetic oversampling underperforms class weighting on this task (Section V-D).

Given approximately 11% of encounters result in readmission, overall accuracy is a misleading metric: a model predicting "no readmission" for every patient reaches 88.9% accuracy with 0% sensitivity [16]. We therefore report AUROC and the area under the precision-recall curve (AUPRC) as primary metrics [17], alongside sensitivity and specificity at a stated threshold, for every model. A second recurring risk in this literature is data leakage from encounters of the same patient appearing in both training and test sets [16]; our evaluation protocol addresses this directly (Section IV).

## III. Data

We use the Diabetes 130-US Hospitals dataset (UCI Machine Learning Repository #296; Strack et al. [7]), covering 101,766 inpatient encounters across 130 US hospitals, 1999–2008. After excluding 2,423 encounters (2.4%) ending in death or hospice discharge — these patients cannot be readmitted — the analytical cohort is 99,343 encounters from 69,990 patients, 11.4% readmitted within 30 days. A total of 23.5% of patients had more than one stay, accounting for 46% of rows (up to 40 stays per patient), so rows are not independent; this motivates our patient-level split (Section IV).

Missingness is substantial for some fields: weight (96.9%), HbA1c result (83.8%), medical specialty (49.1%). The recorded population is 76.5% White, 19.3% Black, 2.0% Hispanic, 1.5% Other, and 0.6% Asian; approximately 45% of patients were 70 or older. The dataset uses ICD-9 diagnosis codes reflecting 1999–2008 coding practice and does not include dates, hospital identity, vital signs, exact lab values, social factors, or clinical notes — limitations we return to in Section VI.

Features include demographics, admission/discharge characteristics, length of stay, counts of labs/procedures/medications/diagnoses, prior-year utilization (outpatient/emergency/inpatient visit counts), HbA1c and glucose test results, and 23 diabetes-medication fields with dosage-change flags. Three ICD-9 diagnosis fields are grouped into nine broad clinical categories, and we derive three features: total prior-year healthcare visits, number of medication changes, and an indicator combining HbA1c testing with a treatment change. Model training is feasible on standard CPU resources.

## IV. Evaluation Protocol

The dataset is split 70/15/15 into training (69,519 encounters), validation (14,911), and test (14,913) sets, with every encounter from a given patient assigned to a single split to prevent leakage. Model parameters are learned on the training set; hyperparameters, classification thresholds, and calibration are selected on the validation set; the test set is scored once, after all modeling decisions were finalized.

Given the class imbalance, AUROC and AUPRC [17] are the primary metrics; accuracy, sensitivity, and specificity are reported alongside at a stated threshold, with the majority-class baseline (88.9% accuracy, 0% sensitivity) reported for reference. Uncertainty is quantified with 95% confidence intervals from patient-level bootstrap resampling (patients, not rows, are resampled, preserving within-patient encounter structure), including paired comparisons between models where a difference is claimed. Class weighting addresses the imbalance for training; synthetic oversampling (SMOTE-NC) was evaluated as an alternative and found to underperform (Section V-D). Calibration is assessed with reliability curves, Brier score, and expected calibration error; SHAP values [18] and decision-curve analysis assess interpretability and potential clinical usefulness. Performance is further evaluated across age, sex, and race subgroups.

## V. Results

### A. Model comparison

Table I reports test-set performance for all headline models, with patient-level bootstrap 95% CIs (1,000 resamples). All models cluster within about 0.01–0.015 AUROC of each other; pairwise differences for the five original candidates fall inside their respective confidence intervals, i.e., are not distinguishable at this sample size. The multilayer perceptron attains the highest raw sensitivity but at markedly lower accuracy and specificity, reflecting a different point on the sensitivity/specificity trade-off rather than a stronger model.

**Table I. Test-set performance, patient-level bootstrap 95% CI, n = 14,913 encounters / 10,499 patients (always-"no" accuracy = 0.889)**

| Model | AUROC (95% CI) | AUPRC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Baseline (majority) | 0.500 | 0.111 | 0.889 | 0.000 | 1.000 |
| Logistic Regression | 0.668 (0.652–0.683) | 0.214 | 0.670 | 0.547 | 0.685 |
| XGBoost (tuned) | 0.673 (0.657–0.689) | 0.224 | 0.673 | 0.561 | 0.687 |
| CatBoost | 0.676 (0.660–0.692) | 0.227 | 0.672 | 0.555 | 0.686 |
| Ensemble (fixed-weight blend) | 0.676 (0.660–0.692) | 0.228 | 0.674 | 0.560 | 0.688 |
| MLP | 0.668 (0.652–0.686) | 0.222 | 0.616 | 0.617 | 0.616 |
| Stacked meta-learner (post-midterm)* | 0.677 (0.661–0.694) | 0.230 | 0.699 | 0.524 | 0.721 |
| CatBoost, wider search (post-midterm)* | 0.677 (0.660–0.694) | 0.229 | 0.669 | 0.563 | 0.683 |

*Accuracy/sensitivity/specificity for the five original candidates use a fixed 0.5 threshold; the two post-midterm rows instead use the Youden (max sensitivity+specificity−1) threshold from the validation ROC curve, matching the operating-threshold convention used for the deployed model. AUROC/AUPRC, the comparison this paper's conclusions rest on, are threshold-independent and directly comparable across all seven rows; accuracy/sensitivity/specificity across the dashed line above are not. **Neither post-midterm row is a statistically significant improvement over plain CatBoost** — see Section V-B.

### B. Beyond the midterm model

After the midterm, we tested two further levers. First, a true stacked meta-learner — a logistic regression trained on the three base models' validation-set predictions, rather than the fixed blend weights used for the ensemble row above. Second, a 30-draw random search over CatBoost's depth, learning rate, L2 regularization, and bagging temperature (versus a 9-combination grid used for the midterm model). An initial run (2026-11-09) reported both as small, statistically significant gains over a freshly-fit CatBoost reference; a subsequent automated test (`8_final_deliverables/test_pipeline_integrity.py`) caught that this run had used a local `data/*.csv` that had drifted from the documented, cleaned cohort (it held 15,219 rows including death/hospice discharges, rather than the correct 14,913), so the result was discarded and the experiment rerun on the corrected data. On the corrected cohort, the stacked meta-learner reached AUROC 0.677, a paired gain of only +0.0016 over the same-run CatBoost reference (95% CI −0.0011 to +0.0042, crossing zero), and the best configuration from the wider search (depth 4, learning rate 0.168, L2 4.0, bagging temperature 1.48) reached AUROC 0.677, a paired gain of +0.0011 (95% CI −0.0024 to +0.0046, also crossing zero). **Neither gain is statistically significant** — unlike every other modeling change tried (Section V-D), none of which cleared this bar either. We report both the initial, since-corrected result and this honest null finding transparently, as a reminder that an automated data-integrity check, not just model evaluation, is part of a trustworthy pipeline.

### C. Is there a ceiling, and is it information or modeling?

A learning curve (CatBoost, 5%–100% of training data) shows AUROC rising from 0.640 to 0.677, still gaining roughly +0.005 per doubling of data at full size, while logistic regression plateaus near 0.668. Extrapolating (indicatively) suggests roughly 0.683 at 2× and 0.69 at 5× the current training set. This is consistent with Chowdhury et al.'s [21] ceiling estimate of 0.669 on the identical cohort (within 0.002–0.010 of our results depending on model), and supports a specific reading: the limit on this task is set predominantly by which variables are recorded (no social factors, no follow-up plan, no vital signs, historical ICD-9 coding), not by insufficient model capacity — the ceiling is soft, not a hard wall, but it is low, and more data alone would close it only slowly.

### D. What did not help

Beyond the stacking and tuning gains above (Section V-B), none of the following moved AUROC outside the noise of a paired bootstrap comparison: synthetic oversampling (SMOTE-NC underperformed class weighting by 0.03–0.12 AUROC depending on model and oversampling ratio); entity embeddings versus one-hot encoding for an MLP; three alternative ways of exploiting the dataset's ">30 days" readmission label to help predict the <30-day target; LACE index and Charlson comorbidity index features (clinically motivated but redundant with features already present); and stronger tabular deep-learning architectures (RealMLP, TabM, FT-Transformer), consistent with recent evidence that tree-based models remain competitive with deep learning on typical tabular data [19]. Real, reproducible gains instead came from data handling: retaining `payer_code` and `medical_specialty` as categorical features rather than dropping them (+0.008 AUROC), encoding administrative ID fields as categories rather than raw integers (+0.004), and adding patient-history features — prior stay count and prior readmission label (+0.007). The full experiment log (83 logged runs) is in the project repository.

### E. Calibration and interpretability

The deployed model's calibrated probabilities are well-aligned with observed outcomes: Brier score 0.094 (vs. 0.099 for a model that always predicts the base rate, and 0.219 for the uncalibrated raw score), expected calibration error 0.007, and a calibration slope of 0.96 (1.0 is ideal), indicating slight over-confidence only at the highest predicted risks. A decision-curve analysis shows the model provides positive net clinical benefit, relative to both "treat everyone" and "treat no one" defaults, across a usable range of risk thresholds. SHAP value analysis [18] attributes individual predictions to contributing features (e.g., prior inpatient visit count, length of stay, medication count), supporting the interface's per-patient explanation and a plausible clinical narrative for the model's reasoning, though SHAP attributions describe association within the model, not causal effect.

### F. Subgroup audit (corrected finding — see note)

> **Note on this result:** an earlier analysis of this subgroup gap, reported in our midterm submission, used a methodology inconsistent with the protocol in Section IV (row-level rather than patient-level bootstrap resampling, an arbitrary 0.5 threshold rather than the model's real operating threshold, and a freshly retrained rather than the deployed model) and concluded the gap was not significant. The patient-level, protocol-compliant analysis below — using the deployed, calibrated model at its real operating threshold — reverses that conclusion. We report the corrected result here, as the midterm submission could not be revised after the error was found.

At the model's operating threshold (11.6% predicted risk), sensitivity is significantly higher for women than men: 0.622 (95% CI 0.580–0.662, n = 8,079) versus 0.519 (95% CI 0.476–0.555, n = 6,834); AUROC is correspondingly higher for women (0.694 vs. 0.658). The model does not take sex as a direct input (SHAP importance rank 23 of 49 features), so the gap likely reflects how the available features associate differently with readmission by sex. A preliminary test of sex-specific operating thresholds narrows the sensitivity gap from about 10 to about 4.5 points at a small (−0.004) accuracy cost, with the residual gap's confidence interval including zero; whether to adopt such a threshold, and under what fairness framework, is an ethical and clinical-governance question we flag for human review, not one this analysis resolves. By race, no reliable difference was found between the two largest groups (Caucasian, African American); smaller race groups have too few readmissions (<35) to assess. By age, AUROC falls from approximately 0.72 (under 50) to 0.64 (70 and over), with older patients flagged more often at lower precision.

## VI. Discussion

### A. What the evidence supports

Across model families, honest (leak-free, patient-level) evaluation on this dataset consistently lands in the 0.65–0.70 AUROC range, matching the credible subset of the published literature we reviewed [3], [4], [21], [22] and roughly 0.10 AUROC above a standard clinical risk score (LACE alone, AUROC 0.575; we did not validate LACE against its original index's external cohorts, so this comparison is indicative, not a formal benchmark). Machine learning adds real but modest value over a simple clinical heuristic; model family choice within "reasonable" options (logistic regression through gradient boosting to deep tabular networks) adds comparatively little once data handling (categorical encoding, retained administrative fields, patient-history features) is done carefully. This is consistent with, and adds a reproducible local case study to, the broader finding that tree-based models remain strong baselines for tabular clinical data [19].

### B. A methodological case study: how 0.68 becomes 0.97

We separately reproduced the mechanism behind the EmbPred30 outlier [20] on a comparable cohort: applying SMOTE oversampling before cross-validation and scoring on the resulting mixed real/synthetic held-out folds reaches AUROC 0.964 — a near-match to the published 0.974 — using nothing more than a standard gradient-boosted model, no embeddings required. Scoring the identical model on only the real held-out rows from that same (leaky) split drops the score to 0.640; applying SMOTE correctly, confined to training folds only, with real rows scored, gives 0.614, in line with Gupta and Choudhary's independently reported 0.608 [22] under a correct protocol. The implication for this literature is direct: reported AUROC on this dataset is far more sensitive to evaluation protocol than to model architecture, a concern directly addressed by reporting standards such as TRIPOD+AI [16] and one we recommend readers of this and related work apply when comparing headline numbers across papers.

### C. Limitations

1. **Information ceiling.** Performance is bounded at roughly 0.68–0.70 AUROC by the variables this dataset records; a learning curve indicates more data would close this only slowly (≈+0.005 AUROC per doubling for tree models, essentially flat for simpler models). The project's contribution is methodological rigor and honest reporting, not a large absolute accuracy gain.
2. **Data age, scope, and generalizability.** 1999–2008, US-only, ICD-9-coded; no external validation on another site or period was available. Practice patterns and diabetes management have changed materially since this period.
3. **Missing clinically relevant information.** No vital signs, exact lab values, social determinants, or discharge follow-up plans — variables repeatedly identified elsewhere as important for readmission risk.
4. **Small subgroups cannot be reliably assessed** (e.g., Asian, Hispanic, Other race categories each have under 35 readmission events in the test set).
5. **Label and ordering noise.** Readmissions at a different facility, and deaths occurring after discharge, are recorded as "not readmitted"; encounter ordering for the history features was inferred, not directly dated.
6. **Proxy variables.** Discharge disposition, prior utilization, and payer code plausibly encode social circumstances and access to care; we retained them because they measurably improved performance, which is itself a fairness trade-off worth stating explicitly rather than leaving implicit.
7. **Fairness.** The sensitivity gap by sex (Section V-F) is real at our current sample size and operating threshold and has not been resolved, only characterized; it requires human/ethical review before any deployment-adjacent decision, including whether sex-specific thresholds are an appropriate response.

### D. Intended use and what this is not

The resulting model and accompanying interface are a research/teaching prototype illustrating leak-free evaluation practice on a real but outdated and limited clinical dataset. It is explicitly **not a medical device and not intended for any real clinical, administrative, or payment decision**, consistent with its stated scope in `MODEL_CARD.md`.

## VII. Conclusion

We built and rigorously evaluated several models for 30-day readmission prediction in diabetic inpatients, with particular attention to patient-level evaluation, honest uncertainty quantification, calibration, interpretability, and subgroup fairness — areas we found inconsistently handled across the comparable published literature on this exact dataset. The best model (CatBoost) reaches AUROC 0.677; a stacked meta-learner and a more extensively tuned CatBoost, tried after the midterm, did not produce a statistically significant improvement once evaluated on the correctly cleaned cohort, reinforcing rather than breaking an information-limited ceiling near 0.68–0.70 for this dataset. We additionally show, as a direct, reproducible demonstration rather than a secondhand claim, how a single evaluation-protocol choice (oversampling before cross-validation) can manufacture a near-perfect-looking AUROC on real clinical tabular data, and we report — correcting our own earlier analysis — a statistically significant sex-based sensitivity gap that warrants human ethical review before any use of this or a similar model beyond the research setting it was built for.

## Work Plan, Roles and Risks

*(carried forward from the midterm report; update for the final submission once presentation recording and repository finalization are assigned)*

Siham Khaldoun: data and modelling lead (data preparation, model development, evaluation, this manuscript's Methods/Results). Kayla Arab and Magalie Achim: writing leads (literature review, manuscript drafting/editing). Larissa Chiu: interface lead (interface development, potential deployment). All members contribute to coding and scientific writing, and must confirm their own CRediT line below before submission.

**Risks:** limited clinical variables and the historical, US-specific dataset restrict predictive performance and generalizability (addressed via explicit limitations, Section VI-C); time constraints around reference-reading and final proofreading before Dec 11; the sex-based fairness finding requires genuine team/instructor discussion, not just a sentence in the limitations section.

## CRediT Contribution Statement

*(PROPOSED, not yet confirmed — `6_project_progress/statements.md` currently shows Larissa Chiu's and Magalie Achim's roles as "TBD," and the repository's commit history so far shows commits from Siham Khaldoun and Kayla Arab only. Larissa and Magalie must review, correct, and sign off on their own lines below, and should commit under their own accounts before submission, since commit history is part of the contribution-disclosure assessment. Do not submit this section unedited.)*

**Kayla Arab:** Conceptualization, Investigation, Writing – original draft, Writing – review & editing.
**Magalie Achim:** Investigation, Writing – original draft, Writing – review & editing. *(proposed — not yet confirmed by Magalie)*
**Siham Khaldoun:** Data curation, Methodology, Software, Formal analysis, Validation, Writing – review & editing.
**Larissa Chiu:** Investigation, Writing – review & editing. *(proposed — not yet confirmed by Larissa)*

## Use of Generative AI Statement

*(draft — team must verify accuracy and completeness before submission; full draft also in `6_project_progress/statements.md`)*

The authors used ChatGPT (OpenAI) to assist with scientific writing, language editing, manuscript organization, and clarification of technical concepts, and to identify potential scientific references and support drafting of selected sections. Claude (Anthropic) was used, directed by team members, to assist with dataset analysis, code for preprocessing/modeling/evaluation/calibration/interpretability, drafting of analysis documents and manuscript text from computed results and retrieved literature, and literature search/retrieval. AI-generated literature summaries were used only as screening aids, not as a substitute for human reading of cited sources. All generated content was reviewed and revised by the authors, who take full responsibility for the accuracy and integrity of the submitted work.

## References

*([1]-[19] carried forward from the midterm report; full bibliographic detail (journal/venue, volume, pages, DOI) still needs to be pulled from the midterm's reference list and verified against each source, not just author/year as listed here. Plus 3 new references needed for claims added in this manuscript: EmbPred30 [20], the ceiling-audit preprint [21], and Gupta & Choudhary [22]. Every reference must be confirmed read by a team member before submission — track in `4_benchmarking/team_literature_log.md`; as of this draft, no reference has a confirmed human reader recorded there.)*

[1] American Diabetes Association, "Standards of Care in Diabetes—2026," 2026.
[2] A. Rubin et al., "DiaTOHC," 2022.
[3] Tan et al., 2025.
[4] E. Emi-Johnson and Nkrumah, 2025.
[5] Liu, Sue, and Wu, 2024.
[6] Guo et al., 2025.
[7] B. Strack et al., "Impact of HbA1c Measurement on Hospital Readmission Rates," 2014.
[8] Mishra, Tanniru, and Sreedharan, 2025.
[9] Salim and Ibrahim, 2026.
[10] Afridi et al., 2026.
[11] Huang et al., 2022.
[12] Chen et al., 2022.
[13] Maisonnave et al., 2025.
[14] Steiner et al., 2026.
[15] Yau et al., 2024.
[16] "TRIPOD+AI statement," 2024.
[17] T. Saito and M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot," 2015.
[18] S. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," 2017.
[19] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?," 2022.

[20] Sarthak, S. Shukla, and S. Tripathi, "EmbPred30: Assessing 30-days Readmission for Diabetic Patients using Categorical Embeddings," arXiv:2002.11215, 2020.

[21] Chowdhury et al., "The Ceiling Is in the Channel: Auditing Learner Gaps and Measurement Frontiers in Clinical Prediction," arXiv:2609.01909, 2026.

[22] Gupta and A. Choudhary, "Benchmarking Machine Learning Classifiers for Hospital Readmission Prediction in Diabetes Care," 2026.
