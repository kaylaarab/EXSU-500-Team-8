# Slide deck (7-minute recorded presentation)

**Status:** built, numbers verified against the repo 2026-11-17. Speaker-slot
assignment and the actual recording are still open — see `PROJECT_STATUS.md`.

The deck is a Claude Artifact (editable, exportable to Google Slides/PPTX),
not a file in this repo, because that is where it was built and where the
team can keep editing it without a local PowerPoint/Keynote install:

**Deck link:** https://claude.ai/artifact/ABbRNb5wSw4CUc6Qc9KvAr
("30-Day Readmission Prediction — Team 8 Final Presentation")

To export it to a local `.pptx` or Google Slides file, open the link and use
Share / Export.

## Slides (10, matching `presentation_script_draft.md`)

| # | Slide | Content |
|---|---|---|
| 1 | Title | Team names, course, "leak-free, honestly evaluated" framing |
| 2 | Problem | 1 in 9 diabetic stays readmitted in 30 days; the prediction task |
| 3 | Data | Diabetes 130-US Hospitals, 101,766 -> 99,343 stays after death/hospice exclusion, 11.4% readmitted |
| 4 | Protocol | Patient-level split; why sensitivity is reported alongside accuracy (89%/0% baseline trap) |
| 5 | Results | Model comparison table (CatBoost 0.677 AUROC, best) + SHAP summary figure |
| 6 | What didn't help | 80+ experiments; the stacked meta-learner / wider-search result that looked promising but didn't hold up |
| 7 | Cautionary result | EmbPred30 SMOTE-leakage demo: 0.96 (leaked) vs 0.64 (correct) on the same model |
| 8 | Fairness | Calibration + decision-curve figure; sex-sensitivity gap figure (62% women vs. 52% men, CIs don't overlap) |
| 9 | Limits & demo | Research prototype, not a medical device; interface demo mention |
| 10 | Conclusion | Three take-aways + thank you |

Each slide carries speaker notes (adapted from `presentation_script_draft.md`)
in an on-slide notes field, visible to presenters when editing the deck.

## Figures embedded
1. SHAP summary plot (`5_results/main_models/shap_summary.png`) — slide 5.
2. Calibration + decision-curve panel (`5_results/evaluation/`) — slide 8.
3. Sex-gap sensitivity bar chart (`5_results/evaluation/`) — slide 8.

These are the same 3 figures embedded in `Manuscript_Draft.docx`.

## Verified against the repo, 2026-11-17
Every number on the deck was checked against current source files and
matches: CatBoost 0.677 AUROC (`PROJECT_STATUS.md`, `README.md`), XGBoost
0.673 (`model_comparison_patient_ci.csv`), sex-gap 0.622/0.519 rounding to
62%/52% (`PROJECT_STATUS.md`, Script 18). No corrections were needed.

## Still needed before recording
- [ ] Assign who presents which slide(s) (placeholders `[A]`/`[B]`/`[C]`/`[D]`
      in `presentation_script_draft.md`).
- [ ] Rehearse to the 7-minute limit.
- [ ] Record and upload per the course submission instructions.
