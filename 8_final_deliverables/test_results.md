# Automated test results

Two test suites are run for the final deliverables: the existing Streamlit
interface regression tests, and a pipeline/model integrity suite written
for this pass. Raw `pytest` output is in `interface_test_output.txt` and
`pipeline_test_output.txt`.

## 1. Interface regression tests (`7_interface/test_app.py`)

```
python -m pytest 7_interface/test_app.py -v
```

**8/8 passed.** Covers: app loads without exception, the "not a medical
device" disclaimer is shown, the default and both example patients score
without error, a died/hospice patient is correctly excluded from scoring,
and the race selectbox default matches `config.json`.

## 2. Pipeline/model integrity tests (`test_pipeline_integrity.py`)

```
python -m pytest 8_final_deliverables/test_pipeline_integrity.py -v
```

**9/9 passed (as of 2026-11-10).** Covers:
- Splits are non-empty; no patient appears in more than one of train/val/test
  (the project's core no-leakage guarantee).
- Split sizes match `MODEL_CARD.md` (69,519 / 14,911 / 14,913).
- No death/hospice discharge codes remain in any split.
- Target column is binary in every split.
- The deployed model (`7_interface/model/catboost.cbm`) loads correctly and
  its declared feature schema matches `data/*.csv`.
- On a synthetic default patient it returns a probability in [0, 1], not a
  raw/unbounded score.
- Giving a patient more prior inpatient/ER visits does not lower its
  predicted risk (basic sanity check on model behaviour).

### Finding caught and fixed (2026-11-10)

The first run of this suite (3/9 failing) found that the local `data/*.csv`
on this machine had drifted from what `scripts/01_preprocess.py` is supposed
to produce: it held the full 101,766-row raw dataset (still containing
death/hospice discharges, and missing the `payer_code`/`medical_specialty`
columns the deployed model expects) rather than the documented 99,343-row
cleaned cohort. This did not affect any already-reported number — the
committed model and every table in `5_results/` were built before the
drift — but it meant a fresh "clone and rerun" would not have reproduced
them.

**Fix:** re-ran `python scripts/01_preprocess.py`, which regenerates
`data/{train,val,test}.csv` from the raw UCI CSV. The rebuilt splits now
match `MODEL_CARD.md` exactly (69,519 / 14,911 / 14,913 encounters, no
death/hospice codes, correct schema), and all 9 tests pass. `data/` is
git-ignored (per `.gitignore`), so this was a local fix only — nothing to
commit for the data itself, and the old (drifted) files were kept as a
backup in `data/_backup_pre_rerun_20261110/` (also git-ignored) in case
anyone needs to compare.

## Downstream catch (2026-11-10)

This suite's fix (rerunning `scripts/01_preprocess.py`) was necessary but not sufficient: a two-reviewer audit of the final deliverables found that `scripts/22_stacking_and_wider_search.py` had been run on 2026-11-09, on the drifted data, *before* this suite caught and fixed the drift — and its already-logged "new best model" result (AUROC 0.679, stacking + wider search, reported as a significant gain) was never rerun afterward. Rerunning it on the corrected cohort found the gain **does not replicate** (both changes land at AUROC 0.677, not significant over plain CatBoost). This has been corrected throughout the repo (`manuscript_draft.md`, `MODEL_CARD.md`, `PROJECT_STATUS.md`, `progress_log.md`, `5_results/model_variations_log.md`). Lesson for next time: when a reproducibility test like this one finds and fixes a data drift, also re-check any analysis script logged around the same time, not just the ones that triggered the test.

## Files in this folder
- `test_pipeline_integrity.py` — the pipeline/model integrity suite.
- `test_results.md` — this file.
- `interface_test_output.txt`, `pipeline_test_output.txt` — raw pytest logs.
