# Automated test results (run 2026-11-10)

Two test suites were run for the final deliverables: the existing Streamlit
interface regression tests, and a new pipeline/model integrity suite written
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

## 2. New pipeline/model integrity tests (`test_pipeline_integrity.py`)

```
python -m pytest 8_final_deliverables/test_pipeline_integrity.py -v
```

**6/9 passed, 3 failed.** The 3 failures are a real finding, not a test bug
— see below.

**Passed:**
- Splits are non-empty; no patient appears in more than one of train/val/test
  (the project's core no-leakage guarantee holds).
- Target column is binary in every split.
- The deployed model (`7_interface/model/catboost.cbm`) loads correctly.
- On a synthetic default patient it returns a probability in [0, 1], not a
  raw/unbounded score.
- Giving a patient more prior inpatient/ER visits does not lower its
  predicted risk (basic sanity check on model behaviour).

**Failed — local `data/*.csv` has drifted from what's documented and from
what the deployed model expects:**

| Test | What it found |
|---|---|
| `test_split_sizes_match_model_card` | Local splits are 71,520 / 15,027 / 15,219 rows, vs. the 69,519 / 14,911 / 14,913 documented in `MODEL_CARD.md`. Total = 101,766, i.e. the **full raw UCI dataset**, not the 99,343-row cohort after exclusions. |
| `test_death_hospice_excluded` | The local train split still contains discharge codes 11, 13, 14, 19, 20 (death/hospice) — these should have been dropped per `scripts/01_preprocess.py` and the project's settled decisions. |
| `test_local_data_schema_matches_deployed_model` | Local `data/test.csv` is missing `payer_code` and `medical_specialty`, two columns the committed model (`7_interface/model/config.json`) was trained on and still requires. |

**What this means:** the `data/*.csv` files currently sitting on this
machine (git-ignored, never committed — see `3_dataset/dataset_analysis_report.md`
section 10) do not match the output `01_preprocess.py` is supposed to
produce today, and do not match the schema the committed model and
`MODEL_CARD.md` numbers assume. They look like an older or partial
regeneration. **This does not put any already-reported number in doubt** —
the committed `catboost.cbm`, `config.json`, and every result in
`5_results/` were produced before this drift and are untouched — but it
means a from-scratch rebuild on this machine would NOT currently reproduce
those numbers.

**Recommended before the Dec 11 submission:** re-run
`python scripts/01_preprocess.py` (and, if changed, scripts 02 onward) fresh
on this machine, then re-run this test suite to confirm it's green, before
anyone claims "clone and rerun" reproducibility in the manuscript or video.
The 3 failing tests above are left failing on purpose (not loosened) so
they'll catch it when that's done.

## Files added this pass
- `test_pipeline_integrity.py` — the new pytest suite described above.
- `test_results.md` — this file.
- `interface_test_output.txt`, `pipeline_test_output.txt` — raw pytest logs.
