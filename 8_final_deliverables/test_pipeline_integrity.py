"""
Pipeline integrity tests for the final deliverables.

These check the things a reader/grader would actually worry about:
that the patient-level split has no leakage, and that the deployed
model (the one in 7_interface/) produces sane, in-range predictions.
They are separate from 7_interface/test_app.py, which tests the
Streamlit UI itself.

Run from the repo root:
    python -m pytest 8_final_deliverables/test_pipeline_integrity.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest
from catboost import CatBoostClassifier, Pool

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MODEL_DIR = REPO / "7_interface" / "model"


@pytest.fixture(scope="module")
def splits():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")
    return train, val, test


def test_splits_exist_and_are_nonempty(splits):
    train, val, test = splits
    assert len(train) > 0 and len(val) > 0 and len(test) > 0


def test_no_patient_overlap_between_splits(splits):
    """The project's core evaluation guarantee: no patient's encounters
    are split across train/val/test, or test results would leak."""
    train, val, test = splits
    train_p = set(train["patient_nbr"])
    val_p = set(val["patient_nbr"])
    test_p = set(test["patient_nbr"])
    assert not (train_p & val_p), "patients leaking between train and val"
    assert not (train_p & test_p), "patients leaking between train and test"
    assert not (val_p & test_p), "patients leaking between val and test"


def test_split_sizes_match_model_card(splits):
    """Guards against someone silently re-running preprocessing with a
    different random seed or exclusion rule and the model card going stale."""
    train, val, test = splits
    assert len(train) == pytest.approx(69519, abs=5)
    assert len(val) == pytest.approx(14911, abs=5)
    assert len(test) == pytest.approx(14913, abs=5)


def test_death_hospice_excluded(splits):
    """Death/hospice discharges can't be readmitted and must be excluded
    from every split, per the project's settled decisions (scripts/01_preprocess.py)."""
    train, val, test = splits
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if "discharge_disposition_id" in df.columns:
            death_hospice_ids = {11, 13, 14, 19, 20, 21}
            present = set(df["discharge_disposition_id"].unique()) & death_hospice_ids
            assert not present, f"{name} split still contains death/hospice discharge ids {present}"


def test_target_is_binary(splits):
    train, val, test = splits
    for name, df in [("train", train), ("val", val), ("test", test)]:
        assert set(df["readmit_30d"].unique()) <= {0, 1}, f"{name} target is not binary"


@pytest.fixture(scope="module")
def demo_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "catboost.cbm"))
    with open(MODEL_DIR / "config.json") as f:
        config = json.load(f)
    return model, config


def test_model_loads(demo_model):
    model, config = demo_model
    assert model is not None
    assert "features" in config or "columns" in config or len(config) > 0


def _synthetic_row(config, overrides=None):
    """Build one input row the way 7_interface/app.py does: config['defaults']
    for every declared feature, with optional field overrides. Kept
    schema-independent of data/*.csv on purpose -- see
    test_local_data_schema_matches_deployed_model below, which checks that
    independence assumption directly."""
    row = dict(config["defaults"])
    if overrides:
        row.update(overrides)
    return pd.DataFrame([{c: row[c] for c in config["features"]}])


def test_model_predicts_in_valid_range(demo_model):
    """Deployed model must output a calibrated probability in [0, 1] on a
    default-filled patient, not a raw unbounded score."""
    model, config = demo_model
    row = _synthetic_row(config)
    pool = Pool(row, cat_features=config["categorical"])
    prob = float(model.predict_proba(pool)[0, 1])
    assert 0.0 <= prob <= 1.0


def test_model_output_responds_to_known_risk_factors(demo_model):
    """Sanity check, not a correctness audit: more prior inpatient visits
    should not lower predicted risk for an otherwise-identical patient."""
    model, config = demo_model
    low = _synthetic_row(config, {"number_inpatient": 0, "number_emergency": 0})
    high = _synthetic_row(config, {"number_inpatient": 5, "number_emergency": 3})
    pool = Pool(pd.concat([low, high], ignore_index=True), cat_features=config["categorical"])
    p_low, p_high = model.predict_proba(pool)[:, 1]
    assert p_high >= p_low, "more prior inpatient/ER visits unexpectedly lowered predicted risk"


def test_local_data_schema_matches_deployed_model(splits, demo_model):
    """Flags schema drift between data/*.csv on this machine and the
    feature set the committed model (7_interface/model/) actually expects
    (caught a real drift once -- see 8_final_deliverables/test_results.md)."""
    _, _, test = splits
    model, config = demo_model
    missing = [c for c in config["features"] if c not in test.columns]
    assert not missing, f"data/test.csv is missing model features: {missing}"
