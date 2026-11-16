import pandas as pd
import pytest

from hypertension_risk.config import FEATURES, TARGET
from hypertension_risk.data import generate_synthetic_data, load_dataset


def test_synthetic_generation_is_reproducible_and_has_no_blood_pressure_features():
    first = generate_synthetic_data(n_rows=300, random_seed=7)
    second = generate_synthetic_data(n_rows=300, random_seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == FEATURES + [TARGET]
    assert not any("blood_pressure" in column for column in first.columns)
    assert first[TARGET].notna().all()
    assert set(first[TARGET].unique()) == {0, 1}
    assert first[FEATURES].isna().any().any()


def test_generator_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 100"):
        generate_synthetic_data(n_rows=99)


def test_loader_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_dataset.py"):
        load_dataset(tmp_path / "missing.csv")
