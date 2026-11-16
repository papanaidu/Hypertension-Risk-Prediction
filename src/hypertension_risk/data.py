"""Synthetic demonstration data generation and file I/O."""

from pathlib import Path

import numpy as np
import pandas as pd

from hypertension_risk.config import (
    BINARY_FEATURES,
    DEFAULT_ROWS,
    FEATURES,
    RAW_DATA_PATH,
    RANDOM_SEED,
    TARGET,
)


def generate_synthetic_data(
    n_rows: int = DEFAULT_ROWS, random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Create reproducible toy data; labels are not clinical outcomes."""
    if n_rows < 100:
        raise ValueError("n_rows must be at least 100 for a demonstration dataset.")

    rng = np.random.default_rng(random_seed)
    age = rng.integers(18, 91, size=n_rows)
    bmi = np.clip(rng.normal(27.0, 5.2, size=n_rows), 15.0, 55.0)
    smoking = rng.binomial(1, 0.22, size=n_rows)
    activity = np.clip(rng.gamma(shape=2.2, scale=2.0, size=n_rows), 0.0, 20.0)
    family_history = rng.binomial(1, 0.32, size=n_rows)
    sodium = np.clip(rng.normal(2_800, 850, size=n_rows), 500.0, 6_000.0)
    diabetes_history = rng.binomial(1, 0.12, size=n_rows)

    frame = pd.DataFrame(
        {
            "age_years": age.astype(float),
            "bmi_demo": bmi,
            "smoking_demo": smoking.astype(float),
            "activity_hours_week_demo": activity,
            "family_history_demo": family_history.astype(float),
            "sodium_intake_mg_day_demo": sodium,
            "diabetes_history_demo": diabetes_history.astype(float),
        }
    )

    # Arbitrary toy signal for a learnable example, not a medical model.
    demo_score = (
        -1.1
        + 0.018 * (frame["age_years"] - 50.0)
        + 0.045 * (frame["bmi_demo"] - 27.0)
        + 0.35 * frame["smoking_demo"]
        - 0.035 * (frame["activity_hours_week_demo"] - 4.0)
        + 0.3 * frame["family_history_demo"]
        + 0.00012 * (frame["sodium_intake_mg_day_demo"] - 2_800.0)
        + 0.4 * frame["diabetes_history_demo"]
    )
    probabilities = 1.0 / (1.0 + np.exp(-demo_score))
    frame[TARGET] = rng.binomial(1, probabilities).astype(int)

    # Missing predictor values demonstrate train-time imputation. Labels remain complete.
    for column in FEATURES:
        missing_mask = rng.random(n_rows) < 0.025
        frame.loc[missing_mask, column] = np.nan

    return frame[FEATURES + [TARGET]]


def save_synthetic_data(
    path: Path = RAW_DATA_PATH,
    n_rows: int = DEFAULT_ROWS,
    random_seed: int = RANDOM_SEED,
) -> Path:
    """Generate and write the synthetic CSV, returning its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_data(n_rows=n_rows, random_seed=random_seed).to_csv(
        path, index=False
    )
    return path


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load a generated CSV and check its required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run scripts\\generate_dataset.py first."
        )
    frame = pd.read_csv(path)
    expected_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(expected_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values.")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"Target column {TARGET!r} must contain only 0 and 1.")
    if not set(BINARY_FEATURES).issubset(frame.columns):
        raise ValueError("Dataset is missing one or more binary demonstration features.")
    return frame[expected_columns]
