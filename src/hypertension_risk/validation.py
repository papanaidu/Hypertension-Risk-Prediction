"""Validation helpers for model input data and app form values."""

import math
from collections.abc import Mapping

import pandas as pd

from hypertension_risk.config import FEATURES, INPUT_BOUNDS


def validate_feature_frame(frame: pd.DataFrame) -> None:
    """Raise a useful error when a prediction frame has an invalid schema."""
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    extra = [column for column in frame.columns if column not in FEATURES]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected feature(s): {', '.join(extra)}")
    for feature in FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        invalid_text = values.isna() & frame[feature].notna()
        if invalid_text.any():
            raise ValueError(f"{feature} must contain numeric values.")
        non_finite = values.dropna().map(lambda value: not math.isfinite(value))
        if non_finite.any():
            raise ValueError(f"{feature} must contain finite numeric values.")


def validate_user_input(values: Mapping[str, float]) -> dict[str, float]:
    """Validate form data and return an ordered, numeric feature mapping."""
    missing = [feature for feature in FEATURES if feature not in values]
    extra = [feature for feature in values if feature not in FEATURES]
    if missing:
        raise ValueError(f"Missing input(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"Unexpected input(s): {', '.join(extra)}")

    clean_values: dict[str, float] = {}
    for feature in FEATURES:
        try:
            value = float(values[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{feature} must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{feature} must be finite.")
        if feature in ("smoking_demo", "family_history_demo", "diabetes_history_demo"):
            if value not in (0.0, 1.0):
                raise ValueError(f"{feature} must be either 0 or 1.")
        lower, upper = INPUT_BOUNDS[feature]
        if not lower <= value <= upper:
            raise ValueError(
                f"{feature} must be between {lower:g} and {upper:g} "
                "(demonstration input bounds)."
            )
        clean_values[feature] = value
    return clean_values
