import pandas as pd
import pytest

from hypertension_risk.config import FEATURES
from hypertension_risk.validation import validate_feature_frame, validate_user_input


def valid_values():
    return {
        "age_years": 45,
        "bmi_demo": 27.0,
        "smoking_demo": 0,
        "activity_hours_week_demo": 4.0,
        "family_history_demo": 1,
        "sodium_intake_mg_day_demo": 2800,
        "diabetes_history_demo": 0,
    }


def test_valid_input_is_returned_in_feature_order_as_floats():
    validated = validate_user_input(valid_values())
    assert list(validated) == FEATURES
    assert all(isinstance(value, float) for value in validated.values())


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("age_years", 17, "between"),
        ("bmi_demo", float("nan"), "finite"),
        ("smoking_demo", 2, "either 0 or 1"),
        ("activity_hours_week_demo", -1, "between"),
        ("sodium_intake_mg_day_demo", 10_001, "between"),
    ],
)
def test_invalid_user_values_raise_clear_errors(feature, value, message):
    values = valid_values()
    values[feature] = value
    with pytest.raises(ValueError, match=message):
        validate_user_input(values)


def test_missing_and_unexpected_input_keys_are_rejected():
    values = valid_values()
    values.pop("age_years")
    with pytest.raises(ValueError, match="Missing input"):
        validate_user_input(values)
    values = valid_values()
    values["systolic_bp"] = 120
    with pytest.raises(ValueError, match="Unexpected input"):
        validate_user_input(values)


def test_prediction_frame_requires_exact_schema_and_finite_values():
    frame = pd.DataFrame([{**valid_values()}], dtype=float)
    validate_feature_frame(frame)
    with pytest.raises(ValueError, match="Missing required"):
        validate_feature_frame(frame.drop(columns=["age_years"]))
    frame.loc[0, "age_years"] = float("inf")
    with pytest.raises(ValueError, match="finite numeric"):
        validate_feature_frame(frame)
