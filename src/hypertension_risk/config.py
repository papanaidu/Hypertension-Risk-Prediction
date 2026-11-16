"""Shared project settings and feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_hypertension_demo.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "model_metrics.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"

TARGET = "synthetic_demo_label"
FEATURES = [
    "age_years",
    "bmi_demo",
    "smoking_demo",
    "activity_hours_week_demo",
    "family_history_demo",
    "sodium_intake_mg_day_demo",
    "diabetes_history_demo",
]
BINARY_FEATURES = [
    "smoking_demo",
    "family_history_demo",
    "diabetes_history_demo",
]
CONTINUOUS_FEATURES = [
    "age_years",
    "bmi_demo",
    "activity_hours_week_demo",
    "sodium_intake_mg_day_demo",
]
RANDOM_SEED = 42
DEFAULT_ROWS = 2_000
SELECTED_FEATURE_COUNT = 5

# These are input guards for a toy application, not clinical cutoffs.
INPUT_BOUNDS = {
    "age_years": (18.0, 100.0),
    "bmi_demo": (12.0, 70.0),
    "smoking_demo": (0.0, 1.0),
    "activity_hours_week_demo": (0.0, 40.0),
    "family_history_demo": (0.0, 1.0),
    "sodium_intake_mg_day_demo": (0.0, 10_000.0),
    "diabetes_history_demo": (0.0, 1.0),
}
