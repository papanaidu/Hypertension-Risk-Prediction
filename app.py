"""Streamlit interface for the synthetic hypertension-risk demo."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hypertension_risk.config import (
    FEATURES,
    FIGURES_DIR,
    INPUT_BOUNDS,
    METRICS_PATH,
    MODEL_PATH,
)
from hypertension_risk.modeling import load_pipeline
from hypertension_risk.validation import validate_user_input
from hypertension_risk.visualization import save_probability_gauge

st.set_page_config(
    page_title="Hypertension Risk Prediction Demo",
    page_icon="🧪",
    layout="wide",
)

st.title("Hypertension Risk Prediction")
st.caption("An educational machine-learning demonstration")
st.error(
    "**Not medical advice or a diagnosis.** This app uses synthetic data and "
    "artificial labels. Its score is not a clinical risk estimate or validated tool."
)
st.markdown(
    """
    This demo intentionally does **not** ask for systolic or diastolic blood
    pressure: a target defined from blood pressure must not be predicted using
    those same measurements. All labels and relationships here are artificial.
    Do not enter real personal health information.
    """
)

try:
    model = load_pipeline(MODEL_PATH)
except (FileNotFoundError, TypeError) as error:
    st.warning(f"{error}\n\nGenerate data and train the model before launching the app.")
    st.stop()

with st.sidebar:
    st.header("Demo inputs")
    st.caption("Input limits are software guards for this toy example, not medical cutoffs.")
    with st.form("demo_inputs"):
        col_left, col_right = st.columns(2)
        with col_left:
            age = st.number_input("Age (demo years)", min_value=18, max_value=100, value=45)
            bmi = st.number_input(
                "BMI (demo value)", min_value=12.0, max_value=70.0, value=27.0, step=0.1
            )
            smoking_label = st.selectbox("Smoking (demo)", ["No", "Yes"])
            activity = st.number_input(
                "Activity (demo hours/week)",
                min_value=0.0,
                max_value=40.0,
                value=4.0,
                step=0.5,
            )
        with col_right:
            family_history_label = st.selectbox("Family history (demo)", ["No", "Yes"])
            sodium = st.number_input(
                "Sodium intake (demo mg/day)",
                min_value=0,
                max_value=10_000,
                value=2_800,
                step=100,
            )
            diabetes_label = st.selectbox("Diabetes history (demo)", ["No", "Yes"])
        submitted = st.form_submit_button("Estimate synthetic demo score", type="primary")

if submitted:
    raw_values = {
        "age_years": age,
        "bmi_demo": bmi,
        "smoking_demo": int(smoking_label == "Yes"),
        "activity_hours_week_demo": activity,
        "family_history_demo": int(family_history_label == "Yes"),
        "sodium_intake_mg_day_demo": sodium,
        "diabetes_history_demo": int(diabetes_label == "Yes"),
    }
    try:
        input_values = validate_user_input(raw_values)
        input_frame = pd.DataFrame([input_values], columns=FEATURES)
        probability = float(model.predict_proba(input_frame)[0, 1])
        if not 0.0 <= probability <= 1.0:
            raise ValueError("The model returned a probability outside the valid range.")
    except (ValueError, TypeError) as error:
        st.error(f"Could not calculate the demo score: {error}")
    else:
        st.subheader("Synthetic model output")
        metric_column, chart_column = st.columns([1, 2])
        with metric_column:
            st.metric("Demo probability", f"{probability:.1%}")
        with chart_column:
            figure = save_probability_gauge(probability)
            st.pyplot(figure, clear_figure=True)
        st.info(
            "This percentage describes only the output of a model trained on "
            "artificial data. It does not estimate your chance of hypertension."
        )

st.divider()
st.subheader("Model evaluation (synthetic held-out split)")
st.caption(
    "These metrics are generated from artificial labels and must not be interpreted "
    "as clinical performance."
)
if METRICS_PATH.exists():
    try:
        metrics_frame = pd.read_csv(METRICS_PATH)
    except (OSError, pd.errors.ParserError) as error:
        st.error(f"Could not read the generated evaluation table: {error}")
    else:
        st.dataframe(
            metrics_frame.style.format(
                {
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1_score": "{:.3f}",
                    "roc_auc": "{:.3f}",
                    "brier_score": "{:.3f}",
                    "accuracy": "{:.3f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        chart_files = [
            ("confusion_matrices.png", "Confusion matrices"),
            ("roc_curves.png", "ROC curves"),
            ("calibration_curve.png", "Probability calibration"),
        ]
        chart_columns = st.columns(len(chart_files))
        for column, (filename, caption) in zip(chart_columns, chart_files):
            image_path = FIGURES_DIR / filename
            with column:
                if image_path.exists():
                    st.image(str(image_path), caption=caption, width="stretch")
                else:
                    st.caption(f"Run training to generate {caption.lower()}.")
else:
    st.info("Run `python scripts\\train_model.py` to generate the comparison table.")

with st.expander("About the inputs and model"):
    st.write(
        "The trained features are age, demo BMI, smoking, activity, family history, "
        "demo sodium intake, and diabetes history. Blood-pressure measurements are "
        "deliberately absent. Input bounds exist only to reject invalid demo values."
    )
    st.write(f"Demo input guards: {INPUT_BOUNDS}")
