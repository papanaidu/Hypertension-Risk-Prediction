import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hypertension_risk.config import FEATURES, TARGET
from hypertension_risk.data import generate_synthetic_data
from hypertension_risk.modeling import load_pipeline, train_and_evaluate


def test_pipeline_training_evaluation_and_artifact(tmp_path):
    frame = generate_synthetic_data(n_rows=400, random_seed=9)
    model_path = tmp_path / "artifacts" / "pipeline.joblib"
    metadata_path = tmp_path / "artifacts" / "metadata.json"

    metrics, outcome = train_and_evaluate(
        frame, model_path=model_path, metadata_path=metadata_path, random_seed=5
    )

    assert set(metrics["model"]) == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }
    assert {
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "brier_score",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
    }.issubset(metrics.columns)
    assert metrics["roc_auc"].between(0, 1).all()
    assert metrics["brier_score"].between(0, 1).all()
    assert model_path.exists()
    assert metadata_path.exists()
    pipeline = load_pipeline(model_path)
    assert isinstance(pipeline, Pipeline)
    predictions = pipeline.predict_proba(frame[FEATURES].iloc[:3])[:, 1]
    assert np.isfinite(predictions).all()
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert outcome["metadata"]["input_features"] == FEATURES
    assert TARGET not in outcome["metadata"]["input_features"]


def test_training_rejects_non_binary_target(tmp_path):
    frame = generate_synthetic_data(n_rows=300, random_seed=11)
    frame[TARGET] = 2
    with pytest.raises(ValueError, match="only binary"):
        train_and_evaluate(
            frame,
            model_path=tmp_path / "model.joblib",
            metadata_path=tmp_path / "metadata.json",
        )
