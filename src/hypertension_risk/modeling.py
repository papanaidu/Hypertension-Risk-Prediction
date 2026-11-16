"""Model pipelines, held-out evaluation, and artifact persistence."""

import json
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hypertension_risk.config import (
    FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    SELECTED_FEATURE_COUNT,
    TARGET,
)
from hypertension_risk.validation import validate_feature_frame


def build_pipeline(classifier: Any, k: int = SELECTED_FEATURE_COUNT) -> Pipeline:
    """Build a full imputation, scaling, feature-selection, and classifier pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "feature_selection",
                SelectKBest(
                    score_func=partial(
                        mutual_info_classif, random_state=RANDOM_SEED
                    ),
                    k=min(k, len(FEATURES)),
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_model_candidates() -> dict[str, Pipeline]:
    """Return fresh instances of the requested classifier pipelines."""
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_SEED)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(random_state=RANDOM_SEED)
        ),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select by training CV, evaluate once on held-out data, and save the winner."""
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must include target column {TARGET!r}.")
    x = frame[FEATURES].copy()
    validate_feature_frame(x)
    y = frame[TARGET]
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must be complete and contain only binary values 0 and 1.")
    if y.nunique() != 2:
        raise ValueError("Training requires both target classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_seed,
        stratify=y,
    )

    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cross_validation = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=random_seed
    )
    for name, pipeline in build_model_candidates().items():
        cv_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cross_validation,
            scoring="roc_auc",
            n_jobs=1,
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probability = pipeline.predict_proba(x_test)[:, 1]
        results.append(
            {
                "model": name,
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1_score": f1_score(y_test, predicted, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probability),
                "brier_score": brier_score_loss(y_test, probability),
                "accuracy": accuracy_score(y_test, predicted),
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std()),
            }
        )
        fitted_models[name] = pipeline
        predictions[name] = (predicted, probability)

    metrics = pd.DataFrame(results).sort_values(
        ["cv_roc_auc_mean", "cv_roc_auc_std"], ascending=[False, True]
    ).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    best_pipeline = fitted_models[best_name]
    selected_mask = best_pipeline.named_steps["feature_selection"].get_support()
    selected_features = [
        feature for feature, selected in zip(FEATURES, selected_mask) if selected
    ]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    metadata = {
        "best_model": best_name,
        "selection_metric": (
            "mean 3-fold stratified training ROC-AUC "
            "(synthetic demonstration only)"
        ),
        "selected_features": selected_features,
        "input_features": FEATURES,
        "target": TARGET,
        "random_seed": random_seed,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "disclaimer": (
            "Synthetic educational output only; not clinical validation or medical advice."
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions[best_name][0], labels=[0, 1]
        ).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, {
        "best_model": best_name,
        "pipeline": best_pipeline,
        "metadata": metadata,
        "y_test": y_test.to_numpy(),
        "predictions": {
            name: {"labels": values[0], "probabilities": values[1]}
            for name, values in predictions.items()
        },
    }


def load_pipeline(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the complete fitted model pipeline, with an actionable error."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run scripts\\train_model.py first."
        )
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Saved model artifact is not a scikit-learn Pipeline.")
    return pipeline
