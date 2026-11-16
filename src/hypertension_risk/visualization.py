"""EDA and model-evaluation charts."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from hypertension_risk.config import FEATURES, TARGET

sns.set_theme(style="whitegrid", palette="muted")


def save_eda_figures(frame: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Save target-distribution and correlation plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=frame, x=TARGET, ax=axis, color="#0F766E")
    axis.set(title="Synthetic demo label distribution", xlabel="Synthetic label", ylabel="Rows")
    output = figures_dir / "target_distribution.png"
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    outputs.append(output)

    fig, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        frame[FEATURES + [TARGET]].corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=axis,
    )
    axis.set_title("Synthetic demonstration correlations (not clinical evidence)")
    output = figures_dir / "correlation_matrix.png"
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    outputs.append(output)
    return outputs


def save_evaluation_figures(
    y_true: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
    figures_dir: Path,
) -> list[Path]:
    """Save confusion matrix, ROC, and calibration plots for all candidates."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(predictions)
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4))
    for axis, name in zip(np.atleast_1d(axes), model_names):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            predictions[name]["labels"],
            labels=[0, 1],
            display_labels=["Demo 0", "Demo 1"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(name)
    output = figures_dir / "confusion_matrices.png"
    fig.suptitle("Held-out synthetic demo confusion matrices")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    outputs.append(output)

    fig, axis = plt.subplots(figsize=(7, 5))
    for name in model_names:
        RocCurveDisplay.from_predictions(
            y_true, predictions[name]["probabilities"], name=name, ax=axis
        )
    axis.set_title("ROC curves — synthetic held-out split")
    output = figures_dir / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    outputs.append(output)

    fig, axis = plt.subplots(figsize=(7, 5))
    for name in model_names:
        fraction_positive, mean_predicted = calibration_curve(
            y_true, predictions[name]["probabilities"], n_bins=10, strategy="quantile"
        )
        axis.plot(mean_predicted, fraction_positive, marker="o", label=name)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Ideal reference")
    axis.set(
        title="Probability calibration — synthetic held-out split",
        xlabel="Mean predicted demo probability",
        ylabel="Fraction of positive demo labels",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend()
    output = figures_dir / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    outputs.append(output)
    return outputs


def save_probability_gauge(probability: float) -> plt.Figure:
    """Return a simple visualization of the model's synthetic demo score."""
    fig, axis = plt.subplots(figsize=(6, 1.6))
    axis.barh(["Synthetic demo score"], [1], color="#DCE5EB", height=0.35)
    axis.barh(["Synthetic demo score"], [probability], color="#0F766E", height=0.35)
    axis.set_xlim(0, 1)
    axis.set_xticks(np.linspace(0, 1, 6), [f"{value:.0%}" for value in np.linspace(0, 1, 6)])
    axis.set_yticks([])
    axis.set_title("Model output (not a clinical risk estimate)", loc="left")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig
