"""Train, compare, evaluate, and persist the demonstration classifiers."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypertension_risk.config import FIGURES_DIR, METRICS_PATH
from hypertension_risk.data import load_dataset
from hypertension_risk.modeling import train_and_evaluate
from hypertension_risk.visualization import save_evaluation_figures


def main() -> None:
    frame = load_dataset()
    metrics, outcome = train_and_evaluate(frame)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    figure_paths = save_evaluation_figures(
        outcome["y_test"], outcome["predictions"], FIGURES_DIR
    )
    print("Held-out synthetic demonstration metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSelected model: {outcome['best_model']}")
    print("Selection used training cross-validation ROC-AUC for demonstration only.")
    print(f"Pipeline input features: {outcome['metadata']['input_features']}")
    print(f"Metrics: {METRICS_PATH}")
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")
    print("\nThese measurements are not clinical validation or medical advice.")


if __name__ == "__main__":
    main()
