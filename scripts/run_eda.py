"""Run a basic missingness and exploratory data analysis report."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypertension_risk.config import EDA_SUMMARY_PATH, FIGURES_DIR, FEATURES, TARGET
from hypertension_risk.data import load_dataset
from hypertension_risk.visualization import save_eda_figures


def main() -> None:
    frame = load_dataset()
    EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "synthetic educational demonstration; not clinical data",
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "features": FEATURES,
        "target": TARGET,
        "missing_values": {
            column: int(count)
            for column, count in frame.isna().sum().items()
        },
        "target_counts": {
            str(label): int(count)
            for label, count in frame[TARGET].value_counts().sort_index().items()
        },
        "numeric_summary": json.loads(
            frame[FEATURES].describe().to_json(orient="index")
        ),
    }
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_paths = save_eda_figures(frame, FIGURES_DIR)
    print(f"EDA summary written to {EDA_SUMMARY_PATH}")
    print("Missing values by column:")
    print(frame.isna().sum().to_string())
    print("Figures:")
    for path in figure_paths:
        print(f"  {path}")
    print("All findings describe synthetic data only.")


if __name__ == "__main__":
    main()
