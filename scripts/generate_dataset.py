"""Generate the project's clearly labeled synthetic demonstration CSV."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hypertension_risk.config import DEFAULT_ROWS, RANDOM_SEED, RAW_DATA_PATH
from hypertension_risk.data import save_synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    output_path = save_synthetic_data(
        path=RAW_DATA_PATH, n_rows=args.rows, random_seed=args.seed
    )
    print(f"Generated {args.rows} synthetic demonstration rows: {output_path}")
    print("The data and labels are artificial and have no clinical meaning.")


if __name__ == "__main__":
    main()
