# Hypertension Risk Prediction Using Machine Learning

An educational, end-to-end machine-learning project built with Python and Streamlit. It demonstrates reproducible synthetic-data generation, exploratory analysis, preprocessing, feature selection, model comparison, probability calibration, and a small interactive app.

> **Important:** This project is not medical advice, a medical diagnosis, or a validated clinical tool. Its data and outcome labels are synthetic. Its estimated probabilities have no clinical meaning and must not be used for health decisions.

## Folder structure

```text
hypertension-risk-prediction/
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── artifacts/
│   └── .gitkeep
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── reports/
│   └── figures/
│       └── .gitkeep
├── scripts/
│   ├── generate_dataset.py
│   ├── run_eda.py
│   └── train_model.py
├── src/
│   └── hypertension_risk/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── modeling.py
│       ├── validation.py
│       └── visualization.py
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_modeling.py
    └── test_validation.py
```

Generated datasets, model files, and reports are created when the commands below are run and are ignored by Git.

## What this project does

- Generates a reproducible **synthetic demonstration dataset**; it does not download, redistribute, or claim to use a real clinical dataset.
- Gives the generated target label a synthetic, non-clinical definition based on non-blood-pressure demo inputs.
- Excludes systolic and diastolic blood pressure from the features, model, and app to avoid defining and then predicting the label from blood-pressure measurements.
- Reports missingness and summary information in EDA. The synthetic dataset deliberately includes missing predictor values so the training pipeline can demonstrate imputation.
- Uses a stratified train/test split, a `SimpleImputer`, `StandardScaler`, `SelectKBest(mutual_info_classif)`, and three classifiers.
- Compares Logistic Regression, Random Forest, and Gradient Boosting using stratified cross-validation on the training split; it selects by mean CV ROC-AUC, then reports metrics once on a separate held-out split. This is a teaching example, not reliable model selection for clinical deployment.
- Reports precision, recall, F1-score, ROC-AUC, confusion matrices, and Brier score. It also writes ROC, confusion-matrix, and calibration plots.
- Saves the best fitted preprocessing-and-model pipeline as a Joblib artifact.
- Validates interactive inputs and presents the model output as a **synthetic demo score**.

The generated target is a Bernoulli draw from an arbitrary formula included only to create a learnable toy classification problem. Its coefficients, feature ranges, class balance, test metrics, and probabilities are not estimates of clinical relationships or real-world performance. Results may vary if the generator or its seed is changed.

## Windows and VS Code setup

1. Install Python 3.10 or newer and VS Code with the Python extension.
2. Open the `hypertension-risk-prediction` folder in VS Code.
3. Open **Terminal → New Terminal** (PowerShell) and run:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PowerShell blocks activation, allow it for this terminal session with
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

4. Generate the CSV, run EDA, train and evaluate models, then launch the app:

   ```powershell
   python scripts\generate_dataset.py
   python scripts\run_eda.py
   python scripts\train_model.py
   streamlit run app.py
   ```

5. Run the tests from the project root:

   ```powershell
   python -m pytest
   ```

The app expects the generated model artifact, so run training before starting Streamlit. The data generator accepts `--rows` and `--seed`; EDA and training use the generated dataset and project defaults.

## Outputs

| Output | Purpose |
|---|---|
| `data/raw/synthetic_hypertension_demo.csv` | Reproducible, clearly labeled synthetic data |
| `reports/eda_summary.json` | Dataset shape, missingness, class counts, and summary statistics |
| `reports/figures/target_distribution.png` | Synthetic target distribution |
| `reports/figures/correlation_matrix.png` | Predictor and target correlation overview |
| `artifacts/best_pipeline.joblib` | Complete fitted preprocessing and classifier pipeline |
| `artifacts/model_metadata.json` | Selected model, features, and run configuration |
| `reports/model_metrics.csv` | Training-CV selection scores and held-out evaluation metrics |
| `reports/figures/confusion_matrices.png` | Held-out confusion matrices |
| `reports/figures/roc_curves.png` | Held-out ROC curves |
| `reports/figures/calibration_curve.png` | Held-out probability calibration curves |

These are generated outputs and are intentionally not committed by default.

## File-by-file guide

- `app.py` builds the Streamlit interface, collects the same feature inputs used at training, validates them, and shows a demo probability and a visual gauge.
- `src/hypertension_risk/config.py` centralizes feature names, output paths, and synthetic-data settings.
- `src/hypertension_risk/data.py` creates the seeded synthetic dataset and reads/writes its CSV.
- `src/hypertension_risk/validation.py` checks the feature schema, finite values, and demonstration input bounds.
- `src/hypertension_risk/modeling.py` creates preprocessing/model pipelines, splits data, evaluates candidates, and saves the chosen complete pipeline.
- `src/hypertension_risk/visualization.py` creates EDA and evaluation figures with Matplotlib and Seaborn.
- `scripts/generate_dataset.py` generates the synthetic CSV.
- `scripts/run_eda.py` audits missing values and writes summaries and EDA charts.
- `scripts/train_model.py` trains and compares the three requested models and writes the model and reports.
- `tests/` contains focused tests for reproducible data, pipeline behavior, and input validation.
- `.streamlit/config.toml` sets a simple Streamlit theme.
- `requirements.txt` lists the Python dependencies.
- `.gitignore` excludes virtual environments, caches, and generated outputs while retaining folder placeholders.

## Reproducibility and responsible use

The data generator and train/test split have explicit seeds. Metrics describe only the held-out portion of this generated dataset; they are not clinical validation, and no accuracy or medical utility is claimed. Do not enter real personal health information. Do not use the app to screen, diagnose, treat, or make decisions about a person.

## License

No separate software license is declared. Add a license only after choosing terms appropriate for your intended use. The project generates its own synthetic data and does not bundle an external dataset.
