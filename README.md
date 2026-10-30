# NASA C-MAPSS Remaining Useful Life Prediction

An end-to-end machine learning project for predicting the remaining useful life (RUL) of simulated turbofan engines using the NASA C-MAPSS dataset. It includes data preparation, model comparison and training pipelines, saved models for all four C-MAPSS subsets, and a Flask dashboard for interactive predictions and model diagnostics.

This is a Machine Learning Laboratory project. The C-MAPSS data are simulated; the results are not certified aircraft maintenance predictions.

## Highlights

- Trains and compares six scikit-learn regressors per subset, with optional XGBoost support.
- Uses engine-wise validation so rows from the same engine never appear in both training and validation.
- Reports validation, grouped cross-validation, near-failure, and last-cycle test metrics.
- Performs paired t-tests with Holm correction to compare the selected model against the alternatives.
- Ships with trained models for FD001-FD004, so the dashboard works immediately.
- Provides manual sensor input, CSV upload, per-engine RUL trend charts, and a model performance page.
- Stores a consolidated `artifacts/findings.pkl` snapshot for reporting and further analysis.

## Dataset

The repository uses four C-MAPSS subsets:

| Subset | Train engines | Train rows | Test engines | Operating conditions | Fault modes |
|---|---:|---:|---:|---:|---:|
| FD001 | 100 | 20,631 | 100 | 1 | 1 |
| FD002 | 260 | 53,759 | 259 | 6 | 1 |
| FD003 | 100 | 24,720 | 100 | 1 | 2 |
| FD004 | 249 | 61,249 | 248 | 6 | 2 |

Each observation contains:

- `engine_id` - engine group identifier
- `cycle` - operating cycle, starting at 1
- `operating_setting_1` to `operating_setting_3`
- `sensor_01` to `sensor_21`
- `RUL` - remaining cycles before failure

The app and training pipeline use these files:

- `notebooks/data/train_FD00x.csv` - training histories with RUL labels.
- `notebooks/data/test_last_cycle_FD00x_labeled.csv` - the last observed cycle of each test engine, joined with the official RUL label.

The original `test_FD00x.csv` and `RUL_FD00x.csv` files are also included. `notebooks/data/data_dictionary.csv` documents every column.

## Project Structure

```text
.
|-- app.py                              # Flask dashboard
|-- artifacts/                          # trained models and evaluation outputs
|   |-- FD001_model.pkl
|   |-- FD001_comparison.csv
|   |-- FD001_predictions.csv
|   |-- FD001_t_tests.csv
|   |-- ...
|   |-- findings.pkl                    # consolidated results snapshot
|   `-- training_summary.csv            # one row per subset
|-- notebooks/
|   |-- 01_FD001_Statistical_Analysis.ipynb
|   |-- NASA_CMAPSS_FD001_to_FD004_Easy.ipynb
|   `-- data/                           # C-MAPSS CSVs and data dictionary
|-- src/
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_transformation.py
|   |   `-- model_trainer.py
|   |-- pipeline/
|   |   |-- train_pipeline.py
|   |   `-- predict_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- static/                             # CSS, JavaScript, images
|-- templates/                          # dashboard pages
|-- logs/                               # generated at runtime
|-- requirements.txt
`-- setup.py
```

## Requirements

- Python 3.10 or newer (developed with Python 3.13)
- pip

Main dependencies: pandas, NumPy, scikit-learn, SciPy, statsmodels, Flask, matplotlib, seaborn, and Jupyter. XGBoost is optional; the code skips it when it is not installed. `catboost` is listed for future experiments but is not used by the current pipelines.

### Setup on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Quick Start

### Run the dashboard

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in a browser.

The repository already contains trained models for FD001-FD004. The dataset selector enables every subset whose `artifacts/<DATASET>_model.pkl` file exists, so newly trained models are discovered automatically.

### Train or retrain models

Train one subset:

```powershell
python -m src.pipeline.train_pipeline --dataset FD001
```

Train all four subsets:

```powershell
python -m src.pipeline.train_pipeline --dataset all
```

Training one subset updates only that dataset's row in `artifacts/training_summary.csv`; existing rows are preserved. The full run can take several minutes depending on the machine.

Outputs written to `artifacts/`:

- `<DATASET>_model.pkl`
- `<DATASET>_comparison.csv`
- `<DATASET>_predictions.csv`
- `<DATASET>_t_tests.csv`
- `training_summary.csv`

### Predict from a CSV

```powershell
python -m src.pipeline.predict_pipeline --dataset FD001 --csv notebooks/data/test_last_cycle_FD001_labeled.csv --output artifacts/FD001_cli_predictions.csv
```

The input CSV must contain the model's feature columns: `cycle`, the three operating settings, and the 21 sensor columns. `engine_id` is optional and, when present, is copied to the output. Extra columns are ignored. The output contains:

- `Dataset`
- `engine_id` and `cycle` when available
- `Predicted RUL`
- `Lower RUL`
- `Upper RUL`

## How It Works

### Feature preparation

- All columns except `engine_id` and `RUL` are used as predictors (25 features).
- A `VarianceThreshold` step removes constant columns separately inside each training pipeline.
- Linear, Ridge, and SVR pipelines add `StandardScaler`.
- Preprocessing is fitted inside the model pipeline, so it learns only from training data.

### Validation and evaluation

1. **Engine-wise holdout:** 80% of engines for training and 20% for validation (`GroupShuffleSplit`, seed 42).
2. **Model comparison:** validation MAE, RMSE, and R2 for each candidate.
3. **Grouped cross-validation:** three-fold `GroupKFold` on the training split for the selected model.
4. **Statistical comparison:** paired t-tests on per-engine mean absolute errors, with Holm correction across models.
5. **Final test:** the best model is refit on all training engines and evaluated on the labeled last-cycle test set.

The selected model is the candidate with the lowest validation RMSE.

### Candidate models

- Linear Regression
- Ridge (`alpha=10`)
- Decision Tree (`max_depth=8`, `min_samples_leaf=10`)
- Random Forest (50 trees, `max_depth=12`, `min_samples_leaf=5`)
- Gradient Boosting (80 estimators, `max_depth=3`)
- SVR (`C=100`, `epsilon=10`)
- XGBoost (optional, 80 estimators, `max_depth=4`, learning rate 0.1)

### Prediction interval

The saved model stores a margin equal to the 90th percentile of absolute validation errors. The dashboard reports `Predicted RUL +/- margin`, with the lower bound clipped at zero. This is a heuristic error band, not a calibrated confidence interval.

## Results

Metrics below are from `artifacts/training_summary.csv`. Test metrics are computed on the labeled last observed cycle of each test engine.

| Subset | Selected model | Validation RMSE | CV RMSE | Near-failure MAE | Test MAE | Test RMSE | Test R2 | 90% margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FD001 | SVR | 29.57 | 39.68 | 6.37 | 17.76 | 24.02 | 0.666 | 52.00 |
| FD002 | SVR | 33.57 | 39.16 | 11.37 | 21.13 | 28.13 | 0.726 | 52.91 |
| FD003 | SVR | 53.45 | 62.38 | 5.86 | 24.78 | 37.92 | 0.161 | 85.75 |
| FD004 | Random Forest | 55.77 | 50.69 | 9.44 | 28.63 | 39.55 | 0.474 | 91.72 |

Notes:

- **Validation RMSE** is measured on the 20% held-out engines.
- **CV RMSE** is the mean three-fold grouped cross-validation score on the training split.
- **Near-failure MAE** uses validation observations with `RUL <= 30`.
- **Test** results use one labeled endpoint per test engine, so they are not directly comparable to row-level validation scores.
- FD003 is the weakest subset at the last-cycle test (`R2 = 0.161`). Treat it as a baseline and retrain or tune before relying on it.

## Artifacts and Findings

The app reads the model files and per-dataset CSVs directly. `findings.pkl` is a consolidated snapshot for reporting and analysis; it is not required for serving predictions.

Each `<DATASET>_model.pkl` contains:

```python
{
    "pipeline": fitted_scikit_learn_pipeline,
    "features": [...],       # required input columns
    "dataset": "FD001",
    "model_name": "SVR",
    "margin": 51.996884      # 90th percentile absolute validation error
}
```

`findings.pkl` contains validation, test, comparison, and significance records for every subset:

```python
import pickle

with open("artifacts/findings.pkl", "rb") as handle:
    findings = pickle.load(handle)

print(findings["datasets"]["FD001"]["metrics"])
print(findings["training_summary"])
```

Generated copies `<DATASET>_train.csv` and `<DATASET>_test.csv` are written by the data ingestion step but excluded from version control.

## Dashboard

| Page | Route | Contents |
|---|---|---|
| Dashboard | `/` | Dataset selector, manual input, CSV upload, result tiles, trend chart, recent predictions |
| Prediction | `/predict` or `/predictdata` | Form handler for manual and upload modes |
| Performance | `/performance` | Model comparison, significance tests, and summary metrics |
| About | `/about` | Dataset statistics and project background |

### Manual input

- All 25 model features are required.
- Input fields start with the median value from the selected subset's training data.
- `engine_id` is used to look up that engine's history for the trend chart and recent-prediction list; it is not a model feature.
- If the engine ID does not exist in the training data, the prediction still works but no trend chart is shown.

### CSV upload

- Only `.csv` files are accepted.
- The file must contain `cycle`, the three operating settings, and all 21 sensor columns.
- The page reports missing columns, predicts every row, shows the first 20 results, and adds up to 8 rows to the recent-prediction panel.

### Recent predictions

Recent predictions are kept in memory for the current server process (maximum 10) and reset when the server restarts.

## Reproducibility and Configuration

- Random seed `42` is used for data splits and model randomness.
- Data paths and artifact paths are resolved relative to the project root.
- The app expects data under `notebooks/data/` and models under `artifacts/`.
- Logs are written to `logs/` at the project root.
- Retraining replaces the model file for that subset and updates its metrics.
- Metrics can vary slightly across operating systems and library versions.
- The dashboard uses Flask's development server. For a public deployment, put it behind a production server and add authentication.

## Limitations and Future Work

- Each row is modeled independently; no rolling windows, degradation slopes, or sequence models are used.
- The last-cycle test set provides only one prediction per test engine, which limits statistical power.
- Prediction intervals are based on a validation error quantile and are not calibrated.
- FD003 performs poorly at the endpoint test and needs further investigation.
- Recent predictions are not persisted to a database.
- The dashboard has no authentication and should not be exposed publicly in its current form.
- Pickled models depend on the library versions that created them; retrain if loading fails after an upgrade.

Possible improvements:

- Add rolling-window and trend features, or sequence models such as LSTM/GRU.
- Cap or transform RUL labels and tune hyperparameters per subset.
- Calibrate prediction intervals and evaluate with NASA's scoring function.
- Add automated tests, Docker packaging, and a production WSGI server.

## Notebooks

- `notebooks/01_FD001_Statistical_Analysis.ipynb` - FD001 training-data statistics and exploratory charts; no models are trained.
- `notebooks/NASA_CMAPSS_FD001_to_FD004_Easy.ipynb` - end-to-end analysis for all four subsets, including model comparison, engine-wise validation, grouped CV, t-tests, test evaluation, and CSV prediction.
- `notebooks/data/validation_summary.csv` - structural checks for row counts, duplicates, missing values, and constant columns.

## Acknowledgements

The C-MAPSS dataset is provided by NASA. Please cite the original work when using the data:

> Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. *Proceedings of the International Conference on Prognostics and Health Management (PHM)*.

This repository is intended for academic and educational use. The C-MAPSS dataset retains its original terms of use.
