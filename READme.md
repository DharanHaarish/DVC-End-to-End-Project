# DVC End-to-End Time Series Forecasting Project 📊

A complete end-to-end machine learning pipeline demonstrating DVC (Data Version Control) best practices for experiment tracking, data versioning, and reproducible ML workflows. This project uses Facebook Prophet for time-series forecasting with real data from Google BigQuery.

## 🎯 What This Project Does

This is a production-ready ML pipeline that:
- Fetches time-series data from Google BigQuery
- Preprocesses data with configurable outlier detection methods
- Trains Prophet models with customizable hyperparameters
- Evaluates performance with multiple metrics (RMSE, MAE, MAPE, MPE)
- Versions everything (data, code, models, experiments) using DVC
- Stores large artifacts in Google Cloud Storage

**The best part?** Change a parameter, run one command, and DVC automatically figures out what needs to be recomputed. No more manual pipeline management!

---

## 🏗️ Project Architecture

DVC-End-to-End-Project/
├── data/
│   ├── raw/                     # Raw data from BigQuery
│   │   └── raw_data.csv
│   └── processed/               # Preprocessed train/test splits
│       ├── df_train.csv
│       └── df_test.csv
├── models/
│   └── prophet_model.pkl        # Trained Prophet model (versioned)
├── src/
│   ├── fetch.py                 # Fetch data from BigQuery
│   ├── preprocess.py            # Outlier detection & train/test split
│   ├── train.py                 # Train Prophet with log transforms
│   └── evaluate.py              # Calculate evaluation metrics
├── dvc.yaml                     # Pipeline definition (stages & dependencies)
├── params.yaml                  # All hyperparameters (single source of truth)
├── metrics.json                 # Latest evaluation metrics
├── dvc.lock                     # Locked pipeline state (like package-lock.json)
├── requirements.txt             # Python dependencies


The pipeline is defined in `dvc.yaml`:

### Stage 1: Fetch Data
```bash
dvc repro fetch
```
- Queries BigQuery for raw time-series data
- Saves to `data/raw/raw_data.csv`
- Requires GCP credentials

### Stage 2: Preprocess
```bash
dvc repro preprocess
```
- Detects and removes outliers using:
  - **IQR** (Interquartile Range)
  - **Z-Score** (standard deviation)
  - **Rolling IQR** (window-based)
- Splits into train (80%) / test (20%)
- Outputs: `data/processed/df_train.csv`, `df_test.csv`

**Configure via `params.yaml`:**
```yaml
preprocess:
  method: "iqr"        # or "zscore", "rolling_iqr"
  window: 15           # for rolling methods
```

### Stage 3: Train
```bash
dvc repro train
```
- Trains Facebook Prophet model
- Optional log transformation (base 10, e, or 2)
- Configurable hyperparameters
- Saves model to `models/prophet_model.pkl`

**Configure via `params.yaml`:**
```yaml
train:
  transform: "log"     # or "none"
  base: 10             # or "e", 2
  hyperparameters:
    seasonality_mode: "multiplicative"
    changepoint_prior_scale: 0.05
    seasonality_prior_scale: 10.0
    holidays_prior_scale: 10.0
    changepoint_range: 0.8
```

### Stage 4: Evaluate
```bash
dvc repro evaluate
```
- Generates forecasts on test set
- Reverses log transform if applied
- Calculates metrics: RMSE, MAE, MAPE, MPE
- Saves to `metrics.json`

**Configure via `params.yaml`:**
```yaml
evaluate:
  transform: "log"     # must match train transform
  base: 10             # must match train base
```

---

## 📊 Understanding params.yaml

All hyperparameters live in one place for easy experimentation:

```yaml
# Preprocessing configuration
preprocess:
  method: "iqr"              # Outlier detection: "iqr", "zscore", "rolling_iqr"
  window: 15                 # Window size for rolling methods

# Training configuration
train:
  transform: "log"           # Apply log transform: "log" or "none"
  base: 10                   # Log base: 10, "e", or 2
  hyperparameters:
    seasonality_mode: "multiplicative"    # or "additive"
    changepoint_prior_scale: 0.05         # Flexibility of trend (0.001-0.5)
    seasonality_prior_scale: 10.0         # Strength of seasonality
    holidays_prior_scale: 10.0            # Strength of holiday effects
    changepoint_range: 0.8                # Portion of data for changepoints

# Evaluation configuration
evaluate:
  transform: "log"           # Must match train transform
  base: 10                   # Must match train base
```

**Pro tip:** Change any value, run `dvc repro`, and DVC automatically re-runs only affected stages!

---

## 📈 Metrics Explained

After running the pipeline, `metrics.json` contains:

- **RMSE** (Root Mean Squared Error): Average prediction error, penalizes large errors
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **MAPE** (Mean Absolute Percentage Error): Average % error (scale-independent)
- **MPE** (Mean Percentage Error): Average % error (shows bias direction)

**Lower is better for all metrics.**


---
