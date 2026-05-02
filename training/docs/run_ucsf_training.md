# `run_ucsf_training.py`

## File

- `training/scripts/run_ucsf_training.py`

## Purpose

This is the **main training entry point**.
It runs the full UCSF model-training workflow end to end.

## What It Does

1. Loads the preprocessed imaging dataset.
2. Drops rows with missing `risk_cluster_label`.
3. Selects the 17 imaging features only.
4. Encodes labels 0,1,2.
5. Checks whether GPU training is available.
6. Creates plots and feature summary tables.
7. Splits data into:
   - train: `350`
   - validation: `75`
   - test: `75`
8. Runs baseline models.
9. Trains an initial XGBoost validation model.
10. Tunes XGBoost with Optuna:
    - `25` trials
    - `5-fold` cross-validation
11. Retrains the best model on `train + validation = 425` samples.
12. Evaluates once on the held-out test set of `75` samples.
13. Computes feature importance and SHAP-style plots.
14. Runs final `10-fold` cross-validation on all `500` samples.
15. Saves the trained model, metrics, figures, and reports.

## Inputs

- `training/configs/training_config.json`
- `outputs/new/step3_ucsf_preprocessed_features.csv`

## Main Outputs

- `training/models/best_xgb_risk_classifier.pkl`
- `training/models/best_params.json`
- `training/models/label_map.json`
- `training/models/feature_list.json`
- `training/outputs/metrics/*`
- `training/outputs/figures/*`
- `training/outputs/tables/*`
- `training/outputs/logs/*`
- `training/outputs/reports/training_summary.md`

## How To Run

```bash
venv/bin/python training/scripts/run_ucsf_training.py
```

## Run Order

- Run this file **first**.
- This is the script that creates the training outputs and saved model artifacts.
