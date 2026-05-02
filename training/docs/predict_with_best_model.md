# `predict_with_best_model.py`

## File

- `training/scripts/predict_with_best_model.py`

## Purpose

This script loads the saved trained XGBoost model and uses it for inference.

## What It Does

1. Loads:
   - `training/models/best_xgb_risk_classifier.pkl`
   - `training/models/feature_list.json`
   - `training/models/label_map.json`
2. Checks that the input dataframe contains the required imaging features.
3. Runs the trained model on those features.
4. Converts ordinal predictions back to string labels:
   - `0 -> low_risk`
   - `1 -> medium_risk`
   - `2 -> high_risk`

## Inputs

Required model artifacts:

- `training/models/best_xgb_risk_classifier.pkl`
- `training/models/feature_list.json`
- `training/models/label_map.json`

Default sample input inside the script:

- `outputs/new/step3_ucsf_preprocessed_features.csv`

## Output

When run directly, the script prints a small preview of:

- `patient_id`
- `predicted_risk_label`

The reusable function is:

- `predict_risk(input_csv: Path) -> pd.DataFrame`

## How To Run

```bash
venv/bin/python training/scripts/predict_with_best_model.py
```

## Run Order

- Run this file **after** `run_ucsf_training.py`.
- It depends on the saved model artifacts created by the training script.
