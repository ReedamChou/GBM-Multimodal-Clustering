# UCSF Risk Classification Training

This folder contains a complete training workflow for ordinal 3-class risk classification on the UCSF imaging-feature dataset.

## Structure

- `configs/training_config.json`: paths and runtime settings
- `scripts/common.py`: shared helpers
- `scripts/run_ucsf_training.py`: end-to-end training pipeline
- `scripts/predict_with_best_model.py`: inference helper for saved model artifacts
- `models/`: trained model and metadata
- `outputs/figures/`: plots
- `outputs/metrics/`: JSON and CSV metric artifacts
- `outputs/reports/`: markdown summaries
- `outputs/tables/`: tabular intermediate outputs
- `outputs/logs/`: tuning logs and GPU status
- `cache/`: writable runtime cache for matplotlib

## Run

```bash
venv/bin/python training/scripts/run_ucsf_training.py
```

## Notes

- The input dataset is `outputs/new/step3_ucsf_preprocessed_features.csv`.
- The pipeline excludes `OS`, `survival_months`, and `risk_cluster_id` from training features to avoid leakage.
- XGBoost is configured to use GPU when available and falls back to CPU when no visible CUDA device is found.
