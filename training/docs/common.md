# `common.py`

## File

- `training/scripts/common.py`

## Purpose

This script contains the shared helpers used by the training workflow.
It is a support module and is **not meant to be run directly**.

## What It Defines

- project and training directory paths
- the 17 imaging feature names
- target column name
- label mapping:
  - `low_risk -> 0`
  - `medium_risk -> 1`
  - `high_risk -> 2`
- output directory creation helpers
- config loading helpers
- dataset loading and sanity-check helpers
- JSON and markdown save helpers
- XGBoost device detection
- shared base XGBoost parameter builder

## Inputs

- `training/configs/training_config.json`
- `outputs/new/step3_ucsf_preprocessed_features.csv`

## Outputs

This module does not create final outputs by itself.
It is imported by `run_ucsf_training.py` and provides helper functions used to create:

- training metrics
- model artifacts
- reports
- plots
- GPU status logs

## Used By

- `training/scripts/run_ucsf_training.py`

## Run Order

- Do **not** run this file directly.
- It runs automatically when `run_ucsf_training.py` imports it.
