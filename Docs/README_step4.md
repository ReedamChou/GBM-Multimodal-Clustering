# Step 4 Train/Test Split and Leakage-Safe Scaling

## What this script does

This README documents scripts/step4_split_train_test.py.

The script implements Step 4 of your GBM pipeline using Step 3 outputs.

For each dataset (UCSF and UPenn), it does the following:

1. Loads Step 3 master table and metadata.
2. Builds a 70/30 train/test split.
3. Uses stratification to keep train and test similar for:
- MGMT proportion (using mgmt_bin)
- OS distribution (quantile bins)
4. Falls back safely to simpler stratification (or random split) if strict stratification is not feasible.
5. Fits StandardScaler on train only for continuous clustering features.
6. Transforms both train and test using that train-fitted scaler.
7. Saves train/test master tables, train/test clustering feature tables, scaler, metadata, and log file.

This enforces the Step 4 anti-leakage rule: no fitting on test data.


## Logger behavior

Each dataset gets its own log file at:

- outputs/step4/ucsf/step4.log
- outputs/step4/upenn/step4.log

The logger records:

1. Input files used.
2. Selected stratification strategy and OS bin count.
3. Row counts and feature counts.
4. Train-only scaling step.
5. Quick train vs test checks for MGMT mean and OS median.
6. Output file locations.


## Input requirements

Step 3 must already be completed.

Expected default input folders and files:

- outputs/step3/ucsf/ucsf_master_table_step3.csv
- outputs/step3/ucsf/ucsf_step3_metadata.json
- outputs/step3/upenn/upenn_master_table_step3.csv
- outputs/step3/upenn/upenn_step3_metadata.json


## Setup

1. Open terminal at project root:

C:/Users/Husain/Documents/AIml sideproject/GBM multi model

2. Create and activate virtual environment (if needed):

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies:

pip install pandas numpy scikit-learn joblib


## Quick start

Run Step 4 for both datasets with defaults:

python scripts/step4_split_train_test.py --output-dir outputs/step4 --log-level INFO


## More run options

Run only UCSF:

python scripts/step4_split_train_test.py --datasets ucsf --output-dir outputs/step4 --log-level INFO

Run only UPenn:

python scripts/step4_split_train_test.py --datasets upenn --output-dir outputs/step4 --log-level INFO

Change split ratio and seed:

python scripts/step4_split_train_test.py --test-size 0.30 --random-state 42 --output-dir outputs/step4

Use custom Step 3 input folder:

python scripts/step4_split_train_test.py --step3-dir outputs/step3 --output-dir outputs/step4


## Output structure

For each dataset, the script writes:

- {dataset}_train_master_table_step4.csv
- {dataset}_test_master_table_step4.csv
- {dataset}_train_clustering_features_step4.csv
- {dataset}_test_clustering_features_step4.csv
- {dataset}_train_scaler_step4.joblib
- {dataset}_step4_split_metadata.json
- step4.log

Folders:

- outputs/step4/ucsf/
- outputs/step4/upenn/


## Notes

1. Step 3 may already contain standardized columns; Step 4 still re-fits scaling on train only to follow leakage-safe protocol.
2. Clustering should be run on train clustering features from Step 4, not Step 3 full data.
3. Keep test set untouched until validation steps.


## Troubleshooting

If you see ModuleNotFoundError for pandas or sklearn:

pip install --upgrade pandas numpy scikit-learn joblib

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If Step 4 says missing Step 3 files, run Step 3 first:

python scripts/step3_build_master_feature_table.py --output-dir outputs/step3
