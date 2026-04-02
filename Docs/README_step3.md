# Step 3 Master Feature Table Preprocessing

## What this script does

This README documents [step3_build_master_feature_table.py](step3_build_master_feature_table.py).

The script implements Step 3 of your GBM pipeline for already merged atlas plus clinical CSV files.

For each dataset input (UCSF and or UPenn), it performs:

1. Raw cleaning only
- Converts common text placeholders to missing values: unknown, indeterminate, not available, NA, etc.
- Converts numeric-looking text columns to numeric dtype when parsing is reliable.
- Resolves key column aliases for patient ID, dominant lobe, MGMT, IDH, and survival fields.

2. Leakage prevention by design
- Does not fit imputers on the full cohort.
- Does not encode categorical clustering features on the full cohort.
- Does not standardize continuous features on the full cohort.
- Leaves those train-derived transformations for Step 4.

3. Outcome leakage prevention
- Excludes outcome columns from clustering feature export:
  - OS or Survival_from_surgery_days_UPDATED
  - 1-dead 0-alive or Survival_Censor
  - Survival_Status (if present)

4. Saved artifacts per dataset
- Raw-cleaned master table CSV.
- Metadata JSON describing resolved columns and raw predictor columns.


## Expected input

You should provide merged CSV files produced after your atlas registration and clinical merge.

By default, the script now auto-detects these files in:

Clinical+Atlas Merged Data/
- UCSF_sri24_atlas_features_merged.csv
- UPenn_sri24_atlas_features_merged.csv

Required minimum columns:
- Patient ID column (for example ID)
- Dominant lobe column from atlas mapping (for example dominant_lobe)
- Clinical columns for Sex or Gender, MGMT, and IDH or IDH1

The script supports multiple alias names for UCSF and UPenn style schemas.


## Setup

1. Open a terminal in the project root:

C:/Users/Husain/Documents/AIml sideproject/GBM multi model

2. Create and activate a Python environment (recommended):

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies:

pip install pandas numpy


## Quick start for your current project

Your merged files are already in:

Clinical+Atlas Merged Data/
- UCSF_sri24_atlas_features_merged.csv
- UPenn_sri24_atlas_features_merged.csv

So you can run Step 3 directly with one command from the project root:

python scripts/step3_build_master_feature_table.py --output-dir outputs/step3

This command will process both datasets automatically.


## How to run

Run with auto-detected default merged files:

python scripts/step3_build_master_feature_table.py --output-dir outputs/step3

Run for both datasets:

python scripts/step3_build_master_feature_table.py --ucsf-merged "PATH_TO_UCSF_MERGED.csv" --upenn-merged "PATH_TO_UPENN_MERGED.csv" --output-dir outputs/step3

Run only UCSF:

python scripts/step3_build_master_feature_table.py --ucsf-merged "PATH_TO_UCSF_MERGED.csv" --output-dir outputs/step3

Run only UPenn:

python scripts/step3_build_master_feature_table.py --upenn-merged "PATH_TO_UPENN_MERGED.csv" --output-dir outputs/step3


## Output structure

Inside outputs/step3, the script creates one folder per dataset:

- ucsf/
  - ucsf_master_table_step3.csv
  - ucsf_step3_metadata.json

- upenn/
  - upenn_master_table_step3.csv
  - upenn_step3_metadata.json


## Notes

- Step 3 is now a leakage-safe raw-cleaning step only.
- Train-only imputation, categorical encoding, and scaling are all performed in Step 4.
- Keep the current Step 3 outputs as the master tables that Step 4 will split before any learned preprocessing is fit.


## Troubleshooting

- If you get import errors for pandas or sklearn, reinstall dependencies:

pip install --upgrade pandas numpy

- If the script says no input files found, confirm these exact file names exist inside Clinical+Atlas Merged Data:
  - UCSF_sri24_atlas_features_merged.csv
  - UPenn_sri24_atlas_features_merged.csv

- If PowerShell blocks venv activation, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
