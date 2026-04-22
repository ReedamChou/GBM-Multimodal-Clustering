# Step 3: Preprocess Imaging Features

## Script

`scripts/step3_preprocess_imaging_features.py`

## Input

- `outputs/new/step2_ucsf_clustered.csv`

## Outputs

- `outputs/new/step3_ucsf_preprocessed_features.csv`
- `outputs/new/step3_imputation_report.csv`
- `outputs/new/step3_dropped_correlated_features.csv`

## Procedure

1. Load the clustered dataset.
2. Separate metadata columns from imaging feature columns.
3. Impute missing values by data type:
   categorical features use the mode,
   discrete integer features use the mode,
   continuous integer features use the median rounded to an integer,
   and continuous numeric features use the median.
4. One-hot encode categorical imaging features such as `dominant_brain_lobe`.
5. Scale all processed imaging feature columns to the `[0, 1]` range.
6. Compute feature-to-feature Pearson correlation on the processed imaging matrix.
7. Drop features whose absolute Pearson correlation exceeds `0.90`.
8. Write the updated feature matrix and the preprocessing reports to `outputs/new`.

## Metadata Kept Unchanged

- `case_id`
- `patient_id`
- `OS`
- `survival_months`
- `risk_cluster_id`
- `risk_cluster_label`

## Notes

- Imputation is applied before scaling and correlation filtering so the downstream computations operate on a complete matrix.
- Pearson correlation is invariant to min-max scaling, so the requested `[0, 1]` scaling and the correlation filter remain consistent.
