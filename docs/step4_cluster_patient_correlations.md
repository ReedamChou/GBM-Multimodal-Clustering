# Step 4: Cluster-Wise Patient Pearson Correlations

## Script

`scripts/step4_cluster_patient_correlations.py`

## Input

- `outputs/new/step3_ucsf_preprocessed_features.csv`

## Outputs

- `outputs/new/ucsf_cluster_0_high_risk_patient_correlation_matrix.csv`
- `outputs/new/ucsf_cluster_1_medium_risk_patient_correlation_matrix.csv`
- `outputs/new/ucsf_cluster_2_low_risk_patient_correlation_matrix.csv`

## Procedure

1. Load the preprocessed dataset from step 3.
2. Exclude identifiers, survival fields, and cluster labels from the feature set.
3. Use only the processed imaging features.
4. Split patients by `risk_cluster_id`.
5. For each cluster, compute a patient-by-patient Pearson correlation matrix across imaging features.
6. Save one CSV per cluster in `outputs/new`.

## Matrix Shape

- If a cluster contains `N` patients, the resulting matrix is `N x N`.
- For the current UCSF data this produces matrices for the high-, medium-, and low-risk cohorts.
