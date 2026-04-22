# GBM Survival-Clustering And Patient-Correlation Pipeline

This repository contains a four-step UCSF workflow built around `data/raw/ucsf_with_clinical.csv`.
The pipeline removes non-survival clinical variables, creates survival-based risk groups, preprocesses imaging features, and then generates patient-wise Pearson correlation matrices inside each risk cluster.

## Pipeline Scripts

- `scripts/step1_remove_clinical_features.py`
- `scripts/step2_assign_survival_clusters.py`
- `scripts/step3_preprocess_imaging_features.py`
- `scripts/step4_cluster_patient_correlations.py`

## Inputs

- Raw dataset: `data/raw/ucsf_with_clinical.csv`
- Virtual environment: `venv`

## Outputs

Main pipeline outputs are written to `outputs/new`:

- `step1_ucsf_imaging_os.csv`
- `step2_ucsf_clustered.csv`
- `step3_ucsf_preprocessed_features.csv`
- `step3_imputation_report.csv`
- `step3_dropped_correlated_features.csv`
- `ucsf_cluster_0_high_risk_patient_correlation_matrix.csv`
- `ucsf_cluster_1_medium_risk_patient_correlation_matrix.csv`
- `ucsf_cluster_2_low_risk_patient_correlation_matrix.csv`

## Risk Groups

- `high_risk`: survival lower than 6 months
- `medium_risk`: survival between 6 and 18 months, inclusive
- `low_risk`: survival greater than 18 months

`OS` is treated as survival in days and converted to months using `30.4375` days per month.
Rows without `OS` are excluded at the clustering step because they cannot be assigned to a risk group.

## Quick Start

Activate the virtual environment:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full workflow step by step:

```bash
python scripts/step1_remove_clinical_features.py
python scripts/step2_assign_survival_clusters.py
python scripts/step3_preprocess_imaging_features.py
python scripts/step4_cluster_patient_correlations.py
```

## What Each Step Does

1. Step 1 keeps identifiers, imaging-derived columns, and `OS`, while dropping the remaining clinical columns.
2. Step 2 converts `OS` to months and assigns `risk_cluster_id` and `risk_cluster_label`.
3. Step 3 imputes missing values by feature type, one-hot encodes categorical imaging variables, scales imaging features to `[0, 1]`, and removes features with absolute Pearson correlation above `0.90`.
4. Step 4 builds one patient-by-patient Pearson correlation matrix per cluster using only the processed imaging features.

## Documentation

Detailed step docs are available in `docs/`:

- `docs/step1_remove_clinical_features.md`
- `docs/step2_assign_survival_clusters.md`
- `docs/step3_preprocess_imaging_features.md`
- `docs/step4_cluster_patient_correlations.md`
- `docs/atlas_feature_nan_readme.md`
