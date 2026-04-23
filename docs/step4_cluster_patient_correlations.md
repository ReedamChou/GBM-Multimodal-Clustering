# Step 4: Cluster-Wise Patient Pearson Correlations

## Script

`scripts/step4_cluster_patient_correlations.py`

## Goal

Generate a **patient-by-patient similarity matrix** inside each survival-risk cohort.
The script measures similarity using **Pearson correlation** across the processed imaging
features produced in Step 3.

This means:

- Patients are compared only against other patients in the same `risk_cluster_id`.
- Similarity is based on full feature profiles (not on OS or metadata).
- One correlation matrix is saved per cluster.

## Input

- `outputs/new/step3_ucsf_preprocessed_features.csv`

Expected key columns in the input:

- A patient identifier column (auto-detected from aliases such as `case_id`, `patient_id`, `ID`, etc.)
- `risk_cluster_id`
- Processed imaging feature columns from Step 3

The script automatically excludes metadata fields from feature calculations.

## Outputs

- `outputs/new/ucsf_cluster_0_high_risk_patient_correlation_matrix.csv`
- `outputs/new/ucsf_cluster_1_medium_risk_patient_correlation_matrix.csv`
- `outputs/new/ucsf_cluster_2_low_risk_patient_correlation_matrix.csv`

Each output CSV is a square matrix where:

- Rows = patients in that cluster
- Columns = the same patients
- Value `(i, j)` = Pearson correlation between patient `i` and patient `j`

## Procedure

1. Parse command-line args.
2. Validate that the input CSV exists.
3. Read the Step 3 dataset into a pandas DataFrame.
4. Resolve the patient ID column by checking known aliases.
5. Build the imaging feature list by dropping metadata columns (ID/survival/cluster labels).
6. Validate `risk_cluster_id` exists.
7. For each cluster ID in `{0, 1, 2}`:
	 - Filter rows to that cluster.
	 - Convert selected feature columns to float.
	 - Compute patient-wise Pearson correlations.
	 - Replace index/columns with patient IDs for readability.
	 - Write matrix to cluster-specific output CSV.
8. Print per-cluster row count and output path.

## Code Walkthrough

### Constants and Defaults

- `DEFAULT_INPUT_CSV`: points to Step 3 output in `outputs/new`.
- `DEFAULT_OUTPUT_DIR`: points to `outputs/new`.
- `PATIENT_ID_ALIASES`: fallback names used to find the patient ID column.
- `METADATA_COLUMNS`: excluded from feature calculations.
- `RISK_CLUSTER_LABELS`: maps cluster IDs to filename labels:
	- `0 -> high_risk`
	- `1 -> medium_risk`
	- `2 -> low_risk`

### `resolve_patient_id_column(columns)`

- Builds a lowercase map of column names.
- Searches aliases in order.
- Returns the first match.
- Raises `ValueError` if none is found.

Why this matters:
different datasets can use different patient-ID naming conventions.

### `get_imaging_feature_columns(df)`

- Returns all columns not listed in `METADATA_COLUMNS`.
- These remaining columns are treated as imaging features used for similarity.

### `compute_patient_correlation_matrix(cluster_df, patient_id_column, feature_columns)`

Core logic:

1. `cluster_df[feature_columns].astype(float)` creates a numeric feature matrix.
2. `.T.corr(method="pearson")` computes correlations between patients, not features.
	 - `corr` in pandas computes column-wise by default.
	 - Transposing (`.T`) makes original rows (patients) become columns.
3. Row/column labels are replaced with patient IDs for interpretability.

Result:
a symmetric matrix with 1.0 on the diagonal.

### `main()`

- Handles I/O, validation, cluster loop, saving, and logging.
- Stops early with clear errors if required input structure is missing.

## Why the Matrix Uses `.T.corr()`

Without transpose, correlation would be computed **between features**.
Step 4 needs correlation **between patients**.

So if the original data shape is:

- Rows = patients
- Columns = imaging features

then transposing gives:

- Rows = features
- Columns = patients

and `corr()` now returns patient-to-patient similarity.

## Error Handling and Assumptions

The script raises errors when:

- Input CSV is missing.
- No patient ID alias is found.
- `risk_cluster_id` is missing.
- Any expected cluster (0, 1, or 2) has zero patients.

Assumptions:

- Step 3 already handled missing-value imputation/scaling as needed.
- Imaging feature columns are numeric or castable to float.
- Cluster IDs are exactly 0, 1, 2.

## Matrix Shape

- If a cluster contains `N` patients, the resulting matrix is `N x N`.
- For the current UCSF data this produces matrices for the high-, medium-, and low-risk cohorts.

## How to Run

Default paths:

```bash
python scripts/step4_cluster_patient_correlations.py
```

Custom paths:

```bash
python scripts/step4_cluster_patient_correlations.py \
	--input-csv outputs/new/step3_ucsf_preprocessed_features.csv \
	--output-dir outputs/new
```

## Interpreting Values

- `+1.0`: very similar feature patterns between two patients
- `0.0`: no linear relationship
- `-1.0`: opposite feature patterns

These matrices are typically used downstream for subgroup similarity analysis,
cohort cohesion checks, or patient-level network analysis.
