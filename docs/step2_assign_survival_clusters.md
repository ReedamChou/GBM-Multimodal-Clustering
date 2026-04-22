# Step 2: Assign Survival Clusters

## Script

`scripts/step2_assign_survival_clusters.py`

## Input

- `outputs/new/step1_ucsf_imaging_os.csv`

## Output

- `outputs/new/step2_ucsf_clustered.csv`

## Procedure

1. Load the step 1 dataset.
2. Read `OS` as survival in days.
3. Convert survival to months with `OS / 30.4375`.
4. Exclude rows where `OS` is missing because they cannot be clustered reliably.
5. Assign three survival-based risk groups:
   `high_risk` when survival is less than 6 months,
   `medium_risk` when survival is between 6 and 18 months inclusive,
   and `low_risk` when survival is greater than 18 months.
6. Add `risk_cluster_id` and `risk_cluster_label`.
7. Write the clustered dataset to `outputs/new`.

## Output Columns Added

- `survival_months`
- `risk_cluster_id`
- `risk_cluster_label`
