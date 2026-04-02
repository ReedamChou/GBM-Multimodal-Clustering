# Step 7 Test-Set Validation by Nearest Centroid Assignment

## What this script does

This README documents scripts/step7_validate_test_set.py.

The script implements Step 7 of your GBM pipeline using:

- Step 4 test master table and test clustering features
- Step 5 train cluster centroids and train labels
- Step 6 train cluster summary and high-risk cluster label

For each dataset (UCSF and UPenn), the script:

1. Loads the held-out Step 4 test set.
2. Loads the Step 5 cluster centroids selected from the train set.
3. Uses the Step 5 feature list and train-time median imputations to clean the test feature matrix.
4. Assigns each test patient to the nearest cluster centroid using Euclidean distance.
5. Builds a test-set cluster summary table with:
- n per cluster
- median OS with IQR
- percentage MGMT methylated
- percentage IDH mutant
- mean global NC/EN ratio
- mean global ED/EN ratio
- mean tumor burden index
- dominant lobe mode
- mean age
6. Compares train and test cluster proportions side by side.
7. Runs test-set Kaplan-Meier and log-rank validation across assigned clusters.
8. Carries forward the Step 6 high-risk cluster label and checks whether its size and imaging signature still look consistent on the test set.
9. Saves CSV outputs, metadata JSON, plot, and logs.


## Logger

Per-dataset log files are written to:

- outputs/step7/ucsf/step7.log
- outputs/step7/upenn/step7.log

The log includes:

1. Input files loaded.
2. Test row counts and feature counts.
3. Test-set cluster counts after centroid assignment.
4. Test-set log-rank result.
5. Output files written.


## Inputs required

Steps 4, 5, and 6 must already be completed.

Expected default files:

- outputs/step4/ucsf/ucsf_test_master_table_step4.csv
- outputs/step4/ucsf/ucsf_test_clustering_features_step4.csv
- outputs/step4/upenn/upenn_test_master_table_step4.csv
- outputs/step4/upenn/upenn_test_clustering_features_step4.csv

- outputs/step5/ucsf/ucsf_step5_cluster_centroids.csv
- outputs/step5/ucsf/ucsf_step5_train_cluster_labels.csv
- outputs/step5/upenn/upenn_step5_cluster_centroids.csv
- outputs/step5/upenn/upenn_step5_train_cluster_labels.csv

- outputs/step6/ucsf/ucsf_step6_metadata.json
- outputs/step6/ucsf/ucsf_step6_cluster_summary_train.csv
- outputs/step6/upenn/upenn_step6_metadata.json
- outputs/step6/upenn/upenn_step6_cluster_summary_train.csv


## Setup

1. Open terminal at project root:

C:/Users/Husain/Documents/AIml sideproject/GBM multi model

2. Create and activate a virtual environment if needed:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies:

pip install pandas numpy lifelines matplotlib


## Quick start

Run Step 7 for both datasets:

python scripts/step7_validate_test_set.py --output-dir outputs/step7 --log-level INFO


## Optional commands

Run only UCSF:

python scripts/step7_validate_test_set.py --datasets ucsf --output-dir outputs/step7

Run only UPenn:

python scripts/step7_validate_test_set.py --datasets upenn --output-dir outputs/step7

Use custom input directories:

python scripts/step7_validate_test_set.py --step4-dir outputs/step4 --step5-dir outputs/step5 --step6-dir outputs/step6 --output-dir outputs/step7


## Output files

For each dataset, Step 7 writes:

- {dataset}_step7_test_with_assigned_clusters.csv
- {dataset}_step7_cluster_summary_test.csv
- {dataset}_step7_cluster_proportions_comparison.csv
- {dataset}_step7_kaplan_meier_test.png (if matplotlib available)
- {dataset}_step7_metadata.json
- step7.log

Folders:

- outputs/step7/ucsf/
- outputs/step7/upenn/


## Notes

1. Test assignment uses the exact Step 5 feature space and train-derived centroids.
2. Missing values in test clustering features are filled using the Step 5 train medians saved in the selection JSON.
3. The Step 6 high-risk cluster label is preserved and checked again on the test set rather than redefined.


## Troubleshooting

If you see missing packages:

pip install --upgrade pandas numpy lifelines matplotlib

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If Step 7 says Step 5 or Step 6 files are missing, run:

python scripts/step5_spectral_clustering.py --output-dir outputs/step5 --log-level INFO
python scripts/step6_characterize_clusters.py --output-dir outputs/step6 --log-level INFO