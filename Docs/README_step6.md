# Step 6 Cluster Characterization (Train Set)

## What this script does

This README documents scripts/step6_characterize_clusters.py.

Step 6 characterizes the training clusters created in Step 5 and produces the statistical summary needed for your results section.

For each dataset (UCSF and UPenn), the script:

1. Loads Step 4 train master table and Step 5 train cluster labels.
2. Merges patient rows with cluster labels.
3. Builds a cluster summary table with:
- n per cluster
- median OS with IQR (Q1, Q3)
- percentage MGMT methylated
- percentage IDH mutant
- mean global NC/EN ratio
- mean global ED/EN ratio
- mean tumor burden index (TBI)
- dominant lobe mode
- mean age
4. Runs survival analysis across clusters:
- Kaplan-Meier curves (train set)
- log-rank test
5. Runs Kruskal-Wallis tests for continuous features across clusters.
6. Runs chi-square tests for MGMT and lobe distribution across clusters.
7. Applies Benjamini-Hochberg correction to all inferential tests.
8. Identifies a high-risk cluster using rule-based scoring (low OS, high NC/EN, low MGMT methylation, temporal bonus).
9. Saves CSV outputs, metadata JSON, plot, and logs.


## Logger

Per-dataset log files are written to:

- outputs/step6/ucsf/step6.log
- outputs/step6/upenn/step6.log

The log includes:

1. Input files loaded.
2. Merge and cluster counts.
3. Survival test result.
4. High-risk cluster picked.
5. Output files written.


## Inputs required

Step 4 and Step 5 must already be completed.

Expected default files:

- outputs/step4/ucsf/ucsf_step4_split_metadata.json
- outputs/step4/ucsf/ucsf_train_master_table_step4.csv
- outputs/step4/upenn/upenn_step4_split_metadata.json
- outputs/step4/upenn/upenn_train_master_table_step4.csv

- outputs/step5/ucsf/ucsf_step5_selection.json
- outputs/step5/ucsf/ucsf_step5_train_cluster_labels.csv
- outputs/step5/upenn/upenn_step5_selection.json
- outputs/step5/upenn/upenn_step5_train_cluster_labels.csv


## Setup

1. Open terminal at project root:

C:/Users/Husain/Documents/AIml sideproject/GBM multi model

2. Create and activate a virtual environment (if needed):

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies:

pip install pandas numpy scipy lifelines matplotlib


## Quick start

Run Step 6 for both datasets:

python scripts/step6_characterize_clusters.py --output-dir outputs/step6 --log-level INFO


## Optional commands

Run only UCSF:

python scripts/step6_characterize_clusters.py --datasets ucsf --output-dir outputs/step6

Run only UPenn:

python scripts/step6_characterize_clusters.py --datasets upenn --output-dir outputs/step6

Use custom input directories:

python scripts/step6_characterize_clusters.py --step4-dir outputs/step4 --step5-dir outputs/step5 --output-dir outputs/step6


## Output files

For each dataset, Step 6 writes:

- {dataset}_step6_cluster_summary_train.csv
- {dataset}_step6_train_with_clusters.csv
- {dataset}_step6_kruskal_tests.csv
- {dataset}_step6_chi_square_tests.csv
- {dataset}_step6_inferential_tests_bh.csv
- {dataset}_step6_kaplan_meier_train.png (if matplotlib available)
- {dataset}_step6_metadata.json
- step6.log

Folders:

- outputs/step6/ucsf/
- outputs/step6/upenn/


## Notes

1. BH correction is applied across all inferential tests generated in Step 6.
2. Step 6 is train-set characterization only. Test-set validation is Step 7.
3. The high-risk label is rule-based and intended to support your project hypothesis.


## Troubleshooting

If you see missing packages:

pip install --upgrade pandas numpy scipy lifelines matplotlib

If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If Step 6 says Step 5 files are missing, run Step 5 first:

python scripts/step5_spectral_clustering.py --output-dir outputs/step5 --log-level INFO