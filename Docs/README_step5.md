# Step 5 Spectral Clustering on Train Set

## What this script does

This README documents scripts/step5_spectral_clustering.py.

The script implements Step 5 of your pipeline using Step 4 train clustering features.

For each dataset (UCSF and UPenn), it performs:

1. Loads Step 4 split metadata and train feature table.
2. Uses k values 2, 3, 4, 5, and 6 by default.
3. Runs sklearn.cluster.SpectralClustering for each k.
4. Computes silhouette score for each successful k using sklearn.metrics.silhouette_score.
5. Selects the best k based on highest silhouette score.
6. Saves train patient cluster labels for the selected best k.
7. Computes and saves cluster centroids as mean feature vectors for each cluster.
8. Saves a silhouette scores table and a summary selection JSON.
9. Generates a silhouette elbow plot PNG if matplotlib is available.


## Logger behavior

Each dataset has a dedicated log file:

- outputs/step5/ucsf/step5.log
- outputs/step5/upenn/step5.log

The logger records:

1. Input files and row or feature counts.
2. Per-k clustering status and silhouette values.
3. Selected best k and best silhouette.
4. Output paths written for labels, centroids, and scores.
5. Plot generation status.


## Input requirements

Step 4 must already be completed.

Expected default files:

- outputs/step4/ucsf/ucsf_step4_split_metadata.json
- outputs/step4/ucsf/ucsf_train_clustering_features_step4.csv
- outputs/step4/upenn/upenn_step4_split_metadata.json
- outputs/step4/upenn/upenn_train_clustering_features_step4.csv


## Setup

1. Open terminal at project root:

C:/Users/Husain/Documents/AIml sideproject/GBM multi model

2. Create and activate virtual environment if needed:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install required packages:

pip install pandas numpy scikit-learn joblib matplotlib


## Quick start

Run Step 5 for both datasets with defaults:

python scripts/step5_spectral_clustering.py --output-dir outputs/step5 --log-level INFO


## More run options

Run only UCSF:

python scripts/step5_spectral_clustering.py --datasets ucsf --output-dir outputs/step5

Run only UPenn:

python scripts/step5_spectral_clustering.py --datasets upenn --output-dir outputs/step5

Override k values:

python scripts/step5_spectral_clustering.py --k-values 2 3 4 5 6 --output-dir outputs/step5

Change random seed and neighbors:

python scripts/step5_spectral_clustering.py --random-state 42 --n-neighbors 10 --output-dir outputs/step5

Use custom Step 4 folder:

python scripts/step5_spectral_clustering.py --step4-dir outputs/step4 --output-dir outputs/step5


## Output structure

For each dataset, Step 5 writes:

- {dataset}_step5_silhouette_scores.csv
- {dataset}_step5_train_cluster_labels.csv
- {dataset}_step5_cluster_centroids.csv
- {dataset}_step5_selection.json
- {dataset}_step5_silhouette_elbow.png (if matplotlib is available)
- step5.log

Folders:

- outputs/step5/ucsf/
- outputs/step5/upenn/


## Notes

1. Spectral clustering in sklearn does not provide a direct predict method for unseen data.
2. For Step 7, use centroids and nearest-centroid assignment for test patients, as your pipeline specifies.
3. Centroids are computed in feature space as the mean vector of training samples per cluster.


## Troubleshooting

If you see ModuleNotFoundError for pandas or sklearn:

pip install --upgrade pandas numpy scikit-learn joblib matplotlib

If PowerShell blocks virtual environment activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If Step 5 reports missing Step 4 files, run Step 4 first:

python scripts/step4_split_train_test.py --output-dir outputs/step4 --log-level INFO
