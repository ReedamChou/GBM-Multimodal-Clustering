# Step 5 Spectral Clustering on Train Set

## What this script does

This README documents scripts/step5_spectral_clustering.py.

The script implements Step 5 of your pipeline using Step 4 train clustering features.

For each dataset (UCSF and UPenn), it performs:

1. Loads Step 4 split metadata and train feature table.
2. Uses k values 2, 3, 4, 5, and 6 by default.
3. Runs sklearn.cluster.SpectralClustering for each k.
4. Computes multiple k-selection criteria for each successful k:
- silhouette score
- gap statistic
- train-resample stability using ARI
- train survival separation
- holdout survival consistency
- biological interpretability of the high-risk cluster
5. Combines those criteria into a multi-metric selection score instead of using silhouette alone.
6. Saves train patient cluster labels for the selected best k.
7. Computes and saves cluster centroids as mean feature vectors for each cluster.
8. Saves the full k-evaluation table and a summary selection JSON.
9. Generates a multi-metric k-evaluation plot PNG if matplotlib is available.


## Logger behavior

Each dataset has a dedicated log file:

- outputs/step5/ucsf/step5.log
- outputs/step5/upenn/step5.log

The logger records:

1. Input files and row or feature counts.
2. Per-k clustering status and multi-metric scores.
3. Selected best k and the weighted selection score.
4. Output paths written for labels, centroids, and evaluation tables.
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

pip install pandas numpy scikit-learn joblib matplotlib lifelines scipy


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

Control the new k-selection metrics:

python scripts/step5_spectral_clustering.py --gap-refs 5 --stability-resamples 8 --stability-sample-fraction 0.80 --output-dir outputs/step5

Use custom Step 4 folder:

python scripts/step5_spectral_clustering.py --step4-dir outputs/step4 --output-dir outputs/step5


## Output structure

For each dataset, Step 5 writes:

- {dataset}_step5_silhouette_scores.csv
- {dataset}_step5_k_evaluation.csv
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
4. Step 5 now uses the Step 4 holdout split as a model-selection validation proxy when scoring candidate k values.
5. That improves k selection, but it means the Step 4 holdout is no longer a pristine untouched final test set. For publication-grade final reporting, keep an additional external or nested test layer.


## Troubleshooting

If you see ModuleNotFoundError for pandas or sklearn:

pip install --upgrade pandas numpy scikit-learn joblib matplotlib

If PowerShell blocks virtual environment activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If Step 5 reports missing Step 4 files, run Step 4 first:

python scripts/step4_split_train_test.py --output-dir outputs/step4 --log-level INFO
