# README - Step 9 Cluster Stability

## Purpose

Step 9 measures whether the train-set clusters discovered in Step 5 are reproducible under resampling. This complements Step 7 validation:

- Step 7 asks: do train-derived clusters still separate held-out test patients?
- Step 9 asks: if we perturb the train set, do we recover essentially the same clusters?

The step adds:

- bootstrap resampling
- consensus clustering
- pairwise adjusted Rand index (ARI) across bootstrap runs
- co-clustering frequency heatmaps

## Inputs

Step 9 reads:

- `outputs/step4/<dataset>/<dataset>_train_clustering_features_step4.csv`
- `outputs/step5/<dataset>/<dataset>_step5_selection.json`
- `outputs/step5/<dataset>/<dataset>_step5_train_cluster_labels.csv`

It uses the Step 5 `best_k` as the fixed cluster count for stability analysis.

## What It Does

For each dataset:

1. Load the Step 4 train feature matrix.
2. Load the Step 5 full-train cluster labels and selected `k`.
3. Run spectral clustering on many bootstrap resamples of the train set.
4. For each bootstrap run:
   - collapse duplicate bootstrap samples back to original patients
   - compare the bootstrap clustering to the full-train clustering with ARI
5. Compare every pair of bootstrap runs with ARI on their overlapping patients.
6. Build a consensus matrix:
   - numerator = how often two patients were assigned to the same cluster
   - denominator = how often those two patients were sampled together
7. Cluster the consensus matrix to obtain a consensus partition.
8. Save a heatmap of the co-clustering matrix ordered by consensus cluster.

## Main Outputs

For each dataset, Step 9 writes:

- `*_step9_bootstrap_runs.csv`
  - one row per bootstrap run
  - includes ARI vs the full-train clustering
- `*_step9_pairwise_ari.csv`
  - ARI between pairs of bootstrap runs
- `*_step9_consensus_matrix.csv`
  - patient-by-patient co-clustering frequencies
- `*_step9_consensus_labels.csv`
  - baseline Step 5 labels and consensus-cluster labels
- `*_step9_patient_stability.csv`
  - per-patient stability summaries
- `*_step9_consensus_heatmap.png`
  - visualization of the consensus matrix
- `*_step9_summary.json`
  - compact summary of ARI and consensus-strength metrics

## How To Run

```bash
venv-gbm/bin/python scripts/step9_cluster_stability.py \
  --datasets ucsf upenn \
  --step4-dir outputs/step4 \
  --step5-dir outputs/step5 \
  --output-dir outputs/step9 \
  --n-bootstraps 100 \
  --bootstrap-fraction 1.0 \
  --random-state 42
```

## How To Read The Results

- Higher `consensus_vs_full_train_ari` means the consensus partition agrees well with the original Step 5 clustering.
- Higher `pairwise_bootstrap_ari` means repeated bootstrap clusterings are more reproducible.
- Higher `mean_within_cluster` and lower `mean_between_cluster` in the consensus summary mean cleaner cluster separation.
- A strong heatmap shows block-like structure along the diagonal after ordering by consensus cluster.

Rule of thumb:

- ARI near `1` suggests strong reproducibility.
- ARI around `0.5` to `0.7` suggests moderate reproducibility.
- ARI near `0` suggests weak reproducibility.

These thresholds are only rough heuristics and should be interpreted together with the biological and survival findings from Steps 6 to 8.
