# UCSF Comparative Analysis: Iteration 4 vs Iteration 7 vs My New Pipeline

This report extends your friend's UCSF comparison by adding a third run column for your current new pipeline.

## Scope
- Iteration 4 and Iteration 7 values are from your provided comparison text.
- My New Pipeline values are from current artifacts under outputs (step3, step4, step4b, step5, step6, step7, step5b).
- This is UCSF-only.

## 1) Metrics Comparison (UCSF only)

### 1.1 Single-split pipeline metrics (Steps 4-7)

| Metric | Iteration 4 (UCSF) | Iteration 7 (UCSF) | My New Pipeline (UCSF) | My New Pipeline source |
|---|---:|---:|---:|---|
| Step 3 cohort rows / columns | 501 / 31 | 501 / 31 | 501 / 85 | outputs/step3/ucsf/ucsf_step3_metadata.json |
| Step 4 train / test rows | 350 / 151 | 400 / 101 | 350 / 151 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Step 4 test size | 0.30 | 0.20 | 0.30 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Step 5 selected k | 2 | 3 | 5 | outputs/step5/ucsf/ucsf_step5_selection.json |
| Step 6 train log-rank p | 4.79e-11 | 1.322e-10 | 1.1297e-09 | outputs/step6/ucsf/ucsf_step6_metadata.json |
| Step 7 test log-rank p | 1.702e-06 | 0.01429 | 9.4013e-05 | outputs/step7/ucsf/ucsf_step7_metadata.json |
| High-risk cluster label | cluster 0 | cluster 1 | cluster 3 | outputs/step6/ucsf/ucsf_step6_metadata.json |
| High-risk proportion (train / test) | 83.14% / 75.50% | 10.50% / 9.90% | 13.43% / 19.21% | outputs/step7/ucsf/ucsf_step7_metadata.json |
| High-risk cluster dominant lobe (train / test) | frontal / frontal | parietal / parietal | frontal / frontal | outputs/step7/ucsf/ucsf_step7_metadata.json |
| High-risk NC/EN rank on test | 1 | 1 | 1 | outputs/step7/ucsf/ucsf_step7_metadata.json |

### 1.2 Repeated validation metrics (30 seeded splits)

| Metric | Iteration 4 (UCSF) | Iteration 7 (UCSF) | My New Pipeline (UCSF) | My New Pipeline source |
|---|---:|---:|---:|---|
| Repeated splits (seeds) | 30 | 30 | N/A (not exported in current outputs) | N/A |
| Outer test size | 0.30 | 0.20 | N/A (not exported in current outputs) | N/A |
| Chosen k counts | k=2: 17; k=3: 11; k=4: 2 | k=2: 2; k=3: 24; k=4: 3; k=6: 1 | N/A (not exported in current outputs) | N/A |
| Train survival significant | 30/30 | 30/30 | N/A (not exported in current outputs) | N/A |
| Test survival significant | 30/30 | 29/30 | N/A (not exported in current outputs) | N/A |
| Median train log-rank p | 5.087e-10 | 5.708e-09 | N/A (not exported in current outputs) | N/A |
| Median test log-rank p | 4.185e-05 | 6.851e-04 | N/A (not exported in current outputs) | N/A |
| High-risk proportion within 10pp | 29/30 (96.7%) | 30/30 (100%) | N/A (not exported in current outputs) | N/A |
| Dominant lobe matched train | 29/30 (96.7%) | 28/30 (93.3%) | N/A (not exported in current outputs) | N/A |
| High-risk NC/EN rank-1 on test | 22/30 (73.3%) | 25/30 (83.3%) | N/A (not exported in current outputs) | N/A |
| High-risk NC/EN rank 1 or 2 | 29/30 (96.7%) | 29/30 (96.7%) | N/A (not exported in current outputs) | N/A |

### 1.3 Train-split cluster stability (100 bootstraps)

| Metric | Iteration 4 (UCSF) | Iteration 7 (UCSF) | My New Pipeline (UCSF) | My New Pipeline source |
|---|---:|---:|---:|---|
| best_k_from_step5 | 2 | 3 | 5 | outputs/step5/ucsf/ucsf_step5_selection.json |
| Baseline silhouette | 0.1519 | 0.1784 | 0.1059 | outputs/step5/ucsf/ucsf_step5_selection.json |
| Consensus vs original ARI | 0.9430 | 0.9278 | 0.6437 (at k=5) | outputs/step5b/ucsf/ucsf_step5b_summary.json |
| Bootstrap ARI vs full train (mean / median) | 0.9431 / 0.9598 | 0.9491 / 0.9557 | N/A (not exported in current outputs) | N/A |
| Bootstrap ARI vs full train (p05 / p95) | 0.8705 / 1.0000 | 0.9029 / 0.9863 | N/A (not exported in current outputs) | N/A |
| Pairwise bootstrap ARI (mean / median) | 0.9108 / 0.9591 | 0.9316 / 0.9349 | N/A (not exported in current outputs) | N/A |
| Consensus strength within / between | 0.9716 / 0.03236 | 0.9620 / 0.01711 | N/A (not exported in current outputs) | N/A |
| Consensus stability gap | 0.9393 | 0.9449 | N/A (not exported in current outputs) | N/A |

## 2) Step 4 Configuration Comparison (UCSF)

| Step 4 setting | Iteration 4 | Iteration 7 | My New Pipeline | Evidence |
|---|---:|---:|---:|---|
| test_size | 0.30 | 0.20 | 0.30 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| random_state | 42 | 42 | 42 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Train / test rows | 350 / 151 | 400 / 101 | 350 / 151 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Scaler | StandardScaler | StandardScaler | StandardScaler | outputs/step4/ucsf/ucsf_train_scaler_step4.joblib |
| Power transform | None | Yeo-Johnson | None at Step 4 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Correlation pruning | Disabled | Enabled (|r| >= 0.90) | Disabled at Step 4 | outputs/step4/ucsf/ucsf_step4_split_metadata.json |
| Feature count (excl. id) | 27 | 23 | 58 at Step 4, then 35 after Step 4b VIF | outputs/step4/ucsf/ucsf_step4_split_metadata.json and outputs/step4b/ucsf/ucsf_step4b_metadata.json |
| Dropped features | none | 4 x *_en_ratio | 23 removed by VIF in Step 4b (58 -> 35) | outputs/step4b/ucsf/ucsf_step4b_metadata.json |

## 3) Practical Reading Notes

- Iteration 7 and My New Pipeline both move away from the very large high-risk group seen in Iteration 4.
- My New Pipeline currently has stronger single-split test separation than Iteration 7, but weaker than Iteration 4.
- Not all repeated-validation and bootstrap stability fields are available in current outputs, so those cells are marked N/A.

## 4) Artifacts Used for My New Pipeline Column

- outputs/step3/ucsf/ucsf_step3_metadata.json
- outputs/step4/ucsf/ucsf_step4_split_metadata.json
- outputs/step4b/ucsf/ucsf_step4b_metadata.json
- outputs/step5/ucsf/ucsf_step5_selection.json
- outputs/step6/ucsf/ucsf_step6_metadata.json
- outputs/step7/ucsf/ucsf_step7_metadata.json
- outputs/step5b/ucsf/ucsf_step5b_summary.json
