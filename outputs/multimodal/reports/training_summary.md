# GBM Training Summary

## Dataset

- Rows after label cleaning: `446`
- Imaging features used: `64`
- Class distribution: `{0: 237, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `gpu`
- GPU available: `True`
- Device note: `XGBoost configured with {'device': 'cuda', 'tree_method': 'hist'}`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `GradientBoosting`
- Best baseline balanced accuracy: `0.5475`
- Best baseline macro F1: `0.5473`

## XGBoost

- Validation balanced accuracy: `0.5591`
- Validation macro F1: `0.5573`
- Best Optuna CV balanced accuracy: `0.6453`
- Best params: `{'max_depth': 8, 'learning_rate': 0.19894207370181136, 'n_estimators': 120, 'subsample': 0.7447439543868675, 'colsample_bytree': 0.7013263590091331, 'gamma': 0.0870604853187095, 'min_child_weight': 9, 'reg_alpha': 0.005345974934598897, 'reg_lambda': 0.1758668915442357}`

## Final Test Metrics

- Balanced accuracy: `0.5681`
- Macro F1: `0.5668`
- Quadratic kappa: `0.1353`
- AUC: `0.5708`

## Final Cross-Validation

- AUC: `0.5611`
- 95% CI: `0.5166` to `0.6056`
- One-sided p-value vs 0.5: `0.00627409`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`t2_frontal_en_ratio`, `t1_frontal_en_ratio`, `t1_parietal_ed_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `t2_frontal_en_ratio` | 0.312866 |
| 2 | `t1_frontal_en_ratio` | 0.292053 |
| 3 | `t1_parietal_ed_ratio` | 0.236139 |
| 4 | `t2_temporal_nc_ratio` | 0.206280 |
| 5 | `t1_frontal_nc_ratio` | 0.204205 |
| 6 | `t1_occipital_nc_ratio` | 0.166503 |
| 7 | `t1_global_ed_en_ratio` | 0.165393 |
| 8 | `t1_frontal_ed_ratio` | 0.158047 |
| 9 | `t1_global_nc_en_ratio` | 0.156750 |
| 10 | `t2_temporal_en_ratio` | 0.156232 |
| 11 | `t2_frontal_ed_ratio` | 0.153247 |
| 12 | `t1_temporal_ed_ratio` | 0.146156 |
| 13 | `t1_global_ed_total_ratio` | 0.132261 |
| 14 | `t2_temporal_ed_ratio` | 0.130935 |
| 15 | `t2_occipital_ed_ratio` | 0.130097 |
| 16 | `t1gd_temporal_en_ratio` | 0.126441 |
| 17 | `t1_temporal_nc_ratio` | 0.122118 |
| 18 | `t1gd_frontal_en_ratio` | 0.119906 |
| 19 | `t1_parietal_en_ratio` | 0.119474 |
| 20 | `t1gd_temporal_ed_ratio` | 0.114222 |
| 21 | `t2_global_ed_total_ratio` | 0.110629 |
| 22 | `t1_tumor_burden_index` | 0.106158 |
| 23 | `t2_parietal_en_ratio` | 0.103180 |
| 24 | `t1_occipital_ed_ratio` | 0.101585 |
| 25 | `t2_parietal_ed_ratio` | 0.099563 |
| 26 | `t1_temporal_en_ratio` | 0.096940 |
| 27 | `t1gd_global_nc_en_ratio` | 0.092423 |
| 28 | `t2_global_ed_en_ratio` | 0.090493 |
| 29 | `t1gd_occipital_ed_ratio` | 0.089501 |
| 30 | `t1gd_parietal_en_ratio` | 0.086161 |
| 31 | `t1_parietal_nc_ratio` | 0.081000 |
| 32 | `t2_tumor_burden_index` | 0.079175 |
| 33 | `flair_global_ed_en_ratio` | 0.075567 |
| 34 | `t1gd_tumor_burden_index` | 0.073582 |
| 35 | `flair_occipital_en_ratio` | 0.071709 |
| 36 | `t2_global_nc_en_ratio` | 0.070647 |
| 37 | `t1gd_frontal_nc_ratio` | 0.069688 |
| 38 | `t2_frontal_nc_ratio` | 0.067738 |
| 39 | `flair_temporal_nc_ratio` | 0.067156 |
| 40 | `t1gd_parietal_ed_ratio` | 0.063679 |
| 41 | `t2_occipital_nc_ratio` | 0.058105 |
| 42 | `t2_parietal_nc_ratio` | 0.056953 |
| 43 | `t1_occipital_en_ratio` | 0.054845 |
| 44 | `flair_parietal_ed_ratio` | 0.053648 |
| 45 | `flair_frontal_en_ratio` | 0.051740 |
| 46 | `t1gd_global_ed_total_ratio` | 0.045440 |
| 47 | `flair_frontal_nc_ratio` | 0.041182 |
| 48 | `flair_frontal_ed_ratio` | 0.040107 |
| 49 | `flair_tumor_burden_index` | 0.039857 |
| 50 | `flair_temporal_en_ratio` | 0.033578 |
| 51 | `t1gd_frontal_ed_ratio` | 0.032112 |
| 52 | `t1gd_temporal_nc_ratio` | 0.032111 |
| 53 | `t1gd_occipital_nc_ratio` | 0.030436 |
| 54 | `flair_temporal_ed_ratio` | 0.028240 |
| 55 | `t2_occipital_en_ratio` | 0.025612 |
| 56 | `t1gd_parietal_nc_ratio` | 0.024675 |
| 57 | `flair_global_ed_total_ratio` | 0.014929 |
| 58 | `flair_global_nc_en_ratio` | 0.004988 |
| 59 | `t1gd_global_ed_en_ratio` | 0.000000 |
| 60 | `t1gd_occipital_en_ratio` | 0.000000 |
| 61 | `flair_parietal_en_ratio` | 0.000000 |
| 62 | `flair_parietal_nc_ratio` | 0.000000 |
| 63 | `flair_occipital_ed_ratio` | 0.000000 |
| 64 | `flair_occipital_nc_ratio` | 0.000000 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
