# GBM Training Summary

## Dataset

- Rows after label cleaning: `446`
- Imaging features used: `64`
- Class distribution: `{0: 237, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cpu`
- GPU available: `False`
- Device note: `SVM uses CPU training`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `GradientBoosting`
- Best baseline balanced accuracy: `0.5475`
- Best baseline macro F1: `0.5473`

## SVM (RBF)

- Validation balanced accuracy: `0.5072`
- Validation macro F1: `0.4007`
- Best Optuna CV balanced accuracy: `0.5389`
- Best params: `{'C': 4.148801564779672, 'gamma': 0.002372493834791724}`

## Final Test Metrics

- Balanced accuracy: `0.5865`
- Macro F1: `0.5820`
- Quadratic kappa: `0.1706`
- AUC: `0.6443`

## Final Cross-Validation

- AUC: `0.5604`
- 95% CI: `0.4923` to `0.6285`
- One-sided p-value vs 0.5: `0.0378048`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`t1_occipital_en_ratio`, `flair_occipital_en_ratio`, `t1_temporal_nc_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `t1_occipital_en_ratio` | 0.005662 |
| 2 | `flair_occipital_en_ratio` | 0.002159 |
| 3 | `t1_temporal_nc_ratio` | 0.001754 |
| 4 | `t1_frontal_en_ratio` | 0.001714 |
| 5 | `t1gd_occipital_en_ratio` | 0.001637 |
| 6 | `flair_parietal_nc_ratio` | 0.001582 |
| 7 | `t2_temporal_nc_ratio` | 0.001501 |
| 8 | `t1gd_frontal_en_ratio` | 0.001443 |
| 9 | `flair_frontal_en_ratio` | 0.001415 |
| 10 | `t2_frontal_en_ratio` | 0.001348 |
| 11 | `t2_parietal_nc_ratio` | 0.001337 |
| 12 | `t1gd_global_ed_en_ratio` | 0.001231 |
| 13 | `flair_temporal_en_ratio` | 0.001208 |
| 14 | `t1gd_parietal_nc_ratio` | 0.001179 |
| 15 | `t1_global_ed_en_ratio` | 0.001175 |
| 16 | `t2_global_ed_en_ratio` | 0.001168 |
| 17 | `flair_temporal_nc_ratio` | 0.001136 |
| 18 | `t1gd_temporal_nc_ratio` | 0.001034 |
| 19 | `t1_parietal_nc_ratio` | 0.001032 |
| 20 | `t1gd_temporal_en_ratio` | 0.001005 |
| 21 | `t2_occipital_en_ratio` | 0.000887 |
| 22 | `flair_global_ed_en_ratio` | 0.000886 |
| 23 | `t1gd_parietal_en_ratio` | 0.000869 |
| 24 | `flair_frontal_nc_ratio` | 0.000835 |
| 25 | `t2_frontal_nc_ratio` | 0.000802 |
| 26 | `t1gd_frontal_nc_ratio` | 0.000789 |
| 27 | `t1_temporal_en_ratio` | 0.000618 |
| 28 | `flair_occipital_ed_ratio` | 0.000594 |
| 29 | `t2_temporal_en_ratio` | 0.000570 |
| 30 | `t1_frontal_nc_ratio` | 0.000543 |
| 31 | `t1gd_global_ed_total_ratio` | 0.000474 |
| 32 | `t2_tumor_burden_index` | 0.000450 |
| 33 | `t1_parietal_en_ratio` | 0.000445 |
| 34 | `t1gd_parietal_ed_ratio` | 0.000417 |
| 35 | `t1gd_occipital_ed_ratio` | 0.000411 |
| 36 | `t2_global_ed_total_ratio` | 0.000381 |
| 37 | `t1_occipital_ed_ratio` | 0.000368 |
| 38 | `flair_global_nc_en_ratio` | 0.000353 |
| 39 | `t2_occipital_ed_ratio` | 0.000328 |
| 40 | `t1_tumor_burden_index` | 0.000326 |
| 41 | `t1_frontal_ed_ratio` | 0.000319 |
| 42 | `t1_global_nc_en_ratio` | 0.000301 |
| 43 | `t2_parietal_en_ratio` | 0.000300 |
| 44 | `flair_occipital_nc_ratio` | 0.000295 |
| 45 | `t2_global_nc_en_ratio` | 0.000293 |
| 46 | `t1_global_ed_total_ratio` | 0.000290 |
| 47 | `flair_global_ed_total_ratio` | 0.000233 |
| 48 | `t1_temporal_ed_ratio` | 0.000207 |
| 49 | `t1gd_tumor_burden_index` | 0.000194 |
| 50 | `t2_parietal_ed_ratio` | 0.000191 |
| 51 | `flair_frontal_ed_ratio` | 0.000180 |
| 52 | `t1_parietal_ed_ratio` | 0.000178 |
| 53 | `t1_occipital_nc_ratio` | 0.000158 |
| 54 | `flair_parietal_en_ratio` | 0.000142 |
| 55 | `t2_temporal_ed_ratio` | 0.000133 |
| 56 | `t2_frontal_ed_ratio` | 0.000118 |
| 57 | `t2_occipital_nc_ratio` | 0.000088 |
| 58 | `flair_temporal_ed_ratio` | 0.000080 |
| 59 | `flair_tumor_burden_index` | 0.000078 |
| 60 | `t1gd_global_nc_en_ratio` | 0.000078 |
| 61 | `t1gd_frontal_ed_ratio` | 0.000075 |
| 62 | `t1gd_temporal_ed_ratio` | 0.000071 |
| 63 | `t1gd_occipital_nc_ratio` | 0.000038 |
| 64 | `flair_parietal_ed_ratio` | 0.000006 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
