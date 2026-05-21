# GBM Training Summary

## Dataset

- Rows after label cleaning: `449`
- Imaging features used: `16`
- Class distribution: `{0: 240, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cuda`
- GPU available: `True`
- Device note: `GPU training available`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6285`
- Best baseline macro F1: `0.6284`

## XGBoost

- Validation balanced accuracy: `0.5712`
- Validation macro F1: `0.5712`
- Best Optuna CV balanced accuracy: `0.5868`
- Best params: `{'n_estimators': 638, 'learning_rate': 0.07405974009235358, 'max_depth': 4, 'subsample': 0.863768631339208, 'colsample_bytree': 0.9918704573610522, 'min_child_weight': 9, 'gamma': 3.4904832000270054, 'reg_alpha': 0.00011431541785080484, 'reg_lambda': 9.315666345772176}`

## Final Test Metrics

- Balanced accuracy: `0.6267`
- Macro F1: `0.6123`
- Quadratic kappa: `0.2483`
- AUC: `0.6337`

## Final Cross-Validation

- AUC: `0.6088`
- 95% CI: `0.5351` to `0.6825`
- One-sided p-value vs 0.5: `0.00432496`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `temporal_en_ratio`, `global_nc_en_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.247184 |
| 2 | `temporal_en_ratio` | 0.184973 |
| 3 | `global_nc_en_ratio` | 0.142530 |
| 4 | `frontal_ed_ratio` | 0.124135 |
| 5 | `occipital_en_ratio` | 0.112097 |
| 6 | `global_ed_en_ratio` | 0.105900 |
| 7 | `frontal_nc_ratio` | 0.070239 |
| 8 | `global_ed_total_ratio` | 0.052297 |
| 9 | `temporal_ed_ratio` | 0.044441 |
| 10 | `temporal_nc_ratio` | 0.040539 |
| 11 | `parietal_en_ratio` | 0.034944 |
| 12 | `occipital_ed_ratio` | 0.034149 |
| 13 | `tumor_burden_index` | 0.030920 |
| 14 | `parietal_ed_ratio` | 0.016911 |
| 15 | `occipital_nc_ratio` | 0.010862 |
| 16 | `parietal_nc_ratio` | 0.000000 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
