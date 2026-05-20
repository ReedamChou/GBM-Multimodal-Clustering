# GBM Training Summary

## Dataset

- Rows after label cleaning: `493`
- Imaging features used: `16`
- Class distribution: `{0: 276, 1: 217}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cuda`
- GPU available: `True`
- Device note: `GPU training available`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6347`
- Best baseline macro F1: `0.6356`

## XGBoost

- Validation balanced accuracy: `0.5796`
- Validation macro F1: `0.5795`
- Best Optuna CV balanced accuracy: `0.5965`
- Best params: `{'n_estimators': 283, 'learning_rate': 0.013940346079873234, 'max_depth': 8, 'subsample': 0.7200762468698007, 'colsample_bytree': 0.5610191174223894, 'min_child_weight': 5, 'gamma': 0.17194260557609198, 'reg_alpha': 3.5204810455260365, 'reg_lambda': 0.0019674328025306126}`

## Final Test Metrics

- Balanced accuracy: `0.6349`
- Macro F1: `0.6205`
- Quadratic kappa: `0.2605`
- AUC: `0.6644`

## Final Cross-Validation

- AUC: `0.6412`
- 95% CI: `0.5334` to `0.7490`
- One-sided p-value vs 0.5: `0.0079553`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `global_nc_en_ratio`, `temporal_en_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.236399 |
| 2 | `global_nc_en_ratio` | 0.147877 |
| 3 | `temporal_en_ratio` | 0.129662 |
| 4 | `frontal_ed_ratio` | 0.112372 |
| 5 | `temporal_ed_ratio` | 0.106476 |
| 6 | `global_ed_total_ratio` | 0.098073 |
| 7 | `frontal_nc_ratio` | 0.088950 |
| 8 | `global_ed_en_ratio` | 0.079278 |
| 9 | `parietal_ed_ratio` | 0.066235 |
| 10 | `temporal_nc_ratio` | 0.062399 |
| 11 | `tumor_burden_index` | 0.054821 |
| 12 | `occipital_ed_ratio` | 0.038139 |
| 13 | `occipital_nc_ratio` | 0.034870 |
| 14 | `occipital_en_ratio` | 0.033584 |
| 15 | `parietal_nc_ratio` | 0.018662 |
| 16 | `parietal_en_ratio` | 0.018554 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
