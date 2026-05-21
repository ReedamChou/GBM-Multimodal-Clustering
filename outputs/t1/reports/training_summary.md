# GBM Training Summary

## Dataset

- Rows after label cleaning: `446`
- Imaging features used: `16`
- Class distribution: `{0: 237, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cuda`
- GPU available: `True`
- Device note: `GPU training available`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `GradientBoosting`
- Best baseline balanced accuracy: `0.5175`
- Best baseline macro F1: `0.5171`

## XGBoost

- Validation balanced accuracy: `0.5797`
- Validation macro F1: `0.5797`
- Best Optuna CV balanced accuracy: `0.6095`
- Best params: `{'n_estimators': 557, 'learning_rate': 0.22905398292065163, 'max_depth': 10, 'subsample': 0.5479436509786584, 'colsample_bytree': 0.6021349895060872, 'min_child_weight': 10, 'gamma': 1.3860777742685086, 'reg_alpha': 0.0035022259288765485, 'reg_lambda': 0.0006705809023851751}`

## Final Test Metrics

- Balanced accuracy: `0.6142`
- Macro F1: `0.6119`
- Quadratic kappa: `0.2265`
- AUC: `0.6496`

## Final Cross-Validation

- AUC: `0.5805`
- 95% CI: `0.5084` to `0.6527`
- One-sided p-value vs 0.5: `0.0162174`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `temporal_nc_ratio`, `temporal_en_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.812087 |
| 2 | `temporal_nc_ratio` | 0.639669 |
| 3 | `temporal_en_ratio` | 0.605806 |
| 4 | `parietal_en_ratio` | 0.519120 |
| 5 | `parietal_nc_ratio` | 0.491990 |
| 6 | `frontal_nc_ratio` | 0.370429 |
| 7 | `temporal_ed_ratio` | 0.364798 |
| 8 | `parietal_ed_ratio` | 0.348610 |
| 9 | `global_nc_en_ratio` | 0.340516 |
| 10 | `global_ed_en_ratio` | 0.316683 |
| 11 | `occipital_nc_ratio` | 0.295932 |
| 12 | `global_ed_total_ratio` | 0.285907 |
| 13 | `occipital_ed_ratio` | 0.222648 |
| 14 | `frontal_ed_ratio` | 0.208269 |
| 15 | `tumor_burden_index` | 0.178661 |
| 16 | `occipital_en_ratio` | 0.026495 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
