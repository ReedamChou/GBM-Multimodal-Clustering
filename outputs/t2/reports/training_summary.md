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

- Validation balanced accuracy: `0.5439`
- Validation macro F1: `0.5440`
- Best Optuna CV balanced accuracy: `0.5969`
- Best params: `{'n_estimators': 541, 'learning_rate': 0.02617851131913525, 'max_depth': 7, 'subsample': 0.5940628175743564, 'colsample_bytree': 0.5208403597369262, 'min_child_weight': 8, 'gamma': 4.471732207284308, 'reg_alpha': 0.16503714191870786, 'reg_lambda': 0.22075306926862642}`

## Final Test Metrics

- Balanced accuracy: `0.6622`
- Macro F1: `0.6476`
- Quadratic kappa: `0.3133`
- AUC: `0.6696`

## Final Cross-Validation

- AUC: `0.6430`
- 95% CI: `0.5455` to `0.7406`
- One-sided p-value vs 0.5: `0.00449898`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `global_nc_en_ratio`, `temporal_ed_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.280930 |
| 2 | `global_nc_en_ratio` | 0.170304 |
| 3 | `temporal_ed_ratio` | 0.152042 |
| 4 | `temporal_en_ratio` | 0.135426 |
| 5 | `global_ed_total_ratio` | 0.133875 |
| 6 | `frontal_ed_ratio` | 0.123366 |
| 7 | `temporal_nc_ratio` | 0.104387 |
| 8 | `global_ed_en_ratio` | 0.082334 |
| 9 | `frontal_nc_ratio` | 0.067600 |
| 10 | `occipital_nc_ratio` | 0.057634 |
| 11 | `tumor_burden_index` | 0.053800 |
| 12 | `parietal_ed_ratio` | 0.049388 |
| 13 | `occipital_ed_ratio` | 0.047876 |
| 14 | `parietal_en_ratio` | 0.035587 |
| 15 | `occipital_en_ratio` | 0.033912 |
| 16 | `parietal_nc_ratio` | 0.004748 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
