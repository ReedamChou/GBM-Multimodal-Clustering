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

- Validation balanced accuracy: `0.5714`
- Validation macro F1: `0.5716`
- Best Optuna CV balanced accuracy: `0.6071`
- Best params: `{'n_estimators': 249, 'learning_rate': 0.011569046886474444, 'max_depth': 8, 'subsample': 0.7347049478829597, 'colsample_bytree': 0.5496480501205234, 'min_child_weight': 5, 'gamma': 1.3170773995337703, 'reg_alpha': 3.717642135537067, 'reg_lambda': 0.0023232967562617536}`

## Final Test Metrics

- Balanced accuracy: `0.6438`
- Macro F1: `0.6146`
- Quadratic kappa: `0.2730`
- AUC: `0.7103`

## Final Cross-Validation

- AUC: `0.6432`
- 95% CI: `0.5441` to `0.7423`
- One-sided p-value vs 0.5: `0.00485937`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `temporal_en_ratio`, `global_nc_en_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.198355 |
| 2 | `temporal_en_ratio` | 0.117982 |
| 3 | `global_nc_en_ratio` | 0.097059 |
| 4 | `frontal_ed_ratio` | 0.084464 |
| 5 | `global_ed_total_ratio` | 0.078764 |
| 6 | `temporal_ed_ratio` | 0.061800 |
| 7 | `tumor_burden_index` | 0.053977 |
| 8 | `frontal_nc_ratio` | 0.052009 |
| 9 | `global_ed_en_ratio` | 0.045906 |
| 10 | `temporal_nc_ratio` | 0.039622 |
| 11 | `parietal_ed_ratio` | 0.037321 |
| 12 | `occipital_nc_ratio` | 0.016846 |
| 13 | `occipital_ed_ratio` | 0.015085 |
| 14 | `parietal_en_ratio` | 0.010010 |
| 15 | `occipital_en_ratio` | 0.008197 |
| 16 | `parietal_nc_ratio` | 0.007419 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
