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
- Best Optuna CV balanced accuracy: `0.5806`
- Best params: `{'n_estimators': 550, 'learning_rate': 0.013105962038574032, 'max_depth': 7, 'subsample': 0.7674467470112066, 'colsample_bytree': 0.8024727592169401, 'min_child_weight': 8, 'gamma': 0.41047614636699714, 'reg_alpha': 0.3765727380736835, 'reg_lambda': 8.093768208366937}`

## Final Test Metrics

- Balanced accuracy: `0.6545`
- Macro F1: `0.6443`
- Quadratic kappa: `0.3038`
- AUC: `0.6406`

## Final Cross-Validation

- AUC: `0.5934`
- 95% CI: `0.5176` to `0.6693`
- One-sided p-value vs 0.5: `0.0106036`

## SHAP Feature Importance (High-Risk Class)

Top 3 dominant features: **`frontal_en_ratio`, `temporal_en_ratio`, `global_nc_en_ratio`**

| Rank | Feature | Mean |SHAP| |
|------|---------|-------------|
| 1 | `frontal_en_ratio` | 0.275313 |
| 2 | `temporal_en_ratio` | 0.195932 |
| 3 | `global_nc_en_ratio` | 0.169167 |
| 4 | `frontal_ed_ratio` | 0.143836 |
| 5 | `global_ed_en_ratio` | 0.133848 |
| 6 | `temporal_nc_ratio` | 0.121770 |
| 7 | `frontal_nc_ratio` | 0.112231 |
| 8 | `occipital_en_ratio` | 0.104958 |
| 9 | `global_ed_total_ratio` | 0.091814 |
| 10 | `occipital_ed_ratio` | 0.077660 |
| 11 | `temporal_ed_ratio` | 0.077412 |
| 12 | `tumor_burden_index` | 0.065304 |
| 13 | `parietal_ed_ratio` | 0.063832 |
| 14 | `parietal_en_ratio` | 0.055929 |
| 15 | `occipital_nc_ratio` | 0.051997 |
| 16 | `parietal_nc_ratio` | 0.033646 |

> SHAP plots saved to `outputs/figures/shap_summary.png` and `shap_importance.png`.
