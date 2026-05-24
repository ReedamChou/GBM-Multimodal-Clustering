# GBM Training Summary

## Dataset

- Rows after label cleaning: `446`
- Imaging features used: `16`
- Class distribution: `{0: 237, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cpu`
- GPU available: `False`
- Device note: `SVM uses CPU training`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `GradientBoosting`
- Best baseline balanced accuracy: `0.5175`
- Best baseline macro F1: `0.5171`

## SVM (RBF)

- Validation balanced accuracy: `0.4772`
- Validation macro F1: `0.3665`
- Best Optuna CV balanced accuracy: `0.5213`
- Best params: `{'C': 0.17963887106388227, 'gamma': 0.20150165915947318}`

## Final Test Metrics

- Balanced accuracy: `0.5116`
- Macro F1: `0.3671`
- Quadratic kappa: `0.0217`
- AUC: `0.5806`

## Final Cross-Validation

- AUC: `0.4524`
- 95% CI: `0.3749` to `0.5298`
- One-sided p-value vs 0.5: `0.90112`

## SHAP Feature Importance

SHAP analysis failed or was skipped — see `outputs/logs/shap_error.txt`.
