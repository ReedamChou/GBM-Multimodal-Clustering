# GBM Training Summary

## Dataset

- Rows after label cleaning: `493`
- Imaging features used: `16`
- Class distribution: `{0: 276, 1: 217}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cpu`
- GPU available: `False`
- Device note: `SVM uses CPU training`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6347`
- Best baseline macro F1: `0.6356`

## SVM (RBF)

- Validation balanced accuracy: `0.5000`
- Validation macro F1: `0.3019`
- Best Optuna CV balanced accuracy: `0.5844`
- Best params: `{'C': 541.2582843534118, 'gamma': 0.004210818577542578}`

## Final Test Metrics

- Balanced accuracy: `0.5381`
- Macro F1: `0.5375`
- Quadratic kappa: `0.0757`
- AUC: `0.4619`

## Final Cross-Validation

- AUC: `0.5584`
- 95% CI: `0.4933` to `0.6236`
- One-sided p-value vs 0.5: `0.0364562`

## SHAP Feature Importance

SHAP analysis failed or was skipped — see `outputs/logs/shap_error.txt`.
