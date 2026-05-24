# GBM Training Summary

## Dataset

- Rows after label cleaning: `449`
- Imaging features used: `16`
- Class distribution: `{0: 240, 1: 209}`
- Total feature nulls at training input: `0`

## Runtime

- Selected device: `cpu`
- GPU available: `False`
- Device note: `SVM uses CPU training`
- Optuna trials run: `50`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6285`
- Best baseline macro F1: `0.6284`

## SVM (RBF)

- Validation balanced accuracy: `0.5139`
- Validation macro F1: `0.3503`
- Best Optuna CV balanced accuracy: `0.5497`
- Best params: `{'C': 992.6866770500909, 'gamma': 0.030403976631756376}`

## Final Test Metrics

- Balanced accuracy: `0.5573`
- Macro F1: `0.5573`
- Quadratic kappa: `0.1146`
- AUC: `0.4332`

## Final Cross-Validation

- AUC: `0.4875`
- 95% CI: `0.4442` to `0.5309`
- One-sided p-value vs 0.5: `0.73418`

## SHAP Feature Importance

SHAP analysis failed or was skipped — see `outputs/logs/shap_error.txt`.
