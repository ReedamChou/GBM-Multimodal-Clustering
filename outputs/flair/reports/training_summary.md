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
- Best Optuna CV balanced accuracy: `0.5521`
- Best params: `{'C': 992.6866770500909, 'gamma': 0.030403976631756376}`

## Final Test Metrics

- Balanced accuracy: `0.5434`
- Macro F1: `0.5432`
- Quadratic kappa: `0.0867`
- AUC: `0.4392`

## Final Cross-Validation

- AUC: `0.4854`
- 95% CI: `0.4419` to `0.5290`
- One-sided p-value vs 0.5: `0.765891`

## SHAP Feature Importance

SHAP analysis failed or was skipped — see `outputs/logs/shap_error.txt`.
