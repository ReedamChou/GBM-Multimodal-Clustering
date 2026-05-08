# UCSF Training Summary

## Dataset

- Rows after label cleaning: `500`
- Imaging features used: `17`
- Class distribution: `{'high_risk': 218, 'low_risk': 282}`
- Total feature nulls at training input: `0`

## Runtime

- Requested accelerator: `cuda`
- Selected device: `cuda`
- GPU available: `True`
- Device note: `XGBoost CUDA probe succeeded.`
- Optuna trials run: `25`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.6147`
- Best baseline macro F1: `0.6151`

## XGBoost

- Validation balanced accuracy: `0.5725`
- Validation macro F1: `0.5717`
- Best Optuna CV balanced accuracy: `0.6042`
- Best params: `{'n_estimators': 527, 'learning_rate': 0.017901479162808983, 'max_depth': 6, 'subsample': 0.5034237651213853, 'colsample_bytree': 0.7766688519794932, 'min_child_weight': 7, 'gamma': 4.074537193978972, 'reg_alpha': 0.27615159740375017, 'reg_lambda': 0.19939973559342072}`

## Final Test Metrics

- Balanced accuracy: `0.7154`
- Macro F1: `0.7066`
- Quadratic kappa: `0.4198`
- Macro AUC OvR: `0.7763`

## Final Cross-Validation

- Balanced accuracy: `0.5981`
- AUC: `0.6516`
