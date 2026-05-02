# UCSF Training Summary

## Dataset

- Rows after label cleaning: `500`
- Imaging features used: `17`
- Class distribution: `{'low_risk': 202, 'medium_risk': 183, 'high_risk': 115}`
- Total feature nulls at training input: `0`

## Runtime

- Requested accelerator: `cuda`
- Selected device: `cpu`
- GPU available: `False`
- Device note: `nvidia-smi could not communicate with the NVIDIA driver.`
- Optuna trials run: `25`

## Baselines

- Best validation baseline: `LogisticRegression`
- Best baseline balanced accuracy: `0.4537`
- Best baseline macro F1: `0.4440`

## XGBoost

- Validation balanced accuracy: `0.4307`
- Validation macro F1: `0.4267`
- Best Optuna CV balanced accuracy: `0.4842`
- Best params: `{'n_estimators': 505, 'learning_rate': 0.28725032911724113, 'max_depth': 8, 'subsample': 0.775687946662355, 'colsample_bytree': 0.8559610257613437, 'min_child_weight': 1, 'gamma': 3.430305198888086, 'reg_alpha': 0.08789485815463152, 'reg_lambda': 0.00013833222430044656}`

## Final Test Metrics

- Balanced accuracy: `0.4912`
- Macro F1: `0.4904`
- Quadratic kappa: `0.3899`
- Macro AUC OvR: `0.6471`

## Final Cross-Validation

- Balanced accuracy: `0.4486 ± 0.0507`
- Quadratic kappa: `0.2629 ± 0.1599`
- Macro F1: `0.4443 ± 0.0539`
