# Training

This step runs a research-grade training workflow: baselines, Optuna
hyperparameter tuning, train/val/test split, SHAP, confusion matrix,
and final stratified 10-fold CV.

## Inputs

- Input file: `outputs/features_processed.csv`
- Target column: `risk_label` (binary 0/1)
- Feature columns: the fixed 16 radiomic features

## Baseline models

- Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF), KNN
- Baselines are evaluated on the validation split and saved to
  `outputs/metrics/baseline_results.csv`

## XGBoost classifier

- Binary classifier with parameters from `config.json` and auto
  `scale_pos_weight` from class imbalance.

## Optuna tuning

- Hyperparameters are optimized with Optuna using stratified CV on the
  train+val set.
- Trials are logged to `outputs/logs/optuna_trials.csv`.

## Train/Val/Test split

- 70/15/15 stratified split (train/val/test).
- Split assignments are saved to `outputs/tables/split_assignments.csv`.

## Final cross-validation

- Stratified 10-fold CV on the full dataset after tuning.
- Reports mean AUC and 95% CI via `scipy.stats.t.interval`.
- One-sided p-value vs AUC = 0.5 is reported.

## SHAP

- SHAP summary and bar plots are saved to:
  - `outputs/figures/shap_summary.png`
  - `outputs/figures/shap_importance.png`

## Usage

```bash
python src/train.py
```

## Outputs

- `outputs/metrics/baseline_results.csv`
- `outputs/metrics/final_cv_summary.json`
- `outputs/metrics/final_cv_fold_metrics.csv`
- `outputs/metrics/test_metrics.json`
- `outputs/metrics/test_classification_report.csv`
- `outputs/metrics/validation_xgboost_metrics.json`
- `outputs/metrics/dataset_summary.json`
- `outputs/metrics/feature_summary.json`
- `outputs/metrics/final_cv_summary.json`
- `outputs/figures/class_distribution.png`
- `outputs/figures/feature_distributions.png`
- `outputs/figures/correlation_heatmap.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/shap_importance.png`
- `outputs/logs/optuna_trials.csv`
- `outputs/models/best_xgb_risk_classifier.pkl`
- `outputs/models/best_params.json`
- `outputs/reports/training_summary.md`
- `outputs/final_model.json`
- `outputs/feature_list.json`
- `outputs/cv_results.json`
- `outputs/training_report.txt`
