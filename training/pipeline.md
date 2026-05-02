# Training Pipeline

This document describes only the **training workflow** for UCSF risk classification.
It starts from the already prepared feature file:

- `outputs/new/step3_ucsf_preprocessed_features.csv`

and ends with trained model artifacts, evaluation metrics, plots, and reports under `training/`.

## Fixed Run Settings

These are the actual values used by the current training pipeline:

- Input file: `outputs/new/step3_ucsf_preprocessed_features.csv`
- Random state: `42`
- Total rows used for training workflow: `500`
- Number of imaging features: `17`
- Test split size: `15%` = `75` patients
- Validation split size: `15%` of total = `75` patients
- Train split size: `70%` = `350` patients
- Optuna trials: `25`
- Optuna cross-validation folds: `5`
- Final cross-validation folds: `10`
- XGBoost early stopping rounds: `30`

## 1. Input Dataset

Source file:

- `outputs/new/step3_ucsf_preprocessed_features.csv`

The training script loads:

- 17 imaging features
- target label: `risk_cluster_label`

It explicitly excludes leakage columns from training features:

- `OS`
- `survival_months`
- `risk_cluster_id`

## 2. Label Cleaning

Script:

- `training/scripts/run_ucsf_training.py`

Processing:

1. Read the CSV into a dataframe.
2. Drop rows with missing `risk_cluster_label`.
3. Keep only the 17 imaging features plus the label for modeling.
4. Map labels to ordinal integers:
   - `low_risk -> 0`
   - `medium_risk -> 1`
   - `high_risk -> 2`

Actual dataset used in the current run:

- Total rows after label cleaning: `500`
- `low_risk`: `202`
- `medium_risk`: `183`
- `high_risk`: `115`

Outputs:

- `training/outputs/metrics/dataset_summary.json`

## 3. Runtime / Device Check

Before model training, the pipeline checks whether XGBoost can use CUDA.

Processing:

1. Request GPU use with XGBoost.
2. Check whether `nvidia-smi` is healthy.
3. If CUDA is usable, train with `device='cuda'`.
4. Otherwise fall back to CPU automatically.

Outputs:

- `training/outputs/logs/gpu_status.json`
- `training/outputs/reports/gpu_troubleshooting.md`

## 4. Exploratory Outputs

The pipeline generates summary plots from the training input file.

Processing:

1. Plot class distribution.
2. Plot feature histograms split by class.
3. Plot the feature-feature correlation heatmap.
4. Save feature summary statistics.
5. Save any feature pairs with absolute correlation above `0.90`.

Outputs:

- `training/outputs/figures/class_distribution.png`
- `training/outputs/figures/feature_distributions.png`
- `training/outputs/figures/correlation_heatmap.png`
- `training/outputs/tables/feature_summary.csv`
- `training/outputs/tables/high_correlation_pairs.csv`

## 5. Train / Validation / Test Split

The dataset is split with stratification so class proportions are preserved.

Processing:

1. Split into `train+validation` and `test`
   - test size: `15%`
   - actual test count: `75`
2. Split `train+validation` again into:
   - train
   - validation
   - actual train count: `350`
   - actual validation count: `75`
3. Preserve label proportions in all splits using stratified sampling.
4. Save patient-level split assignments.

Actual split counts from `training/outputs/tables/split_assignments.csv`:

- Train: `350`
- Validation: `75`
- Test: `75`

Outputs:

- `training/outputs/tables/split_assignments.csv`

## 6. Baseline Models

The pipeline runs baseline models before tuned XGBoost.

Models evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM (RBF)
- KNN

Processing:

1. Apply `VarianceThreshold` to remove zero-variance features where needed.
2. Apply `RobustScaler` for scale-sensitive models.
3. Fit each baseline on the training split.
4. Evaluate on the validation split.
5. Save balanced accuracy and macro F1.

Outputs:

- `training/outputs/metrics/baseline_results.csv`

## 7. Initial XGBoost Validation Model

The pipeline trains a first XGBoost model before tuning.

Processing:

1. Use class-balanced sample weights.
2. Fit XGBoost on the training split.
3. Evaluate on the validation split.
4. Save validation metrics.

Outputs:

- `training/outputs/metrics/validation_xgboost_metrics.json`

## 8. Optuna Hyperparameter Tuning

The primary model is tuned with Optuna.

Processing:

1. Use `X_trainval` and `y_trainval`.
2. Run stratified `5-fold` cross-validation during tuning.
3. Optimize for:
   - balanced accuracy
4. Search parameters such as:
   - `n_estimators`
   - `learning_rate`
   - `max_depth`
   - `subsample`
   - `colsample_bytree`
   - `min_child_weight`
   - `gamma`
   - `reg_alpha`
   - `reg_lambda`
5. Save full Optuna trial history and best parameters.

Actual tuning settings:

- Optuna trials: `25`
- Tuning folds: `5`
- Samples used for tuning (`train + validation`): `425`

Outputs:

- `training/outputs/logs/optuna_trials.csv`
- `training/models/best_params.json`

## 9. Final Model Fit

After tuning, the pipeline trains the final model on the combined train+validation set.

Processing:

1. Merge train and validation splits.
   - actual merged count: `425`
2. Apply class-balanced sample weights.
3. Fit the final XGBoost model with the best Optuna parameters.
4. Save the trained model and metadata.

Outputs:

- `training/models/best_xgb_risk_classifier.pkl`
- `training/models/best_params.json`
- `training/models/label_map.json`
- `training/models/feature_list.json`

## 10. Held-Out Test Evaluation

The final model is evaluated on the held-out test split.

Metrics reported:

- balanced accuracy
- macro F1
- quadratic kappa
- macro AUC OvR
- confusion matrix

Outputs:

- `training/outputs/metrics/test_metrics.json`
- `training/outputs/metrics/test_classification_report.csv`
- `training/outputs/figures/confusion_matrix.png`

## 11. Feature Importance And Explainability

The pipeline produces feature importance and SHAP-style contribution outputs.

Processing:

1. Save XGBoost feature importance values.
2. Plot horizontal feature importance bars.
3. Generate high-risk class contribution summaries.

Outputs:

- `training/outputs/tables/feature_importance.csv`
- `training/outputs/figures/feature_importance.png`
- `training/outputs/figures/shap_high_risk_bar.png`
- `training/outputs/figures/shap_high_risk_beeswarm.png`

## 12. Final Cross-Validation Estimate

The pipeline computes a final performance estimate using the full dataset.

Processing:

1. Use stratified k-fold cross-validation on the full training dataframe.
   - actual folds: `10`
   - actual rows used: `500`
2. Refit the tuned XGBoost configuration fold by fold.
3. Save fold-level metrics and summary metrics.

Outputs:

- `training/outputs/metrics/final_cv_fold_metrics.csv`
- `training/outputs/metrics/final_cv_summary.json`

## 13. Final Report

The pipeline writes a compact summary of the full run.

Output:

- `training/outputs/reports/training_summary.md`

This report includes:

- dataset size
- class distribution
- runtime device used
- best baseline model
- validation XGBoost metrics
- best Optuna score and parameters
- final test metrics
- final cross-validation summary

## 14. How To Interpret The Results

The pipeline produces **three different kinds of model performance numbers**.
They do **not** mean the same thing.

### 14.1 Validation XGBoost Result

This is the result from the first untuned XGBoost model trained on:

- Train: `350`

and evaluated on:

- Validation: `75`

For the current run:

- Validation balanced accuracy: `0.4307`
- Validation macro F1: `0.4267`

Purpose:

- quick check before tuning
- not the final model result

### 14.2 Optuna 5-Fold Cross-Validation Result

This is the score used to choose the best XGBoost hyperparameters.

Data used:

- `train + validation = 425` samples

Method:

- stratified `5-fold` cross-validation
- `25` Optuna trials
- scoring metric: balanced accuracy

For the current run:

- Best Optuna CV balanced accuracy: `0.4842`

Purpose:

- model selection
- hyperparameter tuning
- not the final held-out test result

### 14.3 Final Held-Out Test Result

After tuning, the model is retrained on:

- `train + validation = 425` samples

and then evaluated once on the untouched:

- Test: `75` samples

This is the **single held-out final evaluation** of the selected model.

For the current run:

- Balanced accuracy: `0.4912`
- Macro F1: `0.4904`
- Quadratic kappa: `0.3899`
- Macro AUC OvR: `0.6471`

This is usually the main answer to:

- “What did the final trained model score on the test set?”

### 14.4 Final 10-Fold Cross-Validation Result

This is a separate robustness estimate of the full modeling approach.

Data used:

- all `500` samples

Method:

- stratified `10-fold` cross-validation
- refit the tuned XGBoost configuration in each fold

For the current run:

- Balanced accuracy: `0.4486 ± 0.0507`
- Quadratic kappa: `0.2629 ± 0.1599`
- Macro F1: `0.4443 ± 0.0539`

Purpose:

- estimate how stable the pipeline is across different splits
- reduce dependence on one lucky or unlucky test split
- not the same as the final held-out test result

### 14.5 Which Number Is The Main Final Result?

If someone asks for the **final model test performance**, report:

- Held-out test balanced accuracy: `0.4912`

If someone asks for the **more robust overall estimate of pipeline performance**, report:

- 10-fold CV balanced accuracy: `0.4486 ± 0.0507`

If someone asks for the **tuning score**, report:

- Optuna 5-fold CV balanced accuracy: `0.4842`

## 15. Run Command

To execute the full training workflow:

```bash
venv/bin/python training/scripts/run_ucsf_training.py
```

## 16. End-To-End Flow Summary

```text
step3_ucsf_preprocessed_features.csv
    ->
load 17 imaging features + risk_cluster_label
    ->
drop null labels
    ->
encode labels to 0 / 1 / 2
    ->
device check (GPU if available, else CPU)
    ->
EDA plots + feature summary tables
    ->
stratified train / validation / test split
350 / 75 / 75
    ->
baseline models
    ->
initial XGBoost validation run
validation on 75 samples
    ->
Optuna tuning on train+validation
25 trials, 5-fold CV, 425 samples
    ->
fit final XGBoost model
train on 425 samples
    ->
test-set evaluation
evaluate once on 75 samples
    ->
feature importance + SHAP-style plots
    ->
full-dataset cross-validation
10 folds, 500 samples
    ->
save model, metrics, figures, and training summary
```
