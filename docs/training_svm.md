# Training (SVM)

This step runs the SVM (RBF) training workflow with mandatory robust scaling,
Optuna tuning over `C` and `gamma`, SHAP analysis, confusion matrix, and final
stratified cross-validation.

---

## Supported Modalities

Each modality's processed CSV trains to its own isolated output directory.
Results, models, figures, and metrics for each modality are never mixed.

| Modality | Input CSV | Output Directory |
|----------|-----------|-----------------|
| T1 | `outputs/features_processed_t1.csv` | `outputs/t1/` |
| T2 | `outputs/features_processed_t2.csv` | `outputs/t2/` |
| T1GD | `outputs/features_processed_t1gd.csv` | `outputs/t1gd/` |
| FLAIR | `outputs/features_processed_flair.csv` | `outputs/flair/` |

> Prerequisite: Run `preprocessing.py` for the chosen modality before training.
> See `docs/preprocessing.md`.

---

## CLI Usage

### T1
```bash
python src/train_svm.py \
  --input  outputs/features_processed_t1.csv \
  --output outputs/t1
```

### T2
```bash
python src/train_svm.py \
  --input  outputs/features_processed_t2.csv \
  --output outputs/t2
```

### T1GD
```bash
python src/train_svm.py \
  --input  outputs/features_processed_t1gd.csv \
  --output outputs/t1gd
```

### FLAIR
```bash
python src/train_svm.py \
  --input  outputs/features_processed_flair.csv \
  --output outputs/flair
```

### All 4 in sequence (PowerShell)
```powershell
foreach ($mod in @("t1","t2","t1gd","flair")) {
    python src/train_svm.py `
        --input  "outputs/features_processed_$mod.csv" `
        --output "outputs/$mod"
}
```

### All 4 in sequence (bash)
```bash
for mod in t1 t2 t1gd flair; do
    python src/train_svm.py \
        --input  "outputs/features_processed_${mod}.csv" \
        --output "outputs/${mod}"
done
```

---

## All CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.json` | Path to config.json |
| `--modality` | none | Uses modality defaults for input/output when provided |
| `--input` | `training.input_csv` from config | Processed features CSV |
| `--output` | `training.output_dir` from config | Root output directory |

---

## Training Workflow (in order)

### 1. Dataset Loading
- Reads the processed CSV and validates 16 feature columns + `risk_label`.
- Drops rows where `risk_label` is not 0 or 1.
- Imputes feature nulls using training-set medians.

### 2. Exploratory Plots
Saved to `{output}/figures/`:
- `class_distribution.png` — class balance bar chart
- `feature_distributions.png` — 4x4 histograms per feature
- `correlation_heatmap.png` — Pearson correlation matrix heatmap

### 3. Train / Val / Test Split
- 70 / 15 / 15 stratified split.
- Split assignments saved to `{output}/tables/split_assignments.csv`.

### 4. Baseline Models
Evaluated on the validation split. Saved to `{output}/metrics/baseline_results.csv`.

| Model | Scaling |
|-------|---------|
| Logistic Regression | RobustScaler + VarianceThreshold |
| SVM (RBF) | RobustScaler + VarianceThreshold |
| KNN | RobustScaler + VarianceThreshold |
| Random Forest | None (tree model) |
| Gradient Boosting | None (tree model) |

### 5. Mandatory Scaling
- `RobustScaler` is fit on the training split and applied to validation/test.
- The final SVM model is trained on scaled features only.

### 6. SVM Validation Run
- Fixed-parameter SVM evaluates the validation split for sanity checks.
- Metrics saved to `{output}/metrics/validation_xgboost_metrics.json` (file name retained for compatibility).

### 7. Optuna Tuning
- CV on the train+val set, optimizing balanced accuracy.
- Search space: `C` (1e-2 to 1e3, log scale), `gamma` (1e-4 to 1e1, log scale).
- Trials saved to `{output}/logs/optuna_trials.csv`.

### 8. Final Test Evaluation
- Best SVM is fit on train+val and evaluated on the test set.
- Metrics in `{output}/metrics/test_metrics.json` and `test_classification_report.csv`.
- Confusion matrix saved to `{output}/figures/confusion_matrix.png`.

### 9. Feature Importance + SHAP
- Permutation importance saved to `{output}/tables/feature_importance.csv`.
- SHAP uses `KernelExplainer` and can be slow; outputs are:
  - `{output}/figures/shap_summary.png`
  - `{output}/figures/shap_importance.png`
  - `{output}/logs/shap_error.txt` if SHAP fails

### 10. Final Cross-Validation
- Stratified CV with the tuned hyperparameters.
- Fold metrics in `{output}/metrics/final_cv_fold_metrics.csv`.
- Summary in `{output}/metrics/final_cv_summary.json` and `{output}/cv_results.json`.

---

## Output Tree (per run)

```
{output}/
├── figures/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── shap_importance.png
│   └── shap_summary.png
├── metrics/
│   ├── baseline_results.csv
│   ├── dataset_summary.json
│   ├── feature_summary.json
│   ├── validation_xgboost_metrics.json
│   ├── test_classification_report.csv
│   ├── test_metrics.json
│   ├── final_cv_fold_metrics.csv
│   └── final_cv_summary.json
├── models/
│   ├── best_xgb_risk_classifier.pkl
│   └── best_params.json
├── tables/
│   ├── split_assignments.csv
│   ├── feature_importance.csv
│   └── high_correlation_pairs.csv
├── logs/
│   ├── optuna_trials.csv
│   ├── gpu_status.json
│   └── shap_error.txt
├── reports/
│   └── training_summary.md
├── final_model.json
├── feature_list.json
├── cv_results.json
└── training_report.txt
```

Notes:
- `best_xgb_risk_classifier.pkl` contains a dict with the SVM model and the fitted scaler.
- `final_model.json` records model type, parameters, and feature list.

---

## config.json - Relevant Keys

```json
"training": {
  "input_csv":                         "outputs/features_processed.csv",
  "output_dir":                        "outputs",
  "test_size":                         0.15,
  "validation_fraction_of_trainval":   0.176470588,
  "optuna_trials":                     50,
  "optuna_cv_folds":                   5,
  "final_cv_folds":                    10,
  "random_seed":                       42
}
```

When passing `--input` and `--output` on the CLI, the `input_csv` and
`output_dir` values in config are overridden by the CLI values.
