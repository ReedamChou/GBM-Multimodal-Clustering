# Step 5: Multi-Class Risk Classification (XGBoost)

## Script

`scripts/step5_multiclass_classification.py`

## Goal

Train and evaluate an XGBoost multi-class classifier for patient risk prediction using
**imaging features only**.

The target is `risk_cluster_label` with 3 classes:

- low risk
- medium risk
- high risk

The script performs Stratified 5-fold cross validation, reports classification metrics,
then trains a final XGBoost model on the full dataset to generate patient-level predictions.

## Input

Primary source:

- `outputs/new/step3_ucsf_preprocessed_features.csv`

Path resolution behavior:

1. `outputs/new/step3_ucsf_preprocessed_features.csv`
2. `../outputs/new/step3_ucsf_preprocessed_features.csv`

Required columns:

- `risk_cluster_label` (target)
- patient identifier column (resolved from aliases such as `patient_id`, `case_id`, `ID`, etc.)
- numeric imaging features from Step 3

## Output

Output folder:

- `outputs/new/step5_multiclass/`

Output files:

- `step5_model_comparison.csv`
- `step5_xgboost_confusion_matrix.csv`
- `step5_xgboost_patient_predictions.csv`

`step5_xgboost_patient_predictions.csv` contains:

- `patient_id`
- `predicted_risk`

## Core Logic

### 1) Configuration

The script uses:

- `RANDOM_STATE = 42`
- `N_SPLITS = 5`
- `TARGET_COLUMN = "risk_cluster_label"`
- `APPLY_SCALING = False`

`APPLY_SCALING` is retained for compatibility and future baselines. For tree models,
scaling is generally unnecessary.

### 2) Input and Patient ID Resolution

The CSV path is auto-resolved from candidate locations.

Patient ID column is automatically detected using aliases:

- `patient_id`, `case_id`, `PatientID`, `ID`, `id`

This allows the script to work even if ID naming differs across datasets.

### 3) Imaging-Only Feature Selection

The script excludes non-imaging columns (IDs, survival fields, cluster metadata) and keeps
numeric columns only.

Result:

- `X`: numeric imaging features
- `y_raw`: target labels from `risk_cluster_label`

### 4) Target Encoding

`LabelEncoder` converts string labels into integer class IDs for model training.

The script prints the class mapping so predictions can be interpreted correctly.

### 5) XGBoost Model Definition

A single `XGBClassifier` is configured for multiclass learning with:

- objective: `multi:softprob`
- class count: `num_class = len(class_names)`
- reproducible seed: `random_state=42`

### 6) Stratified 5-Fold Cross Validation

Cross validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

For each fold:

1. Split data into train/test fold.
2. Optionally apply scaling (if enabled).
3. Train a fresh cloned XGBoost model.
4. Predict fold labels.
5. Compute:
   - Accuracy
   - Macro F1-score
   - Confusion matrix

Fold confusion matrices are aggregated into one final matrix.

### 7) CV Aggregation

The script reports:

- `accuracy_mean`, `accuracy_std`
- `f1_macro_mean`, `f1_macro_std`
- aggregated confusion matrix

These values are saved to CSV files in the output directory.

### 8) Final Full-Dataset Training and Prediction

After CV, the script trains a final XGBoost model on all samples (`X`, `y`), then:

1. predicts encoded labels for all patients
2. inverse-transforms predictions back to original class names
3. writes predictions into a new dataframe column: `predicted_risk`

### 9) Patient Grouping and Export

The script groups patients by predicted class and prints patient IDs per group,
prioritizing this display order:

1. `low_risk`
2. `medium_risk`
3. `high_risk`

Any additional class labels (if present) are printed after these.

Finally, it saves:

- `step5_xgboost_patient_predictions.csv`

with columns:

- `patient_id`
- `predicted_risk`

## Block-by-Block Walkthrough (`# %%`)

### Block 1
Imports dependencies and defines constants, path candidates, exclusion lists, and helper
functions (`resolve_input_csv`, `resolve_patient_id_column`).

### Block 2
Loads the dataset, validates target presence, and resolves patient ID column.

### Block 3
Builds imaging-only numeric feature matrix and target vector source labels.

### Block 4
Encodes labels and defines the XGBoost model configuration.

### Helper Section (function definition)
Defines `evaluate_model_cv(...)` for fold-wise training, scoring, and confusion matrix
aggregation.

### Block 5
Runs Stratified 5-fold CV and prints fold and averaged metrics.

### Block 6
Builds and prints model summary table (single XGBoost row).

### Block 7
Saves model comparison and confusion matrix outputs.

### Block 8
Fits final XGBoost on full dataset, generates class-name predictions, prints grouped
patient IDs, and saves patient-level prediction CSV.

## Error Checks Included

The script raises clear errors when:

- input CSV cannot be found
- target column is missing
- patient ID column cannot be resolved from aliases
- no numeric imaging features remain after exclusions

## How to Run

From project root:

```bash
python scripts/step5_multiclass_classification.py
```

From `scripts/` directory:

```bash
python step5_multiclass_classification.py
```

## How to Read the Results

- `step5_model_comparison.csv`: CV summary metrics for XGBoost.
- `step5_xgboost_confusion_matrix.csv`: aggregate confusion matrix across folds.
- `step5_xgboost_patient_predictions.csv`: final patient-level predicted risk labels.

Macro F1 remains the primary quality indicator for balanced multi-class interpretation,
especially when class sizes differ.
