# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# Reproducibility and CV configuration.
RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COLUMN = "risk_cluster_label"
APPLY_SCALING = False  # Tree models do not need scaling; keep False unless testing alternatives.

# Candidate locations to support running from project root or from scripts/.
INPUT_CSV_CANDIDATES = [
    Path("outputs/new/step3_ucsf_preprocessed_features.csv"),
    Path("../outputs/new/step3_ucsf_preprocessed_features.csv"),
]

# Non-imaging columns are excluded so training uses atlas/imaging features only.
NON_IMAGING_COLUMNS = {
    "case_id",
    "patient_id",
    "PatientID",
    "ID",
    "id",
    "OS",
    "os",
    "overall_survival",
    "survival",
    "survival_days",
    "survival_months",
    "risk_cluster_id",
    "risk_cluster_label",
}

PATIENT_ID_ALIASES = ["patient_id", "case_id", "PatientID", "ID", "id"]


def resolve_input_csv(candidates: list[Path]) -> Path:
    # Return the first existing CSV path from the candidate list.
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not find input CSV. Searched:\n{searched}")


def resolve_patient_id_column(columns: pd.Index) -> str:
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for alias in PATIENT_ID_ALIASES:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not resolve patient ID column from aliases: {PATIENT_ID_ALIASES}")


# %%
# Load the preprocessed Step 3 dataset and validate target availability.
input_csv = resolve_input_csv(INPUT_CSV_CANDIDATES)
print(f"Using input CSV: {input_csv}")

df = pd.read_csv(input_csv)
print(f"Loaded shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")

patient_id_column = resolve_patient_id_column(df.columns)


# %%
# Build X from imaging-only numeric features and keep y as the risk label.
candidate_feature_columns = [column for column in df.columns if column not in NON_IMAGING_COLUMNS]
numeric_feature_columns = [
    column for column in candidate_feature_columns if pd.api.types.is_numeric_dtype(df[column])
]

non_numeric_candidates = sorted(set(candidate_feature_columns) - set(numeric_feature_columns))
if non_numeric_candidates:
    print("Excluded non-numeric candidate feature columns:")
    print(non_numeric_candidates)

if not numeric_feature_columns:
    raise ValueError("No numeric imaging features were found after excluding non-imaging columns.")

X = df[numeric_feature_columns].copy()
y_raw = df[TARGET_COLUMN].astype(str).copy()

print(f"Selected imaging feature count: {X.shape[1]}")
print("Class distribution:")
print(y_raw.value_counts(dropna=False))


# %%
# Encode string labels for model training and define model configurations.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
class_names = list(label_encoder.classes_)

print("Label encoding:")
for encoded_value, class_name in enumerate(class_names):
    print(f"  {encoded_value} -> {class_name}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
num_classes = len(class_names)

models = {
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
}



# Cross-validation helper: trains one fold at a time and aggregates metrics.
def evaluate_model_cv(
    model,
    X_data: pd.DataFrame,
    y_data: np.ndarray,
    cv_splitter: StratifiedKFold,
    labels: list[str],
    apply_scaling: bool = False,
) -> dict[str, object]:
    # Store per-fold metrics so we can report mean and standard deviation.
    fold_accuracy: list[float] = []
    fold_f1_macro: list[float] = []
    aggregate_confusion = np.zeros((len(labels), len(labels)), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X_data, y_data), start=1):
        X_train = X_data.iloc[train_idx]
        X_test = X_data.iloc[test_idx]
        y_train = y_data[train_idx]
        y_test = y_data[test_idx]

        # Scaling is optional and mainly useful for non-tree baselines.
        if apply_scaling:
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train)
            X_test_model = scaler.transform(X_test)
        else:
            X_train_model = X_train
            X_test_model = X_test

        # clone(model) prevents state leakage across folds.
        fold_model = clone(model)
        fold_model.fit(X_train_model, y_train)
        y_pred = fold_model.predict(X_test_model)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))

        fold_accuracy.append(acc)
        fold_f1_macro.append(f1_macro)
        aggregate_confusion += cm

        print(f"Fold {fold_idx}: accuracy={acc:.4f}, f1_macro={f1_macro:.4f}")

    confusion_df = pd.DataFrame(
        aggregate_confusion,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    return {
        "accuracy_mean": float(np.mean(fold_accuracy)),
        "accuracy_std": float(np.std(fold_accuracy)),
        "f1_macro_mean": float(np.mean(fold_f1_macro)),
        "f1_macro_std": float(np.std(fold_f1_macro)),
        "confusion_matrix": confusion_df,
    }


# %%
# Run Stratified 5-fold CV for each model and print fold/average metrics.
evaluation_results: dict[str, dict[str, object]] = {}

for model_name, model in models.items():
    print(f"\n=== {model_name}: Stratified {N_SPLITS}-Fold CV ===")
    result = evaluate_model_cv(
        model=model,
        X_data=X,
        y_data=y,
        cv_splitter=cv,
        labels=class_names,
        apply_scaling=APPLY_SCALING,
    )
    evaluation_results[model_name] = result

    print("Average CV metrics:")
    print(
        f"  Accuracy : {result['accuracy_mean']:.4f} +/- {result['accuracy_std']:.4f}\n"
        f"  F1 Macro : {result['f1_macro_mean']:.4f} +/- {result['f1_macro_std']:.4f}"
    )
    print("Confusion matrix (aggregated across folds):")
    print(result["confusion_matrix"])


# %%
# Build a comparison table and choose the best model by macro F1.
summary_rows = []
for model_name, result in evaluation_results.items():
    summary_rows.append(
        {
            "model": model_name,
            "accuracy_mean": result["accuracy_mean"],
            "accuracy_std": result["accuracy_std"],
            "f1_macro_mean": result["f1_macro_mean"],
            "f1_macro_std": result["f1_macro_std"],
        }
    )

summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)

print("\n=== Model Comparison (sorted by F1 macro) ===")
print(summary_df.to_string(index=False))

best_model_name = str(summary_df.iloc[0]["model"])
best_model_f1 = float(summary_df.iloc[0]["f1_macro_mean"])
print(f"\nBest model by F1 macro: {best_model_name} ({best_model_f1:.4f})")


# %%
# Save summary and per-model confusion matrices for downstream analysis.
output_dir = input_csv.parent / "step5_multiclass"
output_dir.mkdir(parents=True, exist_ok=True)

summary_path = output_dir / "step5_model_comparison.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved model comparison: {summary_path}")

for model_name, result in evaluation_results.items():
    matrix_filename = f"step5_{model_name.lower()}_confusion_matrix.csv"
    matrix_path = output_dir / matrix_filename
    result["confusion_matrix"].to_csv(matrix_path, index=True)
    print(f"Saved confusion matrix: {matrix_path}")


# %%
# Train final XGBoost model on full dataset and generate patient-level predictions.
final_model = clone(models["XGBoost"])

if APPLY_SCALING:
    final_scaler = StandardScaler()
    X_full_model = final_scaler.fit_transform(X)
else:
    X_full_model = X

final_model.fit(X_full_model, y)
final_pred_encoded = final_model.predict(X_full_model)
final_pred_labels = label_encoder.inverse_transform(final_pred_encoded.astype(int))

df["predicted_risk"] = final_pred_labels

print("\nPredicted patient groups:")
preferred_order = ["low_risk", "medium_risk", "high_risk"]
remaining_classes = [label for label in class_names if label not in preferred_order]
print_order = preferred_order + remaining_classes

for risk_label in print_order:
    patient_ids = (
        df.loc[df["predicted_risk"] == risk_label, patient_id_column]
        .astype(str)
        .tolist()
    )
    if patient_ids:
        print(f"\n{risk_label} ({len(patient_ids)} patients):")
        print(", ".join(patient_ids))

patient_prediction_df = pd.DataFrame(
    {
        "patient_id": df[patient_id_column].astype(str),
        "predicted_risk": df["predicted_risk"],
    }
)

prediction_path = output_dir / "step5_xgboost_patient_predictions.csv"
patient_prediction_df.to_csv(prediction_path, index=False)
print(f"\nSaved patient predictions: {prediction_path}")
