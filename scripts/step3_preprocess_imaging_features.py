from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step2_ucsf_clustered.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step3_ucsf_preprocessed_features.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "new"

IDENTIFIER_COLUMNS = {"case_id", "patient_id"}
METADATA_COLUMNS = {"OS", "os", "overall_survival", "survival", "survival_days", "survival_months", "risk_cluster_id", "risk_cluster_label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Impute, encode, scale, and feature-filter the UCSF imaging dataset. "
            "Feature correlation filtering uses a Pearson threshold of 0.90."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    return parser.parse_args()


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = IDENTIFIER_COLUMNS | METADATA_COLUMNS
    return [column for column in df.columns if column not in excluded]


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_integer_like(series: pd.Series) -> bool:
    numeric = numeric_series(series).dropna()
    if numeric.empty:
        return False
    return bool(np.all(np.isclose(numeric % 1, 0)))


def classify_feature(series: pd.Series) -> str:
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    if is_integer_like(series):
        unique_values = numeric_series(series).dropna().nunique()
        return "integer_discrete" if unique_values <= 20 else "integer_continuous"
    return "continuous"


def safe_mode(series: pd.Series) -> object:
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else None


def impute_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputed = df[feature_columns].copy()
    report_rows: list[dict[str, object]] = []

    for column in feature_columns:
        feature_type = classify_feature(imputed[column])

        if feature_type == "categorical":
            categorical = imputed[column].astype("string")
            fill_value = safe_mode(categorical)
            if fill_value is None:
                fill_value = "missing"
            imputed[column] = categorical.fillna(str(fill_value))
        elif feature_type == "integer_discrete":
            numeric = numeric_series(imputed[column])
            fill_value = safe_mode(numeric)
            if fill_value is None or pd.isna(fill_value):
                fill_value = 0
            imputed[column] = numeric.fillna(fill_value).round().astype(int)
        elif feature_type == "integer_continuous":
            numeric = numeric_series(imputed[column])
            fill_value = numeric.median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0
            imputed[column] = numeric.fillna(fill_value).round().astype(int)
        else:
            numeric = numeric_series(imputed[column])
            fill_value = numeric.median(skipna=True)
            if pd.isna(fill_value):
                fill_value = 0.0
            imputed[column] = numeric.fillna(fill_value).astype(float)

        report_rows.append(
            {
                "feature": column,
                "feature_type": feature_type,
                "missing_before": int(df[column].isna().sum()),
                "missing_after": int(imputed[column].isna().sum()),
                "imputation_value": str(fill_value),
            }
        )

    return imputed, pd.DataFrame(report_rows)


def scale_zero_to_one(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.astype(float).copy()
    for column in scaled.columns:
        column_min = scaled[column].min()
        column_max = scaled[column].max()
        if pd.isna(column_min) or pd.isna(column_max) or np.isclose(column_min, column_max):
            scaled[column] = 0.0
        else:
            scaled[column] = (scaled[column] - column_min) / (column_max - column_min)
    return scaled


def drop_highly_correlated_features(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_matrix = df.corr(method="pearson").abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if (upper_triangle[column] > threshold).any()]

    dropped_rows: list[dict[str, object]] = []
    for column in to_drop:
        correlations = upper_triangle[column][upper_triangle[column] > threshold].sort_values(ascending=False)
        dropped_rows.append(
            {
                "dropped_feature": column,
                "max_absolute_correlation": float(correlations.iloc[0]),
                "correlated_with": correlations.index[0],
            }
        )

    filtered = df.drop(columns=to_drop)
    return filtered, pd.DataFrame(dropped_rows)


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    feature_columns = get_feature_columns(df)
    metadata_columns = [column for column in df.columns if column not in feature_columns]

    imputed_features, imputation_report = impute_features(df, feature_columns)
    categorical_columns = [
        column for column in feature_columns if classify_feature(df[column]) == "categorical"
    ]

    encoded_features = pd.get_dummies(
        imputed_features,
        columns=categorical_columns,
        dtype=float,
    )
    scaled_features = scale_zero_to_one(encoded_features)
    filtered_features, dropped_feature_report = drop_highly_correlated_features(
        scaled_features,
        threshold=args.correlation_threshold,
    )

    output_df = pd.concat([df[metadata_columns].copy(), filtered_features], axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(args.output_csv, index=False)
    imputation_report.to_csv(args.report_dir / "step3_imputation_report.csv", index=False)
    dropped_feature_report.to_csv(args.report_dir / "step3_dropped_correlated_features.csv", index=False)

    print(f"Input rows: {len(df)}")
    print(f"Input feature columns: {len(feature_columns)}")
    print(f"Categorical feature columns encoded: {len(categorical_columns)}")
    print(f"Encoded feature columns: {encoded_features.shape[1]}")
    print(f"Features dropped by correlation threshold {args.correlation_threshold:.2f}: {len(dropped_feature_report)}")
    print(f"Final feature columns: {filtered_features.shape[1]}")
    print(f"Output written: {args.output_csv}")
    print(f"Imputation report written: {args.report_dir / 'step3_imputation_report.csv'}")
    print(f"Correlation-drop report written: {args.report_dir / 'step3_dropped_correlated_features.csv'}")


if __name__ == "__main__":
    main()
