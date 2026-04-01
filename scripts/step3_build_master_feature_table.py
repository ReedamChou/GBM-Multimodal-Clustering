from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_DIR = PROJECT_ROOT / "Clinical+Atlas Merged Data"
DEFAULT_UCSF_MERGED = DEFAULT_MERGED_DIR / "UCSF_sri24_atlas_features_merged.csv"
DEFAULT_UPENN_MERGED = DEFAULT_MERGED_DIR / "UPenn_sri24_atlas_features_merged.csv"


MISSING_MARKERS = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not available",
    "not applicable",
    "unknown",
    "indeterminate",
}


COLUMN_ALIASES = {
    "id": ["ID", "id", "PatientID", "patient_id"],
    "dominant_lobe": [
        "dominant_lobe",
        "Dominant_Lobe",
        "Dominant lobe",
        "dominant brain lobe",
    ],
    "sex": ["Sex", "Gender", "sex", "gender"],
    "mgmt": ["MGMT status", "MGMT", "mgmt_status", "mgmt"],
    "idh": ["IDH", "IDH1", "idh", "idh1"],
    "os": ["OS", "Survival_from_surgery_days_UPDATED", "survival_days"],
    "censor": ["1-dead 0-alive", "Survival_Censor", "survival_censor"],
    "survival_status": ["Survival_Status", "survival_status"],
}


def first_present_column(columns: pd.Index, aliases: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for alias in aliases:
        key = alias.lower()
        if key in lookup:
            return str(lookup[key])
    return None


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def safe_mode(series: pd.Series, default: str | int = "unknown") -> str | int:
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return default
    return mode_values.iloc[0]


def replace_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = out[col].astype("string").str.strip()
            lower_s = s.str.lower()
            out[col] = s.mask(lower_s.isin(MISSING_MARKERS), pd.NA)
    return out


def maybe_convert_to_numeric(df: pd.DataFrame, minimum_parse_rate: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            non_null_count = out[col].notna().sum()
            if non_null_count == 0:
                continue
            converted = pd.to_numeric(out[col], errors="coerce")
            parse_rate = converted.notna().sum() / float(non_null_count)
            if parse_rate >= minimum_parse_rate:
                out[col] = converted
    return out


def map_sex_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(["m", "male"])] = 1
    out[s.isin(["f", "female"])] = 0
    return out


def map_mgmt_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("positive", na=False) | s.str.contains("methylated", na=False)] = 1
    out[s.str.contains("negative", na=False) | s.str.contains("unmethylated", na=False)] = 0
    return out


def map_idh_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("wild", na=False)] = 0
    out[s.str.contains("mutat", na=False) | s.str.contains("idh1 p\\.", na=False) | s.str.contains("idh2 p\\.", na=False)] = 1
    return out


def canonicalize_lobe(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[s.str.contains("frontal", na=False)] = "frontal"
    out[s.str.contains("temporal", na=False)] = "temporal"
    out[s.str.contains("parietal", na=False)] = "parietal"
    out[s.str.contains("occipital", na=False)] = "occipital"
    return out


def encode_binary_with_unknown_rule(
    df: pd.DataFrame,
    source_col: str,
    output_col: str,
    mapper: Callable[[pd.Series], pd.Series],
    unknown_threshold: float = 0.10,
) -> tuple[str, str | None, float, int]:
    mapped = mapper(df[source_col])
    missing_rate = float(mapped.isna().mean())

    unknown_col = None
    if missing_rate > unknown_threshold:
        unknown_col = f"{output_col}_unknown"
        df[unknown_col] = mapped.isna().astype(int)

    fill_value = int(safe_mode(mapped, default=0))
    df[output_col] = mapped.fillna(fill_value).astype(int)
    return output_col, unknown_col, missing_rate, fill_value


def process_step3(dataset_name: str, input_csv: Path, output_dir: Path) -> None:
    df_raw = pd.read_csv(input_csv)
    df = replace_missing_markers(df_raw)
    df = maybe_convert_to_numeric(df)

    resolved = {
        key: first_present_column(df.columns, aliases)
        for key, aliases in COLUMN_ALIASES.items()
    }

    id_col = resolved["id"]
    if id_col is None:
        raise ValueError(f"{dataset_name}: no patient ID column found.")

    outcome_cols = {
        c
        for c in [resolved["os"], resolved["censor"], resolved["survival_status"]]
        if c is not None
    }

    categorical_fill_values: dict[str, str] = {}
    numeric_fill_values: dict[str, float] = {}
    binary_columns: list[str] = []

    for key, mapper, out_col in [
        ("sex", map_sex_to_binary, "sex_bin"),
        ("mgmt", map_mgmt_to_binary, "mgmt_bin"),
        ("idh", map_idh_to_binary, "idh_bin"),
    ]:
        src = resolved[key]
        if src is None:
            continue
        encoded_col, unknown_col, _, _ = encode_binary_with_unknown_rule(df, src, out_col, mapper)
        binary_columns.append(encoded_col)
        if unknown_col is not None:
            binary_columns.append(unknown_col)

    dominant_lobe_col = resolved["dominant_lobe"]
    if dominant_lobe_col is not None:
        lobe = canonicalize_lobe(df[dominant_lobe_col])
        lobe_missing_rate = float(lobe.isna().mean())
        if lobe_missing_rate > 0.10:
            fill_value = "unknown"
        else:
            fill_value = str(safe_mode(lobe, default="unknown"))
        lobe = lobe.fillna(fill_value)
        df["dominant_lobe_clean"] = lobe
        categorical_fill_values["dominant_lobe_clean"] = fill_value

        for lobe_name in ["frontal", "temporal", "parietal", "occipital"]:
            col_name = f"dominant_lobe_{lobe_name}"
            df[col_name] = (lobe == lobe_name).astype(int)
            binary_columns.append(col_name)

        if fill_value == "unknown":
            df["dominant_lobe_unknown"] = (lobe == "unknown").astype(int)
            binary_columns.append("dominant_lobe_unknown")

    for col in df.columns:
        if col == id_col or col in outcome_cols:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            missing_rate = float(df[col].isna().mean())
            if missing_rate == 0:
                continue
            if missing_rate > 0.10:
                fill_value = "unknown"
            else:
                fill_value = str(safe_mode(df[col], default="unknown"))
            df[col] = df[col].fillna(fill_value)
            categorical_fill_values[col] = fill_value

    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != id_col and c not in outcome_cols
    ]

    continuous_cols = [
        c
        for c in numeric_cols
        if c not in binary_columns and df[c].nunique(dropna=True) > 2
    ]

    for col in continuous_cols:
        median_val = float(df[col].median(skipna=True))
        df[col] = df[col].fillna(median_val)
        numeric_fill_values[col] = median_val

    scaler = StandardScaler()
    if continuous_cols:
        df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

    feature_cols = [
        c
        for c in df.columns
        if c not in outcome_cols and c != id_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    dataset_out_dir = output_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    master_out = dataset_out_dir / f"{dataset_name}_master_table_step3.csv"
    features_out = dataset_out_dir / f"{dataset_name}_clustering_features_step3.csv"
    scaler_out = dataset_out_dir / f"{dataset_name}_step3_scaler.joblib"
    metadata_out = dataset_out_dir / f"{dataset_name}_step3_metadata.json"

    df.to_csv(master_out, index=False)
    df[[id_col] + feature_cols].to_csv(features_out, index=False)
    joblib.dump({"scaler": scaler, "continuous_columns": continuous_cols}, scaler_out)

    metadata = {
        "dataset": dataset_name,
        "input_csv": str(input_csv),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "resolved_columns": resolved,
        "outcome_columns_excluded_from_clustering": sorted(list(outcome_cols)),
        "binary_columns": sorted(binary_columns),
        "continuous_columns": sorted(continuous_cols),
        "clustering_feature_columns": feature_cols,
        "categorical_imputation": categorical_fill_values,
        "continuous_median_imputation": numeric_fill_values,
        "outputs": {
            "master_table": str(master_out),
            "clustering_features": str(features_out),
            "scaler": str(scaler_out),
        },
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[{dataset_name}] rows={df.shape[0]} features={len(feature_cols)}")
    print(f"[{dataset_name}] saved: {master_out}")
    print(f"[{dataset_name}] saved: {features_out}")
    print(f"[{dataset_name}] saved: {scaler_out}")
    print(f"[{dataset_name}] saved: {metadata_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 preprocessing for UCSF and UPenn merged atlas+clinical files: "
            "missing-value handling, encoding, standardization, and clustering feature export."
        )
    )
    parser.add_argument(
        "--ucsf-merged",
        type=Path,
        help=(
            "Path to merged UCSF atlas+clinical CSV. "
            "If omitted, defaults to Clinical+Atlas Merged Data/UCSF_sri24_atlas_features_merged.csv when present."
        ),
    )
    parser.add_argument(
        "--upenn-merged",
        type=Path,
        help=(
            "Path to merged UPenn atlas+clinical CSV. "
            "If omitted, defaults to Clinical+Atlas Merged Data/UPenn_sri24_atlas_features_merged.csv when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "step3",
        help="Output directory for Step 3 artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ucsf_merged = args.ucsf_merged or (DEFAULT_UCSF_MERGED if DEFAULT_UCSF_MERGED.exists() else None)
    upenn_merged = args.upenn_merged or (DEFAULT_UPENN_MERGED if DEFAULT_UPENN_MERGED.exists() else None)

    if ucsf_merged is None and upenn_merged is None:
        raise ValueError(
            "No input files found. Provide --ucsf-merged and/or --upenn-merged, "
            "or place merged files in Clinical+Atlas Merged Data with expected names."
        )

    if ucsf_merged is not None and not ucsf_merged.exists():
        raise FileNotFoundError(f"UCSF merged file not found: {ucsf_merged}")

    if upenn_merged is not None and not upenn_merged.exists():
        raise FileNotFoundError(f"UPenn merged file not found: {upenn_merged}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if ucsf_merged is not None:
        process_step3("ucsf", ucsf_merged, args.output_dir)

    if upenn_merged is not None:
        process_step3("upenn", upenn_merged, args.output_dir)


if __name__ == "__main__":
    main()