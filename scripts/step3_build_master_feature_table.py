from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
        "dominant_brain_lobe",
    ],
    "sex": ["Sex", "Gender", "sex", "gender"],
    "mgmt": ["MGMT status", "MGMT", "mgmt_status", "mgmt"],
    "idh": ["IDH", "IDH1", "idh", "idh1"],
    "os": ["OS", "Survival_from_surgery_days_UPDATED", "survival_days"],
    "censor": ["1-dead 0-alive", "Survival_Censor", "survival_censor"],
    "survival_status": ["Survival_Status", "survival_status"],
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_DIR = PROJECT_ROOT / "Clinical+Atlas Merged Data"
DEFAULT_UCSF_MERGED = DEFAULT_MERGED_DIR / "UCSF_sri24_atlas_features_merged.csv"
DEFAULT_UPENN_MERGED = DEFAULT_MERGED_DIR / "UPenn_sri24_atlas_features_merged.csv"


def first_present_column(columns: pd.Index, aliases: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for alias in aliases:
        key = alias.lower()
        if key in lookup:
            return str(lookup[key])
    return None


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

    raw_numeric_predictor_columns = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != id_col and c not in outcome_cols
    ]
    raw_string_predictor_columns = [
        c
        for c in df.columns
        if c != id_col and c not in outcome_cols and (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
    ]

    dataset_out_dir = output_dir / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    master_out = dataset_out_dir / f"{dataset_name}_master_table_step3.csv"
    metadata_out = dataset_out_dir / f"{dataset_name}_step3_metadata.json"

    # Remove old learned Step 3 artifacts so the output folder cannot mix leaky and leakage-safe versions.
    for obsolete_path in [
        dataset_out_dir / f"{dataset_name}_clustering_features_step3.csv",
        dataset_out_dir / f"{dataset_name}_step3_scaler.joblib",
    ]:
        if obsolete_path.exists():
            obsolete_path.unlink()

    df.to_csv(master_out, index=False)

    metadata = {
        "dataset": dataset_name,
        "input_csv": str(input_csv),
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "resolved_columns": resolved,
        "outcome_columns_excluded_from_clustering": sorted(list(outcome_cols)),
        "raw_numeric_predictor_columns": raw_numeric_predictor_columns,
        "raw_string_predictor_columns": raw_string_predictor_columns,
        "step3_mode": "raw_cleaning_only",
        "outputs": {
            "master_table": str(master_out),
        },
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[{dataset_name}] rows={df.shape[0]} raw_numeric_predictors={len(raw_numeric_predictor_columns)}")
    print(f"[{dataset_name}] saved: {master_out}")
    print(f"[{dataset_name}] saved: {metadata_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 raw cleaning for UCSF and UPenn merged atlas+clinical files: "
            "replace text missing markers, coerce numeric-looking columns, resolve key aliases, and save a leakage-safe master table."
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