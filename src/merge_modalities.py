#!/usr/bin/env python3
"""Merge per-modality processed features into a multimodal dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUTS = {
    "t1": "outputs/features_processed_t1.csv",
    "t2": "outputs/features_processed_t2.csv",
    "t1gd": "outputs/features_processed_t1gd.csv",
    "flair": "outputs/features_processed_flair.csv",
}
DEFAULT_OUTPUT = "outputs/features_multimodal.csv"


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _load_modality(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"patient_id", "risk_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}. "
            "Ensure preprocessing outputs include patient_id and risk_label."
        )

    feature_cols = [
        col for col in df.columns if col not in ("patient_id", "risk_label")
    ]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {path.name}.")

    rename = {col: f"{prefix}_{col}" for col in feature_cols}
    df = df.rename(columns=rename)
    df = df[["patient_id", "risk_label", *rename.values()]]
    df = df.rename(columns={"risk_label": f"risk_label_{prefix}"})
    return df


def _validate_risk_labels(merged: pd.DataFrame) -> None:
    risk_cols = [col for col in merged.columns if col.startswith("risk_label_")]
    if not risk_cols:
        raise ValueError("No risk_label columns found after merge.")
    mismatched = merged[risk_cols].nunique(axis=1) > 1
    if mismatched.any():
        examples = merged.loc[mismatched, "patient_id"].astype(str).head(10).tolist()
        raise ValueError(
            "Risk label mismatch across modalities. "
            f"Example patient_ids: {examples}"
        )

    merged["risk_label"] = merged[risk_cols[0]]
    merged.drop(columns=risk_cols, inplace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge T1/T2/T1GD/FLAIR processed CSVs into a multimodal dataset."
    )
    parser.add_argument("--input-t1", default=DEFAULT_INPUTS["t1"])
    parser.add_argument("--input-t2", default=DEFAULT_INPUTS["t2"])
    parser.add_argument("--input-t1gd", default=DEFAULT_INPUTS["t1gd"])
    parser.add_argument("--input-flair", default=DEFAULT_INPUTS["flair"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    paths = {
        "t1": _resolve_path(root, args.input_t1),
        "t2": _resolve_path(root, args.input_t2),
        "t1gd": _resolve_path(root, args.input_t1gd),
        "flair": _resolve_path(root, args.input_flair),
    }
    out_path = _resolve_path(root, args.output)

    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {key} input CSV: {path}")

    t1 = _load_modality(paths["t1"], "t1")
    t2 = _load_modality(paths["t2"], "t2")
    t1gd = _load_modality(paths["t1gd"], "t1gd")
    flair = _load_modality(paths["flair"], "flair")

    merged = t1.merge(t2, on="patient_id", how="inner")
    merged = merged.merge(t1gd, on="patient_id", how="inner")
    merged = merged.merge(flair, on="patient_id", how="inner")

    if merged.empty:
        raise ValueError("Merge produced 0 rows. Check patient_id overlap.")

    _validate_risk_labels(merged)

    feature_cols = [
        col for col in merged.columns if col not in ("patient_id", "risk_label")
    ]
    merged = merged[["patient_id", "risk_label", *feature_cols]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("Multimodal merge complete.")
    print(f"  T1 rows: {len(t1)}")
    print(f"  T2 rows: {len(t2)}")
    print(f"  T1GD rows: {len(t1gd)}")
    print(f"  FLAIR rows: {len(flair)}")
    print(f"  Merged rows: {len(merged)}")
    print(f"  Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
