from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step1_ucsf_imaging_os.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step2_ucsf_clustered.csv"

SURVIVAL_ALIASES = ["OS", "os", "overall_survival", "survival", "survival_days"]
DAYS_PER_MONTH = 30.4375

RISK_CLUSTER_LABELS = {
    0: "high_risk",
    1: "low_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create three survival-based risk clusters from the UCSF dataset. "
            "OS is interpreted as days and converted to months for thresholding."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def resolve_survival_column(columns: pd.Index) -> str:
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for alias in SURVIVAL_ALIASES:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not resolve a survival column from aliases: {SURVIVAL_ALIASES}")


def assign_risk_cluster(survival_months: pd.Series) -> tuple[pd.Series, pd.Series]:
    conditions = [
        survival_months < 12,
        survival_months >= 12,
    ]
    cluster_ids = np.select(conditions, [0, 1], default=-1)
    cluster_ids_series = pd.Series(cluster_ids, index=survival_months.index)
    cluster_labels = cluster_ids_series.map(RISK_CLUSTER_LABELS).fillna("unassigned")
    return cluster_ids_series, cluster_labels


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    survival_column = resolve_survival_column(df.columns)

    survival_days = pd.to_numeric(df[survival_column], errors="coerce")
    missing_survival_mask = survival_days.isna()

    clustered_df = df.loc[~missing_survival_mask].copy()
    clustered_df["survival_months"] = survival_days.loc[~missing_survival_mask] / DAYS_PER_MONTH

    risk_cluster_id, risk_cluster_label = assign_risk_cluster(clustered_df["survival_months"])
    clustered_df["risk_cluster_id"] = risk_cluster_id.astype(int)
    clustered_df["risk_cluster_label"] = risk_cluster_label.astype(str)

    ordered_columns = [
        column
        for column in ["case_id", "patient_id", survival_column, "survival_months", "risk_cluster_id", "risk_cluster_label"]
        if column in clustered_df.columns
    ]
    ordered_columns += [column for column in clustered_df.columns if column not in ordered_columns]
    clustered_df = clustered_df[ordered_columns]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(args.output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Rows excluded due to missing survival: {int(missing_survival_mask.sum())}")
    print(f"Clustered rows: {len(clustered_df)}")
    for cluster_id, label in RISK_CLUSTER_LABELS.items():
        count = int((clustered_df["risk_cluster_id"] == cluster_id).sum())
        print(f"{label}: {count}")
    print(f"Output written: {args.output_csv}")


if __name__ == "__main__":
    main()
