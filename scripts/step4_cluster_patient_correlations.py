from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step3_ucsf_preprocessed_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "new"

PATIENT_ID_ALIASES = ["patient_id", "case_id", "PatientID", "ID", "id", "case_id"]
METADATA_COLUMNS = {"case_id", "patient_id", "OS", "os", "overall_survival", "survival", "survival_days", "survival_months", "risk_cluster_id", "risk_cluster_label"}
RISK_CLUSTER_LABELS = {
    0: "high_risk",
    1: "low_risk",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one patient-by-patient Pearson correlation matrix per survival cluster "
            "using processed imaging features only."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_patient_id_column(columns: pd.Index) -> str:
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for alias in PATIENT_ID_ALIASES:
        match = lower_to_original.get(alias.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not resolve a patient ID column from aliases: {PATIENT_ID_ALIASES}")


def get_imaging_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in METADATA_COLUMNS]


def compute_patient_correlation_matrix(cluster_df: pd.DataFrame, patient_id_column: str, feature_columns: list[str]) -> pd.DataFrame:
    feature_matrix = cluster_df[feature_columns].astype(float)
    patient_correlation = feature_matrix.T.corr(method="pearson")
    patient_labels = cluster_df[patient_id_column].astype(str)
    patient_correlation.index = patient_labels
    patient_correlation.columns = patient_labels
    return patient_correlation


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    print(df["risk_cluster_id"].value_counts())
    patient_id_column = resolve_patient_id_column(df.columns)
    feature_columns = get_imaging_feature_columns(df)

    if "risk_cluster_id" not in df.columns:
        raise ValueError("Input CSV must include the 'risk_cluster_id' column.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, label in RISK_CLUSTER_LABELS.items():
        cluster_df = df.loc[df["risk_cluster_id"] == cluster_id].copy()
        if cluster_df.empty:
            raise ValueError(f"No patients found for cluster {cluster_id} ({label}).")

        matrix = compute_patient_correlation_matrix(cluster_df, patient_id_column, feature_columns)
        output_csv = args.output_dir / f"ucsf_cluster_{cluster_id}_{label}_patient_correlation_matrix.csv"
        matrix.to_csv(output_csv, index=True)

        print(f"Cluster {cluster_id} ({label}) patients: {len(cluster_df)}")
        print(f"Cluster {cluster_id} output written: {output_csv}")


if __name__ == "__main__":
    main()
