from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "ucsf_with_clinical.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "new" / "step1_ucsf_imaging_os.csv"

SURVIVAL_ALIASES = ["OS", "os", "overall_survival", "survival", "survival_days"]
IDENTIFIER_COLUMNS = {"case_id", "patient_id"}
CLINICAL_COLUMNS = {
    "Sex",
    "Age at MRI",
    "WHO CNS Grade",
    "Final pathologic diagnosis (WHO 2021)",
    "MGMT status",
    "MGMT index",
    "1p/19q",
    "IDH",
    "1-dead 0-alive",
    "EOR",
    "Biopsy prior to imaging",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove clinical columns from the UCSF dataset while keeping identifiers, "
            "imaging-derived columns, and the survival/OS column."
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


def build_output_frame(df: pd.DataFrame, survival_column: str) -> tuple[pd.DataFrame, list[str]]:
    keep_columns: list[str] = []
    dropped_columns: list[str] = []

    for column in df.columns:
        if column in IDENTIFIER_COLUMNS or column == survival_column:
            keep_columns.append(column)
            continue

        if column in CLINICAL_COLUMNS:
            dropped_columns.append(column)
            continue

        keep_columns.append(column)

    return df[keep_columns].copy(), dropped_columns


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    survival_column = resolve_survival_column(df.columns)
    output_df, dropped_columns = build_output_frame(df, survival_column)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")
    print(f"Survival column kept: {survival_column}")
    print(f"Output columns: {len(output_df.columns)}")
    print(f"Clinical columns removed: {len(dropped_columns)}")
    if dropped_columns:
        print("Removed columns:")
        for column in dropped_columns:
            print(f"- {column}")
    print(f"Output written: {args.output_csv}")


if __name__ == "__main__":
    main()
