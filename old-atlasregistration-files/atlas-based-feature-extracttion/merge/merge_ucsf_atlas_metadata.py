#!/usr/bin/env python3
"""Merge UCSF atlas features with UCSF metadata on patient ID."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-csv",
        type=Path,
        default=Path("/home/pleb/Codes/GBM-multimodal-clustering/UCSF_sri24_atlas_features_all.csv"),
        help="Atlas feature CSV with a patient_id column.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("/home/pleb/Codes/GBM-multimodal-clustering/UCSF-PDGM-metadata_v5.csv"),
        help="Metadata CSV with an ID column.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/pleb/Codes/GBM-multimodal-clustering/UCSF_sri24_atlas_features_merged.csv"),
        help="Path for the merged CSV.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    atlas_rows = read_rows(args.atlas_csv)
    metadata_rows = read_rows(args.metadata_csv)

    metadata_by_id = {row["ID"]: row for row in metadata_rows}
    metadata_columns = [column for column in metadata_rows[0].keys() if column != "ID"]

    merged_rows: list[dict[str, str]] = []
    for atlas_row in atlas_rows:
        patient_id = atlas_row["patient_id"]
        metadata_row = metadata_by_id.get(patient_id)
        if metadata_row is None:
            continue

        merged_row = dict(atlas_row)
        for column in metadata_columns:
            merged_row[column] = metadata_row[column]
        merged_rows.append(merged_row)

    fieldnames = list(atlas_rows[0].keys()) + metadata_columns
    write_rows(args.out_csv, merged_rows, fieldnames)

    print(f"atlas_rows={len(atlas_rows)}")
    print(f"metadata_rows={len(metadata_rows)}")
    print(f"merged_rows={len(merged_rows)}")
    print(f"wrote={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
