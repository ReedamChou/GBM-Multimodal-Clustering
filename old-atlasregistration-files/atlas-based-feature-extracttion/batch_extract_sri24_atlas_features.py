#!/usr/bin/env python3
"""Run SRI24 atlas-based lobar feature extraction for the UCSF cohort."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_sri24_atlas_features import (
    compute_feature_row,
    load_3d_volume,
    load_atlas_context,
    project_atlas_to_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=Path("/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-IMAGE-STRUCTURAL"),
        help="Directory containing one folder per UCSF patient.",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-AUTOMATED-SEGMENT"),
        help="Directory containing UCSF tumor segmentation files.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path("/tmp/sri24-master/inst/extdata"),
        help="Directory containing the extracted SRI24 atlas files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/pleb/Codes/GBM-multimodal-clustering/UCSF_sri24_atlas_features_all.csv"),
        help="Path for the cohort-level CSV output.",
    )
    parser.add_argument(
        "--dilation-voxels",
        type=int,
        default=3,
        help="Binary dilation applied to the supratentorial atlas mask before lobe fill.",
    )
    return parser.parse_args()


def preferred_segmentation_path(segmentation_dir: Path, case_id: str) -> Path | None:
    candidates = [
        segmentation_dir / f"{case_id}_tumor_segmentation.nii",
        segmentation_dir / f"{case_id}_tumor_segmentation.nii.gz",
        segmentation_dir / f"{case_id}_automated_approx_segm.nii",
        segmentation_dir / f"{case_id}_automated_approx_segm.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No feature rows were produced.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    rows: list[dict[str, object]] = []
    missing_segmentations: list[str] = []
    case_dirs = sorted(path for path in args.structural_dir.iterdir() if path.is_dir())
    atlas = load_atlas_context(args.atlas_dir, args.dilation_voxels)
    projected_cache = None
    projected_shape = None
    projected_affine = None
    reused_projection_count = 0
    rebuilt_projection_count = 0

    for index, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        tumor_seg_path = preferred_segmentation_path(args.segmentation_dir, case_id)
        if tumor_seg_path is None:
            missing_segmentations.append(case_id)
            continue

        tumor_img, tumor_data = load_3d_volume(tumor_seg_path)
        current_shape = tuple(tumor_img.shape[:3])
        current_affine = tumor_img.affine

        if (
            projected_cache is None
            or projected_shape != current_shape
            or not np.allclose(projected_affine, current_affine, atol=1e-4)
        ):
            projected_cache = project_atlas_to_target(atlas, tumor_img)
            projected_shape = current_shape
            projected_affine = current_affine.copy()
            rebuilt_projection_count += 1
        else:
            reused_projection_count += 1

        row = compute_feature_row(
            tumor_data=tumor_data.astype(np.uint8),
            tumor_seg_path=tumor_seg_path,
            case_id=case_id,
            projected=projected_cache,
        )
        row["patient_id"] = case_id
        row["structural_folder"] = str(case_dir)
        row["segmentation_file_used"] = str(tumor_seg_path)
        rows.append(row)

        if index % 25 == 0 or index == len(case_dirs):
            print(f"Processed {index}/{len(case_dirs)} patients", flush=True)

    if missing_segmentations:
        print(f"Skipped {len(missing_segmentations)} patients without segmentations", flush=True)

    print(
        f"Atlas projections rebuilt={rebuilt_projection_count}, reused={reused_projection_count}",
        flush=True,
    )
    write_rows(rows, args.out_csv)
    print(f"Wrote cohort features to {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
