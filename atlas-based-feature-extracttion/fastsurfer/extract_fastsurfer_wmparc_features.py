#!/usr/bin/env python3
"""Extract research-grade lobar tumor features from FastSurfer + tumor labels.

This pipeline is built around subject-specific FastSurfer outputs, preferring
surface-derived `wmparc` labels for white matter coverage and DKT cortical
labels for cortical coverage. Tumor voxels are sampled into the subject-space
anatomical labels using affine-based nearest-neighbor mapping, which is
equivalent to resampling the anatomical labels into tumor space without
interpolating the tumor mask.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable

import nibabel as nib
import numpy as np


NC_LABEL = 1
ED_LABEL = 2
EN_LABEL = 4

LOBE_IDS = {
    "frontal": 1,
    "temporal": 2,
    "parietal": 3,
    "occipital": 4,
}
LOBE_NAMES = tuple(LOBE_IDS.keys())

# DKT cortical parcel labels. Cingulate and insula are intentionally excluded
# because the requested output space is restricted to the 4 major lobes.
CORTICAL_LOBE_LABELS = {
    "frontal": {
        1003, 1012, 1014, 1017, 1018, 1019, 1020, 1024, 1027, 1028,
        2003, 2012, 2014, 2017, 2018, 2019, 2020, 2024, 2027, 2028,
    },
    "temporal": {
        1006, 1007, 1009, 1015, 1016, 1030, 1034,
        2006, 2007, 2009, 2015, 2016, 2030, 2034,
    },
    "parietal": {
        1008, 1022, 1025, 1029, 1031,
        2008, 2022, 2025, 2029, 2031,
    },
    "occipital": {
        1005, 1011, 1013, 1021,
        2005, 2011, 2013, 2021,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-dir",
        type=Path,
        required=True,
        help="FastSurfer subject directory, for example /home/pleb/fastsurfer/output/patient1",
    )
    parser.add_argument(
        "--tumor-seg",
        type=Path,
        required=True,
        help="Tumor segmentation NIfTI with labels 1=NC, 2=ED, 4=EN.",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Optional case identifier written to the CSV. Defaults to tumor filename stem.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else float(numerator) / float(denominator)


def load_3d_volume(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got shape {data.shape}")
    return img, data


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(zooms[0] * zooms[1] * zooms[2])


def expected_wmparc_labels(cortical_labels: Iterable[int]) -> set[int]:
    out: set[int] = set()
    for label in cortical_labels:
        if 1000 <= label < 2000:
            out.add(3000 + (label - 1000))
        elif 2000 <= label < 3000:
            out.add(4000 + (label - 2000))
    return out


def pick_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {description}. Tried: {joined}")


def build_lobar_label_volume(aparc_data: np.ndarray, wmparc_data: np.ndarray) -> np.ndarray:
    if aparc_data.shape != wmparc_data.shape:
        raise ValueError(
            f"Shape mismatch between aparc and wmparc: {aparc_data.shape} vs {wmparc_data.shape}"
        )

    lobar = np.zeros(aparc_data.shape, dtype=np.uint8)

    for lobe_name, lobe_id in LOBE_IDS.items():
        cortical_labels = CORTICAL_LOBE_LABELS[lobe_name]
        wm_labels = expected_wmparc_labels(cortical_labels)

        cortical_mask = np.isin(aparc_data, list(cortical_labels))
        wm_mask = np.isin(wmparc_data, list(wm_labels))

        lobar[cortical_mask | wm_mask] = lobe_id

    return lobar


def sample_source_labels_at_target_voxels(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    target_coords: np.ndarray,
    target_affine: np.ndarray,
    fill_value: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    world_coords = nib.affines.apply_affine(target_affine, target_coords)
    src_float = nib.affines.apply_affine(np.linalg.inv(source_affine), world_coords)
    src_idx = np.rint(src_float).astype(np.int32)

    src_shape = np.asarray(source_data.shape, dtype=np.int32)
    valid = np.all((src_idx >= 0) & (src_idx < src_shape), axis=1)

    sampled = np.full(target_coords.shape[0], fill_value, dtype=source_data.dtype)
    if np.any(valid):
        sampled[valid] = source_data[tuple(src_idx[valid].T)]

    return sampled, valid


def extract_features(
    subject_dir: Path,
    tumor_seg_path: Path,
    case_id: str,
) -> dict[str, object]:
    mri_dir = subject_dir / "mri"

    wmparc_path = pick_existing(
        [
            mri_dir / "wmparc.DKTatlas.mapped.mgz",
            mri_dir / "wmparc.mgz",
        ],
        "wmparc volume",
    )
    aparc_path = pick_existing(
        [
            mri_dir / "aparc.DKTatlas+aseg.mapped.mgz",
            mri_dir / "aparc.DKTatlas+aseg.deep.mgz",
            mri_dir / "aparc.DKTatlas+aseg.orig.mgz",
        ],
        "aparc+DKT volume",
    )
    brainmask_path = pick_existing(
        [
            mri_dir / "brainmask.mgz",
            mri_dir / "mask.mgz",
        ],
        "brain mask",
    )

    wmparc_img, wmparc_data = load_3d_volume(wmparc_path)
    aparc_img, aparc_data = load_3d_volume(aparc_path)
    brainmask_img, brainmask_data = load_3d_volume(brainmask_path)
    tumor_img, tumor_data = load_3d_volume(tumor_seg_path)

    if not np.allclose(wmparc_img.affine, aparc_img.affine, atol=1e-4):
        raise ValueError("wmparc and aparc affines do not match.")
    if wmparc_data.shape != aparc_data.shape:
        raise ValueError("wmparc and aparc shapes do not match.")

    lobar_subject = build_lobar_label_volume(aparc_data.astype(np.int32), wmparc_data.astype(np.int32))

    tumor_mask = tumor_data > 0
    tumor_coords = np.argwhere(tumor_mask)
    tumor_values = tumor_data[tuple(tumor_coords.T)].astype(np.uint8)

    sampled_lobes, valid_hits = sample_source_labels_at_target_voxels(
        source_data=lobar_subject,
        source_affine=wmparc_img.affine,
        target_coords=tumor_coords,
        target_affine=tumor_img.affine,
        fill_value=0,
    )

    total_tumor_voxels = int(tumor_mask.sum())
    nc_voxels = int(np.count_nonzero(tumor_data == NC_LABEL))
    ed_voxels = int(np.count_nonzero(tumor_data == ED_LABEL))
    en_voxels = int(np.count_nonzero(tumor_data == EN_LABEL))

    brain_voxels = int(np.count_nonzero(brainmask_data > 0))
    brain_volume_mm3 = brain_voxels * voxel_volume_mm3(brainmask_img)
    tumor_volume_mm3 = total_tumor_voxels * voxel_volume_mm3(tumor_img)

    mapped_mask = sampled_lobes > 0
    mapped_voxels = int(np.count_nonzero(mapped_mask))
    outside_lobes_voxels = total_tumor_voxels - mapped_voxels
    affine_out_of_bounds_voxels = int(np.count_nonzero(~valid_hits))

    per_lobe_counts: Dict[str, Dict[str, int]] = {
        lobe: {"NC": 0, "ED": 0, "EN": 0, "TOTAL": 0} for lobe in LOBE_NAMES
    }

    for lobe_name, lobe_id in LOBE_IDS.items():
        lobe_mask = sampled_lobes == lobe_id
        lobe_values = tumor_values[lobe_mask]
        lobe_nc = int(np.count_nonzero(lobe_values == NC_LABEL))
        lobe_ed = int(np.count_nonzero(lobe_values == ED_LABEL))
        lobe_en = int(np.count_nonzero(lobe_values == EN_LABEL))
        lobe_total = lobe_nc + lobe_ed + lobe_en
        per_lobe_counts[lobe_name] = {
            "NC": lobe_nc,
            "ED": lobe_ed,
            "EN": lobe_en,
            "TOTAL": lobe_total,
        }

    dominant_lobe = ""
    if mapped_voxels > 0:
        dominant_lobe = max(LOBE_NAMES, key=lambda name: per_lobe_counts[name]["TOTAL"])
        if per_lobe_counts[dominant_lobe]["TOTAL"] == 0:
            dominant_lobe = ""

    row: dict[str, object] = {
        "case_id": case_id,
        "subject_dir": str(subject_dir),
        "tumor_seg_path": str(tumor_seg_path),
        "aparc_path": str(aparc_path),
        "wmparc_path": str(wmparc_path),
        "brainmask_path": str(brainmask_path),
        "dominant_brain_lobe": dominant_lobe,
        "global_nc_en_ratio": safe_div(nc_voxels, en_voxels),
        "global_ed_en_ratio": safe_div(ed_voxels, en_voxels),
        "global_ed_total_ratio": safe_div(ed_voxels, total_tumor_voxels),
        "tumor_burden_index": safe_div(tumor_volume_mm3, brain_volume_mm3),
        "tumor_nc_voxels": nc_voxels,
        "tumor_ed_voxels": ed_voxels,
        "tumor_en_voxels": en_voxels,
        "tumor_total_voxels": total_tumor_voxels,
        "brain_voxels": brain_voxels,
        "tumor_volume_mm3": tumor_volume_mm3,
        "brain_volume_mm3": brain_volume_mm3,
        "tumor_voxels_in_requested_lobes": mapped_voxels,
        "tumor_voxels_outside_requested_lobes": outside_lobes_voxels,
        "tumor_requested_lobe_fraction": safe_div(mapped_voxels, total_tumor_voxels),
        "affine_out_of_bounds_voxels": affine_out_of_bounds_voxels,
        "lobe_assignment_reliable": int(safe_div(mapped_voxels, total_tumor_voxels) >= 0.8),
    }

    for lobe_name in LOBE_NAMES:
        counts = per_lobe_counts[lobe_name]
        row[f"{lobe_name}_ed_ratio"] = safe_div(counts["ED"], counts["TOTAL"])
        row[f"{lobe_name}_en_ratio"] = safe_div(counts["EN"], counts["TOTAL"])
        row[f"{lobe_name}_nc_ratio"] = safe_div(counts["NC"], counts["TOTAL"])
        row[f"{lobe_name}_tumor_voxels"] = counts["TOTAL"]
        row[f"{lobe_name}_ed_voxels"] = counts["ED"]
        row[f"{lobe_name}_en_voxels"] = counts["EN"]
        row[f"{lobe_name}_nc_voxels"] = counts["NC"]

    if dominant_lobe:
        dominant_total = per_lobe_counts[dominant_lobe]["TOTAL"]
        row["dominant_lobe_fraction_of_total_tumor"] = safe_div(dominant_total, total_tumor_voxels)
        row["dominant_lobe_fraction_of_mapped_tumor"] = safe_div(dominant_total, mapped_voxels)
    else:
        row["dominant_lobe_fraction_of_total_tumor"] = float("nan")
        row["dominant_lobe_fraction_of_mapped_tumor"] = float("nan")

    return row


def write_csv(row: dict[str, object], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    case_id = args.case_id or args.tumor_seg.stem
    row = extract_features(args.subject_dir, args.tumor_seg, case_id)
    write_csv(row, args.out_csv)

    print(f"Wrote features to {args.out_csv}")
    print(f"dominant_brain_lobe={row['dominant_brain_lobe']}")
    print(f"tumor_requested_lobe_fraction={row['tumor_requested_lobe_fraction']}")
    print(f"lobe_assignment_reliable={row['lobe_assignment_reliable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
