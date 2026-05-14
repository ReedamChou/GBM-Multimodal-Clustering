#!/usr/bin/env python3
"""Test single-case tumor feature extraction using nibabel and numpy.

Run with the project environment, for example:
`venv-gbm/bin/python test_single_case_features.py`

This script loads:
- a FastSurfer atlas: `aparc.DKTatlas+aseg.deep.mgz`
- a tumor segmentation: `UCSF-PDGM-0004_tumor_segmentation.nii`

It computes:
- global NC/EN ratio
- global ED/EN ratio
- global ED total ratio
- tumor burden index

It also attempts to derive:
- dominant brain lobe
- per-lobe ED/EN/NC ratios across frontal, temporal, parietal, occipital

If the lobe features are not reliable from these two files alone, the script
records that explicitly in the output CSV.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


MGZ_PATH = Path("/home/pleb/fastsurfer/output/patient1/mri/aparc.DKTatlas+aseg.deep.mgz")
TUMOR_PATH = Path(
    "/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-AUTOMATED-SEGMENT/"
    "UCSF-PDGM-0004_tumor_segmentation.nii"
)
OUT_CSV = Path("/home/pleb/Codes/GBM-multimodal-clustering/test_single_case_features.csv")

NC_LABEL = 1
ED_LABEL = 2
EN_LABEL = 4

# DKT cortical labels grouped into the 4 requested lobes.
# These labels only cover cortical parcels, not the full white-matter extent.
LOBE_LABELS = {
    "frontal": {
        1002,
        1003,
        1012,
        1014,
        1017,
        1018,
        1019,
        1020,
        1024,
        1026,
        1027,
        1028,
        2002,
        2003,
        2012,
        2014,
        2017,
        2018,
        2019,
        2020,
        2024,
        2026,
        2027,
        2028,
    },
    "temporal": {
        1006,
        1007,
        1009,
        1015,
        1016,
        1030,
        1034,
        2006,
        2007,
        2009,
        2015,
        2016,
        2030,
        2034,
    },
    "parietal": {
        1008,
        1010,
        1022,
        1023,
        1025,
        1029,
        1031,
        2008,
        2010,
        2022,
        2023,
        2025,
        2029,
        2031,
    },
    "occipital": {
        1005,
        1011,
        1013,
        1021,
        2005,
        2011,
        2013,
        2021,
    },
}

LABEL_TO_LOBE = {
    label: lobe for lobe, labels in LOBE_LABELS.items() for label in labels
}


def safe_div(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else float(numerator) / float(denominator)


def load_images() -> tuple[nib.spatialimages.SpatialImage, np.ndarray, nib.spatialimages.SpatialImage, np.ndarray]:
    mgz_img = nib.load(str(MGZ_PATH))
    tumor_img = nib.load(str(TUMOR_PATH))

    mgz_data = np.asarray(mgz_img.dataobj, dtype=np.int32)
    tumor_data = np.asarray(tumor_img.dataobj, dtype=np.uint8)

    if mgz_data.ndim == 4 and mgz_data.shape[-1] == 1:
        mgz_data = mgz_data[..., 0]

    if mgz_data.ndim != 3:
        raise ValueError(f"Expected a 3D atlas volume, got shape {mgz_data.shape}")
    if tumor_data.ndim != 3:
        raise ValueError(f"Expected a 3D tumor mask, got shape {tumor_data.shape}")

    return mgz_img, mgz_data, tumor_img, tumor_data


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(zooms[0] * zooms[1] * zooms[2])


def compute_global_features(
    mgz_img: nib.spatialimages.SpatialImage,
    mgz_data: np.ndarray,
    tumor_img: nib.spatialimages.SpatialImage,
    tumor_data: np.ndarray,
) -> dict[str, object]:
    nc_voxels = int(np.count_nonzero(tumor_data == NC_LABEL))
    ed_voxels = int(np.count_nonzero(tumor_data == ED_LABEL))
    en_voxels = int(np.count_nonzero(tumor_data == EN_LABEL))
    total_tumor_voxels = nc_voxels + ed_voxels + en_voxels

    brain_voxels = int(np.count_nonzero(mgz_data))
    tumor_volume_mm3 = total_tumor_voxels * voxel_volume_mm3(tumor_img)
    brain_volume_mm3 = brain_voxels * voxel_volume_mm3(mgz_img)

    return {
        "global_nc_en_ratio": safe_div(nc_voxels, en_voxels),
        "global_ed_en_ratio": safe_div(ed_voxels, en_voxels),
        "global_ed_total_ratio": safe_div(ed_voxels, total_tumor_voxels),
        "tumor_burden_index": safe_div(tumor_volume_mm3, brain_volume_mm3),
        "tumor_nc_voxels": nc_voxels,
        "tumor_ed_voxels": ed_voxels,
        "tumor_en_voxels": en_voxels,
        "tumor_total_voxels": total_tumor_voxels,
        "brain_voxels_from_mgz": brain_voxels,
        "tumor_volume_mm3": tumor_volume_mm3,
        "brain_volume_mm3": brain_volume_mm3,
        "mgz_shape": "x".join(str(x) for x in mgz_data.shape),
        "tumor_shape": "x".join(str(x) for x in tumor_data.shape),
    }


def map_tumor_voxels_to_lobes(
    mgz_img: nib.spatialimages.SpatialImage,
    mgz_data: np.ndarray,
    tumor_img: nib.spatialimages.SpatialImage,
    tumor_data: np.ndarray,
) -> tuple[dict[str, Counter], int, int, int]:
    per_lobe = {lobe: Counter() for lobe in LOBE_LABELS}

    tumor_coords = np.argwhere(tumor_data > 0)
    if tumor_coords.size == 0:
        return per_lobe, 0, 0, 0

    tumor_values = tumor_data[tuple(tumor_coords.T)]
    world_coords = nib.affines.apply_affine(tumor_img.affine, tumor_coords)
    mgz_float = nib.affines.apply_affine(np.linalg.inv(mgz_img.affine), world_coords)
    mgz_idx = np.rint(mgz_float).astype(np.int32)

    shape = np.array(mgz_data.shape, dtype=np.int32)
    in_bounds_mask = np.all((mgz_idx >= 0) & (mgz_idx < shape), axis=1)

    out_of_bounds = int(np.count_nonzero(~in_bounds_mask))
    unassigned = 0
    mapped = 0

    for idx, value, in_bounds in zip(mgz_idx, tumor_values, in_bounds_mask):
        if not in_bounds:
            continue

        atlas_label = int(mgz_data[tuple(idx)])
        lobe = LABEL_TO_LOBE.get(atlas_label)
        if lobe is None:
            unassigned += 1
            continue

        per_lobe[lobe][int(value)] += 1
        mapped += 1

    return per_lobe, mapped, unassigned, out_of_bounds


def build_output_row() -> dict[str, object]:
    mgz_img, mgz_data, tumor_img, tumor_data = load_images()
    row: dict[str, object] = {
        "mgz_path": str(MGZ_PATH),
        "tumor_path": str(TUMOR_PATH),
        "dominant_brain_lobe": "",
        "frontal_ed_ratio": "",
        "frontal_en_ratio": "",
        "frontal_nc_ratio": "",
        "temporal_ed_ratio": "",
        "temporal_en_ratio": "",
        "temporal_nc_ratio": "",
        "parietal_ed_ratio": "",
        "parietal_en_ratio": "",
        "parietal_nc_ratio": "",
        "occipital_ed_ratio": "",
        "occipital_en_ratio": "",
        "occipital_nc_ratio": "",
        "lobe_feature_status": "unavailable",
        "lobe_feature_reason": "",
        "mapped_lobar_tumor_voxels": 0,
        "mapped_lobar_fraction": 0.0,
    }
    row.update(compute_global_features(mgz_img, mgz_data, tumor_img, tumor_data))

    per_lobe, mapped, unassigned, out_of_bounds = map_tumor_voxels_to_lobes(
        mgz_img, mgz_data, tumor_img, tumor_data
    )

    total_tumor_voxels = row["tumor_total_voxels"]
    row["mapped_lobar_tumor_voxels"] = mapped
    row["mapped_lobar_fraction"] = safe_div(mapped, total_tumor_voxels)

    if mapped == 0:
        row["lobe_feature_reason"] = (
            "No tumor voxels landed on the selected DKT cortical lobe labels after atlas mapping."
        )
        return row

    if mapped / total_tumor_voxels < 0.5:
        row["lobe_feature_reason"] = (
            "Most tumor voxels are not represented by the 4-lobe DKT cortical labels in "
            f"`aparc.DKTatlas+aseg.deep.mgz` (mapped={mapped}, unassigned={unassigned}, "
            f"out_of_bounds={out_of_bounds}). A dedicated lobar mask or subject-space lobe "
            "segmentation is needed for reliable dominant-lobe and per-lobe ratios."
        )
        return row

    lobe_totals = {}
    for lobe, counts in per_lobe.items():
        lobe_nc = counts[NC_LABEL]
        lobe_ed = counts[ED_LABEL]
        lobe_en = counts[EN_LABEL]
        lobe_total = lobe_nc + lobe_ed + lobe_en
        lobe_totals[lobe] = lobe_total
        row[f"{lobe}_ed_ratio"] = safe_div(lobe_ed, lobe_total)
        row[f"{lobe}_en_ratio"] = safe_div(lobe_en, lobe_total)
        row[f"{lobe}_nc_ratio"] = safe_div(lobe_nc, lobe_total)

    dominant_total = max(lobe_totals.values())
    if dominant_total > 0:
        row["dominant_brain_lobe"] = max(lobe_totals, key=lobe_totals.get)
        row["lobe_feature_status"] = "ok"

    return row


def write_csv(row: dict[str, object], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    row = build_output_row()
    write_csv(row, OUT_CSV)
    print(f"Wrote feature row to {OUT_CSV}")
    print(f"Lobe feature status: {row['lobe_feature_status']}")
    if row["lobe_feature_reason"]:
        print(row["lobe_feature_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
