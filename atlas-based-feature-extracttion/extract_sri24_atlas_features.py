#!/usr/bin/env python3
"""Extract coarse lobar tumor features in SRI24 template space.

This script assumes the tumor segmentation is already aligned to the common
240x240x155, 1 mm template space used by the local UCSF/UPenn cohort. It uses
the SRI24 TZO atlas as cortical lobe seeds, expands those seeds through a
slightly dilated supratentorial mask, samples the resulting lobar atlas at the
tumor voxels, and writes the requested features to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


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

LOBE_PREFIXES = {
    "frontal": (
        "Frontal_",
        "Precentral_",
        "Rolandic_Oper_",
        "Supp_Motor_Area_",
        "Olfactory_",
        "Rectus_",
    ),
    "temporal": (
        "Temporal_",
        "Heschl_",
        "ParaHippocampal_",
        "Hippocampus_",
        "Amygdala_",
        "Fusiform_",
    ),
    "parietal": (
        "Postcentral_",
        "Parietal_",
        "SupraMarginal_",
        "Angular_",
        "Precuneus_",
        "Paracentral_Lobule_",
    ),
    "occipital": (
        "Calcarine_",
        "Cuneus_",
        "Lingual_",
        "Occipital_",
    ),
}


@dataclass(frozen=True)
class AtlasContext:
    atlas_dir: Path
    label_file: Path
    tzo_path: Path
    suptent_path: Path
    tissues_path: Path
    source_affine: np.ndarray
    seed_volume: np.ndarray
    lobar_atlas: np.ndarray
    dilated_suptent_mask: np.ndarray
    source_brainmask: np.ndarray
    seed_sets: dict[str, set[int]]
    dilation_voxels: int


@dataclass(frozen=True)
class ProjectedAtlasContext:
    atlas: AtlasContext
    target_shape: tuple[int, int, int]
    target_affine: np.ndarray
    lobar_labels: np.ndarray
    seed_labels: np.ndarray
    suptent_mask: np.ndarray
    brainmask: np.ndarray
    lobar_valid: np.ndarray
    brainmask_valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tumor-seg",
        type=Path,
        required=True,
        help="Tumor segmentation NIfTI in common template space.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Optional case identifier written to the CSV.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path("/tmp/sri24-master/inst/extdata"),
        help="Directory containing SRI24 atlas files such as tzo116plus.nii.gz.",
    )
    parser.add_argument(
        "--dilation-voxels",
        type=int,
        default=3,
        help="Binary dilation applied to the supratentorial mask before lobe fill.",
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
        raise ValueError(f"Expected a 3D volume at {path}, got shape {data.shape}")
    return img, data


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


def sample_source_into_target_grid(
    source_data: np.ndarray,
    source_affine: np.ndarray,
    target_img: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, np.ndarray]:
    target_coords = np.indices(target_img.shape[:3], dtype=np.int32).reshape(3, -1).T
    sampled, valid = sample_source_labels_at_target_voxels(
        source_data=source_data,
        source_affine=source_affine,
        target_coords=target_coords,
        target_affine=target_img.affine,
        fill_value=0,
    )
    return sampled.reshape(target_img.shape[:3]), valid.reshape(target_img.shape[:3])


def parse_tzo_label_map(label_file: Path) -> dict[int, str]:
    label_map: dict[int, str] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        label_map[int(parts[0])] = parts[1]
    return label_map


def build_seed_label_volume(
    tzo_data: np.ndarray,
    label_map: dict[int, str],
) -> tuple[np.ndarray, dict[str, set[int]]]:
    seed_sets = {lobe_name: set() for lobe_name in LOBE_NAMES}

    for idx, name in label_map.items():
        if name == "Background" or name.startswith("Cerebelum_") or name.startswith("Vermis_"):
            continue
        if "Ventricle" in name:
            continue

        for lobe_name, prefixes in LOBE_PREFIXES.items():
            if name.startswith(prefixes):
                seed_sets[lobe_name].add(idx)
                break

    seed_volume = np.zeros(tzo_data.shape, dtype=np.uint8)
    for lobe_name, lobe_id in LOBE_IDS.items():
        seed_volume[np.isin(tzo_data, list(seed_sets[lobe_name]))] = lobe_id

    return seed_volume, seed_sets


def build_filled_lobar_atlas(
    seed_volume: np.ndarray,
    supratentorial_mask: np.ndarray,
    dilation_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = supratentorial_mask > 0
    if dilation_voxels > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_voxels)

    nearest_seed_indices = ndimage.distance_transform_edt(
        seed_volume == 0,
        return_distances=False,
        return_indices=True,
    )

    filled = seed_volume.copy()
    filled[mask] = seed_volume[tuple(index_array[mask] for index_array in nearest_seed_indices)]
    filled[~mask] = 0
    return filled, mask


def load_atlas_context(atlas_dir: Path, dilation_voxels: int) -> AtlasContext:
    tzo_path = atlas_dir / "tzo116plus.nii.gz"
    suptent_path = atlas_dir / "suptent.nii.gz"
    tissues_path = atlas_dir / "tissues.nii.gz"
    label_file = atlas_dir / "SRI24-tzo116plus.txt"

    for path in (tzo_path, suptent_path, tissues_path, label_file):
        if not path.exists():
            raise FileNotFoundError(f"Required atlas file does not exist: {path}")

    tzo_img, tzo_data = load_3d_volume(tzo_path)
    suptent_img, suptent_data = load_3d_volume(suptent_path)
    tissues_img, tissues_data = load_3d_volume(tissues_path)

    if not np.allclose(tzo_img.affine, suptent_img.affine, atol=1e-4):
        raise ValueError("tzo116plus and suptent affines do not match.")
    if not np.allclose(tzo_img.affine, tissues_img.affine, atol=1e-4):
        raise ValueError("tzo116plus and tissues affines do not match.")

    label_map = parse_tzo_label_map(label_file)
    seed_volume, seed_sets = build_seed_label_volume(tzo_data.astype(np.int32), label_map)
    lobar_atlas, dilated_suptent_mask = build_filled_lobar_atlas(
        seed_volume=seed_volume,
        supratentorial_mask=suptent_data.astype(bool),
        dilation_voxels=dilation_voxels,
    )

    return AtlasContext(
        atlas_dir=atlas_dir,
        label_file=label_file,
        tzo_path=tzo_path,
        suptent_path=suptent_path,
        tissues_path=tissues_path,
        source_affine=tzo_img.affine.copy(),
        seed_volume=seed_volume,
        lobar_atlas=lobar_atlas,
        dilated_suptent_mask=dilated_suptent_mask.astype(np.uint8),
        source_brainmask=(tissues_data > 0).astype(np.uint8),
        seed_sets=seed_sets,
        dilation_voxels=dilation_voxels,
    )


def project_atlas_to_target(
    atlas: AtlasContext,
    target_img: nib.spatialimages.SpatialImage,
) -> ProjectedAtlasContext:
    lobar_labels, lobar_valid = sample_source_into_target_grid(
        source_data=atlas.lobar_atlas,
        source_affine=atlas.source_affine,
        target_img=target_img,
    )
    seed_labels, _ = sample_source_into_target_grid(
        source_data=atlas.seed_volume,
        source_affine=atlas.source_affine,
        target_img=target_img,
    )
    suptent_mask, _ = sample_source_into_target_grid(
        source_data=atlas.dilated_suptent_mask,
        source_affine=atlas.source_affine,
        target_img=target_img,
    )
    brainmask, brainmask_valid = sample_source_into_target_grid(
        source_data=atlas.source_brainmask,
        source_affine=atlas.source_affine,
        target_img=target_img,
    )

    return ProjectedAtlasContext(
        atlas=atlas,
        target_shape=tuple(target_img.shape[:3]),
        target_affine=target_img.affine.copy(),
        lobar_labels=lobar_labels.astype(np.uint8),
        seed_labels=seed_labels.astype(np.uint8),
        suptent_mask=suptent_mask.astype(np.uint8),
        brainmask=brainmask.astype(np.uint8),
        lobar_valid=lobar_valid.astype(bool),
        brainmask_valid=brainmask_valid.astype(bool),
    )


def compute_feature_row(
    tumor_data: np.ndarray,
    tumor_seg_path: Path,
    case_id: str,
    projected: ProjectedAtlasContext,
) -> dict[str, object]:
    tumor_mask = tumor_data > 0

    total_tumor_voxels = int(np.count_nonzero(tumor_mask))
    nc_voxels = int(np.count_nonzero(tumor_data == NC_LABEL))
    ed_voxels = int(np.count_nonzero(tumor_data == ED_LABEL))
    en_voxels = int(np.count_nonzero(tumor_data == EN_LABEL))

    brain_voxels = int(np.count_nonzero(projected.brainmask > 0))
    tumor_volume_mm3 = float(total_tumor_voxels)
    brain_volume_mm3 = float(brain_voxels)

    seed_mapped_voxels = int(np.count_nonzero(projected.seed_labels[tumor_mask] > 0))
    mapped_voxels = int(np.count_nonzero(projected.lobar_labels[tumor_mask] > 0))
    suptent_mapped_voxels = int(np.count_nonzero(projected.suptent_mask[tumor_mask] > 0))
    out_of_bounds_voxels = int(np.count_nonzero(~projected.lobar_valid[tumor_mask]))

    per_lobe_counts = {
        lobe_name: {"NC": 0, "ED": 0, "EN": 0, "TOTAL": 0}
        for lobe_name in LOBE_NAMES
    }

    for lobe_name, lobe_id in LOBE_IDS.items():
        lobe_mask = tumor_mask & (projected.lobar_labels == lobe_id)
        lobe_nc = int(np.count_nonzero(lobe_mask & (tumor_data == NC_LABEL)))
        lobe_ed = int(np.count_nonzero(lobe_mask & (tumor_data == ED_LABEL)))
        lobe_en = int(np.count_nonzero(lobe_mask & (tumor_data == EN_LABEL)))
        per_lobe_counts[lobe_name] = {
            "NC": lobe_nc,
            "ED": lobe_ed,
            "EN": lobe_en,
            "TOTAL": lobe_nc + lobe_ed + lobe_en,
        }

    dominant_lobe = ""
    if mapped_voxels > 0:
        dominant_lobe = max(LOBE_NAMES, key=lambda lobe_name: per_lobe_counts[lobe_name]["TOTAL"])
        if per_lobe_counts[dominant_lobe]["TOTAL"] == 0:
            dominant_lobe = ""

    row: dict[str, object] = {
        "case_id": case_id,
        "tumor_seg_path": str(tumor_seg_path),
        "atlas_dir": str(projected.atlas.atlas_dir),
        "atlas_label_file": str(projected.atlas.label_file),
        "atlas_tzo_path": str(projected.atlas.tzo_path),
        "atlas_suptent_path": str(projected.atlas.suptent_path),
        "atlas_tissues_path": str(projected.atlas.tissues_path),
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
        "atlas_seed_mapped_tumor_voxels": seed_mapped_voxels,
        "atlas_seed_mapped_fraction": safe_div(seed_mapped_voxels, total_tumor_voxels),
        "atlas_suptent_mapped_tumor_voxels": suptent_mapped_voxels,
        "atlas_suptent_mapped_fraction": safe_div(suptent_mapped_voxels, total_tumor_voxels),
        "tumor_voxels_in_requested_lobes": mapped_voxels,
        "tumor_voxels_outside_requested_lobes": total_tumor_voxels - mapped_voxels,
        "tumor_requested_lobe_fraction": safe_div(mapped_voxels, total_tumor_voxels),
        "affine_out_of_bounds_voxels": out_of_bounds_voxels,
        "brainmask_affine_out_of_bounds_voxels": int(np.count_nonzero(~projected.brainmask_valid)),
        "atlas_suptent_dilation_voxels": projected.atlas.dilation_voxels,
        "atlas_seed_voxels": int(np.count_nonzero(projected.seed_labels)),
        "atlas_filled_lobe_voxels": int(np.count_nonzero(projected.lobar_labels)),
        "atlas_dilated_suptent_voxels": int(np.count_nonzero(projected.suptent_mask)),
        "frontal_seed_label_count": len(projected.atlas.seed_sets["frontal"]),
        "temporal_seed_label_count": len(projected.atlas.seed_sets["temporal"]),
        "parietal_seed_label_count": len(projected.atlas.seed_sets["parietal"]),
        "occipital_seed_label_count": len(projected.atlas.seed_sets["occipital"]),
        "lobe_assignment_reliable": int(safe_div(mapped_voxels, total_tumor_voxels) >= 0.9),
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


def extract_features(
    tumor_seg_path: Path,
    out_csv: Path,
    atlas_dir: Path,
    case_id: str,
    dilation_voxels: int,
) -> dict[str, object]:
    if not tumor_seg_path.exists():
        raise FileNotFoundError(f"Required file does not exist: {tumor_seg_path}")

    atlas = load_atlas_context(atlas_dir, dilation_voxels)
    tumor_img, tumor_data = load_3d_volume(tumor_seg_path)
    projected = project_atlas_to_target(atlas, tumor_img)
    row = compute_feature_row(
        tumor_data=tumor_data.astype(np.uint8),
        tumor_seg_path=tumor_seg_path,
        case_id=case_id,
        projected=projected,
    )
    write_csv(row, out_csv)
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
    row = extract_features(
        tumor_seg_path=args.tumor_seg,
        out_csv=args.out_csv,
        atlas_dir=args.atlas_dir,
        case_id=case_id,
        dilation_voxels=args.dilation_voxels,
    )

    print(f"Wrote features to {args.out_csv}")
    print(f"dominant_brain_lobe={row['dominant_brain_lobe']}")
    print(f"tumor_requested_lobe_fraction={row['tumor_requested_lobe_fraction']}")
    print(f"lobe_assignment_reliable={row['lobe_assignment_reliable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
