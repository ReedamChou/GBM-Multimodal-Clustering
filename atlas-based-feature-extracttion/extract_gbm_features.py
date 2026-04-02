#!/usr/bin/env python3
"""Extract lobe-aware tumor features from the mounted UPenn/UCSF MRI dataset.

Assumptions used here:
- Structural MRIs and segmentations are already aligned in a common BraTS-style
  240x240x155, 1 mm isotropic space.
- Tumor segmentation labels follow the common BraTS convention:
  1 = necrotic/non-enhancing tumor core (NC)
  2 = peritumoral edema (ED)
  4 = enhancing tumor (EN)
- Per-lobe ED/EN/NC ratios are defined as fractions within the total tumor
  volume of that lobe:
  ED_ratio = ED_lobe / tumor_lobe
  EN_ratio = EN_lobe / tumor_lobe
  NC_ratio = NC_lobe / tumor_lobe
- Tumor burden index is defined as total tumor voxels divided by brain voxels.
  If no explicit brain mask is provided, the union of the four lobe masks is
  used as the denominator.

The script expects one binary NIfTI mask per lobe, already aligned to the same
image space as the segmentations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np


COHORTS = {
    "UPenn": {
        "struct_dir": "UPenn/DATA-IMAGE-STRUCTURAL",
        "seg_dir": "UPenn/DATA-AUTOMATED-SEGMENT",
        "seg_suffix": "_automated_approx_segm.nii.gz",
    },
    "UCSF": {
        "struct_dir": "UCSF/DATA-IMAGE-STRUCTURAL",
        "seg_dir": "UCSF/DATA-AUTOMATED-SEGMENT",
        "seg_suffix": "_tumor_segmentation.nii.gz",
    },
}

LOBES = ("frontal", "temporal", "parietal", "occipital")
LABELS = {"NC": 1, "ED": 2, "EN": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Dataset root that contains the UPenn and UCSF folders.",
    )
    parser.add_argument("--frontal-mask", type=Path, required=True)
    parser.add_argument("--temporal-mask", type=Path, required=True)
    parser.add_argument("--parietal-mask", type=Path, required=True)
    parser.add_argument("--occipital-mask", type=Path, required=True)
    parser.add_argument(
        "--brain-mask",
        type=Path,
        default=None,
        help="Optional whole-brain mask aligned to the same space.",
    )
    parser.add_argument(
        "--include-missing-seg",
        action="store_true",
        help="Include rows with missing segmentation files instead of skipping them.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Where to write the extracted feature table.",
    )
    return parser.parse_args()


def safe_div(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def load_nifti(path: Path) -> Tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img


def load_binary_mask(path: Path) -> Tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    data, img = load_nifti(path)
    return data > 0, img


def assert_same_space(
    ref_shape: Tuple[int, ...],
    ref_affine: np.ndarray,
    ref_name: str,
    cur_shape: Tuple[int, ...],
    cur_affine: np.ndarray,
    cur_name: str,
) -> None:
    if ref_shape != cur_shape:
        raise ValueError(f"Shape mismatch: {ref_name} {ref_shape} vs {cur_name} {cur_shape}")
    if not np.allclose(ref_affine, cur_affine, atol=1e-4):
        raise ValueError(f"Affine mismatch between {ref_name} and {cur_name}")


def build_case_records(base_dir: Path) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for cohort, spec in COHORTS.items():
        struct_dir = base_dir / spec["struct_dir"]
        seg_dir = base_dir / spec["seg_dir"]
        if not struct_dir.exists():
            continue
        for case_dir in sorted(struct_dir.iterdir()):
            if not case_dir.is_dir() or case_dir.name.startswith("."):
                continue
            case_id = case_dir.name
            seg_path = seg_dir / f"{case_id}{spec['seg_suffix']}"
            cases.append(
                {
                    "cohort": cohort,
                    "case_id": case_id,
                    "struct_dir": case_dir,
                    "seg_path": seg_path,
                }
            )
    return cases


def extract_case_features(
    case: Dict[str, object],
    lobe_masks: Dict[str, np.ndarray],
    ref_affine: np.ndarray,
    brain_mask: np.ndarray,
) -> Dict[str, object]:
    seg_path = Path(case["seg_path"])
    seg_data, seg_img = load_nifti(seg_path)
    seg = seg_data.astype(np.uint8, copy=False)
    assert_same_space(
        ref_shape=brain_mask.shape,
        ref_affine=ref_affine,
        ref_name="lobe masks",
        cur_shape=seg.shape,
        cur_affine=seg_img.affine,
        cur_name=str(seg_path),
    )

    nc = seg == LABELS["NC"]
    ed = seg == LABELS["ED"]
    en = seg == LABELS["EN"]
    tumor = seg > 0

    global_nc = int(nc.sum())
    global_ed = int(ed.sum())
    global_en = int(en.sum())
    global_total = int(tumor.sum())
    brain_voxels = int(brain_mask.sum())

    row: Dict[str, object] = {
        "cohort": case["cohort"],
        "case_id": case["case_id"],
        "segmentation_file": str(seg_path),
        "dominant_brain_lobe": "",
        "global_nc_en_ratio": safe_div(global_nc, global_en),
        "global_ed_en_ratio": safe_div(global_ed, global_en),
        "global_ed_total_ratio": safe_div(global_ed, global_total),
        "tumor_burden_index": safe_div(global_total, brain_voxels),
        "global_nc_voxels": global_nc,
        "global_ed_voxels": global_ed,
        "global_en_voxels": global_en,
        "global_total_tumor_voxels": global_total,
        "brain_voxels": brain_voxels,
    }

    lobe_totals: Dict[str, int] = {}
    for lobe, mask in lobe_masks.items():
        lobe_nc = int((nc & mask).sum())
        lobe_ed = int((ed & mask).sum())
        lobe_en = int((en & mask).sum())
        lobe_total = lobe_nc + lobe_ed + lobe_en
        lobe_totals[lobe] = lobe_total

        row[f"{lobe}_nc_ratio"] = safe_div(lobe_nc, lobe_total)
        row[f"{lobe}_ed_ratio"] = safe_div(lobe_ed, lobe_total)
        row[f"{lobe}_en_ratio"] = safe_div(lobe_en, lobe_total)
        row[f"{lobe}_nc_voxels"] = lobe_nc
        row[f"{lobe}_ed_voxels"] = lobe_ed
        row[f"{lobe}_en_voxels"] = lobe_en
        row[f"{lobe}_tumor_voxels"] = lobe_total

    dominant_total = max(lobe_totals.values()) if lobe_totals else 0
    if dominant_total > 0:
        row["dominant_brain_lobe"] = max(lobe_totals, key=lobe_totals.get)

    return row


def load_lobe_masks(
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mask_paths = {
        "frontal": args.frontal_mask,
        "temporal": args.temporal_mask,
        "parietal": args.parietal_mask,
        "occipital": args.occipital_mask,
    }

    lobe_masks: Dict[str, np.ndarray] = {}
    ref_img = None
    for lobe in LOBES:
        mask, img = load_binary_mask(mask_paths[lobe])
        if ref_img is None:
            ref_img = img
        else:
            assert_same_space(
                ref_shape=ref_img.shape,
                ref_affine=ref_img.affine,
                ref_name="first lobe mask",
                cur_shape=img.shape,
                cur_affine=img.affine,
                cur_name=f"{lobe} mask",
            )
        lobe_masks[lobe] = mask

    if ref_img is None:
        raise RuntimeError("No lobe masks were loaded.")

    if args.brain_mask is not None:
        brain_mask, brain_img = load_binary_mask(args.brain_mask)
        assert_same_space(
            ref_shape=ref_img.shape,
            ref_affine=ref_img.affine,
            ref_name="lobe masks",
            cur_shape=brain_img.shape,
            cur_affine=brain_img.affine,
            cur_name="brain mask",
        )
    else:
        brain_mask = np.logical_or.reduce([lobe_masks[lobe] for lobe in LOBES])

    return lobe_masks, ref_img.affine, brain_mask


def write_csv(rows: Iterable[Dict[str, object]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        raise RuntimeError("No rows were generated.")

    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    lobe_masks, ref_affine, brain_mask = load_lobe_masks(args)
    cases = build_case_records(args.base_dir)

    rows: List[Dict[str, object]] = []
    missing = 0
    for case in cases:
        seg_path = Path(case["seg_path"])
        if not seg_path.exists():
            missing += 1
            if args.include_missing_seg:
                rows.append(
                    {
                        "cohort": case["cohort"],
                        "case_id": case["case_id"],
                        "segmentation_file": "",
                        "dominant_brain_lobe": "",
                        "global_nc_en_ratio": float("nan"),
                        "global_ed_en_ratio": float("nan"),
                        "global_ed_total_ratio": float("nan"),
                        "tumor_burden_index": float("nan"),
                        "global_nc_voxels": 0,
                        "global_ed_voxels": 0,
                        "global_en_voxels": 0,
                        "global_total_tumor_voxels": 0,
                        "brain_voxels": int(brain_mask.sum()),
                        **{
                            f"{lobe}_{metric}": (0 if metric.endswith("voxels") else float("nan"))
                            for lobe in LOBES
                            for metric in (
                                "nc_ratio",
                                "ed_ratio",
                                "en_ratio",
                                "nc_voxels",
                                "ed_voxels",
                                "en_voxels",
                                "tumor_voxels",
                            )
                        },
                    }
                )
            continue
        rows.append(extract_case_features(case, lobe_masks, ref_affine, brain_mask))

    write_csv(rows, args.out_csv)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    if missing:
        print(f"Skipped or flagged {missing} cases without segmentation files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
