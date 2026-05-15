#!/usr/bin/env python3
"""SRI24 atlas registration and 16-feature extraction for GBM survival
risk stratification.

Builds a 4-lobe atlas from SRI24 parcellation, registers it to each
patient's native T1 space via ANTs (or affine fallback), then extracts
exactly 16 radiomic features per patient.

Usage:
    python src/atlas_registration.py                            # full batch
    python src/atlas_registration.py --dry-run                  # list patients
    python src/atlas_registration.py --patient UCSF-PDGM-0004  # single patient
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# ── Constants ─────────────────────────────────────────────────────────────
NC_LABEL, ED_LABEL, EN_LABEL = 1, 2, 4
LOBE_IDS = {"frontal": 1, "temporal": 2, "parietal": 3, "occipital": 4}
LOBE_NAMES = list(LOBE_IDS.keys())

LOBE_PREFIXES = {
    "frontal": ("Frontal_", "Precentral_", "Rolandic_Oper_",
                "Supp_Motor_Area_", "Olfactory_", "Rectus_"),
    "temporal": ("Temporal_", "Heschl_", "ParaHippocampal_",
                 "Hippocampus_", "Amygdala_", "Fusiform_"),
    "parietal": ("Postcentral_", "Parietal_", "SupraMarginal_",
                 "Angular_", "Precuneus_", "Paracentral_Lobule_"),
    "occipital": ("Calcarine_", "Cuneus_", "Lingual_", "Occipital_"),
}

FEATURE_COLS = [
    "global_nc_en_ratio", "global_ed_en_ratio",
    "global_ed_total_ratio", "tumor_burden_index",
    *(f"{lb}_{sub}_ratio" for lb in LOBE_NAMES for sub in ("ed", "en", "nc")),
]
assert len(FEATURE_COLS) == 16

OUTPUT_COLS = ["patient_id", *FEATURE_COLS, "OS_months", "lobe_assignment_reliable"]


# ── Helpers ───────────────────────────────────────────────────────────────
def _sdiv(num: float, den: float) -> float:
    """Safe division: 0.0 on zero denominator."""
    return 0.0 if den == 0 else float(num) / float(den)


def _load_vol(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load NIfTI, squeeze trailing 4th dim if present."""
    img = nib.load(str(path))
    d = np.asarray(img.dataobj)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 3:
        raise ValueError(f"Expected 3-D at {path}, got {d.shape}")
    return img, d


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


# ── Section A: Build 4-lobe atlas (run once, cached) ─────────────────────
def _parse_tzo_labels(label_file: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        out[int(parts[0])] = parts[1]
    return out


def _build_seeds(tzo: np.ndarray, label_map: dict[int, str]) -> np.ndarray:
    seeds = np.zeros(tzo.shape, dtype=np.uint8)
    for idx, name in label_map.items():
        if name == "Background" or "Ventricle" in name:
            continue
        if name.startswith(("Cerebelum_", "Vermis_")):
            continue
        for lobe, prefixes in LOBE_PREFIXES.items():
            if name.startswith(prefixes):
                seeds[tzo == idx] = LOBE_IDS[lobe]
                break
    return seeds


def build_4lobe_atlas(sri24_dir: Path, dilation: int,
                      out_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Build or load the filled 4-lobe atlas in SRI24 space."""
    if out_path.exists():
        print(f"  Cached atlas: {out_path}")
        return _load_vol(out_path)

    tzo_path = sri24_dir / "tzo116plus.nii.gz"
    sup_path = sri24_dir / "suptent.nii.gz"
    lbl_path = sri24_dir / "SRI24-tzo116plus.txt"
    for p in (tzo_path, sup_path, lbl_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing SRI24 file: {p}")

    tzo_img, tzo_data = _load_vol(tzo_path)
    _, sup_data = _load_vol(sup_path)
    seeds = _build_seeds(tzo_data.astype(np.int32),
                         _parse_tzo_labels(lbl_path))

    mask = sup_data > 0
    if dilation > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation)

    nn_idx = ndimage.distance_transform_edt(
        seeds == 0, return_distances=False, return_indices=True)
    filled = seeds.copy()
    filled[mask] = seeds[tuple(ix[mask] for ix in nn_idx)]
    filled[~mask] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atlas_img = nib.Nifti1Image(filled, tzo_img.affine, tzo_img.header)
    nib.save(atlas_img, str(out_path))
    print(f"  Built 4-lobe atlas -> {out_path}")
    return atlas_img, filled


# ── Section B: Register atlas -> patient T1 ──────────────────────────────
def _affine_resample_nn(src_img: nib.Nifti1Image,
                        tgt_img: nib.Nifti1Image) -> np.ndarray:
    """Nearest-neighbor resample via NIfTI affines (fallback path)."""
    tgt_shape = tgt_img.shape[:3]
    coords = np.indices(tgt_shape, dtype=np.int32).reshape(3, -1).T
    world = nib.affines.apply_affine(tgt_img.affine, coords)
    src_ijk = nib.affines.apply_affine(
        np.linalg.inv(src_img.affine), world)
    si = np.rint(src_ijk).astype(np.int32)
    ss = np.array(src_img.shape[:3])
    valid = np.all((si >= 0) & (si < ss), axis=1)
    sd = np.asarray(src_img.dataobj)
    if sd.ndim == 4:
        sd = sd[..., 0]
    out = np.zeros(coords.shape[0], dtype=sd.dtype)
    if valid.any():
        out[valid] = sd[tuple(si[valid].T)]
    return out.reshape(tgt_shape)


def _ants_register(patient_t1: Path, sri24_t1: Path,
                   atlas_img: nib.Nifti1Image,
                   brainmask_img: nib.Nifti1Image,
                   tx_dir: Path, reg_type: str,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """ANTs registration: SRI24 -> patient T1, apply to atlas + brainmask."""
    import ants  # type: ignore

    fixed = ants.image_read(str(patient_t1))
    moving = ants.image_read(str(sri24_t1))
    tx_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached transform
    cached_tx = tx_dir / "0GenericAffine.mat"
    if cached_tx.exists():
        tx_list = [str(cached_tx)]
    else:
        reg = ants.registration(fixed=fixed, moving=moving,
                                type_of_transform=reg_type)
        # Cache the forward transforms
        for src in reg["fwdtransforms"]:
            dst = tx_dir / Path(src).name
            if not dst.exists():
                import shutil
                shutil.copy2(src, dst)
        tx_list = [str(tx_dir / Path(t).name) for t in reg["fwdtransforms"]]

    # Apply to atlas (label volume -> nearest neighbor)
    a_ants = ants.from_nibabel(atlas_img)
    reg_atlas = ants.apply_transforms(
        fixed=fixed, moving=a_ants,
        transformlist=tx_list, interpolator="nearestNeighbor")

    # Apply to brainmask
    bm_ants = ants.from_nibabel(brainmask_img)
    reg_bm = ants.apply_transforms(
        fixed=fixed, moving=bm_ants,
        transformlist=tx_list, interpolator="nearestNeighbor")

    return reg_atlas.numpy(), reg_bm.numpy()


def register_atlas(patient_t1: Path, seg_img: nib.Nifti1Image,
                   atlas_img: nib.Nifti1Image,
                   brainmask_img: nib.Nifti1Image,
                   cfg: dict, patient_id: str,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (registered_atlas, registered_brainmask) in patient space."""
    if cfg["atlas"].get("use_ants_registration", True):
        sri24_dir = Path(cfg["data"]["sri24_dir"])
        sri24_t1 = sri24_dir / cfg["data"].get("sri24_t1_filename", "T1.nii.gz")
        if not sri24_t1.exists():
            raise FileNotFoundError(
                f"SRI24 T1 template not found: {sri24_t1}. "
                "Check sri24_t1_filename in config.json.")
        tx_dir = Path(cfg["atlas"]["transforms_cache_dir"]) / patient_id
        return _ants_register(patient_t1, sri24_t1,
                              atlas_img, brainmask_img,
                              tx_dir, cfg["atlas"]["registration_type"])
    else:
        print(f"    [WARN] Affine fallback (no ANTs registration)")
        ra = _affine_resample_nn(atlas_img, seg_img)
        rb = _affine_resample_nn(brainmask_img, seg_img)
        return ra, rb


# ── Section C: Extract 16 features ───────────────────────────────────────
def extract_features(seg_data: np.ndarray, atlas_data: np.ndarray,
                     brainmask: np.ndarray, patient_id: str,
                     os_months: float) -> dict:
    """Compute exactly 16 features + metadata for one patient."""
    seg = seg_data.astype(np.uint8)
    nc_total = int(np.sum(seg == NC_LABEL))
    ed_total = int(np.sum(seg == ED_LABEL))
    en_total = int(np.sum(seg == EN_LABEL))
    wt_total = nc_total + ed_total + en_total
    brain_vox = int(np.sum(brainmask > 0))

    row: dict = {"patient_id": patient_id}

    # Global features (4)
    row["global_nc_en_ratio"] = _sdiv(nc_total, en_total)
    row["global_ed_en_ratio"] = _sdiv(ed_total, en_total)
    row["global_ed_total_ratio"] = _sdiv(ed_total, wt_total)
    row["tumor_burden_index"] = _sdiv(wt_total, brain_vox)

    # Lobe-wise features (12): denominator = total lobe voxels (Option B)
    tumor_mask = seg > 0
    mapped_vox = 0
    for lobe_name, lobe_id in LOBE_IDS.items():
        lobe_mask = atlas_data == lobe_id
        lobe_total = int(np.sum(lobe_mask))
        nc_in = int(np.sum((seg == NC_LABEL) & lobe_mask))
        ed_in = int(np.sum((seg == ED_LABEL) & lobe_mask))
        en_in = int(np.sum((seg == EN_LABEL) & lobe_mask))
        mapped_vox += int(np.sum(tumor_mask & lobe_mask))

        row[f"{lobe_name}_ed_ratio"] = _sdiv(ed_in, lobe_total)
        row[f"{lobe_name}_en_ratio"] = _sdiv(en_in, lobe_total)
        row[f"{lobe_name}_nc_ratio"] = _sdiv(nc_in, lobe_total)

    # QA column
    reliable = _sdiv(mapped_vox, wt_total) >= 0.90 if wt_total > 0 else False
    row["OS_months"] = os_months
    row["lobe_assignment_reliable"] = reliable
    return row


# ── Section D: Discovery + batch runner ───────────────────────────────────
def discover_patients(cfg: dict, root: Path) -> list[dict]:
    """Find patients with both T1 and segmentation files."""
    struct_dir = root / cfg["data"]["ucsf_root"] / cfg["data"]["structural_subdir"]
    seg_dir = root / cfg["data"]["ucsf_root"] / cfg["data"]["segmentation_subdir"]
    t1_sfx = cfg["data"]["t1_suffix"]
    seg_sfx = cfg["data"]["seg_suffix"]

    patients = []
    for d in sorted(struct_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        pid = d.name
        t1 = d / f"{pid}{t1_sfx}"
        seg = seg_dir / f"{pid}{seg_sfx}"
        if not t1.exists() or not seg.exists():
            continue
        patients.append({"id": pid, "t1": t1, "seg": seg})
    return patients


def load_clinical_os(csv_path: Path, days_per_month: float) -> dict[str, float]:
    """Read clinical CSV → {patient_id: OS_months}."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Support both 'ID' and 'patient_id' column names
    id_col = "ID" if "ID" in df.columns else "patient_id"
    assert id_col in df.columns, (
        f"Expected 'ID' or 'patient_id' column in {csv_path}, "
        f"got: {list(df.columns)}")
    assert "OS" in df.columns, f"Expected 'OS' column in {csv_path}"
    os_map: dict[str, float] = {}
    for _, r in df.iterrows():
        pid = str(r[id_col]).strip()
        os_val = r["OS"]
        if pd.isna(os_val):
            os_map[pid] = float("nan")
        else:
            os_map[pid] = float(os_val) / days_per_month
    return os_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--patient", type=str, default=None,
                        help="Run for a single patient ID only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List discovered patients without processing.")
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = _load_config(args.config.resolve())

    # Load clinical OS
    csv_path = root / cfg["data"]["clinical_csv"]
    days_pm = cfg["preprocessing"]["os_days_per_month"]
    os_map = load_clinical_os(csv_path, days_pm)

    # Discover patients
    patients = discover_patients(cfg, root)
    if args.patient:
        patients = [p for p in patients if p["id"] == args.patient]
        if not patients:
            print(f"ERROR: patient '{args.patient}' not found.")
            return 1

    print(f"Found {len(patients)} patients with T1 + segmentation files.")

    if args.dry_run:
        for p in patients:
            os_m = os_map.get(p["id"], float("nan"))
            in_csv = "yes" if p["id"] in os_map else "NO"
            print(f"  {p['id']}  OS={os_m:.1f}mo  csv={in_csv}")
        print("Dry run complete. No processing performed.")
        return 0

    # Validate patient IDs match CSV
    missing = [p["id"] for p in patients if p["id"] not in os_map]
    if missing:
        print(f"WARNING: {len(missing)} patients have no CSV match: "
              f"{missing[:5]}...")

    # Build 4-lobe atlas
    sri24_dir = root / cfg["data"]["sri24_dir"]
    atlas_out = root / "outputs" / "sri24_4lobe_atlas.nii.gz"
    atlas_img, atlas_data = build_4lobe_atlas(
        sri24_dir, cfg["atlas"]["lobe_dilation_voxels"], atlas_out)

    # Load brainmask
    tissues_path = sri24_dir / "tissues.nii.gz"
    if not tissues_path.exists():
        raise FileNotFoundError(f"Missing: {tissues_path}")
    bm_img, bm_data = _load_vol(tissues_path)
    bm_img = nib.Nifti1Image((bm_data > 0).astype(np.uint8),
                              bm_img.affine, bm_img.header)

    # Process patients
    rows: list[dict] = []
    for i, p in enumerate(patients, 1):
        pid = p["id"]
        os_m = os_map.get(pid, float("nan"))
        print(f"[{i}/{len(patients)}] {pid} ...", end=" ", flush=True)

        try:
            seg_img, seg_data = _load_vol(p["seg"])
            reg_atlas, reg_bm = register_atlas(
                p["t1"], seg_img, atlas_img, bm_img, cfg, pid)
            row = extract_features(seg_data, reg_atlas, reg_bm, pid, os_m)
            rows.append(row)
            rel = "OK" if row["lobe_assignment_reliable"] else "UNRELIABLE"
            print(f"done ({rel})")
        except Exception as e:
            print(f"FAILED: {e}")

    # Write output CSV
    out_csv = root / "outputs" / "features_raw.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
