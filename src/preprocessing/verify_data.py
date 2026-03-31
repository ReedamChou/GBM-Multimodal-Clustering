"""
Dataset verification — scan raw UCSF-PDGM & UPENN-GBM directories, check
completeness per patient, and produce a manifest CSV.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()

MODALITIES = ["T1", "T1GD", "T2", "FLAIR"]


# ── UCSF file finders ──────────────────────────────────────────────────────
def _discover_ucsf_patients(cfg: dict) -> pd.DataFrame:
    """Walk UCSF structural directory and build per-patient file manifest."""
    struct_dir = resolve_path(cfg["paths"]["ucsf"]["structural_dir"])
    seg_dir = resolve_path(cfg["paths"]["ucsf"]["segmentation_dir"])

    records: List[Dict] = []
    for patient_dir in sorted(struct_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name  # e.g. UCSF-PDGM-0004

        # Skip follow-up folders (contain _FU in name)
        if "_FU" in pid:
            continue

        rec: Dict = {"patient_id": pid, "cohort": "UCSF"}
        for mod in MODALITIES:
            nii = patient_dir / f"{pid}_{mod}.nii.gz"
            rec[f"{mod}_exists"] = nii.is_file()
            rec[f"{mod}_path"] = str(nii) if nii.is_file() else ""

        # Segmentation mask (flat directory)
        seg_file = seg_dir / f"{pid}_tumor_segmentation.nii.gz"
        rec["seg_exists"] = seg_file.is_file()
        rec["seg_path"] = str(seg_file) if seg_file.is_file() else ""

        records.append(rec)

    return pd.DataFrame(records)


# ── UPenn file finders ─────────────────────────────────────────────────────
def _discover_upenn_patients(cfg: dict) -> pd.DataFrame:
    """Walk UPenn structural directory; use only baseline (_11) scans."""
    struct_dir = resolve_path(cfg["paths"]["upenn"]["structural_dir"])
    seg_dir = resolve_path(cfg["paths"]["upenn"]["segmentation_dir"])

    records: List[Dict] = []
    for patient_dir in sorted(struct_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        folder_name = patient_dir.name  # e.g. UPENN-GBM-00001_11

        # Only baseline scans (suffix _11)
        if not folder_name.endswith("_11"):
            continue

        # Extract core patient ID (without _11 suffix)
        core_id = folder_name.replace("_11", "")  # UPENN-GBM-00001

        rec: Dict = {"patient_id": core_id, "cohort": "UPenn"}
        for mod in MODALITIES:
            nii = patient_dir / f"{folder_name}_{mod}.nii.gz"
            rec[f"{mod}_exists"] = nii.is_file()
            rec[f"{mod}_path"] = str(nii) if nii.is_file() else ""

        seg_file = seg_dir / f"{folder_name}_automated_approx_segm.nii.gz"
        rec["seg_exists"] = seg_file.is_file()
        rec["seg_path"] = str(seg_file) if seg_file.is_file() else ""

        records.append(rec)

    return pd.DataFrame(records)


# ── Clinical CSV match ─────────────────────────────────────────────────────
def _match_clinical(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add a flag indicating whether a clinical record exists for each patient."""
    # UCSF
    ucsf_csv = pd.read_csv(resolve_path(cfg["paths"]["ucsf"]["clinical_csv"]))
    ucsf_ids = set(ucsf_csv.iloc[:, 0].astype(str).str.strip())

    # UPenn
    upenn_csv = pd.read_csv(resolve_path(cfg["paths"]["upenn"]["clinical_csv"]))
    upenn_ids = set(upenn_csv.iloc[:, 0].astype(str).str.strip())

    clinical_ids = ucsf_ids | upenn_ids
    manifest["clinical_exists"] = manifest["patient_id"].isin(clinical_ids)
    return manifest


# ── Public API ──────────────────────────────────────────────────────────────
def verify_dataset(cfg: dict | None = None, save: bool = True) -> pd.DataFrame:
    """
    Run full verification.  Returns a manifest DataFrame and optionally saves
    it to ``data/manifest.csv``.
    """
    if cfg is None:
        cfg = load_config()

    logger.info("Scanning UCSF-PDGM …")
    ucsf_df = _discover_ucsf_patients(cfg)
    logger.info("  Found %d UCSF baseline patients", len(ucsf_df))

    logger.info("Scanning UPENN-GBM …")
    upenn_df = _discover_upenn_patients(cfg)
    logger.info("  Found %d UPenn baseline patients", len(upenn_df))

    manifest = pd.concat([ucsf_df, upenn_df], ignore_index=True)
    manifest = _match_clinical(manifest, cfg)

    # Summary statistics
    for mod in MODALITIES:
        n = manifest[f"{mod}_exists"].sum()
        logger.info("  %s present: %d / %d", mod, n, len(manifest))
    n_seg = manifest["seg_exists"].sum()
    logger.info("  Segmentation present: %d / %d", n_seg, len(manifest))
    n_clin = manifest["clinical_exists"].sum()
    logger.info("  Clinical match: %d / %d", n_clin, len(manifest))

    # Flag complete patients (all 4 modalities + seg + clinical)
    complete_cols = [f"{m}_exists" for m in MODALITIES] + [
        "seg_exists",
        "clinical_exists",
    ]
    manifest["is_complete"] = manifest[complete_cols].all(axis=1)
    n_complete = manifest["is_complete"].sum()
    logger.info("  Fully complete patients: %d / %d", n_complete, len(manifest))

    if save:
        out = resolve_path(cfg["paths"]["output"]["data_dir"]) / "manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out, index=False)
        logger.info("  Manifest saved → %s", out)

    return manifest


# ── CLI entry ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verify_dataset()
