"""
Atlas registration — warp each patient's T1 to MNI152 space using ANTsPy,
then apply the same transform to the segmentation mask.  Produces a per-
patient brain-lobe label map for downstream regional feature extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import ants
import nibabel as nib
import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, load_config, resolve_path, setup_logging

logger = setup_logging()


# ── Atlas utilities ─────────────────────────────────────────────────────────
def _get_mni_template() -> ants.ANTsImage:
    """Load the MNI152 template bundled with ANTsPy."""
    return ants.image_read(ants.get_ants_data("mni"))


def _load_nifti_as_ants(path: str | Path) -> ants.ANTsImage:
    """Load a NIfTI file as an ANTsImage."""
    return ants.image_read(str(path))


# ── Core registration ──────────────────────────────────────────────────────
def register_to_mni(
    t1_path: str | Path,
    seg_path: str | Path,
    output_dir: str | Path,
    patient_id: str,
    reg_type: str = "SyN",
) -> Dict[str, Path]:
    """
    Register a patient's T1 image to MNI152 space and warp the segmentation
    mask with the same transform.

    Returns dict with paths to:
        - ``t1_mni``  : T1 in MNI space
        - ``seg_mni`` : segmentation in MNI space (nearest-neighbour)
        - ``lobe_map``: brain-lobe label volume extracted from the atlas
    """
    output_dir = ensure_dir(output_dir)
    template = _get_mni_template()
    moving = _load_nifti_as_ants(t1_path)

    logger.info("  Registering %s → MNI152 (%s) …", patient_id, reg_type)
    reg = ants.registration(
        fixed=template,
        moving=moving,
        type_of_transform=reg_type,
    )

    # Save warped T1
    t1_mni_path = output_dir / f"{patient_id}_T1_MNI.nii.gz"
    ants.image_write(reg["warpedmovout"], str(t1_mni_path))

    # Warp segmentation mask (nearest-neighbour to preserve labels)
    seg_moving = _load_nifti_as_ants(seg_path)
    seg_warped = ants.apply_transforms(
        fixed=template,
        moving=seg_moving,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    seg_mni_path = output_dir / f"{patient_id}_seg_MNI.nii.gz"
    ants.image_write(seg_warped, str(seg_mni_path))

    # Brain-lobe labelling via atlas parcellation
    lobe_path = _extract_lobe_labels(template, seg_warped, output_dir, patient_id)

    return {
        "t1_mni": t1_mni_path,
        "seg_mni": seg_mni_path,
        "lobe_map": lobe_path,
    }


# ── Lobe labelling ─────────────────────────────────────────────────────────
_LOBE_NAMES = {
    1: "Frontal",
    2: "Parietal",
    3: "Temporal",
    4: "Occipital",
    5: "Cerebellum",
    6: "Brainstem",
    7: "Insula",
    8: "Other",
}


def _extract_lobe_labels(
    template: ants.ANTsImage,
    seg_mni: ants.ANTsImage,
    output_dir: Path,
    patient_id: str,
) -> Path:
    """
    Use ANTsPy's built-in DKT parcellation to assign each tumour voxel a
    brain-lobe label.  Save a NIfTI volume of the lobe assignment and return
    the path.
    """
    # DKT cortical labelling atlas (Desikan-Killiany-Tourville)
    dkt = ants.desikan_killiany_tourville_labeling(template)
    dkt_arr = dkt.numpy()
    seg_arr = seg_mni.numpy()

    # Map DKT labels → coarse lobe labels
    lobe_arr = np.zeros_like(seg_arr, dtype=np.int16)
    tumour_mask = seg_arr > 0

    # DKT label ranges (approximate grouping)
    frontal = set(range(1002, 1029)) | set(range(2002, 2029))
    parietal = set(range(1029, 1036)) | set(range(2029, 2036))
    temporal = set(range(1006, 1016)) | set(range(2006, 2016))
    occipital = set(range(1001, 1006)) | set(range(2001, 2006))

    for vox in zip(*np.where(tumour_mask)):
        dkt_label = int(dkt_arr[vox])
        if dkt_label in frontal:
            lobe_arr[vox] = 1
        elif dkt_label in parietal:
            lobe_arr[vox] = 2
        elif dkt_label in temporal:
            lobe_arr[vox] = 3
        elif dkt_label in occipital:
            lobe_arr[vox] = 4
        else:
            lobe_arr[vox] = 8  # Other / subcortical

    lobe_img = seg_mni.new_image_like(lobe_arr.astype(np.float32))
    lobe_path = output_dir / f"{patient_id}_lobe_labels.nii.gz"
    ants.image_write(lobe_img, str(lobe_path))
    return lobe_path


# ── Batch runner ────────────────────────────────────────────────────────────
def register_cohort(
    manifest: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """
    Register all *complete* patients in the manifest.  Appends columns
    ``t1_mni_path``, ``seg_mni_path``, ``lobe_map_path`` to the manifest.
    """
    if cfg is None:
        cfg = load_config()

    reg_type = cfg["preprocessing"]["registration_type"]
    output_base = ensure_dir(resolve_path(cfg["paths"]["output"]["data_dir"]) / "registered")

    complete = manifest[manifest["is_complete"]].copy()
    logger.info("Registering %d complete patients …", len(complete))

    results = []
    for _, row in complete.iterrows():
        pid = row["patient_id"]
        t1_path = row["T1_path"]
        seg_path = row["seg_path"]
        out_dir = ensure_dir(output_base / pid)
        try:
            paths = register_to_mni(t1_path, seg_path, out_dir, pid, reg_type)
            results.append(
                {
                    "patient_id": pid,
                    "t1_mni_path": str(paths["t1_mni"]),
                    "seg_mni_path": str(paths["seg_mni"]),
                    "lobe_map_path": str(paths["lobe_map"]),
                }
            )
        except Exception as exc:
            logger.warning("  FAILED %s: %s", pid, exc)
            results.append(
                {
                    "patient_id": pid,
                    "t1_mni_path": "",
                    "seg_mni_path": "",
                    "lobe_map_path": "",
                }
            )

    reg_df = pd.DataFrame(results)
    manifest = manifest.merge(reg_df, on="patient_id", how="left")
    return manifest


def compute_lobe_volumes(lobe_map_path: str | Path) -> Dict[str, float]:
    """
    Given a lobe-label NIfTI, compute the fraction of tumour voxels in each
    lobe.  Returns a dict keyed by lobe name.
    """
    img = nib.load(str(lobe_map_path))
    data = np.asarray(img.dataobj)
    total = (data > 0).sum()
    if total == 0:
        return {name: 0.0 for name in _LOBE_NAMES.values()}
    fracs = {}
    for label, name in _LOBE_NAMES.items():
        fracs[f"frac_{name}"] = float((data == label).sum()) / total
    return fracs
