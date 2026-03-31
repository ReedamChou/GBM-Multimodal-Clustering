"""
Radiomics feature extraction — PyRadiomics on each MRI modality × tumour
sub-region combination, plus whole-tumour shape features.

Produces a wide DataFrame: one row per patient, columns =
  {modality}_{region}_{feature_class}_{feature_name}
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from radiomics import featureextractor

from src.utils.helpers import ensure_dir, load_config, resolve_path, setup_logging

logger = setup_logging()

# BraTS convention sub-regions
REGION_MAP: Dict[str, list] = {
    "whole_tumour": [1, 2, 4],
    "tumour_core": [1, 4],
    "enhancing": [4],
    "edema": [2],
    "necrosis": [1],
}

MODALITIES = ["T1", "T1GD", "T2", "FLAIR"]


# ── Extractor factory ──────────────────────────────────────────────────────
def _build_extractor(cfg: dict) -> featureextractor.RadiomicsFeatureExtractor:
    """Instantiate a PyRadiomics extractor from the project config."""
    params = {
        "binWidth": cfg["radiomics"]["bin_width"],
        "resampledPixelSpacing": cfg["radiomics"]["resample_spacing"],
        "interpolator": "sitkBSpline",
        "geometryTolerance": 1e-3,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    for fc in cfg["radiomics"]["feature_classes"]:
        extractor.enableFeatureClassByName(fc)
    return extractor


# ── Per-patient extraction ──────────────────────────────────────────────────
def _binarize_mask(seg_path: str, labels: list) -> sitk.Image:
    """Load a segmentation NIfTI and produce a binary mask for *labels*."""
    seg = sitk.ReadImage(str(seg_path))
    arr = sitk.GetArrayFromImage(seg)
    binary = np.isin(arr, labels).astype(np.uint8)
    out = sitk.GetImageFromArray(binary)
    out.CopyInformation(seg)
    return out


def extract_patient_features(
    patient_id: str,
    modality_paths: Dict[str, str],
    seg_path: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
) -> Dict[str, float]:
    """
    Extract radiomics features for one patient across all modality × region
    combinations.  Returns a flat dict ready for DataFrame conversion.
    """
    features: Dict[str, float] = {"patient_id": patient_id}

    for region_name, labels in REGION_MAP.items():
        mask = _binarize_mask(seg_path, labels)

        # Check mask has nonzero voxels
        mask_arr = sitk.GetArrayFromImage(mask)
        if mask_arr.sum() == 0:
            logger.debug("  %s — region '%s' is empty, skipping", patient_id, region_name)
            continue

        # Shape features only once per region (modality-independent)
        if region_name == "whole_tumour":
            try:
                # Use T1 as reference image for shape features
                img = sitk.ReadImage(str(modality_paths["T1"]))
                result = extractor.execute(img, mask)
                for key, val in result.items():
                    if "shape" in key.lower() and not key.startswith("diagnostics"):
                        features[f"shape_{region_name}_{key}"] = float(val)
            except Exception as e:
                logger.debug("  Shape extraction failed for %s: %s", patient_id, e)

        for mod in MODALITIES:
            mod_path = modality_paths.get(mod, "")
            if not mod_path:
                continue
            try:
                img = sitk.ReadImage(str(mod_path))
                result = extractor.execute(img, mask)
                for key, val in result.items():
                    if key.startswith("diagnostics"):
                        continue
                    if "shape" in key.lower():
                        continue  # already captured above
                    col = f"{mod}_{region_name}_{key}"
                    features[col] = float(val)
            except Exception as e:
                logger.debug(
                    "  Feature extraction error %s / %s / %s: %s",
                    patient_id, mod, region_name, e,
                )

    return features


# ── Batch extraction ────────────────────────────────────────────────────────
def extract_cohort_features(
    manifest: pd.DataFrame,
    cfg: dict | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Extract radiomics features for every *complete* patient in the manifest.
    Returns a features DataFrame (one row per patient) and caches it to disk.
    """
    if cfg is None:
        cfg = load_config()
    if n_jobs is None:
        n_jobs = cfg["preprocessing"]["n_jobs"]

    extractor = _build_extractor(cfg)
    complete = manifest[manifest["is_complete"]].copy()
    logger.info("Extracting radiomics for %d patients (jobs=%d) …", len(complete), n_jobs)

    def _worker(row: pd.Series) -> Dict:
        pid = row["patient_id"]
        mod_paths = {m: row[f"{m}_path"] for m in MODALITIES}
        seg = row["seg_path"]
        return extract_patient_features(pid, mod_paths, seg, extractor)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_worker)(row) for _, row in complete.iterrows()
        )

    feat_df = pd.DataFrame(results)
    logger.info("  Extracted %d features × %d patients", feat_df.shape[1] - 1, len(feat_df))

    # Cache
    out = resolve_path(cfg["paths"]["output"]["data_dir"]) / "radiomics_features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(out, index=False)
    logger.info("  Features saved → %s", out)

    return feat_df
