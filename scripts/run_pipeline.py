#!/usr/bin/env python3
"""
run_pipeline.py — Master orchestrator for the GBM Multi-Modal Clustering
pipeline.  Run from the project root:

    python -m scripts.run_pipeline            # full pipeline
    python -m scripts.run_pipeline --step 1   # only data verification
    python -m scripts.run_pipeline --step 1-5 # steps 1 through 5

Pipeline steps
──────────────
  1. Data verification & manifest creation
  2. Atlas registration (ANTsPy → MNI152)
  3. Radiomics feature extraction (PyRadiomics)
  4. Clinical data harmonisation
  5. Feature engineering (merge, impute, PCA)
  6. Consensus clustering (sweep over k)
  7. Survival analysis (KM, log-rank, Cox PH)
  8. Cluster characterisation (SHAP, enrichment)
  9. External validation (UCSF→UPenn transfer)
 10. Figure generation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Ensure project root is on PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.helpers import load_config, resolve_path, set_seed, setup_logging

logger = setup_logging()


# ── Argument parsing ───────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GBM Clustering Pipeline")
    parser.add_argument(
        "--step",
        type=str,
        default="1-10",
        help="Step or range to run, e.g. '1', '3-7', or '1-10' (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/config.yaml).",
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Skip atlas registration (use cached data if available).",
    )
    return parser.parse_args()


def _parse_step_range(step_str: str) -> Tuple[int, int]:
    if "-" in step_str:
        lo, hi = step_str.split("-", 1)
        return int(lo), int(hi)
    s = int(step_str)
    return s, s


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline steps
# ══════════════════════════════════════════════════════════════════════════════

def step1_verify(cfg: dict) -> pd.DataFrame:
    """Step 1: Data verification & manifest creation."""
    from src.preprocessing.verify_data import verify_dataset
    return verify_dataset(cfg)


def step2_register(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Step 2: Atlas registration."""
    from src.preprocessing.register_atlas import register_cohort
    return register_cohort(manifest, cfg)


def step3_radiomics(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Step 3: Radiomics feature extraction."""
    from src.preprocessing.extract_radiomics import extract_cohort_features
    return extract_cohort_features(manifest, cfg)


def step4_clinical(cfg: dict) -> pd.DataFrame:
    """Step 4: Clinical data harmonisation."""
    from src.features.clinical_harmonization import harmonize_clinical
    return harmonize_clinical(cfg)


def step5_features(
    radiomics: pd.DataFrame,
    clinical: pd.DataFrame,
    lobe_volumes: pd.DataFrame | None,
    cfg: dict,
):
    """Step 5: Feature engineering (merge + PCA)."""
    from src.features.feature_engineering import engineer_features, merge_features
    merged = merge_features(radiomics, clinical, lobe_volumes)
    meta, X_pca, pca = engineer_features(merged, cfg)
    return meta, X_pca, pca


def step6_clustering(X_pca: pd.DataFrame, cfg: dict):
    """Step 6: Consensus clustering."""
    from src.clustering.consensus_clustering import (
        run_consensus_clustering,
        select_best_k,
    )
    metrics_df, all_labels, all_consensus = run_consensus_clustering(X_pca.values, cfg)
    best_k = select_best_k(metrics_df)
    labels = all_labels[best_k]
    consensus = all_consensus[best_k]
    return metrics_df, labels, consensus, best_k, all_labels, all_consensus


def step7_survival(meta: pd.DataFrame, labels: np.ndarray, cfg: dict):
    """Step 7: Survival analysis."""
    from src.validation.survival_analysis import run_survival_analysis
    return run_survival_analysis(meta, labels, cfg)


def step8_characterize(
    meta: pd.DataFrame,
    X_pca: pd.DataFrame,
    labels: np.ndarray,
    cfg: dict,
):
    """Step 8: Cluster characterisation."""
    from src.validation.cluster_characterization import characterize_clusters
    return characterize_clusters(meta, X_pca, labels, cfg)


def step9_external(
    meta: pd.DataFrame,
    X_pca: pd.DataFrame,
    labels: np.ndarray,
    cfg: dict,
):
    """Step 9: External validation."""
    from src.validation.external_validation import run_external_validation
    return run_external_validation(meta, X_pca, labels, cfg)


def step10_figures(
    consensus, labels, best_k, metrics_df, X_pca, meta, pca,
    km_dict, cph, summary, shap_values, X_shap, km_val, cfg,
):
    """Step 10: Figure generation."""
    from src.visualization.plots import generate_all_figures
    generate_all_figures(
        consensus_matrix=consensus,
        labels=labels,
        best_k=best_k,
        metrics_df=metrics_df,
        X_pca=X_pca.values if hasattr(X_pca, "values") else X_pca,
        meta=meta,
        pca=pca,
        km_dict=km_dict,
        cph=cph,
        summary=summary,
        shap_values=shap_values,
        X_shap=X_shap,
        km_val=km_val,
        cfg=cfg,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    lo, hi = _parse_step_range(args.step)
    logger.info("=" * 60)
    logger.info("GBM Multi-Modal Clustering Pipeline — steps %d–%d", lo, hi)
    logger.info("=" * 60)

    data_dir = resolve_path(cfg["paths"]["output"]["data_dir"])

    # Shared state across steps
    manifest = None
    radiomics = None
    clinical = None
    lobe_volumes = None
    meta = None
    X_pca = None
    pca = None
    metrics_df = None
    labels = None
    consensus = None
    best_k = None
    km_dict = None
    lr_pairwise = None
    cph = None
    char_results = None
    ext_results = None

    # ── Step 1 ────────────────────────────────────────────────────────────
    if lo <= 1 <= hi:
        logger.info("\n▸ STEP 1 / 10 — Data Verification")
        manifest = step1_verify(cfg)
    else:
        manifest_path = data_dir / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)

    # ── Step 2 ────────────────────────────────────────────────────────────
    if lo <= 2 <= hi and not args.skip_registration:
        logger.info("\n▸ STEP 2 / 10 — Atlas Registration")
        manifest = step2_register(manifest, cfg)
    else:
        logger.info("  ⏭ Registration skipped (--skip-registration)")

    # ── Step 3 ────────────────────────────────────────────────────────────
    if lo <= 3 <= hi:
        logger.info("\n▸ STEP 3 / 10 — Radiomics Feature Extraction")
        radiomics = step3_radiomics(manifest, cfg)
    else:
        rad_path = data_dir / "radiomics_features.csv"
        if rad_path.exists():
            radiomics = pd.read_csv(rad_path)

    # ── Step 4 ────────────────────────────────────────────────────────────
    if lo <= 4 <= hi:
        logger.info("\n▸ STEP 4 / 10 — Clinical Data Harmonisation")
        clinical = step4_clinical(cfg)
    else:
        clin_path = data_dir / "clinical_harmonized.csv"
        if clin_path.exists():
            clinical = pd.read_csv(clin_path)

    # ── Step 5 ────────────────────────────────────────────────────────────
    if lo <= 5 <= hi:
        logger.info("\n▸ STEP 5 / 10 — Feature Engineering")
        meta, X_pca, pca = step5_features(radiomics, clinical, lobe_volumes, cfg)
    else:
        meta_path = data_dir / "meta.csv"
        pca_path = data_dir / "X_pca.csv"
        if meta_path.exists() and pca_path.exists():
            meta = pd.read_csv(meta_path)
            X_pca = pd.read_csv(pca_path)

    # ── Step 6 ────────────────────────────────────────────────────────────
    if lo <= 6 <= hi:
        logger.info("\n▸ STEP 6 / 10 — Consensus Clustering")
        metrics_df, labels, consensus, best_k, _, all_consensus = step6_clustering(X_pca, cfg)
    else:
        met_path = data_dir / "clustering_metrics.csv"
        if met_path.exists():
            metrics_df = pd.read_csv(met_path, index_col=0)

    # ── Step 7 ────────────────────────────────────────────────────────────
    if lo <= 7 <= hi:
        logger.info("\n▸ STEP 7 / 10 — Survival Analysis")
        km_dict, lr_pairwise, cph = step7_survival(meta, labels, cfg)

    # ── Step 8 ────────────────────────────────────────────────────────────
    if lo <= 8 <= hi:
        logger.info("\n▸ STEP 8 / 10 — Cluster Characterisation")
        char_results = step8_characterize(meta, X_pca, labels, cfg)

    # ── Step 9 ────────────────────────────────────────────────────────────
    if lo <= 9 <= hi:
        logger.info("\n▸ STEP 9 / 10 — External Validation (UCSF→UPenn)")
        ext_results = step9_external(meta, X_pca, labels, cfg)

    # ── Step 10 ───────────────────────────────────────────────────────────
    if lo <= 10 <= hi:
        logger.info("\n▸ STEP 10 / 10 — Figure Generation")
        shap_vals = char_results.get("shap_values") if char_results else None
        summary = char_results["summary"] if char_results else None
        km_val = ext_results.get("km_val") if ext_results else None
        step10_figures(
            consensus, labels, best_k, metrics_df, X_pca, meta, pca,
            km_dict, cph, summary, shap_vals, X_pca, km_val, cfg,
        )

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete — %.1f s elapsed", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
