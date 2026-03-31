"""
Feature engineering — merge radiomics + clinical features, impute, scale,
remove low-variance / high-correlation columns, run PCA, and produce the
final clustering-ready matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()


# ── Merge ──────────────────────────────────────────────────────────────────
def merge_features(
    radiomics: pd.DataFrame,
    clinical: pd.DataFrame,
    lobe_volumes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge radiomics, clinical, and (optional) lobe volumes on patient_id."""
    merged = radiomics.merge(clinical, on="patient_id", how="inner")
    if lobe_volumes is not None and not lobe_volumes.empty:
        merged = merged.merge(lobe_volumes, on="patient_id", how="left")
    logger.info("Merged feature matrix: %d patients × %d columns", *merged.shape)
    return merged


# ── Cleaning pipeline ──────────────────────────────────────────────────────
def _drop_id_and_meta(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Separate patient_id and metadata columns from numeric features."""
    meta_cols = ["patient_id", "cohort", "os_days", "os_event"]
    meta = df[["patient_id"]].copy()
    for c in meta_cols:
        if c in df.columns:
            meta[c] = df[c].values
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return meta, df[feat_cols].copy()


def _remove_low_variance(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop features with variance below *threshold*."""
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    kept = X.columns[sel.get_support()]
    logger.info("  Variance filter: %d → %d features", X.shape[1], len(kept))
    return X[kept]


def _remove_high_correlation(X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Iteratively drop one of each pair with Pearson |r| > *threshold*."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info("  Correlation filter: dropping %d features (|r| > %.2f)",
                len(to_drop), threshold)
    return X.drop(columns=to_drop)


# ── Public API ──────────────────────────────────────────────────────────────
def engineer_features(
    merged: pd.DataFrame,
    cfg: dict | None = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Full feature-engineering pipeline:

    1. Separate metadata from numeric features.
    2. Impute missing values (median).
    3. Remove near-zero-variance features.
    4. Remove highly correlated features.
    5. Standardise (z-score).
    6. PCA dimensionality reduction.

    Returns
    -------
    meta : DataFrame
        patient_id + cohort + survival columns.
    X_pca : DataFrame
        PCA-transformed feature matrix (clustering input).
    pca : PCA
        Fitted PCA object (for explained-variance reporting).
    """
    if cfg is None:
        cfg = load_config()

    meta, X = _drop_id_and_meta(merged)

    # Force numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    n_nan = X.isna().sum().sum()
    logger.info("  Imputed %d NaN values (median strategy)", n_nan)

    # Variance filter
    vt = cfg["features"]["variance_threshold"]
    X_imp = _remove_low_variance(X_imp, vt)

    # Correlation filter
    ct = cfg["features"]["correlation_threshold"]
    X_imp = _remove_high_correlation(X_imp, ct)

    # Standardise
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imp), columns=X_imp.columns, index=X_imp.index
    )

    # PCA
    n_comp = min(cfg["features"]["n_components_pca"], X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_comp, random_state=cfg["project"]["seed"])
    X_pca_arr = pca.fit_transform(X_scaled)
    pca_cols = [f"PC{i+1}" for i in range(n_comp)]
    X_pca = pd.DataFrame(X_pca_arr, columns=pca_cols, index=X_scaled.index)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n95 = int(np.searchsorted(cum_var, 0.95) + 1)
    logger.info(
        "  PCA: %d components capture %.1f%% variance; 95%% at %d components",
        n_comp,
        cum_var[-1] * 100,
        n95,
    )

    if save:
        data_dir = resolve_path(cfg["paths"]["output"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        meta.to_csv(data_dir / "meta.csv", index=False)
        X_pca.to_csv(data_dir / "X_pca.csv", index=False)
        X_scaled.to_csv(data_dir / "X_scaled.csv", index=False)
        logger.info("  Saved meta.csv, X_pca.csv, X_scaled.csv")

    return meta, X_pca, pca
