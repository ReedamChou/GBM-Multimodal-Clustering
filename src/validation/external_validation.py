"""
External validation — train cluster model on UCSF (discovery), predict on
UPenn (validation), and compare survival / enrichment between cohorts.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier

from src.utils.helpers import load_config, resolve_path, setup_logging
from src.validation.cluster_characterization import cluster_summary_table, enrichment_tests
from src.validation.survival_analysis import fit_kaplan_meier, logrank_between_clusters

logger = setup_logging()


# ══════════════════════════════════════════════════════════════════════════════
#  Split cohorts
# ══════════════════════════════════════════════════════════════════════════════
def split_cohorts(
    meta: pd.DataFrame,
    X: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split meta and feature matrices by cohort column."""
    ucsf_mask = meta["cohort"] == "UCSF"
    upenn_mask = meta["cohort"] == "UPenn"
    return (
        meta[ucsf_mask].reset_index(drop=True),
        X[ucsf_mask].reset_index(drop=True),
        meta[upenn_mask].reset_index(drop=True),
        X[upenn_mask].reset_index(drop=True),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Label transfer via kNN
# ══════════════════════════════════════════════════════════════════════════════
def transfer_labels_knn(
    X_train: np.ndarray,
    labels_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 7,
) -> np.ndarray:
    """
    Transfer cluster labels from discovery to validation cohort using a
    k-Nearest-Neighbours classifier trained on the discovery PCA space.
    """
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(X_train, labels_train)
    labels_val = knn.predict(X_test)
    return labels_val


# ══════════════════════════════════════════════════════════════════════════════
#  Agreement metrics
# ══════════════════════════════════════════════════════════════════════════════
def cluster_agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> Dict[str, float]:
    """Adjusted Rand Index and NMI between two label arrays."""
    return {
        "ARI": float(adjusted_rand_score(labels_a, labels_b)),
        "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Full external validation pipeline
# ══════════════════════════════════════════════════════════════════════════════
def run_external_validation(
    meta: pd.DataFrame,
    X_pca: pd.DataFrame,
    labels_discovery: np.ndarray,
    cfg: dict | None = None,
    save: bool = True,
) -> Dict:
    """
    1. Split into UCSF (discovery) and UPenn (validation).
    2. Transfer cluster labels via kNN.
    3. Run survival analysis on validation cohort.
    4. Compare cluster profiles between cohorts.

    Returns dict of results.
    """
    if cfg is None:
        cfg = load_config()

    meta_ucsf, X_ucsf, meta_upenn, X_upenn = split_cohorts(meta, X_pca)

    # Labels for discovery cohort (UCSF part of full labels)
    ucsf_idx = meta["cohort"] == "UCSF"
    labels_ucsf = labels_discovery[ucsf_idx.values]

    # Transfer labels to UPenn
    logger.info("Transferring cluster labels to UPenn via kNN …")
    labels_upenn = transfer_labels_knn(
        X_ucsf.values, labels_ucsf, X_upenn.values
    )

    # Also do de-novo clustering on UPenn for agreement comparison
    logger.info("De-novo clustering on UPenn for agreement check …")
    n_clusters = len(np.unique(labels_ucsf))
    denovo = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_upenn_denovo = denovo.fit_predict(X_upenn.values)

    agreement = cluster_agreement(labels_upenn, labels_upenn_denovo)
    logger.info("  kNN vs de-novo agreement — ARI=%.3f  NMI=%.3f",
                agreement["ARI"], agreement["NMI"])

    # Survival analysis on validation
    logger.info("Survival analysis on UPenn validation cohort …")
    km_val = fit_kaplan_meier(meta_upenn, labels_upenn)
    lr_val = logrank_between_clusters(meta_upenn, labels_upenn, cfg["survival"]["alpha"])

    # Cluster summary comparison
    summary_disc = cluster_summary_table(meta_ucsf, labels_ucsf)
    summary_val = cluster_summary_table(meta_upenn, labels_upenn)

    results = {
        "labels_upenn_knn": labels_upenn,
        "labels_upenn_denovo": labels_upenn_denovo,
        "agreement": agreement,
        "km_val": km_val,
        "logrank_val": lr_val,
        "summary_discovery": summary_disc,
        "summary_validation": summary_val,
    }

    if save:
        tbl_dir = resolve_path(cfg["paths"]["output"]["tables_dir"])
        tbl_dir.mkdir(parents=True, exist_ok=True)
        lr_val.to_csv(tbl_dir / "logrank_validation.csv", index=False)
        summary_val.to_csv(tbl_dir / "cluster_summary_validation.csv", index=False)
        pd.DataFrame([agreement]).to_csv(tbl_dir / "external_agreement.csv", index=False)
        logger.info("  Validation tables saved → %s", tbl_dir)

    return results
