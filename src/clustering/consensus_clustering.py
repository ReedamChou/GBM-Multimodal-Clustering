"""
Consensus clustering — run multiple base algorithms with subsampled
iterations and combine via a co-association matrix.  Select optimal k
using silhouette score, Calinski-Harabasz index, and PAC (Proportion
of Ambiguous Clustering).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None


# ══════════════════════════════════════════════════════════════════════════════
#  Base clustering algorithms
# ══════════════════════════════════════════════════════════════════════════════
def _fit_spectral(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return SpectralClustering(
        n_clusters=k, affinity="rbf", random_state=seed, n_init=10
    ).fit_predict(X)


def _fit_gmm(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return GaussianMixture(
        n_components=k, covariance_type="full", random_state=seed, n_init=5
    ).fit_predict(X)


def _fit_agglomerative(X: np.ndarray, k: int, **_) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def _fit_kmedoids(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    if KMedoids is None:
        raise ImportError("scikit-learn-extra not installed — skipping KMedoids")
    return KMedoids(n_clusters=k, random_state=seed, method="alternate").fit_predict(X)


_ALGO_MAP = {
    "spectral": _fit_spectral,
    "gmm": _fit_gmm,
    "agglomerative": _fit_agglomerative,
    "kmedoids": _fit_kmedoids,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Consensus clustering
# ══════════════════════════════════════════════════════════════════════════════
def build_consensus_matrix(
    X: np.ndarray,
    k: int,
    algorithms: List[str],
    n_iter: int = 100,
    subsample_frac: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    """
    Build a consensus (co-association) matrix by repeatedly subsampling and
    clustering with each base algorithm.

    Returns an n×n matrix where entry (i,j) is the fraction of iterations
    in which patients i and j were assigned to the same cluster.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    coassoc = np.zeros((n, n), dtype=np.float64)
    count = np.zeros((n, n), dtype=np.float64)

    for it in range(n_iter):
        # Subsample indices
        idx = np.sort(rng.choice(n, size=int(n * subsample_frac), replace=False))
        X_sub = X[idx]

        for algo_name in algorithms:
            try:
                labels = _ALGO_MAP[algo_name](X_sub, k, seed=seed + it)
            except Exception:
                continue
            # Update co-association for this subsample
            for i_loc in range(len(idx)):
                for j_loc in range(i_loc + 1, len(idx)):
                    gi, gj = idx[i_loc], idx[j_loc]
                    count[gi, gj] += 1
                    count[gj, gi] += 1
                    if labels[i_loc] == labels[j_loc]:
                        coassoc[gi, gj] += 1
                        coassoc[gj, gi] += 1

    # Normalise
    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.where(count > 0, coassoc / count, 0.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


def consensus_labels(consensus_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Derive final labels from the consensus matrix via hierarchical clustering
    on 1 - consensus (distance).
    """
    dist = 1.0 - consensus_matrix
    np.fill_diagonal(dist, 0.0)
    # Condensed form for scipy
    n = dist.shape[0]
    condensed = dist[np.triu_indices(n, k=1)]
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust") - 1  # 0-indexed
    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  Model selection metrics
# ══════════════════════════════════════════════════════════════════════════════
def pac_score(consensus_matrix: np.ndarray, lower: float = 0.1, upper: float = 0.9) -> float:
    """
    Proportion of Ambiguous Clustering — fraction of off-diagonal entries
    in [lower, upper].  Lower PAC ⇒ more stable clustering.
    """
    n = consensus_matrix.shape[0]
    vals = consensus_matrix[np.triu_indices(n, k=1)]
    ambiguous = ((vals >= lower) & (vals <= upper)).sum()
    return float(ambiguous) / len(vals)


def evaluate_k(
    X: np.ndarray,
    labels: np.ndarray,
    consensus_matrix: np.ndarray,
) -> Dict[str, float]:
    """Compute clustering quality metrics for a given partition."""
    metrics: Dict[str, float] = {}
    metrics["silhouette"] = float(silhouette_score(X, labels))
    metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    metrics["pac"] = pac_score(consensus_matrix)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Full sweep over k
# ══════════════════════════════════════════════════════════════════════════════
def run_consensus_clustering(
    X: np.ndarray,
    cfg: dict | None = None,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Sweep over k values defined in config, build consensus matrices, derive
    final labels, and evaluate.

    Returns
    -------
    metrics_df : DataFrame
        One row per k with silhouette, CH, PAC scores.
    all_labels : dict
        k → label array (length n).
    all_consensus : dict
        k → consensus matrix (n×n).
    """
    if cfg is None:
        cfg = load_config()

    k_range = cfg["clustering"]["k_range"]
    algorithms = cfg["clustering"]["algorithms"]
    n_iter = cfg["clustering"]["consensus"]["n_iterations"]
    sub_frac = cfg["clustering"]["consensus"]["subsample_fraction"]
    seed = cfg["project"]["seed"]

    metrics_rows = []
    all_labels: Dict[int, np.ndarray] = {}
    all_consensus: Dict[int, np.ndarray] = {}

    for k in k_range:
        logger.info("Consensus clustering k=%d …", k)
        C = build_consensus_matrix(X, k, algorithms, n_iter, sub_frac, seed)
        labels = consensus_labels(C, k)
        mets = evaluate_k(X, labels, C)
        mets["k"] = k
        metrics_rows.append(mets)
        all_labels[k] = labels
        all_consensus[k] = C
        logger.info(
            "  k=%d  silhouette=%.3f  CH=%.0f  PAC=%.3f",
            k, mets["silhouette"], mets["calinski_harabasz"], mets["pac"],
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index("k")

    # Save metrics
    data_dir = resolve_path(cfg["paths"]["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(data_dir / "clustering_metrics.csv")
    logger.info("Clustering metrics saved → %s", data_dir / "clustering_metrics.csv")

    return metrics_df, all_labels, all_consensus


def select_best_k(metrics_df: pd.DataFrame) -> int:
    """
    Heuristic selection: rank by silhouette (desc) and PAC (asc), pick the
    k with the best average rank.
    """
    df = metrics_df.copy()
    df["sil_rank"] = df["silhouette"].rank(ascending=False)
    df["pac_rank"] = df["pac"].rank(ascending=True)
    df["mean_rank"] = (df["sil_rank"] + df["pac_rank"]) / 2
    best_k = int(df["mean_rank"].idxmin())
    logger.info("Best k = %d (silhouette=%.3f, PAC=%.3f)",
                best_k, df.loc[best_k, "silhouette"], df.loc[best_k, "pac"])
    return best_k
