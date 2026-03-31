"""
Publication-quality visualisations for every analysis stage.

All plot functions accept a ``save_path`` argument; when provided the figure
is saved at the configured DPI and format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
import umap

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()


def _apply_style(cfg: dict) -> None:
    """Set global Matplotlib / Seaborn styling."""
    sns.set_theme(style="whitegrid", palette=cfg["visualization"]["palette"])
    plt.rcParams.update({
        "figure.dpi": cfg["visualization"]["dpi"],
        "savefig.dpi": cfg["visualization"]["dpi"],
        "savefig.bbox": "tight",
        "font.size": 10,
    })


def _save(fig: plt.Figure, path: Optional[str | Path], cfg: dict) -> None:
    if path is not None:
        fmt = cfg["visualization"]["figure_format"]
        p = Path(path)
        if p.suffix == "":
            p = p.with_suffix(f".{fmt}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
        plt.close(fig)
        logger.info("  Saved figure → %s", p)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Consensus heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_consensus_heatmap(
    consensus_matrix: np.ndarray,
    labels: np.ndarray,
    k: int,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Reordered consensus matrix heatmap with cluster annotations."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    order = np.argsort(labels)
    C_sorted = consensus_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C_sorted, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Co-association probability")
    ax.set_title(f"Consensus Matrix (k={k})")
    ax.set_xlabel("Patient index (sorted by cluster)")
    ax.set_ylabel("Patient index (sorted by cluster)")

    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  2. Clustering metrics (elbow-style)
# ══════════════════════════════════════════════════════════════════════════════
def plot_clustering_metrics(
    metrics_df: pd.DataFrame,
    best_k: int,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Line plots of silhouette, CH, and PAC vs. k."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, label in zip(
        axes,
        ["silhouette", "calinski_harabasz", "pac"],
        ["Silhouette Score", "Calinski-Harabasz Index", "PAC"],
    ):
        ax.plot(metrics_df.index, metrics_df[col], "o-", linewidth=2)
        ax.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"k*={best_k}")
        ax.set_xlabel("k")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()

    fig.tight_layout()
    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  3. UMAP scatter
# ══════════════════════════════════════════════════════════════════════════════
def plot_umap(
    X: np.ndarray,
    labels: np.ndarray,
    cohort: np.ndarray | None = None,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """2-D UMAP coloured by cluster labels (and optionally shaped by cohort)."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    ucfg = cfg["visualization"]["umap"]
    reducer = umap.UMAP(
        n_neighbors=ucfg["n_neighbors"],
        min_dist=ucfg["min_dist"],
        metric=ucfg["metric"],
        random_state=cfg["project"]["seed"],
    )
    emb = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette(cfg["visualization"]["palette"], len(unique_labels))

    if cohort is not None:
        markers = {"UCSF": "o", "UPenn": "^"}
        for cl, color in zip(unique_labels, palette):
            for coh, marker in markers.items():
                mask = (labels == cl) & (cohort == coh)
                if mask.sum() == 0:
                    continue
                ax.scatter(
                    emb[mask, 0], emb[mask, 1],
                    c=[color], marker=marker, s=20, alpha=0.7,
                    label=f"C{cl}-{coh}",
                )
    else:
        for cl, color in zip(unique_labels, palette):
            mask = labels == cl
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=[color], s=20, alpha=0.7,
                label=f"Cluster {cl}",
            )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP Embedding — Cluster Assignments")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  4. Kaplan-Meier curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_kaplan_meier(
    km_dict: Dict[int, KaplanMeierFitter],
    title: str = "Overall Survival by Cluster",
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Kaplan-Meier survival curves with confidence bands."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette(cfg["visualization"]["palette"], len(km_dict))

    for (cl, kmf), color in zip(sorted(km_dict.items()), palette):
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=2)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title)
    ax.legend(loc="lower left")

    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5. SHAP summary / beeswarm
# ══════════════════════════════════════════════════════════════════════════════
def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure | None:
    """SHAP beeswarm plot (top features)."""
    try:
        import shap as shap_lib
    except ImportError:
        logger.warning("shap not available — skipping SHAP summary plot")
        return None

    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap_lib.summary_plot(shap_values, X, show=False, max_display=20)
    fig = plt.gcf()

    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6. PCA explained-variance
# ══════════════════════════════════════════════════════════════════════════════
def plot_pca_variance(
    pca: PCA,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Scree plot + cumulative explained variance."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(1, len(cumvar) + 1)
    ax1.bar(x, pca.explained_variance_ratio_, alpha=0.5, label="Individual")
    ax2 = ax1.twinx()
    ax2.plot(x, cumvar, "r-o", markersize=3, label="Cumulative")
    ax2.axhline(0.95, color="grey", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax2.set_ylabel("Cumulative Variance")
    ax1.set_title("PCA Explained Variance")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  7. Clinical enrichment heatmap
# ══════════════════════════════════════════════════════════════════════════════
def plot_enrichment_heatmap(
    summary: pd.DataFrame,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Heatmap of normalised clinical variables across clusters."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["cluster", "n"]]
    plot_data = summary.set_index("cluster")[numeric_cols]

    # Z-score normalise per variable for visual comparison
    plot_z = (plot_data - plot_data.mean()) / plot_data.std()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(plot_z.T, annot=True, fmt=".1f", cmap="coolwarm", ax=ax,
                linewidths=0.5, center=0)
    ax.set_title("Clinical Feature Enrichment per Cluster (z-scored)")
    ax.set_xlabel("Cluster")

    fig.tight_layout()
    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  8. Cox forest plot
# ══════════════════════════════════════════════════════════════════════════════
def plot_cox_forest(
    cph,
    save_path: Optional[str | Path] = None,
    cfg: dict | None = None,
) -> plt.Figure:
    """Forest plot of Cox PH hazard ratios with confidence intervals."""
    if cfg is None:
        cfg = load_config()
    _apply_style(cfg)

    fig, ax = plt.subplots(figsize=(8, 5))
    cph.plot(ax=ax)
    ax.set_title("Cox Proportional Hazards — Hazard Ratios")
    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)

    fig.tight_layout()
    _save(fig, save_path, cfg)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Master figure generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_all_figures(
    consensus_matrix: np.ndarray,
    labels: np.ndarray,
    best_k: int,
    metrics_df: pd.DataFrame,
    X_pca: np.ndarray,
    meta: pd.DataFrame,
    pca: PCA,
    km_dict: Dict[int, KaplanMeierFitter],
    cph,
    summary: pd.DataFrame,
    shap_values=None,
    X_shap: pd.DataFrame | None = None,
    km_val: Dict | None = None,
    cfg: dict | None = None,
) -> None:
    """Generate and save all publication figures."""
    if cfg is None:
        cfg = load_config()
    fig_dir = resolve_path(cfg["paths"]["output"]["figures_dir"])

    logger.info("Generating all figures → %s", fig_dir)

    plot_consensus_heatmap(consensus_matrix, labels, best_k,
                           fig_dir / "consensus_heatmap", cfg)
    plot_clustering_metrics(metrics_df, best_k,
                            fig_dir / "clustering_metrics", cfg)
    plot_umap(X_pca, labels,
              cohort=meta["cohort"].values if "cohort" in meta.columns else None,
              save_path=fig_dir / "umap_clusters", cfg=cfg)
    plot_kaplan_meier(km_dict, "Overall Survival by Cluster (Discovery)",
                      fig_dir / "km_discovery", cfg)
    plot_pca_variance(pca, fig_dir / "pca_variance", cfg)
    plot_enrichment_heatmap(summary, fig_dir / "enrichment_heatmap", cfg)
    plot_cox_forest(cph, fig_dir / "cox_forest", cfg)

    if shap_values is not None and X_shap is not None:
        plot_shap_summary(shap_values, X_shap, fig_dir / "shap_summary", cfg)

    if km_val is not None:
        plot_kaplan_meier(km_val, "Overall Survival by Cluster (Validation — UPenn)",
                          fig_dir / "km_validation", cfg)

    logger.info("All figures generated ✓")
