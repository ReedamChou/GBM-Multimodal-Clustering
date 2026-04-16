"""Step 8: Publication-quality visualizations.

Generates five key figures:
1. Feature heatmap (z-scored cluster means)
2. Paired Kaplan-Meier curves (train + test)
3. Lobe distribution bar chart
4. UMAP scatter plot (train set)
5. Cluster summary comparison table (train vs test)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from cluster_validation_utils import build_event_observed, first_present
from pipeline_preprocessing import canonicalize_lobe

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None

try:
    import umap
except Exception:  # pragma: no cover
    umap = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_STEP4B_DIR = PROJECT_ROOT / "outputs" / "step4b"
DEFAULT_STEP5_DIR = PROJECT_ROOT / "outputs" / "step5"
DEFAULT_STEP6_DIR = PROJECT_ROOT / "outputs" / "step6"
DEFAULT_STEP7_DIR = PROJECT_ROOT / "outputs" / "step7"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step8"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step8.{log_file.parent.name}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ── Color palette for clusters ─────────────────────────────────────────
CLUSTER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#D35400", "#7F8C8D",
]


def get_cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


# ── Figure 1: Feature heatmap ─────────────────────────────────────────
def plot_feature_heatmap(
    train_labeled_df: pd.DataFrame,
    cluster_col: str,
    feature_cols: list[str],
    dataset: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    # Compute z-scored mean per cluster.
    grouped = train_labeled_df.groupby(cluster_col)[feature_cols].mean()
    # Z-score across clusters for each feature (row-wise normalization).
    z_scored = grouped.apply(lambda x: (x - x.mean()) / max(x.std(), 1e-8), axis=0)

    # Shorten feature names for display.
    display_names = [
        c.replace("_", " ").replace("global ", "").replace("dominant lobe ", "lobe: ")[:35]
        for c in z_scored.columns
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(feature_cols) * 0.4), max(4, len(grouped) * 0.8 + 2)))
    sns.heatmap(
        z_scored.T.values,
        xticklabels=[f"Cluster {int(c)}" for c in z_scored.index],
        yticklabels=display_names,
        cmap="RdBu_r",
        center=0,
        annot=True if len(feature_cols) <= 20 else False,
        fmt=".2f" if len(feature_cols) <= 20 else "",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Z-score (across clusters)"},
    )
    ax.set_title(f"{dataset.upper()} Feature Heatmap by Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature heatmap: %s", output_path)


# ── Figure 2: Paired Kaplan-Meier ──────────────────────────────────────
def plot_paired_km(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cluster_col: str,
    os_col: str,
    censor_col: str | None,
    survival_status_col: str | None,
    dataset: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, df, split_label in [
        (axes[0], train_df, "Train"),
        (axes[1], test_df, "Test"),
    ]:
        durations = pd.to_numeric(df[os_col], errors="coerce")
        events = build_event_observed(df, censor_col, survival_status_col)
        valid = durations.notna() & events.notna()

        # Log-rank p-value.
        groups = pd.to_numeric(df[cluster_col], errors="coerce")
        logrank_p = None
        if valid.sum() >= 10 and groups[valid].nunique() >= 2:
            try:
                result = multivariate_logrank_test(
                    durations[valid],
                    groups[valid].astype(int),
                    events[valid].astype(int),
                )
                logrank_p = float(result.p_value)
            except Exception:
                pass

        kmf = KaplanMeierFitter()
        for cluster_id in sorted(df[cluster_col].dropna().unique()):
            mask = (df[cluster_col] == cluster_id) & valid
            if mask.sum() == 0:
                continue
            kmf.fit(
                durations=durations[mask],
                event_observed=events[mask],
                label=f"Cluster {int(cluster_id)} (n={int(mask.sum())})",
            )
            kmf.plot(ax=ax, ci_show=True, color=get_cluster_color(int(cluster_id)))

        ax.set_title(f"{split_label} Set", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability" if split_label == "Train" else "")
        ax.grid(alpha=0.3)
        if logrank_p is not None:
            ax.text(0.02, 0.04, f"Log-rank p = {logrank_p:.3g}", transform=ax.transAxes, fontsize=10)

    fig.suptitle(f"{dataset.upper()} Kaplan-Meier Curves by Cluster", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved paired KM plot: %s", output_path)


# ── Figure 3: Lobe distribution bar chart ──────────────────────────────
def plot_lobe_distribution(
    train_df: pd.DataFrame,
    cluster_col: str,
    lobe_col: str,
    dataset: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    lobe = canonicalize_lobe(train_df[lobe_col]).fillna("unknown")
    plot_df = pd.DataFrame({"cluster": train_df[cluster_col], "lobe": lobe})
    ct = pd.crosstab(plot_df["cluster"], plot_df["lobe"], normalize="index") * 100

    lobe_colors = {"frontal": "#3498DB", "temporal": "#E74C3C", "parietal": "#2ECC71", "occipital": "#F39C12", "unknown": "#BDC3C7"}
    ordered_lobes = [l for l in ["frontal", "temporal", "parietal", "occipital", "unknown"] if l in ct.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ct))
    width = 0.15
    for i, lobe_name in enumerate(ordered_lobes):
        ax.bar(x + i * width, ct[lobe_name], width, label=lobe_name.capitalize(), color=lobe_colors.get(lobe_name, "#95A5A6"))

    ax.set_xticks(x + width * len(ordered_lobes) / 2)
    ax.set_xticklabels([f"Cluster {int(c)}" for c in ct.index])
    ax.set_ylabel("Percentage of patients (%)")
    ax.set_title(f"{dataset.upper()} Lobe Distribution by Cluster", fontsize=14, fontweight="bold")
    ax.legend(title="Brain Lobe", frameon=True)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved lobe distribution plot: %s", output_path)


# ── Figure 4: UMAP scatter plot ───────────────────────────────────────
def plot_umap_clusters(
    train_features_df: pd.DataFrame,
    labels: pd.Series,
    id_col: str,
    feature_cols: list[str],
    dataset: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    if umap is None:
        logger.warning("umap-learn is not installed; skipping UMAP plot.")
        return

    x = train_features_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(labels.unique())
    for cl in unique_labels:
        mask = labels == cl
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=get_cluster_color(int(cl)),
            label=f"Cluster {int(cl)}",
            alpha=0.6,
            s=20,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_title(f"{dataset.upper()} UMAP Visualization by Cluster", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(title="Cluster", frameon=True, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info("Saved UMAP plot: %s", output_path)


# ── Figure 5: Summary comparison table ────────────────────────────────
def plot_summary_table(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    dataset: str,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    # Merge train and test summaries.
    cols_to_show = [
        "cluster_label", "n", "median_os", "mgmt_methylated_pct",
        "idh_mutant_pct", "mean_global_nc_en_ratio", "dominant_lobe_mode", "mean_age",
    ]
    cols_display = [
        "Cluster", "N", "Median OS", "MGMT Meth %", "IDH Mut %", "NC/EN", "Dom. Lobe", "Mean Age"
    ]

    train_tab = train_summary[[c for c in cols_to_show if c in train_summary.columns]].copy()
    test_tab = test_summary[[c for c in cols_to_show if c in test_summary.columns]].copy()

    # Round values.
    for df in [train_tab, test_tab]:
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(2)

    fig, axes = plt.subplots(2, 1, figsize=(14, max(4, (len(train_tab) + 2) * 1.5)))

    for ax, tab, title in [
        (axes[0], train_tab, "Train Set"),
        (axes[1], test_tab, "Test Set"),
    ]:
        ax.axis("off")
        ax.set_title(f"{title}", fontsize=12, fontweight="bold", loc="left")
        col_labels = [cols_display[cols_to_show.index(c)] if c in cols_to_show else c for c in tab.columns]
        table = ax.table(
            cellText=tab.values,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

    fig.suptitle(f"{dataset.upper()} Cluster Summary: Train vs Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary table: %s", output_path)


# ── Main orchestration ────────────────────────────────────────────────
def process_dataset(
    dataset: str,
    step4_dir: Path,
    step4b_dir: Path,
    step5_dir: Path,
    step6_dir: Path,
    step7_dir: Path,
    output_dir: Path,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step8.log", log_level)
    logger.info("Starting Step 8 visualization for dataset=%s", dataset)

    if plt is None or sns is None:
        logger.error("matplotlib/seaborn not available; cannot generate visualizations.")
        return

    # Load metadata files.
    step4_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = step5_dir / dataset / f"{dataset}_step5_selection.json"
    step6_meta_path = step6_dir / dataset / f"{dataset}_step6_metadata.json"

    for path, name in [
        (step4_meta_path, "Step 4"),
        (step5_selection_path, "Step 5"),
        (step6_meta_path, "Step 6"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} metadata not found: {path}")

    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)
    step6_meta = load_json(step6_meta_path)

    id_col = step4_meta.get("id_column", "patient_id")
    os_col = step4_meta.get("os_column")
    feature_cols = step5_selection.get("feature_columns", [])

    # Load dataframes.
    train_master_df = pd.read_csv(resolve_path(step4_meta["outputs"]["train_master"]))
    test_master_df = pd.read_csv(resolve_path(step4_meta["outputs"]["test_master"]))
    train_labels_df = pd.read_csv(resolve_path(step5_selection["outputs"]["train_cluster_labels"]))
    train_summary_df = pd.read_csv(resolve_path(step6_meta["outputs"]["cluster_summary"]))

    # Merge labels into train master.
    train_merged = train_master_df.merge(train_labels_df, on=id_col, how="inner")

    # Load train features (prefer step4b if available).
    step4b_meta_path = step4b_dir / dataset / f"{dataset}_step4b_metadata.json"
    if step4b_meta_path.exists():
        step4b_meta = load_json(step4b_meta_path)
        train_features_df = pd.read_csv(Path(step4b_meta["outputs"]["train_features"]))
        viz_feature_cols = step4b_meta.get("selected_feature_columns", feature_cols)
    else:
        train_features_df = pd.read_csv(resolve_path(step4_meta["outputs"]["train_features"]))
        viz_feature_cols = feature_cols

    available_viz_features = [c for c in viz_feature_cols if c in train_features_df.columns]

    # Try loading Step 7 test data.
    test_summary_df = None
    test_labeled_df = None
    step7_meta_path = step7_dir / dataset / f"{dataset}_step7_metadata.json"
    if step7_meta_path.exists():
        step7_meta = load_json(step7_meta_path)
        test_summary_path = resolve_path(step7_meta["outputs"]["cluster_summary_test"])
        test_assigned_path = resolve_path(step7_meta["outputs"]["test_with_assigned_clusters"])
        if test_summary_path.exists():
            test_summary_df = pd.read_csv(test_summary_path)
        if test_assigned_path.exists():
            test_labeled_df = pd.read_csv(test_assigned_path)
    else:
        logger.warning("Step 7 metadata not found; test-set visualizations will be limited.")

    # Column resolution.
    cols = step6_meta.get("columns_used", {})
    censor_col = cols.get("censor")
    survival_status_col = cols.get("survival_status")
    lobe_col = cols.get("dominant_lobe")

    cluster_col = "cluster_label"
    dataset_out.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Feature heatmap ──
    try:
        heatmap_features = [c for c in available_viz_features if c in train_merged.columns]
        if heatmap_features:
            plot_feature_heatmap(
                train_labeled_df=train_merged,
                cluster_col=cluster_col,
                feature_cols=heatmap_features,
                dataset=dataset,
                output_path=dataset_out / f"{dataset}_step8_feature_heatmap.png",
                logger=logger,
            )
    except Exception:
        logger.exception("Feature heatmap failed.")

    # ── Figure 2: Paired KM ──
    try:
        if os_col and test_labeled_df is not None:
            plot_paired_km(
                train_df=train_merged,
                test_df=test_labeled_df,
                cluster_col=cluster_col,
                os_col=os_col,
                censor_col=censor_col,
                survival_status_col=survival_status_col,
                dataset=dataset,
                output_path=dataset_out / f"{dataset}_step8_km_paired.png",
                logger=logger,
            )
    except Exception:
        logger.exception("Paired KM plot failed.")

    # ── Figure 3: Lobe distribution ──
    try:
        if lobe_col and lobe_col in train_merged.columns:
            plot_lobe_distribution(
                train_df=train_merged,
                cluster_col=cluster_col,
                lobe_col=lobe_col,
                dataset=dataset,
                output_path=dataset_out / f"{dataset}_step8_lobe_distribution.png",
                logger=logger,
            )
    except Exception:
        logger.exception("Lobe distribution plot failed.")

    # ── Figure 4: UMAP ──
    try:
        # Align labels with features.
        features_with_labels = train_features_df.merge(train_labels_df, on=id_col, how="inner")
        if available_viz_features:
            plot_umap_clusters(
                train_features_df=features_with_labels,
                labels=features_with_labels[cluster_col],
                id_col=id_col,
                feature_cols=available_viz_features,
                dataset=dataset,
                output_path=dataset_out / f"{dataset}_step8_umap_clusters.png",
                logger=logger,
            )
    except Exception:
        logger.exception("UMAP plot failed.")

    # ── Figure 5: Summary table ──
    try:
        if test_summary_df is not None:
            plot_summary_table(
                train_summary=train_summary_df,
                test_summary=test_summary_df,
                dataset=dataset,
                output_path=dataset_out / f"{dataset}_step8_summary_table.png",
                logger=logger,
            )
    except Exception:
        logger.exception("Summary table failed.")

    logger.info("Step 8 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 8: Generate publication-quality visualizations."
    )
    parser.add_argument("--datasets", nargs="+", default=["ucsf"], choices=["ucsf", "upenn"])
    parser.add_argument("--step4-dir", type=Path, default=DEFAULT_STEP4_DIR)
    parser.add_argument("--step4b-dir", type=Path, default=DEFAULT_STEP4B_DIR)
    parser.add_argument("--step5-dir", type=Path, default=DEFAULT_STEP5_DIR)
    parser.add_argument("--step6-dir", type=Path, default=DEFAULT_STEP6_DIR)
    parser.add_argument("--step7-dir", type=Path, default=DEFAULT_STEP7_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            step4_dir=args.step4_dir,
            step4b_dir=args.step4b_dir,
            step5_dir=args.step5_dir,
            step6_dir=args.step6_dir,
            step7_dir=args.step7_dir,
            output_dir=args.output_dir,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
