"""Step 5b: Consensus clustering for robust k selection evidence.

Runs as a supplementary analysis to Step 5.  For each candidate k,
re-clusters 100 bootstrap subsets and builds a consensus matrix.
The CDF area under the curve (delta-area) identifies the optimal k.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4B_DIR = PROJECT_ROOT / "outputs" / "step4b"
DEFAULT_STEP5_DIR = PROJECT_ROOT / "outputs" / "step5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step5b"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step5b.{log_file.parent.name}")
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


def build_consensus_matrix(
    x: np.ndarray,
    k: int,
    n_resamples: int,
    sample_fraction: float,
    n_neighbors: int,
    random_state: int,
    logger: logging.Logger,
) -> np.ndarray:
    """Build an N x N consensus matrix from bootstrap-resampled spectral
    clustering runs."""
    n_samples = x.shape[0]
    sample_size = max(k + 1, int(round(n_samples * sample_fraction)))
    sample_size = min(sample_size, n_samples)

    co_cluster_count = np.zeros((n_samples, n_samples), dtype=float)
    co_sample_count = np.zeros((n_samples, n_samples), dtype=float)

    rng = np.random.default_rng(random_state + k * 100)

    for resample_idx in range(n_resamples):
        indices = np.sort(rng.choice(n_samples, size=sample_size, replace=False))
        sample_x = x[indices]
        nn = min(max(2, n_neighbors), sample_x.shape[0] - 1)

        try:
            model = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=nn,
                assign_labels="kmeans",
                random_state=random_state + resample_idx,
            )
            labels = model.fit_predict(sample_x)
        except Exception:
            continue

        for i_idx, i_sample in enumerate(indices):
            for j_idx, j_sample in enumerate(indices):
                co_sample_count[i_sample, j_sample] += 1
                if labels[i_idx] == labels[j_idx]:
                    co_cluster_count[i_sample, j_sample] += 1

        if (resample_idx + 1) % 25 == 0:
            logger.info("  k=%d: completed %d/%d resamples", k, resample_idx + 1, n_resamples)

    # Avoid division by zero.
    safe_denom = np.where(co_sample_count > 0, co_sample_count, 1.0)
    consensus = co_cluster_count / safe_denom
    return consensus


def compute_cdf_area(consensus: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the CDF of consensus values and its area under the curve."""
    upper_triangle = consensus[np.triu_indices_from(consensus, k=1)]
    bins = np.linspace(0, 1, 101)
    hist, _ = np.histogram(upper_triangle, bins=bins, density=True)
    cdf = np.cumsum(hist) * (bins[1] - bins[0])
    # NumPy 2.0 removed np.trapz in favor of np.trapezoid.
    return bins[1:], cdf, float(np.trapezoid(cdf, bins[1:]))


def process_dataset(
    dataset: str,
    step4b_dir: Path,
    step5_dir: Path,
    output_dir: Path,
    k_values: list[int],
    n_resamples: int,
    sample_fraction: float,
    n_neighbors: int,
    random_state: int,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step5b.log", log_level)
    logger.info("Starting Step 5b consensus clustering for dataset=%s", dataset)

    # Load VIF-filtered features.
    step4b_meta_path = step4b_dir / dataset / f"{dataset}_step4b_metadata.json"
    if not step4b_meta_path.exists():
        raise FileNotFoundError(f"Step 4b metadata not found: {step4b_meta_path}")

    step4b_meta = load_json(step4b_meta_path)
    id_col = step4b_meta.get("id_column", "patient_id")
    feature_cols = step4b_meta.get("selected_feature_columns", [])

    train_path = Path(step4b_meta["outputs"]["train_features"])
    if not train_path.exists():
        raise FileNotFoundError(f"Step 4b train features not found: {train_path}")

    train_df = pd.read_csv(train_path)
    available_features = [c for c in feature_cols if c in train_df.columns]
    x = train_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0).values
    n_samples = x.shape[0]
    logger.info("Loaded %d samples x %d features", n_samples, len(available_features))

    # Optionally load Step 5 labels for ARI comparison.
    step5_labels = None
    step5_selection_path = step5_dir / dataset / f"{dataset}_step5_selection.json"
    step5_labels_path = step5_dir / dataset / f"{dataset}_step5_train_cluster_labels.csv"
    if step5_labels_path.exists():
        labels_df = pd.read_csv(step5_labels_path)
        if "cluster_label" in labels_df.columns:
            # Align with train_df by ID.
            step5_merged = train_df[[id_col]].merge(labels_df, on=id_col, how="inner")
            if len(step5_merged) == n_samples:
                step5_labels = step5_merged["cluster_label"].values

    # Run consensus clustering for each k.
    results: list[dict] = []
    dataset_out.mkdir(parents=True, exist_ok=True)

    for k in sorted(set(k_values)):
        if k < 2 or k >= n_samples:
            logger.warning("Skipping invalid k=%d", k)
            continue

        logger.info("Running consensus clustering for k=%d (%d resamples)...", k, n_resamples)
        consensus = build_consensus_matrix(
            x=x,
            k=k,
            n_resamples=n_resamples,
            sample_fraction=sample_fraction,
            n_neighbors=n_neighbors,
            random_state=random_state,
            logger=logger,
        )

        # Save consensus matrix.
        consensus_out = dataset_out / f"{dataset}_step5b_consensus_matrix_k{k}.csv"
        pd.DataFrame(consensus).to_csv(consensus_out, index=False)

        # CDF area.
        cdf_bins, cdf_values, cdf_area = compute_cdf_area(consensus)

        # Hierarchical clustering on the consensus matrix.
        distance_matrix = 1.0 - consensus
        np.fill_diagonal(distance_matrix, 0)
        condensed = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        linkage_matrix = linkage(condensed, method="average")
        consensus_labels = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1

        # Consensus silhouette.
        try:
            consensus_silhouette = float(silhouette_score(
                distance_matrix,
                consensus_labels,
                metric="precomputed",
            ))
        except Exception:
            consensus_silhouette = None

        # ARI with Step 5.
        ari_vs_step5 = None
        if step5_labels is not None:
            step5_k = int(np.unique(step5_labels).shape[0])
            if step5_k == k:
                ari_vs_step5 = float(adjusted_rand_score(step5_labels, consensus_labels))

        result = {
            "k": k,
            "cdf_area": cdf_area,
            "consensus_silhouette": consensus_silhouette,
            "ari_vs_step5": ari_vs_step5,
        }
        results.append(result)
        logger.info(
            "k=%d: CDF area=%.4f, consensus silhouette=%s, ARI vs step5=%s",
            k,
            cdf_area,
            f"{consensus_silhouette:.4f}" if consensus_silhouette is not None else "N/A",
            f"{ari_vs_step5:.4f}" if ari_vs_step5 is not None else "N/A",
        )

        # Consensus heatmap.
        if plt is not None and sns is not None:
            heatmap_out = dataset_out / f"{dataset}_step5b_consensus_heatmap_k{k}.png"
            fig, ax = plt.subplots(figsize=(10, 8))
            # Reorder by consensus labels for visual clarity.
            order = np.argsort(consensus_labels)
            reordered = consensus[np.ix_(order, order)]
            sns.heatmap(
                reordered,
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "Co-clustering frequency"},
            )
            ax.set_title(f"{dataset.upper()} Consensus Matrix (k={k}, {n_resamples} resamples)")
            ax.set_xlabel("Patient index")
            ax.set_ylabel("Patient index")
            plt.tight_layout()
            plt.savefig(heatmap_out, dpi=150)
            plt.close()
            logger.info("Saved consensus heatmap: %s", heatmap_out)

    # Delta-area plot (CDF areas across k values).
    result_df = pd.DataFrame(results)
    cdf_plot_out: Path | None = None
    if plt is not None and len(result_df) > 1:
        cdf_plot_out = dataset_out / f"{dataset}_step5b_cdf_plot.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: CDF area vs k.
        axes[0].plot(result_df["k"], result_df["cdf_area"], "o-", color="steelblue", linewidth=2)
        axes[0].set_title("CDF Area Under Curve vs k")
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("CDF Area")
        axes[0].grid(alpha=0.3)

        # Right: delta-area (change in CDF area).
        if len(result_df) > 1:
            delta_area = result_df["cdf_area"].diff()
            axes[1].bar(result_df["k"].iloc[1:], delta_area.iloc[1:], color="coral", edgecolor="white")
            axes[1].set_title("Delta CDF Area (change from k-1 to k)")
            axes[1].set_xlabel("Number of clusters (k)")
            axes[1].set_ylabel("Δ CDF Area")
            axes[1].grid(alpha=0.3)

        plt.suptitle(f"{dataset.upper()} Consensus Clustering k Selection", fontsize=14)
        plt.tight_layout()
        plt.savefig(cdf_plot_out, dpi=150)
        plt.close()
        logger.info("Saved CDF plot: %s", cdf_plot_out)

    # Identify the recommended k (largest delta-area drop — the "elbow").
    recommended_k = None
    if len(result_df) > 1:
        delta = result_df["cdf_area"].diff()
        # The best k is where the relative change in CDF area first becomes negligible.
        # A large positive delta means the structure improved substantially.
        delta_df = result_df.copy()
        delta_df["delta_area"] = delta
        # Pick the k before the first small delta (< 10% of max delta).
        max_delta = delta.iloc[1:].max()
        if max_delta > 0:
            small_deltas = delta_df[(delta_df["delta_area"] < 0.1 * max_delta) & (delta_df.index > 0)]
            if not small_deltas.empty:
                recommended_k = int(delta_df.loc[small_deltas.index[0] - 1, "k"]) if small_deltas.index[0] > 0 else int(result_df.iloc[-1]["k"])
            else:
                recommended_k = int(result_df.iloc[-1]["k"])

    # Summary JSON.
    summary = {
        "dataset": dataset,
        "n_resamples": n_resamples,
        "sample_fraction": sample_fraction,
        "k_values_tested": sorted(set(k_values)),
        "recommended_k": recommended_k,
        "results": results,
        "outputs": {
            "results_table": str(dataset_out / f"{dataset}_step5b_results.csv"),
            "summary_json": str(dataset_out / f"{dataset}_step5b_summary.json"),
            "cdf_plot": str(cdf_plot_out) if cdf_plot_out is not None else None,
            "log_file": str(dataset_out / "step5b.log"),
        },
    }
    summary_out = dataset_out / f"{dataset}_step5b_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    result_df.to_csv(dataset_out / f"{dataset}_step5b_results.csv", index=False)

    logger.info("Recommended k by consensus: %s", recommended_k)
    logger.info("Step 5b complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5b: Consensus clustering.  Runs bootstrap-resampled spectral "
            "clustering for multiple k values and builds consensus matrices "
            "to evaluate cluster stability."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["ucsf"], choices=["ucsf", "upenn"])
    parser.add_argument("--step4b-dir", type=Path, default=DEFAULT_STEP4B_DIR)
    parser.add_argument("--step5-dir", type=Path, default=DEFAULT_STEP5_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-resamples", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.80)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            step4b_dir=args.step4b_dir,
            step5_dir=args.step5_dir,
            output_dir=args.output_dir,
            k_values=args.k_values,
            n_resamples=args.n_resamples,
            sample_fraction=args.sample_fraction,
            n_neighbors=args.n_neighbors,
            random_state=args.random_state,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
