from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step5"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step5.{log_file.parent.name}")
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


def clean_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    x = df[feature_cols].copy()
    medians: dict[str, float] = {}

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        median_val = float(x[col].median(skipna=True))
        if np.isnan(median_val):
            median_val = 0.0
        x[col] = x[col].fillna(median_val)
        medians[col] = median_val

    return x, medians


def evaluate_k_values(
    x: pd.DataFrame,
    k_values: list[int],
    random_state: int,
    n_neighbors: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    results: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}

    n_samples = x.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 rows for silhouette-based clustering.")

    for k in k_values:
        if k < 2:
            results.append({"k": k, "silhouette": np.nan, "status": "skipped_k_lt_2", "clusters_found": 0})
            continue
        if k >= n_samples:
            results.append({"k": k, "silhouette": np.nan, "status": "skipped_k_ge_n_samples", "clusters_found": 0})
            continue

        nn = min(max(2, n_neighbors), n_samples - 1)
        try:
            model = SpectralClustering(
                n_clusters=k,
                affinity="nearest_neighbors",
                n_neighbors=nn,
                assign_labels="kmeans",
                random_state=random_state,
            )
            labels = model.fit_predict(x)
            unique_clusters = int(np.unique(labels).shape[0])
            if unique_clusters < 2:
                results.append(
                    {
                        "k": k,
                        "silhouette": np.nan,
                        "status": "failed_single_cluster",
                        "clusters_found": unique_clusters,
                    }
                )
                logger.warning("k=%s produced a single cluster; skipping silhouette.", k)
                continue

            sil = float(silhouette_score(x, labels, metric="euclidean"))
            results.append(
                {
                    "k": k,
                    "silhouette": sil,
                    "status": "ok",
                    "clusters_found": unique_clusters,
                }
            )
            labels_by_k[k] = labels
            logger.info("k=%s silhouette=%.6f clusters=%s", k, sil, unique_clusters)
        except Exception as exc:
            results.append({"k": k, "silhouette": np.nan, "status": f"error: {exc}", "clusters_found": 0})
            logger.exception("Clustering failed for k=%s", k)

    result_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    return result_df, labels_by_k


def process_dataset(
    dataset: str,
    step4_dir: Path,
    output_dir: Path,
    k_values: list[int],
    random_state: int,
    n_neighbors: int,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step5.log", log_level)
    logger.info("Starting Step 5 for dataset=%s", dataset)

    split_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    if not split_meta_path.exists():
        raise FileNotFoundError(f"Step 4 metadata not found: {split_meta_path}")

    split_meta = load_json(split_meta_path)
    id_col = split_meta.get("id_column")
    feature_cols = split_meta.get("feature_columns", [])
    train_features_path = resolve_path(split_meta["outputs"]["train_features"])

    logger.info("Reading Step 4 metadata: %s", split_meta_path)
    logger.info("Reading train feature file: %s", train_features_path)

    df_train = pd.read_csv(train_features_path)
    if id_col not in df_train.columns:
        raise ValueError(f"{dataset}: ID column not found in train features: {id_col}")

    available_features = [c for c in feature_cols if c in df_train.columns]
    if not available_features:
        raise ValueError(f"{dataset}: no Step 4 feature columns found in train feature CSV.")

    x, fill_medians = clean_feature_matrix(df_train, available_features)
    logger.info("Rows=%s | feature_count=%s", x.shape[0], x.shape[1])

    scores_df, labels_by_k = evaluate_k_values(
        x=x,
        k_values=sorted(set(k_values)),
        random_state=random_state,
        n_neighbors=n_neighbors,
        logger=logger,
    )

    valid_scores = scores_df[scores_df["status"] == "ok"].copy()
    if valid_scores.empty:
        raise RuntimeError(f"{dataset}: no valid clustering result was produced for provided k values.")

    best_row = valid_scores.sort_values("silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    best_silhouette = float(best_row["silhouette"])
    best_labels = labels_by_k[best_k]

    logger.info("Selected best_k=%s with silhouette=%.6f", best_k, best_silhouette)

    labels_df = pd.DataFrame(
        {
            id_col: df_train[id_col].values,
            "cluster_label": best_labels.astype(int),
        }
    )

    x_with_cluster = x.copy()
    x_with_cluster["cluster_label"] = best_labels.astype(int)

    centroids = x_with_cluster.groupby("cluster_label")[available_features].mean().reset_index()
    cluster_sizes = x_with_cluster["cluster_label"].value_counts().sort_index().rename("cluster_size")
    centroids = centroids.merge(cluster_sizes, left_on="cluster_label", right_index=True, how="left")

    dataset_out.mkdir(parents=True, exist_ok=True)

    scores_out = dataset_out / f"{dataset}_step5_silhouette_scores.csv"
    labels_out = dataset_out / f"{dataset}_step5_train_cluster_labels.csv"
    centroids_out = dataset_out / f"{dataset}_step5_cluster_centroids.csv"
    selection_out = dataset_out / f"{dataset}_step5_selection.json"

    scores_df.to_csv(scores_out, index=False)
    labels_df.to_csv(labels_out, index=False)
    centroids.to_csv(centroids_out, index=False)

    selection_meta = {
        "dataset": dataset,
        "step4_metadata": str(split_meta_path),
        "train_features_file": str(train_features_path),
        "id_column": id_col,
        "feature_columns": available_features,
        "k_values_tested": sorted(set(k_values)),
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "n_train_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_neighbors": int(min(max(2, n_neighbors), x.shape[0] - 1)),
        "random_state": random_state,
        "imputation_medians": fill_medians,
        "outputs": {
            "silhouette_scores": str(scores_out),
            "train_cluster_labels": str(labels_out),
            "cluster_centroids": str(centroids_out),
            "log_file": str(dataset_out / "step5.log"),
        },
    }
    selection_out.write_text(json.dumps(selection_meta, indent=2), encoding="utf-8")

    if plt is not None:
        fig_out = dataset_out / f"{dataset}_step5_silhouette_elbow.png"
        plot_df = scores_df[scores_df["status"] == "ok"].copy()
        if not plot_df.empty:
            plt.figure(figsize=(7, 4.5))
            plt.plot(plot_df["k"], plot_df["silhouette"], marker="o")
            plt.title(f"{dataset.upper()} Step 5 Silhouette vs k")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Silhouette score")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_out, dpi=160)
            plt.close()
            logger.info("Saved silhouette elbow plot: %s", fig_out)
        else:
            logger.warning("No valid silhouette rows available for plotting.")
    else:
        logger.warning("matplotlib is not available; silhouette elbow plot was not generated.")

    logger.info("Saved silhouette scores: %s", scores_out)
    logger.info("Saved train labels: %s", labels_out)
    logger.info("Saved centroids: %s", centroids_out)
    logger.info("Saved selection metadata: %s", selection_out)
    logger.info("Step 5 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5: run spectral clustering on Step 4 train feature tables for k values, "
            "select best k by silhouette score, and save train labels plus centroids."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucsf", "upenn"],
        choices=["ucsf", "upenn"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--step4-dir",
        type=Path,
        default=DEFAULT_STEP4_DIR,
        help="Directory containing Step 4 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Step 5 outputs will be written.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="List of k values to evaluate.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Nearest neighbors used by spectral clustering affinity.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            step4_dir=args.step4_dir,
            output_dir=args.output_dir,
            k_values=args.k_values,
            random_state=args.random_state,
            n_neighbors=args.n_neighbors,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()