from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_STEP5_DIR = PROJECT_ROOT / "outputs" / "step5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step9"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step9.{log_file.parent.name}")
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
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_feature_matrix(df: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    x = df[feature_cols].copy()
    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        fill_value = float(fill_values.get(col, 0.0))
        if np.isnan(fill_value):
            fill_value = 0.0
        x[col] = x[col].fillna(fill_value)
    return x


def collapse_bootstrap_labels(sample_indices: np.ndarray, sample_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_indices = np.unique(sample_indices)
    collapsed_labels = np.empty(unique_indices.shape[0], dtype=int)

    for i, original_idx in enumerate(unique_indices):
        labels_for_patient = sample_labels[sample_indices == original_idx]
        values, counts = np.unique(labels_for_patient.astype(int), return_counts=True)
        collapsed_labels[i] = int(values[np.argmax(counts)])

    return unique_indices.astype(int), collapsed_labels.astype(int)


def spectral_cluster_bootstrap(
    x: pd.DataFrame,
    n_clusters: int,
    n_neighbors: int,
    random_state: int,
) -> np.ndarray:
    n_samples = x.shape[0]
    nn = min(max(2, n_neighbors), n_samples - 1)
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=nn,
        assign_labels="kmeans",
        random_state=random_state,
    )
    return model.fit_predict(x)


def cluster_consensus_matrix(consensus_matrix: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    affinity = np.nan_to_num(consensus_matrix.copy(), nan=0.0)
    np.fill_diagonal(affinity, 1.0)

    try:
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=random_state,
        )
        return model.fit_predict(affinity).astype(int)
    except Exception:
        distance = 1.0 - affinity
        np.fill_diagonal(distance, 0.0)
        try:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                linkage="average",
            )
        return model.fit_predict(distance).astype(int)


def summarize_values(series: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p05": float(numeric.quantile(0.05)),
        "p95": float(numeric.quantile(0.95)),
    }


def plot_heatmap(
    matrix: np.ndarray,
    order: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    if plt is None:
        return

    ordered_matrix = matrix[np.ix_(order, order)]
    ordered_labels = labels[order]

    plt.figure(figsize=(8, 7))
    plt.imshow(ordered_matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(label="Co-clustering frequency")
    plt.title(title)
    plt.xlabel("Patients ordered by consensus cluster")
    plt.ylabel("Patients ordered by consensus cluster")

    boundaries: list[int] = []
    if ordered_labels.size > 0:
        start = 0
        for i in range(1, ordered_labels.size):
            if ordered_labels[i] != ordered_labels[i - 1]:
                boundaries.append(i)
        for boundary in boundaries:
            plt.axhline(boundary - 0.5, color="white", linewidth=0.7)
            plt.axvline(boundary - 0.5, color="white", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()


def process_dataset(
    dataset: str,
    step4_dir: Path,
    step5_dir: Path,
    output_dir: Path,
    n_bootstraps: int,
    bootstrap_fraction: float,
    random_state: int,
    n_neighbors_override: int | None,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step9.log", log_level)
    logger.info("Starting Step 9 cluster stability for dataset=%s", dataset)

    step4_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = step5_dir / dataset / f"{dataset}_step5_selection.json"
    if not step4_meta_path.exists():
        raise FileNotFoundError(f"Missing Step 4 metadata: {step4_meta_path}")
    if not step5_selection_path.exists():
        raise FileNotFoundError(f"Missing Step 5 selection metadata: {step5_selection_path}")

    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)

    train_features_path = resolve_path(step4_meta["outputs"]["train_features"])
    train_labels_path = resolve_path(step5_selection["outputs"]["train_cluster_labels"])
    id_col = step5_selection.get("id_column") or step4_meta.get("id_column")
    feature_cols = [c for c in step5_selection.get("feature_columns", [])]
    best_k = int(step5_selection["best_k"])
    n_neighbors = int(n_neighbors_override or step5_selection.get("n_neighbors", 10))
    fill_values = step5_selection.get("imputation_medians", {})

    logger.info("Reading train features: %s", train_features_path)
    logger.info("Reading Step 5 labels: %s", train_labels_path)
    train_features_df = pd.read_csv(train_features_path)
    train_labels_df = pd.read_csv(train_labels_path)

    merged = train_features_df[[id_col] + feature_cols].merge(train_labels_df[[id_col, "cluster_label"]], on=id_col, how="inner")
    if merged.empty:
        raise RuntimeError(f"{dataset}: no overlap between train features and Step 5 labels.")

    x = clean_feature_matrix(merged, feature_cols, fill_values)
    baseline_labels = pd.to_numeric(merged["cluster_label"], errors="coerce").astype(int).to_numpy()
    patient_ids = merged[id_col].astype("string").to_numpy()
    n_rows = x.shape[0]
    bootstrap_size = max(2, int(round(n_rows * bootstrap_fraction)))

    logger.info("n_rows=%s best_k=%s n_bootstraps=%s bootstrap_size=%s", n_rows, best_k, n_bootstraps, bootstrap_size)

    rng = np.random.default_rng(random_state)
    pair_sample_counts = np.zeros((n_rows, n_rows), dtype=np.int32)
    pair_same_counts = np.zeros((n_rows, n_rows), dtype=np.int32)
    presence_counts = np.zeros(n_rows, dtype=np.int32)
    assignment_runs: list[np.ndarray] = []
    bootstrap_rows: list[dict] = []

    for bootstrap_idx in range(n_bootstraps):
        sample_indices = rng.integers(0, n_rows, size=bootstrap_size, endpoint=False)
        sample_labels = spectral_cluster_bootstrap(
            x=x.iloc[sample_indices].reset_index(drop=True),
            n_clusters=best_k,
            n_neighbors=n_neighbors,
            random_state=random_state + bootstrap_idx + 1,
        )
        present_indices, collapsed_labels = collapse_bootstrap_labels(sample_indices, sample_labels)

        assignment = np.full(n_rows, -1, dtype=int)
        assignment[present_indices] = collapsed_labels
        assignment_runs.append(assignment)

        pair_sample_counts[np.ix_(present_indices, present_indices)] += 1
        presence_counts[present_indices] += 1
        for cluster_value in np.unique(collapsed_labels):
            members = present_indices[collapsed_labels == cluster_value]
            pair_same_counts[np.ix_(members, members)] += 1

        ari_vs_baseline = None
        if present_indices.shape[0] >= 2:
            ari_vs_baseline = float(adjusted_rand_score(baseline_labels[present_indices], collapsed_labels))

        bootstrap_rows.append(
            {
                "bootstrap_index": bootstrap_idx,
                "unique_patients_sampled": int(present_indices.shape[0]),
                "duplicate_fraction": float(1.0 - (present_indices.shape[0] / float(bootstrap_size))),
                "ari_vs_full_train_labels": ari_vs_baseline,
            }
        )

    sample_counts_float = pair_sample_counts.astype(float)
    consensus_matrix = np.divide(
        pair_same_counts.astype(float),
        sample_counts_float,
        out=np.zeros_like(sample_counts_float, dtype=float),
        where=sample_counts_float > 0,
    )
    np.fill_diagonal(consensus_matrix, 1.0)

    pairwise_ari_rows: list[dict] = []
    for i in range(len(assignment_runs)):
        for j in range(i + 1, len(assignment_runs)):
            common_mask = (assignment_runs[i] >= 0) & (assignment_runs[j] >= 0)
            overlap_n = int(common_mask.sum())
            if overlap_n < 2:
                continue
            ari = float(adjusted_rand_score(assignment_runs[i][common_mask], assignment_runs[j][common_mask]))
            pairwise_ari_rows.append(
                {
                    "bootstrap_i": i,
                    "bootstrap_j": j,
                    "overlap_n": overlap_n,
                    "pairwise_ari": ari,
                }
            )

    consensus_labels = cluster_consensus_matrix(consensus_matrix, n_clusters=best_k, random_state=random_state)
    consensus_vs_full_ari = float(adjusted_rand_score(baseline_labels, consensus_labels))

    order = np.lexsort((baseline_labels, consensus_labels))
    within_values: list[float] = []
    between_values: list[float] = []
    patient_rows: list[dict] = []
    for idx in range(n_rows):
        same_mask = consensus_labels == consensus_labels[idx]
        same_mask[idx] = False
        other_mask = consensus_labels != consensus_labels[idx]

        within_mean = float(consensus_matrix[idx, same_mask].mean()) if same_mask.any() else np.nan
        between_mean = float(consensus_matrix[idx, other_mask].mean()) if other_mask.any() else np.nan
        if not np.isnan(within_mean):
            within_values.append(within_mean)
        if not np.isnan(between_mean):
            between_values.append(between_mean)

        patient_rows.append(
            {
                id_col: patient_ids[idx],
                "baseline_cluster_label": int(baseline_labels[idx]),
                "consensus_cluster_label": int(consensus_labels[idx]),
                "bootstrap_presence_rate": float(presence_counts[idx] / float(n_bootstraps)),
                "mean_within_consensus_cluster": within_mean,
                "mean_outside_consensus_cluster": between_mean,
                "stability_gap": float(within_mean - between_mean) if not np.isnan(within_mean) and not np.isnan(between_mean) else np.nan,
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    pairwise_ari_df = pd.DataFrame(pairwise_ari_rows)
    patient_stability_df = pd.DataFrame(patient_rows)
    consensus_labels_df = pd.DataFrame(
        {
            id_col: patient_ids,
            "baseline_cluster_label": baseline_labels.astype(int),
            "consensus_cluster_label": consensus_labels.astype(int),
        }
    )

    dataset_out.mkdir(parents=True, exist_ok=True)
    bootstrap_out = dataset_out / f"{dataset}_step9_bootstrap_runs.csv"
    pairwise_ari_out = dataset_out / f"{dataset}_step9_pairwise_ari.csv"
    consensus_matrix_out = dataset_out / f"{dataset}_step9_consensus_matrix.csv"
    consensus_labels_out = dataset_out / f"{dataset}_step9_consensus_labels.csv"
    patient_stability_out = dataset_out / f"{dataset}_step9_patient_stability.csv"
    summary_out = dataset_out / f"{dataset}_step9_summary.json"
    heatmap_out = dataset_out / f"{dataset}_step9_consensus_heatmap.png"

    bootstrap_df.to_csv(bootstrap_out, index=False)
    pairwise_ari_df.to_csv(pairwise_ari_out, index=False)
    pd.DataFrame(consensus_matrix, index=patient_ids, columns=patient_ids).to_csv(consensus_matrix_out)
    consensus_labels_df.to_csv(consensus_labels_out, index=False)
    patient_stability_df.to_csv(patient_stability_out, index=False)
    plot_heatmap(
        matrix=consensus_matrix,
        order=order,
        labels=consensus_labels,
        output_path=heatmap_out,
        title=f"{dataset.upper()} Step 9 Consensus Heatmap (k={best_k})",
    )

    summary = {
        "dataset": dataset,
        "n_train_rows": int(n_rows),
        "best_k_from_step5": best_k,
        "baseline_silhouette": float(step5_selection.get("best_silhouette")),
        "n_bootstraps": int(n_bootstraps),
        "bootstrap_fraction": float(bootstrap_fraction),
        "bootstrap_size": int(bootstrap_size),
        "n_neighbors": int(n_neighbors),
        "consensus_vs_full_train_ari": consensus_vs_full_ari,
        "bootstrap_ari_vs_full_train": summarize_values(bootstrap_df["ari_vs_full_train_labels"]),
        "pairwise_bootstrap_ari": summarize_values(pairwise_ari_df["pairwise_ari"]) if not pairwise_ari_df.empty else summarize_values(pd.Series(dtype=float)),
        "consensus_strength": {
            "mean_within_cluster": float(np.nanmean(within_values)) if within_values else None,
            "mean_between_cluster": float(np.nanmean(between_values)) if between_values else None,
            "mean_stability_gap": float(np.nanmean(patient_stability_df["stability_gap"])) if not patient_stability_df.empty else None,
        },
        "cluster_sizes": {
            "baseline": {str(int(k)): int(v) for k, v in pd.Series(baseline_labels).value_counts().sort_index().items()},
            "consensus": {str(int(k)): int(v) for k, v in pd.Series(consensus_labels).value_counts().sort_index().items()},
        },
        "outputs": {
            "bootstrap_runs": str(bootstrap_out),
            "pairwise_ari": str(pairwise_ari_out),
            "consensus_matrix": str(consensus_matrix_out),
            "consensus_labels": str(consensus_labels_out),
            "patient_stability": str(patient_stability_out),
            "consensus_heatmap": str(heatmap_out) if plt is not None else None,
            "log_file": str(dataset_out / "step9.log"),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Consensus vs full-train ARI=%.4f", consensus_vs_full_ari)
    if not pairwise_ari_df.empty:
        logger.info("Pairwise bootstrap ARI median=%.4f", float(pairwise_ari_df["pairwise_ari"].median()))
    logger.info("Saved Step 9 outputs to %s", dataset_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 9: quantify train-set cluster stability using bootstrap resampling, "
            "consensus clustering, pairwise adjusted Rand indices, and co-clustering heatmaps."
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
        "--step5-dir",
        type=Path,
        default=DEFAULT_STEP5_DIR,
        help="Directory containing Step 5 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Step 9 outputs will be written.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=100,
        help="Number of bootstrap clustering runs.",
    )
    parser.add_argument(
        "--bootstrap-fraction",
        type=float,
        default=1.0,
        help="Fraction of train rows sampled with replacement for each bootstrap run.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=None,
        help="Optional override for spectral clustering nearest neighbors.",
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

    if args.n_bootstraps < 2:
        raise ValueError("--n-bootstraps must be at least 2.")
    if not (0.0 < args.bootstrap_fraction <= 1.0):
        raise ValueError("--bootstrap-fraction must be in (0, 1].")

    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            step4_dir=args.step4_dir,
            step5_dir=args.step5_dir,
            output_dir=args.output_dir,
            n_bootstraps=args.n_bootstraps,
            bootstrap_fraction=args.bootstrap_fraction,
            random_state=args.random_state,
            n_neighbors_override=args.n_neighbors,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
