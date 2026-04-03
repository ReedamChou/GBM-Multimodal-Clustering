from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from step6_characterize_clusters import build_event_observed, compute_cluster_summary, first_present, pick_high_risk_cluster
from step7_validate_test_set import assign_to_nearest_centroid, build_cluster_proportion_comparison, build_high_risk_validation

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


def fit_spectral_labels(
    x: pd.DataFrame,
    k: int,
    random_state: int,
    n_neighbors: int,
) -> np.ndarray:
    n_samples = x.shape[0]
    nn = min(max(2, n_neighbors), n_samples - 1)
    model = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=nn,
        assign_labels="kmeans",
        random_state=random_state,
    )
    return model.fit_predict(x)


def within_cluster_dispersion(x: pd.DataFrame, labels: np.ndarray) -> float:
    x_np = x.to_numpy(dtype=float)
    total = 0.0
    for cluster_value in np.unique(labels):
        mask = labels == cluster_value
        cluster_points = x_np[mask]
        if cluster_points.shape[0] == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        total += float(((cluster_points - centroid) ** 2).sum())
    return total


def compute_gap_statistic(
    x: pd.DataFrame,
    labels: np.ndarray,
    k: int,
    random_state: int,
    n_neighbors: int,
    n_refs: int,
) -> tuple[float, float]:
    actual_dispersion = max(within_cluster_dispersion(x, labels), 1e-12)
    ref_logs: list[float] = []

    mins = x.min(axis=0).to_numpy(dtype=float)
    maxs = x.max(axis=0).to_numpy(dtype=float)
    rng = np.random.default_rng(random_state + (k * 1000))

    for ref_idx in range(n_refs):
        ref_np = rng.uniform(mins, maxs, size=x.shape)
        ref_df = pd.DataFrame(ref_np, columns=x.columns)
        ref_labels = fit_spectral_labels(ref_df, k=k, random_state=random_state + ref_idx + 17, n_neighbors=n_neighbors)
        ref_dispersion = max(within_cluster_dispersion(ref_df, ref_labels), 1e-12)
        ref_logs.append(float(np.log(ref_dispersion)))

    gap = float(np.mean(ref_logs) - np.log(actual_dispersion))
    if len(ref_logs) > 1:
        sk = float(np.sqrt(1.0 + (1.0 / len(ref_logs))) * np.std(ref_logs, ddof=1))
    else:
        sk = 0.0
    return gap, sk


def compute_bootstrap_stability(
    x: pd.DataFrame,
    full_labels: np.ndarray,
    k: int,
    random_state: int,
    n_neighbors: int,
    n_resamples: int,
    sample_fraction: float,
) -> tuple[float | None, float | None]:
    n_rows = x.shape[0]
    sample_size = max(k + 1, int(round(n_rows * sample_fraction)))
    sample_size = min(sample_size, n_rows)
    rng = np.random.default_rng(random_state + (k * 100))
    ari_values: list[float] = []

    for resample_idx in range(n_resamples):
        sample_indices = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        sample_x = x.iloc[sample_indices].reset_index(drop=True)
        subset_labels = fit_spectral_labels(
            sample_x,
            k=k,
            random_state=random_state + resample_idx + 101,
            n_neighbors=n_neighbors,
        )
        ari = float(adjusted_rand_score(full_labels[sample_indices], subset_labels))
        ari_values.append(ari)

    if not ari_values:
        return None, None
    return float(np.mean(ari_values)), float(np.median(ari_values))


def compute_logrank_p(
    df: pd.DataFrame,
    cluster_col: str,
    os_col: str | None,
    censor_col: str | None,
    survival_status_col: str | None,
) -> float | None:
    if os_col is None or os_col not in df.columns:
        return None

    durations = pd.to_numeric(df[os_col], errors="coerce")
    events = build_event_observed(df, censor_col, survival_status_col)
    groups = pd.to_numeric(df[cluster_col], errors="coerce")

    valid = durations.notna() & events.notna() & groups.notna()
    if valid.sum() < 10:
        return None
    if groups[valid].astype(int).nunique() < 2:
        return None

    result = multivariate_logrank_test(
        durations[valid],
        groups[valid].astype(int),
        events[valid].astype(int),
    )
    return float(result.p_value)


def find_analysis_columns(df: pd.DataFrame, step4_meta: dict) -> dict[str, str | None]:
    return {
        "os": step4_meta.get("os_column"),
        "censor": first_present(df.columns, ["1-dead 0-alive", "Survival_Censor", "survival_censor"]),
        "survival_status": first_present(df.columns, ["Survival_Status", "survival_status"]),
        "mgmt": first_present(df.columns, ["mgmt_bin", "MGMT_bin", "MGMT status"]),
        "idh": first_present(df.columns, ["idh_bin", "IDH_bin", "IDH"]),
        "global_nc_en_ratio": first_present(df.columns, ["global_nc_en_ratio"]),
        "global_ed_en_ratio": first_present(df.columns, ["global_ed_en_ratio"]),
        "tumor_burden_index": first_present(df.columns, ["tumor_burden_index"]),
        "age": first_present(df.columns, ["Age at MRI", "Age_at_scan_years", "age", "Age"]),
        "dominant_lobe": first_present(df.columns, ["dominant_lobe_clean", "dominant_brain_lobe", "dominant_lobe"]),
    }


def build_centroids(x: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    x_with_cluster = x.copy()
    x_with_cluster["cluster_label"] = labels.astype(int)
    centroids = x_with_cluster.groupby("cluster_label")[feature_cols].mean().reset_index()
    cluster_sizes = x_with_cluster["cluster_label"].value_counts().sort_index().rename("cluster_size")
    return centroids.merge(cluster_sizes, left_on="cluster_label", right_index=True, how="left")


def safe_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmax == vmin:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    return (values - vmin) / (vmax - vmin)


def p_value_strength(p_value: float | None, max_strength: float = 8.0) -> float:
    if p_value is None or pd.isna(p_value):
        return 0.0
    return float(min(max_strength, max(0.0, -np.log10(max(float(p_value), 1e-12)))))


def is_lowest_median_os(summary_df: pd.DataFrame, cluster_label: int) -> bool:
    if summary_df.empty or "median_os" not in summary_df.columns:
        return False
    tmp = summary_df[["cluster_label", "median_os"]].copy()
    tmp["median_os"] = pd.to_numeric(tmp["median_os"], errors="coerce")
    tmp = tmp.dropna(subset=["median_os"])
    if tmp.empty:
        return False
    lowest_cluster = int(tmp.sort_values(["median_os", "cluster_label"], ascending=[True, True]).iloc[0]["cluster_label"])
    return lowest_cluster == int(cluster_label)


def compute_biological_interpretability_score(high_risk_validation: dict) -> float:
    scores: list[float] = []

    proportion = high_risk_validation.get("cluster_proportion_check") or {}
    pct_diff = proportion.get("pct_point_diff")
    if pct_diff is not None and not pd.isna(pct_diff):
        scores.append(max(0.0, 1.0 - (min(abs(float(pct_diff)), 10.0) / 10.0)))

    lobe_match = high_risk_validation.get("dominant_lobe_matches_train")
    if lobe_match is not None:
        scores.append(1.0 if bool(lobe_match) else 0.0)

    temporal_expected = bool(high_risk_validation.get("temporal_replication_expected"))
    temporal_observed = high_risk_validation.get("temporal_replication_observed")
    if temporal_expected:
        scores.append(1.0 if bool(temporal_observed) else 0.0)

    test_signature = high_risk_validation.get("test_signature") or {}
    nc_rank = test_signature.get("nc_en_rank_among_test_clusters")
    if nc_rank is not None and not pd.isna(nc_rank):
        nc_rank = int(nc_rank)
        if nc_rank == 1:
            scores.append(1.0)
        elif nc_rank == 2:
            scores.append(0.5)
        else:
            scores.append(0.0)

    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate_k_values(
    dataset: str,
    x_train: pd.DataFrame,
    train_master_df: pd.DataFrame,
    test_master_df: pd.DataFrame,
    test_features_df: pd.DataFrame,
    id_col: str,
    feature_cols: list[str],
    k_values: list[int],
    random_state: int,
    n_neighbors: int,
    gap_refs: int,
    stability_resamples: int,
    stability_sample_fraction: float,
    logger: logging.Logger,
    step4_meta: dict,
) -> tuple[pd.DataFrame, dict[int, dict], dict[str, object]]:
    results: list[dict] = []
    artifacts: dict[int, dict] = {}

    n_samples = x_train.shape[0]
    if n_samples < 3:
        raise ValueError("Need at least 3 rows for clustering.")

    analysis_cols_train = find_analysis_columns(train_master_df, step4_meta)
    analysis_cols_test = find_analysis_columns(test_master_df, step4_meta)

    for k in sorted(set(k_values)):
        row = {
            "k": k,
            "status": "ok",
        }

        try:
            if k < 2:
                raise ValueError("k must be at least 2")
            if k >= n_samples:
                raise ValueError("k must be smaller than number of train rows")

            labels = fit_spectral_labels(x_train, k=k, random_state=random_state, n_neighbors=n_neighbors)
            unique_clusters = int(np.unique(labels).shape[0])
            if unique_clusters < 2:
                raise RuntimeError("spectral clustering produced a single cluster")

            row["clusters_found"] = unique_clusters
            row["silhouette"] = float(silhouette_score(x_train, labels, metric="euclidean"))
            gap_value, gap_sk = compute_gap_statistic(
                x=x_train,
                labels=labels,
                k=k,
                random_state=random_state,
                n_neighbors=n_neighbors,
                n_refs=gap_refs,
            )
            row["gap_statistic"] = gap_value
            row["gap_standard_error"] = gap_sk

            stability_mean, stability_median = compute_bootstrap_stability(
                x=x_train,
                full_labels=labels,
                k=k,
                random_state=random_state,
                n_neighbors=n_neighbors,
                n_resamples=stability_resamples,
                sample_fraction=stability_sample_fraction,
            )
            row["bootstrap_ari_mean"] = stability_mean
            row["bootstrap_ari_median"] = stability_median

            labels_df = pd.DataFrame({id_col: train_master_df[id_col].values, "cluster_label": labels.astype(int)})
            merged_train = train_master_df.merge(labels_df, on=id_col, how="inner")

            train_summary = compute_cluster_summary(
                df=merged_train,
                cluster_col="cluster_label",
                os_col=analysis_cols_train["os"],
                mgmt_col=analysis_cols_train["mgmt"],
                idh_col=analysis_cols_train["idh"],
                nc_en_col=analysis_cols_train["global_nc_en_ratio"],
                ed_en_col=analysis_cols_train["global_ed_en_ratio"],
                tbi_col=analysis_cols_train["tumor_burden_index"],
                age_col=analysis_cols_train["age"],
                lobe_col=analysis_cols_train["dominant_lobe"],
            )
            high_risk_cluster, high_risk_label = pick_high_risk_cluster(train_summary)
            row["high_risk_cluster"] = int(high_risk_cluster)
            row["high_risk_label"] = high_risk_label

            train_logrank_p = compute_logrank_p(
                df=merged_train,
                cluster_col="cluster_label",
                os_col=analysis_cols_train["os"],
                censor_col=analysis_cols_train["censor"],
                survival_status_col=analysis_cols_train["survival_status"],
            )
            row["train_logrank_p"] = train_logrank_p
            row["train_high_risk_lowest_os"] = is_lowest_median_os(train_summary, high_risk_cluster)

            centroids = build_centroids(x_train, labels, feature_cols)
            x_test = test_features_df[feature_cols].copy()
            for col in feature_cols:
                x_test[col] = pd.to_numeric(x_test[col], errors="coerce").fillna(float(x_train[col].median(skipna=True)))

            assigned_labels, nearest_distances = assign_to_nearest_centroid(x_test, centroids, feature_cols)
            assignments_df = pd.DataFrame(
                {
                    id_col: test_features_df[id_col].values,
                    "cluster_label": assigned_labels.astype(int),
                    "nearest_centroid_distance": nearest_distances,
                }
            )
            merged_test = test_master_df.merge(assignments_df, on=id_col, how="inner")

            test_summary = compute_cluster_summary(
                df=merged_test,
                cluster_col="cluster_label",
                os_col=analysis_cols_test["os"],
                mgmt_col=analysis_cols_test["mgmt"],
                idh_col=analysis_cols_test["idh"],
                nc_en_col=analysis_cols_test["global_nc_en_ratio"],
                ed_en_col=analysis_cols_test["global_ed_en_ratio"],
                tbi_col=analysis_cols_test["tumor_burden_index"],
                age_col=analysis_cols_test["age"],
                lobe_col=analysis_cols_test["dominant_lobe"],
            )
            proportions_df = build_cluster_proportion_comparison(
                train_labels_df=labels_df,
                test_labels_df=assignments_df,
                cluster_col="cluster_label",
            )
            test_logrank_p = compute_logrank_p(
                df=merged_test,
                cluster_col="cluster_label",
                os_col=analysis_cols_test["os"],
                censor_col=analysis_cols_test["censor"],
                survival_status_col=analysis_cols_test["survival_status"],
            )
            row["test_logrank_p"] = test_logrank_p
            row["test_high_risk_lowest_os"] = is_lowest_median_os(test_summary, high_risk_cluster)

            high_risk_validation = build_high_risk_validation(
                train_summary_df=train_summary,
                test_summary_df=test_summary,
                proportions_df=proportions_df,
                high_risk_cluster=high_risk_cluster,
                high_risk_label=high_risk_label,
            )

            row["biological_interpretability"] = compute_biological_interpretability_score(high_risk_validation)
            row["dominant_lobe_matches_train"] = high_risk_validation.get("dominant_lobe_matches_train")
            row["temporal_replication_expected"] = high_risk_validation.get("temporal_replication_expected")
            row["temporal_replication_observed"] = high_risk_validation.get("temporal_replication_observed")

            proportion_check = high_risk_validation.get("cluster_proportion_check") or {}
            test_signature = high_risk_validation.get("test_signature") or {}
            row["high_risk_pct_point_diff"] = proportion_check.get("pct_point_diff")
            row["high_risk_within_10_pct_points"] = proportion_check.get("within_10_pct_points")
            row["test_high_risk_nc_en_rank"] = test_signature.get("nc_en_rank_among_test_clusters")

            artifacts[k] = {
                "labels": labels.astype(int),
                "centroids": centroids,
                "train_summary": train_summary,
                "test_summary": test_summary,
                "proportions": proportions_df,
                "high_risk_cluster": int(high_risk_cluster),
                "high_risk_label": high_risk_label,
                "high_risk_validation": high_risk_validation,
            }
            logger.info(
                "k=%s silhouette=%.4f gap=%.4f stability=%.4f train_p=%s test_p=%s bio=%.3f",
                k,
                row["silhouette"],
                row["gap_statistic"],
                row["bootstrap_ari_median"] if row["bootstrap_ari_median"] is not None else float("nan"),
                row["train_logrank_p"],
                row["test_logrank_p"],
                row["biological_interpretability"],
            )
        except Exception as exc:
            row["status"] = f"error: {exc}"
            row["clusters_found"] = 0
            logger.exception("Clustering failed for k=%s on dataset=%s", k, dataset)

        results.append(row)

    result_df = pd.DataFrame(results).sort_values("k").reset_index(drop=True)
    valid_df = result_df[result_df["status"] == "ok"].copy()
    if valid_df.empty:
        raise RuntimeError(f"{dataset}: no valid clustering result was produced for provided k values.")

    valid_df["silhouette_norm"] = safe_normalize(valid_df["silhouette"])
    valid_df["gap_norm"] = safe_normalize(valid_df["gap_statistic"])
    valid_df["stability_norm"] = safe_normalize(valid_df["bootstrap_ari_median"])
    valid_df["train_survival_strength"] = valid_df["train_logrank_p"].map(p_value_strength)
    valid_df["test_survival_strength"] = valid_df["test_logrank_p"].map(p_value_strength)
    valid_df["train_survival_norm"] = safe_normalize(valid_df["train_survival_strength"])
    valid_df["test_survival_norm"] = safe_normalize(valid_df["test_survival_strength"])

    valid_df["test_high_risk_lowest_score"] = valid_df["test_high_risk_lowest_os"].fillna(False).astype(bool).astype(float)
    valid_df["survival_consistency_score"] = (
        0.7 * valid_df["test_survival_norm"] + 0.3 * valid_df["test_high_risk_lowest_score"]
    )

    valid_df["selection_score"] = (
        0.15 * valid_df["silhouette_norm"]
        + 0.15 * valid_df["gap_norm"]
        + 0.20 * valid_df["stability_norm"]
        + 0.10 * valid_df["train_survival_norm"]
        + 0.20 * valid_df["survival_consistency_score"]
        + 0.20 * valid_df["biological_interpretability"]
    )

    result_df = result_df.merge(
        valid_df[
            [
                "k",
                "silhouette_norm",
                "gap_norm",
                "stability_norm",
                "train_survival_strength",
                "test_survival_strength",
                "train_survival_norm",
                "test_survival_norm",
                "test_high_risk_lowest_score",
                "survival_consistency_score",
                "selection_score",
            ]
        ],
        on="k",
        how="left",
    )

    selection_config = {
        "method": "multi_criteria_step5_v2",
        "weights": {
            "silhouette_norm": 0.15,
            "gap_norm": 0.15,
            "stability_norm": 0.20,
            "train_survival_norm": 0.10,
            "survival_consistency_score": 0.20,
            "biological_interpretability": 0.20,
        },
        "gap_reference_bootstraps": gap_refs,
        "stability_resamples": stability_resamples,
        "stability_sample_fraction": stability_sample_fraction,
        "selection_note": (
            "This Step 5 selector now uses the Step 4 holdout split as a model-selection validation proxy. "
            "That improves k selection, but it also means the same holdout is no longer a pristine final test set. "
            "For publication-grade final evaluation, keep an untouched external or nested test layer beyond this step."
        ),
    }
    return result_df, artifacts, selection_config


def process_dataset(
    dataset: str,
    step4_dir: Path,
    output_dir: Path,
    k_values: list[int],
    random_state: int,
    n_neighbors: int,
    gap_refs: int,
    stability_resamples: int,
    stability_sample_fraction: float,
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
    test_features_path = resolve_path(split_meta["outputs"]["test_features"])
    train_master_path = resolve_path(split_meta["outputs"]["train_master"])
    test_master_path = resolve_path(split_meta["outputs"]["test_master"])

    logger.info("Reading Step 4 metadata: %s", split_meta_path)
    logger.info("Reading train feature file: %s", train_features_path)
    logger.info("Reading test feature file: %s", test_features_path)

    df_train_features = pd.read_csv(train_features_path)
    df_test_features = pd.read_csv(test_features_path)
    df_train_master = pd.read_csv(train_master_path)
    df_test_master = pd.read_csv(test_master_path)

    if id_col not in df_train_features.columns or id_col not in df_test_features.columns:
        raise ValueError(f"{dataset}: ID column not found in Step 4 feature files: {id_col}")

    available_features = [c for c in feature_cols if c in df_train_features.columns and c in df_test_features.columns]
    if not available_features:
        raise ValueError(f"{dataset}: no Step 4 feature columns found in Step 4 train/test feature CSVs.")

    x_train, fill_medians = clean_feature_matrix(df_train_features, available_features)
    logger.info("Rows=%s | feature_count=%s", x_train.shape[0], x_train.shape[1])

    scores_df, artifacts, selection_config = evaluate_k_values(
        dataset=dataset,
        x_train=x_train,
        train_master_df=df_train_master,
        test_master_df=df_test_master,
        test_features_df=df_test_features[[id_col] + available_features].copy(),
        id_col=id_col,
        feature_cols=available_features,
        k_values=sorted(set(k_values)),
        random_state=random_state,
        n_neighbors=n_neighbors,
        gap_refs=gap_refs,
        stability_resamples=stability_resamples,
        stability_sample_fraction=stability_sample_fraction,
        logger=logger,
        step4_meta=split_meta,
    )

    valid_scores = scores_df[scores_df["status"] == "ok"].copy()
    best_row = valid_scores.sort_values(
        ["selection_score", "survival_consistency_score", "stability_norm", "silhouette_norm"],
        ascending=[False, False, False, False],
    ).iloc[0]

    best_k = int(best_row["k"])
    best_labels = artifacts[best_k]["labels"]
    best_centroids = artifacts[best_k]["centroids"]
    best_silhouette = float(best_row["silhouette"])

    logger.info(
        "Selected best_k=%s selection_score=%.4f silhouette=%.4f gap=%.4f stability=%.4f",
        best_k,
        float(best_row["selection_score"]),
        float(best_row["silhouette"]),
        float(best_row["gap_statistic"]),
        float(best_row["bootstrap_ari_median"]),
    )

    labels_df = pd.DataFrame(
        {
            id_col: df_train_features[id_col].values,
            "cluster_label": best_labels.astype(int),
        }
    )

    dataset_out.mkdir(parents=True, exist_ok=True)

    scores_out = dataset_out / f"{dataset}_step5_silhouette_scores.csv"
    eval_out = dataset_out / f"{dataset}_step5_k_evaluation.csv"
    labels_out = dataset_out / f"{dataset}_step5_train_cluster_labels.csv"
    centroids_out = dataset_out / f"{dataset}_step5_cluster_centroids.csv"
    selection_out = dataset_out / f"{dataset}_step5_selection.json"

    scores_df.to_csv(scores_out, index=False)
    scores_df.to_csv(eval_out, index=False)
    labels_df.to_csv(labels_out, index=False)
    best_centroids.to_csv(centroids_out, index=False)

    selection_meta = {
        "dataset": dataset,
        "step4_metadata": str(split_meta_path),
        "train_features_file": str(train_features_path),
        "test_features_file": str(test_features_path),
        "id_column": id_col,
        "feature_columns": available_features,
        "k_values_tested": sorted(set(k_values)),
        "best_k": best_k,
        "best_silhouette": best_silhouette,
        "best_selection_score": float(best_row["selection_score"]),
        "selection_method": selection_config["method"],
        "selection_config": selection_config,
        "selected_k_metrics": {
            "silhouette": float(best_row["silhouette"]),
            "gap_statistic": float(best_row["gap_statistic"]),
            "bootstrap_ari_median": float(best_row["bootstrap_ari_median"]),
            "train_logrank_p": float(best_row["train_logrank_p"]) if not pd.isna(best_row["train_logrank_p"]) else None,
            "test_logrank_p": float(best_row["test_logrank_p"]) if not pd.isna(best_row["test_logrank_p"]) else None,
            "biological_interpretability": float(best_row["biological_interpretability"]),
            "survival_consistency_score": float(best_row["survival_consistency_score"]),
        },
        "n_train_rows": int(x_train.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_neighbors": int(min(max(2, n_neighbors), x_train.shape[0] - 1)),
        "random_state": random_state,
        "imputation_medians": fill_medians,
        "outputs": {
            "silhouette_scores": str(scores_out),
            "k_evaluation": str(eval_out),
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
            plt.figure(figsize=(8, 5))
            plt.plot(plot_df["k"], plot_df["silhouette"], marker="o", label="Silhouette")
            plt.plot(plot_df["k"], plot_df["gap_statistic"], marker="s", label="Gap statistic")
            plt.plot(plot_df["k"], plot_df["bootstrap_ari_median"], marker="^", label="Bootstrap ARI median")
            plt.plot(plot_df["k"], plot_df["selection_score"], marker="D", label="Selection score")
            plt.title(f"{dataset.upper()} Step 5 Multi-Criteria k Evaluation")
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Metric value")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_out, dpi=160)
            plt.close()
            logger.info("Saved Step 5 evaluation plot: %s", fig_out)
        else:
            logger.warning("No valid Step 5 rows available for plotting.")
    else:
        logger.warning("matplotlib is not available; Step 5 evaluation plot was not generated.")

    logger.info("Saved Step 5 evaluation table: %s", eval_out)
    logger.info("Saved train labels: %s", labels_out)
    logger.info("Saved centroids: %s", centroids_out)
    logger.info("Saved selection metadata: %s", selection_out)
    logger.info("Step 5 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 5: run spectral clustering on Step 4 train feature tables, evaluate k with "
            "multi-criteria scoring, and save train labels plus centroids for the selected solution."
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
        "--gap-refs",
        type=int,
        default=5,
        help="Number of reference datasets for the gap statistic.",
    )
    parser.add_argument(
        "--stability-resamples",
        type=int,
        default=8,
        help="Number of train resamples used for Step 5 stability scoring.",
    )
    parser.add_argument(
        "--stability-sample-fraction",
        type=float,
        default=0.80,
        help="Fraction of train rows used for each Step 5 stability resample.",
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
            gap_refs=args.gap_refs,
            stability_resamples=args.stability_resamples,
            stability_sample_fraction=args.stability_sample_fraction,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
