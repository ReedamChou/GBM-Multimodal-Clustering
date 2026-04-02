from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_STEP5_DIR = PROJECT_ROOT / "outputs" / "step5"
DEFAULT_STEP6_DIR = PROJECT_ROOT / "outputs" / "step6"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step7"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step7.{log_file.parent.name}")
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


def first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return str(found)
    return None


def canonicalize_lobe(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series("unknown", index=series.index, dtype="string")
    out[s.str.contains("frontal", na=False)] = "frontal"
    out[s.str.contains("temporal", na=False)] = "temporal"
    out[s.str.contains("parietal", na=False)] = "parietal"
    out[s.str.contains("occipital", na=False)] = "occipital"
    return out


def safe_mode_str(series: pd.Series, default: str = "unknown") -> str:
    mode_values = series.dropna().mode()
    if mode_values.empty:
        return default
    return str(mode_values.iloc[0])


def build_event_observed(
    df: pd.DataFrame,
    censor_col: str | None,
    survival_status_col: str | None,
) -> pd.Series:
    event = pd.Series(np.nan, index=df.index, dtype=float)

    if survival_status_col and survival_status_col in df.columns:
        s = df[survival_status_col].astype("string").str.strip().str.lower()
        deceased_mask = s.str.contains("deceas|dead", na=False)
        alive_mask = s.str.contains("alive|lost", na=False)
        event.loc[deceased_mask] = 1.0
        event.loc[alive_mask] = 0.0

    if censor_col and censor_col in df.columns:
        c = pd.to_numeric(df[censor_col], errors="coerce")
        unique_vals = set(c.dropna().astype(int).unique().tolist())
        if unique_vals.issubset({0, 1}):
            event = event.fillna(c)

    return event


def compute_cluster_summary(
    df: pd.DataFrame,
    cluster_col: str,
    os_col: str | None,
    mgmt_col: str | None,
    idh_col: str | None,
    nc_en_col: str | None,
    ed_en_col: str | None,
    tbi_col: str | None,
    age_col: str | None,
    lobe_col: str | None,
) -> pd.DataFrame:
    records: list[dict] = []
    for cluster_label, grp in df.groupby(cluster_col):
        os_values = pd.to_numeric(grp[os_col], errors="coerce") if os_col and os_col in grp.columns else pd.Series(dtype=float)
        mgmt_values = pd.to_numeric(grp[mgmt_col], errors="coerce") if mgmt_col and mgmt_col in grp.columns else pd.Series(dtype=float)
        idh_values = pd.to_numeric(grp[idh_col], errors="coerce") if idh_col and idh_col in grp.columns else pd.Series(dtype=float)
        nc_en_values = pd.to_numeric(grp[nc_en_col], errors="coerce") if nc_en_col and nc_en_col in grp.columns else pd.Series(dtype=float)
        ed_en_values = pd.to_numeric(grp[ed_en_col], errors="coerce") if ed_en_col and ed_en_col in grp.columns else pd.Series(dtype=float)
        tbi_values = pd.to_numeric(grp[tbi_col], errors="coerce") if tbi_col and tbi_col in grp.columns else pd.Series(dtype=float)
        age_values = pd.to_numeric(grp[age_col], errors="coerce") if age_col and age_col in grp.columns else pd.Series(dtype=float)
        lobe_values = canonicalize_lobe(grp[lobe_col]) if lobe_col and lobe_col in grp.columns else pd.Series(["unknown"] * len(grp), index=grp.index, dtype="string")

        record = {
            "cluster_label": int(cluster_label),
            "n": int(grp.shape[0]),
            "median_os": float(os_values.median(skipna=True)) if not os_values.empty else np.nan,
            "os_iqr_q1": float(os_values.quantile(0.25)) if not os_values.empty else np.nan,
            "os_iqr_q3": float(os_values.quantile(0.75)) if not os_values.empty else np.nan,
            "mgmt_methylated_pct": float(mgmt_values.mean(skipna=True) * 100.0) if not mgmt_values.empty else np.nan,
            "idh_mutant_pct": float(idh_values.mean(skipna=True) * 100.0) if not idh_values.empty else np.nan,
            "mean_global_nc_en_ratio": float(nc_en_values.mean(skipna=True)) if not nc_en_values.empty else np.nan,
            "mean_global_ed_en_ratio": float(ed_en_values.mean(skipna=True)) if not ed_en_values.empty else np.nan,
            "mean_tumor_burden_index": float(tbi_values.mean(skipna=True)) if not tbi_values.empty else np.nan,
            "dominant_lobe_mode": safe_mode_str(lobe_values, default="unknown"),
            "mean_age": float(age_values.mean(skipna=True)) if not age_values.empty else np.nan,
        }
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "cluster_label",
                "n",
                "median_os",
                "os_iqr_q1",
                "os_iqr_q3",
                "mgmt_methylated_pct",
                "idh_mutant_pct",
                "mean_global_nc_en_ratio",
                "mean_global_ed_en_ratio",
                "mean_tumor_burden_index",
                "dominant_lobe_mode",
                "mean_age",
            ]
        )

    return pd.DataFrame(records).sort_values("cluster_label").reset_index(drop=True)


def clean_test_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    imputation_medians: dict[str, float],
) -> pd.DataFrame:
    x = df[feature_cols].copy()

    for col in feature_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        fill_value = imputation_medians.get(col, 0.0)
        if pd.isna(fill_value):
            fill_value = 0.0
        x[col] = x[col].fillna(float(fill_value))

    return x


def assign_to_nearest_centroid(
    x: pd.DataFrame,
    centroids_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    sample_matrix = x[feature_cols].to_numpy(dtype=float)
    centroid_matrix = centroids_df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if sample_matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if np.isnan(centroid_matrix).any():
        raise ValueError("Centroid matrix contains NaN values; cannot assign test patients.")

    # Vectorized Euclidean distance from every test patient to every cluster centroid.
    distances = np.sqrt(((sample_matrix[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2))
    nearest_index = np.argmin(distances, axis=1)
    assigned_labels = centroids_df.iloc[nearest_index]["cluster_label"].to_numpy(dtype=int)
    nearest_distance = distances[np.arange(distances.shape[0]), nearest_index]
    return assigned_labels, nearest_distance


def build_cluster_proportion_comparison(
    train_labels_df: pd.DataFrame,
    test_labels_df: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    train_counts = train_labels_df[cluster_col].value_counts().sort_index()
    test_counts = test_labels_df[cluster_col].value_counts().sort_index()
    all_clusters = sorted(set(train_counts.index.tolist()) | set(test_counts.index.tolist()))

    rows: list[dict] = []
    n_train = int(len(train_labels_df))
    n_test = int(len(test_labels_df))
    for cluster_label in all_clusters:
        train_n = int(train_counts.get(cluster_label, 0))
        test_n = int(test_counts.get(cluster_label, 0))
        train_pct = (100.0 * train_n / n_train) if n_train else np.nan
        test_pct = (100.0 * test_n / n_test) if n_test else np.nan
        pct_point_diff = test_pct - train_pct if not pd.isna(train_pct) and not pd.isna(test_pct) else np.nan
        rows.append(
            {
                "cluster_label": int(cluster_label),
                "train_n": train_n,
                "train_pct": train_pct,
                "test_n": test_n,
                "test_pct": test_pct,
                "pct_point_diff": pct_point_diff,
                "within_10_pct_points": abs(pct_point_diff) <= 10.0 if not pd.isna(pct_point_diff) else False,
            }
        )

    return pd.DataFrame(rows).sort_values("cluster_label").reset_index(drop=True)


def run_logrank_and_km(
    df: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    cluster_col: str,
    os_col: str | None,
    censor_col: str | None,
    survival_status_col: str | None,
    logger: logging.Logger,
) -> tuple[float | None, float | None, Path | None]:
    if os_col is None or os_col not in df.columns:
        logger.warning("OS column is missing; skipping test-set log-rank and KM plot.")
        return None, None, None

    durations = pd.to_numeric(df[os_col], errors="coerce")
    events = build_event_observed(df, censor_col, survival_status_col)
    groups = pd.to_numeric(df[cluster_col], errors="coerce")

    valid = durations.notna() & events.notna() & groups.notna()
    if valid.sum() < 10:
        logger.warning("Insufficient valid rows for survival testing; skipping test-set log-rank and KM plot.")
        return None, None, None

    unique_groups = groups[valid].astype(int).nunique()
    if unique_groups < 2:
        logger.warning("Only one cluster present in valid test rows; skipping log-rank and KM plot.")
        return None, None, None

    test_result = multivariate_logrank_test(
        durations[valid],
        groups[valid].astype(int),
        events[valid].astype(int),
    )
    logrank_stat = float(test_result.test_statistic)
    logrank_p = float(test_result.p_value)
    logger.info("Test-set log-rank statistic=%.6f p=%.6g", logrank_stat, logrank_p)

    fig_out: Path | None = None
    if plt is not None:
        fig_out = output_dir / f"{dataset}_step7_kaplan_meier_test.png"
        plt.figure(figsize=(8, 5))
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
            kmf.plot(ci_show=True)

        plt.title(f"{dataset.upper()} Test Kaplan-Meier by Assigned Cluster")
        plt.xlabel("Time")
        plt.ylabel("Survival probability")
        plt.grid(alpha=0.3)
        plt.text(0.02, 0.04, f"Log-rank p = {logrank_p:.3g}", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(fig_out, dpi=170)
        plt.close()
        logger.info("Saved Kaplan-Meier plot: %s", fig_out)
    else:
        logger.warning("matplotlib not available; Kaplan-Meier plot was skipped.")

    return logrank_stat, logrank_p, fig_out


def build_high_risk_validation(
    train_summary_df: pd.DataFrame,
    test_summary_df: pd.DataFrame,
    proportions_df: pd.DataFrame,
    high_risk_cluster: int,
    high_risk_label: str,
) -> dict:
    train_row = train_summary_df[train_summary_df["cluster_label"] == high_risk_cluster]
    test_row = test_summary_df[test_summary_df["cluster_label"] == high_risk_cluster]
    proportion_row = proportions_df[proportions_df["cluster_label"] == high_risk_cluster]

    if train_row.empty:
        raise ValueError(f"High-risk cluster {high_risk_cluster} not found in train summary.")

    train_row = train_row.iloc[0]
    test_row_dict: dict[str, object] | None = None
    if not test_row.empty:
        test_series = test_row.iloc[0]
        nc_rank = (
            test_summary_df["mean_global_nc_en_ratio"]
            .rank(ascending=False, method="min")
            .loc[test_series.name]
        )
        test_row_dict = {
            "n": int(test_series["n"]),
            "median_os": float(test_series["median_os"]) if not pd.isna(test_series["median_os"]) else None,
            "mgmt_methylated_pct": float(test_series["mgmt_methylated_pct"]) if not pd.isna(test_series["mgmt_methylated_pct"]) else None,
            "mean_global_nc_en_ratio": float(test_series["mean_global_nc_en_ratio"]) if not pd.isna(test_series["mean_global_nc_en_ratio"]) else None,
            "dominant_lobe_mode": str(test_series["dominant_lobe_mode"]),
            "nc_en_rank_among_test_clusters": int(nc_rank) if not pd.isna(nc_rank) else None,
        }

    proportion_dict: dict[str, object] | None = None
    if not proportion_row.empty:
        prop = proportion_row.iloc[0]
        proportion_dict = {
            "train_pct": float(prop["train_pct"]),
            "test_pct": float(prop["test_pct"]),
            "pct_point_diff": float(prop["pct_point_diff"]),
            "within_10_pct_points": bool(prop["within_10_pct_points"]),
        }

    expected_temporal = "temporal" in high_risk_label.lower() or str(train_row["dominant_lobe_mode"]).lower() == "temporal"
    temporal_replication = None
    lobe_match_train = None
    if test_row_dict is not None:
        test_lobe = str(test_row_dict["dominant_lobe_mode"]).lower()
        temporal_replication = (test_lobe == "temporal") if expected_temporal else None
        lobe_match_train = test_lobe == str(train_row["dominant_lobe_mode"]).lower()

    return {
        "cluster_label": int(high_risk_cluster),
        "label": high_risk_label,
        "train_signature": {
            "n": int(train_row["n"]),
            "median_os": float(train_row["median_os"]) if not pd.isna(train_row["median_os"]) else None,
            "mgmt_methylated_pct": float(train_row["mgmt_methylated_pct"]) if not pd.isna(train_row["mgmt_methylated_pct"]) else None,
            "mean_global_nc_en_ratio": float(train_row["mean_global_nc_en_ratio"]) if not pd.isna(train_row["mean_global_nc_en_ratio"]) else None,
            "dominant_lobe_mode": str(train_row["dominant_lobe_mode"]),
        },
        "test_signature": test_row_dict,
        "cluster_proportion_check": proportion_dict,
        "temporal_replication_expected": expected_temporal,
        "temporal_replication_observed": temporal_replication,
        "dominant_lobe_matches_train": lobe_match_train,
    }


def process_dataset(
    dataset: str,
    step4_dir: Path,
    step5_dir: Path,
    step6_dir: Path,
    output_dir: Path,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step7.log", log_level)
    logger.info("Starting Step 7 for dataset=%s", dataset)

    step4_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = step5_dir / dataset / f"{dataset}_step5_selection.json"
    step6_meta_path = step6_dir / dataset / f"{dataset}_step6_metadata.json"

    if not step4_meta_path.exists():
        raise FileNotFoundError(f"Missing Step 4 metadata: {step4_meta_path}")
    if not step5_selection_path.exists():
        raise FileNotFoundError(f"Missing Step 5 selection metadata: {step5_selection_path}")
    if not step6_meta_path.exists():
        raise FileNotFoundError(f"Missing Step 6 metadata: {step6_meta_path}")

    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)
    step6_meta = load_json(step6_meta_path)

    test_master_path = resolve_path(step4_meta["outputs"]["test_master"])
    test_features_path = resolve_path(step4_meta["outputs"]["test_features"])
    train_labels_path = resolve_path(step5_selection["outputs"]["train_cluster_labels"])
    centroids_path = resolve_path(step5_selection["outputs"]["cluster_centroids"])
    train_summary_path = resolve_path(step6_meta["outputs"]["cluster_summary"])

    logger.info("Reading test master table: %s", test_master_path)
    logger.info("Reading test feature table: %s", test_features_path)
    logger.info("Reading train labels: %s", train_labels_path)
    logger.info("Reading centroids: %s", centroids_path)
    logger.info("Reading Step 6 train summary: %s", train_summary_path)

    test_master_df = pd.read_csv(test_master_path)
    test_features_df = pd.read_csv(test_features_path)
    train_labels_df = pd.read_csv(train_labels_path)
    centroids_df = pd.read_csv(centroids_path)
    train_summary_df = pd.read_csv(train_summary_path)

    id_col = step5_selection.get("id_column") or step4_meta.get("id_column")
    if id_col is None or id_col not in test_master_df.columns or id_col not in test_features_df.columns:
        raise ValueError(f"{dataset}: ID column missing from Step 4 test files.")

    feature_cols = [c for c in step5_selection.get("feature_columns", []) if c in test_features_df.columns and c in centroids_df.columns]
    if not feature_cols:
        raise ValueError(f"{dataset}: no overlapping Step 5 feature columns found for test assignment.")

    logger.info("Rows=%s | feature_count=%s", len(test_features_df), len(feature_cols))

    imputation_medians = step5_selection.get("imputation_medians", {})
    x_test = clean_test_feature_matrix(test_features_df, feature_cols, imputation_medians)
    assigned_labels, nearest_distances = assign_to_nearest_centroid(x_test, centroids_df, feature_cols)

    assignments_df = pd.DataFrame(
        {
            id_col: test_features_df[id_col].values,
            "cluster_label": assigned_labels,
            "nearest_centroid_distance": nearest_distances,
        }
    )
    merged = test_master_df.merge(assignments_df, on=id_col, how="inner")
    logger.info("Assigned test rows=%s | unique clusters=%s", merged.shape[0], merged["cluster_label"].nunique())

    step6_columns = step6_meta.get("columns_used", {})
    os_col = step6_columns.get("os") if step6_columns.get("os") in merged.columns else first_present(merged.columns, ["OS", "Survival_from_surgery_days_UPDATED", "survival_days"])
    censor_col = step6_columns.get("censor") if step6_columns.get("censor") in merged.columns else first_present(merged.columns, ["1-dead 0-alive", "Survival_Censor", "survival_censor"])
    survival_status_col = step6_columns.get("survival_status") if step6_columns.get("survival_status") in merged.columns else first_present(merged.columns, ["Survival_Status", "survival_status"])
    mgmt_col = step6_columns.get("mgmt") if step6_columns.get("mgmt") in merged.columns else first_present(merged.columns, ["mgmt_bin", "MGMT_bin", "MGMT status"])
    idh_col = step6_columns.get("idh") if step6_columns.get("idh") in merged.columns else first_present(merged.columns, ["idh_bin", "IDH_bin", "IDH"])
    nc_en_col = step6_columns.get("global_nc_en_ratio") if step6_columns.get("global_nc_en_ratio") in merged.columns else first_present(merged.columns, ["global_nc_en_ratio"])
    ed_en_col = step6_columns.get("global_ed_en_ratio") if step6_columns.get("global_ed_en_ratio") in merged.columns else first_present(merged.columns, ["global_ed_en_ratio"])
    tbi_col = step6_columns.get("tumor_burden_index") if step6_columns.get("tumor_burden_index") in merged.columns else first_present(merged.columns, ["tumor_burden_index"])
    age_col = step6_columns.get("age") if step6_columns.get("age") in merged.columns else first_present(merged.columns, ["Age at MRI", "Age_at_scan_years", "age", "Age"])
    lobe_col = step6_columns.get("dominant_lobe") if step6_columns.get("dominant_lobe") in merged.columns else first_present(merged.columns, ["dominant_lobe_clean", "dominant_brain_lobe", "dominant_lobe"])

    summary_df = compute_cluster_summary(
        df=merged,
        cluster_col="cluster_label",
        os_col=os_col,
        mgmt_col=mgmt_col,
        idh_col=idh_col,
        nc_en_col=nc_en_col,
        ed_en_col=ed_en_col,
        tbi_col=tbi_col,
        age_col=age_col,
        lobe_col=lobe_col,
    )

    high_risk_info = step6_meta.get("high_risk_cluster", {})
    high_risk_cluster = int(high_risk_info["cluster_label"])
    high_risk_label = str(high_risk_info["label"])
    summary_df["is_high_risk_cluster"] = summary_df["cluster_label"] == high_risk_cluster
    summary_df["cluster_description"] = np.where(
        summary_df["cluster_label"] == high_risk_cluster,
        high_risk_label,
        "other cluster",
    )

    proportions_df = build_cluster_proportion_comparison(
        train_labels_df=train_labels_df,
        test_labels_df=assignments_df,
        cluster_col="cluster_label",
    )

    logrank_stat, logrank_p, km_fig_out = run_logrank_and_km(
        df=merged,
        dataset=dataset,
        output_dir=dataset_out,
        cluster_col="cluster_label",
        os_col=os_col,
        censor_col=censor_col,
        survival_status_col=survival_status_col,
        logger=logger,
    )

    high_risk_validation = build_high_risk_validation(
        train_summary_df=train_summary_df,
        test_summary_df=summary_df,
        proportions_df=proportions_df,
        high_risk_cluster=high_risk_cluster,
        high_risk_label=high_risk_label,
    )

    dataset_out.mkdir(parents=True, exist_ok=True)
    assigned_out = dataset_out / f"{dataset}_step7_test_with_assigned_clusters.csv"
    summary_out = dataset_out / f"{dataset}_step7_cluster_summary_test.csv"
    proportions_out = dataset_out / f"{dataset}_step7_cluster_proportions_comparison.csv"
    metadata_out = dataset_out / f"{dataset}_step7_metadata.json"

    merged.to_csv(assigned_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    proportions_df.to_csv(proportions_out, index=False)

    meta = {
        "dataset": dataset,
        "step4_metadata": str(step4_meta_path),
        "step5_selection": str(step5_selection_path),
        "step6_metadata": str(step6_meta_path),
        "input_files": {
            "test_master": str(test_master_path),
            "test_features": str(test_features_path),
            "train_labels": str(train_labels_path),
            "cluster_centroids": str(centroids_path),
            "train_summary": str(train_summary_path),
        },
        "columns_used": {
            "id": id_col,
            "cluster": "cluster_label",
            "os": os_col,
            "censor": censor_col,
            "survival_status": survival_status_col,
            "mgmt": mgmt_col,
            "idh": idh_col,
            "global_nc_en_ratio": nc_en_col,
            "global_ed_en_ratio": ed_en_col,
            "tumor_burden_index": tbi_col,
            "age": age_col,
            "dominant_lobe": lobe_col,
        },
        "assignment": {
            "method": "nearest_centroid_euclidean",
            "n_test_rows": int(len(assignments_df)),
            "feature_columns": feature_cols,
        },
        "survival_validation": {
            "logrank_statistic": logrank_stat,
            "p_value_raw": logrank_p,
            "km_plot": str(km_fig_out) if km_fig_out is not None else None,
        },
        "high_risk_cluster_validation": high_risk_validation,
        "outputs": {
            "test_with_assigned_clusters": str(assigned_out),
            "cluster_summary_test": str(summary_out),
            "cluster_proportions_comparison": str(proportions_out),
            "log_file": str(dataset_out / "step7.log"),
        },
    }
    metadata_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Saved assigned test table: %s", assigned_out)
    logger.info("Saved test cluster summary: %s", summary_out)
    logger.info("Saved cluster proportion comparison: %s", proportions_out)
    logger.info("Saved metadata: %s", metadata_out)
    logger.info("Step 7 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 7: assign held-out test patients to Step 5 train centroids, "
            "then validate cluster proportions and survival separation on the test set."
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
        "--step6-dir",
        type=Path,
        default=DEFAULT_STEP6_DIR,
        help="Directory containing Step 6 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Step 7 outputs are written.",
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
            step5_dir=args.step5_dir,
            step6_dir=args.step6_dir,
            output_dir=args.output_dir,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()