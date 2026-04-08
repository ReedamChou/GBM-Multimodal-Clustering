from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.statistics import multivariate_logrank_test

from pipeline_preprocessing import canonicalize_lobe, first_present_column, safe_mode


def first_present(columns: pd.Index, candidates: list[str]) -> str | None:
    return first_present_column(columns, candidates)


def normalize_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)

    vmin = float(valid.min())
    vmax = float(valid.max())
    if vmax == vmin:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    return (values - vmin) / (vmax - vmin)


def build_event_observed(
    df: pd.DataFrame,
    censor_col: str | None,
    survival_status_col: str | None,
) -> pd.Series:
    event = pd.Series(np.nan, index=df.index, dtype=float)

    if survival_status_col and survival_status_col in df.columns:
        status = df[survival_status_col].astype("string").str.strip().str.lower()
        deceased_mask = status.str.contains("deceas|dead", na=False)
        alive_mask = status.str.contains("alive|lost", na=False)
        event.loc[deceased_mask] = 1.0
        event.loc[alive_mask] = 0.0

    if censor_col and censor_col in df.columns:
        censor = pd.to_numeric(df[censor_col], errors="coerce")
        unique_vals = set(censor.dropna().astype(int).unique().tolist())
        if unique_vals.issubset({0, 1}):
            event = event.fillna(censor)

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
        if lobe_col and lobe_col in grp.columns:
            lobe_values = canonicalize_lobe(grp[lobe_col]).fillna("unknown")
        else:
            lobe_values = pd.Series(["unknown"] * len(grp), index=grp.index, dtype="string")

        records.append(
            {
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
                "dominant_lobe_mode": str(safe_mode(lobe_values.dropna(), default="unknown")),
                "mean_age": float(age_values.mean(skipna=True)) if not age_values.empty else np.nan,
            }
        )

    columns = [
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
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records, columns=columns).sort_values("cluster_label").reset_index(drop=True)


def pick_high_risk_cluster(summary_df: pd.DataFrame) -> tuple[int, str]:
    temp_bonus = (summary_df["dominant_lobe_mode"].astype("string").str.lower() == "temporal").astype(float) * 0.25
    os_norm = normalize_minmax(summary_df["median_os"])
    nc_norm = normalize_minmax(summary_df["mean_global_nc_en_ratio"])
    mgmt_norm = normalize_minmax(summary_df["mgmt_methylated_pct"])

    risk_score = (1.0 - os_norm) + nc_norm + (1.0 - mgmt_norm) + temp_bonus
    best_idx = int(risk_score.idxmax())
    best_cluster = int(summary_df.loc[best_idx, "cluster_label"])
    dominant_lobe = str(summary_df.loc[best_idx, "dominant_lobe_mode"]).lower()

    if dominant_lobe == "temporal":
        label = "temporally-dominant high-necrosis subtype"
    else:
        label = "high-necrosis poor-survival subtype"
    return best_cluster, label


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


def survival_order_from_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty or "median_os" not in summary_df.columns:
        return ""

    ordered = (
        summary_df[["cluster_label", "median_os"]]
        .assign(median_os=lambda df: pd.to_numeric(df["median_os"], errors="coerce"))
        .dropna(subset=["median_os"])
        .sort_values(["median_os", "cluster_label"], ascending=[True, True])
    )
    return ">".join(str(int(v)) for v in ordered["cluster_label"].tolist())


def is_lowest_median_os(summary_df: pd.DataFrame, cluster_label: int) -> bool:
    if summary_df.empty or "median_os" not in summary_df.columns:
        return False

    ordered = (
        summary_df[["cluster_label", "median_os"]]
        .assign(median_os=lambda df: pd.to_numeric(df["median_os"], errors="coerce"))
        .dropna(subset=["median_os"])
        .sort_values(["median_os", "cluster_label"], ascending=[True, True])
    )
    if ordered.empty:
        return False
    return int(ordered.iloc[0]["cluster_label"]) == int(cluster_label)


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
        raise ValueError("Centroid matrix contains NaN values; cannot assign samples.")

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

    train_series = train_row.iloc[0]
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

    expected_temporal = "temporal" in high_risk_label.lower() or str(train_series["dominant_lobe_mode"]).lower() == "temporal"
    temporal_replication = None
    lobe_match_train = None
    if test_row_dict is not None:
        test_lobe = str(test_row_dict["dominant_lobe_mode"]).lower()
        temporal_replication = (test_lobe == "temporal") if expected_temporal else None
        lobe_match_train = test_lobe == str(train_series["dominant_lobe_mode"]).lower()

    return {
        "cluster_label": int(high_risk_cluster),
        "label": high_risk_label,
        "train_signature": {
            "n": int(train_series["n"]),
            "median_os": float(train_series["median_os"]) if not pd.isna(train_series["median_os"]) else None,
            "mgmt_methylated_pct": float(train_series["mgmt_methylated_pct"]) if not pd.isna(train_series["mgmt_methylated_pct"]) else None,
            "mean_global_nc_en_ratio": float(train_series["mean_global_nc_en_ratio"]) if not pd.isna(train_series["mean_global_nc_en_ratio"]) else None,
            "dominant_lobe_mode": str(train_series["dominant_lobe_mode"]),
        },
        "test_signature": test_row_dict,
        "cluster_proportion_check": proportion_dict,
        "temporal_replication_expected": expected_temporal,
        "temporal_replication_observed": temporal_replication,
        "dominant_lobe_matches_train": lobe_match_train,
    }
