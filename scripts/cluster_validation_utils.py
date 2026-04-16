from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

from pipeline_preprocessing import canonicalize_lobe, first_present_column, safe_mode

logger = logging.getLogger(__name__)


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


def _pick_high_risk_by_cox(
    merged_df: pd.DataFrame,
    cluster_col: str,
    os_col: str,
    event_col: pd.Series,
) -> tuple[int, dict]:
    """Fit Cox PH on cluster dummies and return the cluster with the
    highest hazard ratio as the high-risk cluster.

    Returns (cluster_label, descriptive_label, cox_result_dict).
    """
    durations = pd.to_numeric(merged_df[os_col], errors="coerce")
    valid = durations.notna() & event_col.notna()
    df_cox = pd.DataFrame({
        "T": durations[valid].values,
        "E": event_col[valid].astype(int).values,
        "cluster": merged_df.loc[valid, cluster_col].astype(int).values,
    })

    if df_cox["cluster"].nunique() < 2:
        raise ValueError("Need at least 2 clusters for Cox regression.")

    # One-hot encode cluster labels (CoxPH needs dummies, baseline is the
    # cluster with the best median OS so that all HRs are >= 1).
    best_os_cluster = int(
        df_cox.groupby("cluster")["T"].median().idxmax()
    )
    dummies = pd.get_dummies(df_cox["cluster"], prefix="cluster", dtype=float)
    baseline_col = f"cluster_{best_os_cluster}"
    if baseline_col in dummies.columns:
        dummies = dummies.drop(columns=[baseline_col])

    cox_input = pd.concat(
        [df_cox[["T", "E"]].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_input, duration_col="T", event_col="E")

    # The cluster with the highest hazard ratio is the high-risk group.
    hr_series = np.exp(cph.params_)
    dummy_cols = [c for c in hr_series.index if c.startswith("cluster_")]
    # Include baseline (HR=1.0 by definition).
    hr_dict = {best_os_cluster: 1.0}
    for col in dummy_cols:
        cluster_id = int(col.replace("cluster_", ""))
        hr_dict[cluster_id] = float(hr_series[col])

    high_risk_cluster = max(hr_dict, key=hr_dict.get)

    # Build a meaningful label from the summary data.
    cox_results = {
        "method": "cox_proportional_hazards",
        "baseline_cluster": best_os_cluster,
        "hazard_ratios": hr_dict,
        "p_values": {
            col: float(cph.summary.loc[col, "p"])
            for col in dummy_cols
            if col in cph.summary.index
        },
        "concordance_index": float(cph.concordance_index_),
    }

    return int(high_risk_cluster), cox_results


def pick_high_risk_cluster(
    summary_df: pd.DataFrame,
    merged_df: pd.DataFrame | None = None,
    os_col: str | None = None,
    censor_col: str | None = None,
    survival_status_col: str | None = None,
    cluster_col: str = "cluster_label",
) -> tuple[int, str, float | None]:
    """Identify the high-risk cluster.

    When *merged_df* and survival columns are provided, uses Cox
    proportional hazards regression (data-driven).  Otherwise falls
    back to the cluster with the lowest median OS.

    Returns (cluster_label, descriptive_label, hazard_ratio).

    hazard_ratio is only available when Cox PH succeeds; otherwise None.
    """
    # --- Try Cox PH first. ---
    if merged_df is not None and os_col is not None:
        events = build_event_observed(merged_df, censor_col, survival_status_col)
        durations = pd.to_numeric(merged_df[os_col], errors="coerce")
        valid_count = (durations.notna() & events.notna()).sum()

        if valid_count >= 20 and merged_df[cluster_col].nunique() >= 2:
            try:
                high_risk_cluster, cox_results = _pick_high_risk_by_cox(
                    merged_df=merged_df,
                    cluster_col=cluster_col,
                    os_col=os_col,
                    event_col=events,
                )
                hr = cox_results["hazard_ratios"].get(high_risk_cluster, 0)
                # Get dominant lobe for this cluster from summary.
                row = summary_df[summary_df["cluster_label"] == high_risk_cluster]
                if not row.empty:
                    lobe = str(row.iloc[0].get("dominant_lobe_mode", "unknown")).lower()
                else:
                    lobe = "unknown"

                if lobe == "temporal":
                    label = f"temporally-dominant high-necrosis subtype (HR={hr:.2f})"
                else:
                    label = f"high-necrosis poor-survival subtype (HR={hr:.2f})"

                logger.info(
                    "Cox PH identified high-risk cluster=%d (HR=%.2f, method=cox)",
                    high_risk_cluster,
                    hr,
                )
                return high_risk_cluster, label, float(hr)

            except Exception as exc:
                logger.warning("Cox PH failed, falling back to median OS: %s", exc)

    # --- Fallback: cluster with lowest median OS. ---
    if "median_os" in summary_df.columns:
        os_vals = pd.to_numeric(summary_df["median_os"], errors="coerce")
        valid_rows = summary_df[os_vals.notna()]
        if not valid_rows.empty:
            best_idx = os_vals[valid_rows.index].idxmin()
            best_cluster = int(summary_df.loc[best_idx, "cluster_label"])
            lobe = str(summary_df.loc[best_idx, "dominant_lobe_mode"]).lower()

            if lobe == "temporal":
                label = "temporally-dominant high-necrosis subtype"
            else:
                label = "high-necrosis poor-survival subtype"

            logger.info(
                "Fallback identified high-risk cluster=%d (method=lowest_median_os)",
                best_cluster,
            )
            return best_cluster, label, None

    # Absolute fallback.
    return int(summary_df.iloc[0]["cluster_label"]), "uncharacterised high-risk subtype", None


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
