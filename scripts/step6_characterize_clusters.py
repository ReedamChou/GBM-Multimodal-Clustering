from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy import stats

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_STEP5_DIR = PROJECT_ROOT / "outputs" / "step5"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step6"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step6.{log_file.parent.name}")
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
    col_lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        c = col_lookup.get(candidate.lower())
        if c is not None:
            return str(c)
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


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.array(p_values, dtype=float)
    n = p.shape[0]
    if n == 0:
        return []

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(n, dtype=float)

    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        running_min = min(running_min, value)
        adjusted_ranked[i] = min(1.0, running_min)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def normalize_minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x_min = x.min(skipna=True)
    x_max = x.max(skipna=True)
    if pd.isna(x_min) or pd.isna(x_max) or x_max == x_min:
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    return (x - x_min) / (x_max - x_min)


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

    return pd.DataFrame(records).sort_values("cluster_label").reset_index(drop=True)


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
        logger.warning("OS column is missing; skipping log-rank and KM plot.")
        return None, None, None

    durations = pd.to_numeric(df[os_col], errors="coerce")
    events = build_event_observed(df, censor_col, survival_status_col)
    groups = pd.to_numeric(df[cluster_col], errors="coerce")

    valid = durations.notna() & events.notna() & groups.notna()
    if valid.sum() < 10:
        logger.warning("Insufficient valid rows for survival testing; skipping log-rank and KM plot.")
        return None, None, None

    # Use positional arguments for broad compatibility across lifelines versions.
    test_result = multivariate_logrank_test(
        durations[valid],
        groups[valid].astype(int),
        events[valid].astype(int),
    )

    logrank_stat = float(test_result.test_statistic)
    logrank_p = float(test_result.p_value)
    logger.info("Log-rank test statistic=%.6f p=%.6g", logrank_stat, logrank_p)

    fig_out: Path | None = None
    if plt is not None:
        fig_out = output_dir / f"{dataset}_step6_kaplan_meier_train.png"
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

        plt.title(f"{dataset.upper()} Train Kaplan-Meier by Cluster")
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


def run_kruskal_tests(df: pd.DataFrame, cluster_col: str, continuous_cols: list[str], logger: logging.Logger) -> pd.DataFrame:
    rows: list[dict] = []
    for col in continuous_cols:
        if col not in df.columns:
            continue
        col_numeric = pd.to_numeric(df[col], errors="coerce")
        test_df = pd.DataFrame({"cluster": df[cluster_col], "value": col_numeric}).dropna()
        if test_df.empty:
            continue

        groups = [g["value"].values for _, g in test_df.groupby("cluster") if len(g) > 0]
        if len(groups) < 2:
            continue

        try:
            stat, p = stats.kruskal(*groups)
            rows.append(
                {
                    "feature": col,
                    "statistic": float(stat),
                    "p_value_raw": float(p),
                }
            )
        except Exception:
            logger.exception("Kruskal-Wallis failed for feature=%s", col)

    return pd.DataFrame(rows).sort_values("p_value_raw", ascending=True).reset_index(drop=True)


def run_chi_square_tests(
    df: pd.DataFrame,
    cluster_col: str,
    mgmt_col: str | None,
    lobe_col: str | None,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[dict] = []

    if mgmt_col and mgmt_col in df.columns:
        mgmt = pd.to_numeric(df[mgmt_col], errors="coerce")
        table = pd.crosstab(df[cluster_col], mgmt)
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            try:
                chi2, p, dof, _ = stats.chi2_contingency(table)
                rows.append(
                    {
                        "test": "mgmt_distribution",
                        "statistic": float(chi2),
                        "dof": int(dof),
                        "p_value_raw": float(p),
                    }
                )
            except Exception:
                logger.exception("Chi-square failed for MGMT distribution.")

    if lobe_col and lobe_col in df.columns:
        lobe = canonicalize_lobe(df[lobe_col])
        table = pd.crosstab(df[cluster_col], lobe)
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            try:
                chi2, p, dof, _ = stats.chi2_contingency(table)
                rows.append(
                    {
                        "test": "dominant_lobe_distribution",
                        "statistic": float(chi2),
                        "dof": int(dof),
                        "p_value_raw": float(p),
                    }
                )
            except Exception:
                logger.exception("Chi-square failed for lobe distribution.")

    return pd.DataFrame(rows).sort_values("p_value_raw", ascending=True).reset_index(drop=True)


def apply_global_bh(
    logrank_p: float | None,
    kruskal_df: pd.DataFrame,
    chi_df: pd.DataFrame,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inferential_rows: list[dict] = []

    if logrank_p is not None:
        inferential_rows.append({"test_group": "survival", "test_name": "logrank", "p_value_raw": float(logrank_p)})

    for _, row in kruskal_df.iterrows():
        inferential_rows.append(
            {
                "test_group": "continuous",
                "test_name": str(row["feature"]),
                "p_value_raw": float(row["p_value_raw"]),
            }
        )

    for _, row in chi_df.iterrows():
        inferential_rows.append(
            {
                "test_group": "categorical",
                "test_name": str(row["test"]),
                "p_value_raw": float(row["p_value_raw"]),
            }
        )

    if not inferential_rows:
        empty_df = pd.DataFrame(columns=["test_group", "test_name", "p_value_raw", "p_value_bh", "significant_bh_0_05"])
        return logrank_p, kruskal_df, chi_df, empty_df

    inferential_df = pd.DataFrame(inferential_rows)
    inferential_df["p_value_bh"] = bh_adjust(inferential_df["p_value_raw"].tolist())
    inferential_df["significant_bh_0_05"] = inferential_df["p_value_bh"] < 0.05

    if logrank_p is not None:
        logrank_adj = float(
            inferential_df.loc[
                (inferential_df["test_group"] == "survival") & (inferential_df["test_name"] == "logrank"),
                "p_value_bh",
            ].iloc[0]
        )
    else:
        logrank_adj = None

    if not kruskal_df.empty:
        lookup = inferential_df[inferential_df["test_group"] == "continuous"].set_index("test_name")
        kruskal_df = kruskal_df.copy()
        kruskal_df["p_value_bh"] = kruskal_df["feature"].map(lookup["p_value_bh"])
        kruskal_df["significant_bh_0_05"] = kruskal_df["p_value_bh"] < 0.05

    if not chi_df.empty:
        lookup = inferential_df[inferential_df["test_group"] == "categorical"].set_index("test_name")
        chi_df = chi_df.copy()
        chi_df["p_value_bh"] = chi_df["test"].map(lookup["p_value_bh"])
        chi_df["significant_bh_0_05"] = chi_df["p_value_bh"] < 0.05

    return logrank_adj, kruskal_df, chi_df, inferential_df


def process_dataset(
    dataset: str,
    step4_dir: Path,
    step5_dir: Path,
    output_dir: Path,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step6.log", log_level)
    logger.info("Starting Step 6 for dataset=%s", dataset)

    step4_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = step5_dir / dataset / f"{dataset}_step5_selection.json"

    if not step4_meta_path.exists():
        raise FileNotFoundError(f"Missing Step 4 metadata: {step4_meta_path}")
    if not step5_selection_path.exists():
        raise FileNotFoundError(f"Missing Step 5 selection metadata: {step5_selection_path}")

    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)

    train_master_path = resolve_path(step4_meta["outputs"]["train_master"])
    labels_path = resolve_path(step5_selection["outputs"]["train_cluster_labels"])

    logger.info("Reading train master table: %s", train_master_path)
    logger.info("Reading train cluster labels: %s", labels_path)

    train_df = pd.read_csv(train_master_path)
    labels_df = pd.read_csv(labels_path)

    id_col = step4_meta.get("id_column")
    if id_col is None or id_col not in train_df.columns or id_col not in labels_df.columns:
        raise ValueError(f"{dataset}: ID column missing in train table or labels table.")

    merged = train_df.merge(labels_df, on=id_col, how="inner")
    cluster_col = "cluster_label"
    if cluster_col not in merged.columns:
        raise ValueError(f"{dataset}: cluster_label not found after merging Step 5 labels.")

    logger.info("Merged rows=%s | unique clusters=%s", merged.shape[0], merged[cluster_col].nunique())

    os_col = step4_meta.get("os_column")
    mgmt_col = first_present(merged.columns, ["mgmt_bin", "MGMT_bin", "MGMT status"])
    idh_col = first_present(merged.columns, ["idh_bin", "IDH_bin", "IDH"])
    nc_en_col = first_present(merged.columns, ["global_nc_en_ratio"])
    ed_en_col = first_present(merged.columns, ["global_ed_en_ratio"])
    tbi_col = first_present(merged.columns, ["tumor_burden_index"])
    age_col = first_present(merged.columns, ["Age at MRI", "Age_at_scan_years", "age", "Age"])
    lobe_col = first_present(merged.columns, ["dominant_lobe_clean", "dominant_brain_lobe", "dominant_lobe"])

    censor_col = first_present(
        merged.columns,
        [
            "1-dead 0-alive",
            "Survival_Censor",
            "survival_censor",
        ],
    )
    survival_status_col = first_present(merged.columns, ["Survival_Status", "survival_status"])

    summary_df = compute_cluster_summary(
        df=merged,
        cluster_col=cluster_col,
        os_col=os_col,
        mgmt_col=mgmt_col,
        idh_col=idh_col,
        nc_en_col=nc_en_col,
        ed_en_col=ed_en_col,
        tbi_col=tbi_col,
        age_col=age_col,
        lobe_col=lobe_col,
    )

    high_risk_cluster, high_risk_label = pick_high_risk_cluster(summary_df)
    summary_df["is_high_risk_cluster"] = summary_df["cluster_label"] == high_risk_cluster
    summary_df["cluster_description"] = np.where(
        summary_df["cluster_label"] == high_risk_cluster,
        high_risk_label,
        "other cluster",
    )
    logger.info("High-risk cluster identified: cluster=%s label=%s", high_risk_cluster, high_risk_label)

    logrank_stat, logrank_p, km_fig_out = run_logrank_and_km(
        df=merged,
        dataset=dataset,
        output_dir=dataset_out,
        cluster_col=cluster_col,
        os_col=os_col,
        censor_col=censor_col,
        survival_status_col=survival_status_col,
        logger=logger,
    )

    continuous_cols = [c for c in step4_meta.get("continuous_columns", []) if c in merged.columns]
    kruskal_df = run_kruskal_tests(merged, cluster_col, continuous_cols, logger)
    chi_df = run_chi_square_tests(merged, cluster_col, mgmt_col, lobe_col, logger)

    logrank_p_bh, kruskal_df, chi_df, inferential_df = apply_global_bh(logrank_p, kruskal_df, chi_df)

    dataset_out.mkdir(parents=True, exist_ok=True)
    summary_out = dataset_out / f"{dataset}_step6_cluster_summary_train.csv"
    train_labeled_out = dataset_out / f"{dataset}_step6_train_with_clusters.csv"
    kruskal_out = dataset_out / f"{dataset}_step6_kruskal_tests.csv"
    chi_out = dataset_out / f"{dataset}_step6_chi_square_tests.csv"
    inferential_out = dataset_out / f"{dataset}_step6_inferential_tests_bh.csv"
    metadata_out = dataset_out / f"{dataset}_step6_metadata.json"

    summary_df.to_csv(summary_out, index=False)
    merged.to_csv(train_labeled_out, index=False)
    kruskal_df.to_csv(kruskal_out, index=False)
    chi_df.to_csv(chi_out, index=False)
    inferential_df.to_csv(inferential_out, index=False)

    meta = {
        "dataset": dataset,
        "step4_metadata": str(step4_meta_path),
        "step5_selection": str(step5_selection_path),
        "train_master_file": str(train_master_path),
        "train_labels_file": str(labels_path),
        "columns_used": {
            "id": id_col,
            "cluster": cluster_col,
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
        "high_risk_cluster": {
            "cluster_label": int(high_risk_cluster),
            "label": high_risk_label,
        },
        "survival_test": {
            "logrank_statistic": logrank_stat,
            "p_value_raw": logrank_p,
            "p_value_bh": logrank_p_bh,
            "km_plot": str(km_fig_out) if km_fig_out is not None else None,
        },
        "outputs": {
            "cluster_summary": str(summary_out),
            "train_with_clusters": str(train_labeled_out),
            "kruskal_tests": str(kruskal_out),
            "chi_square_tests": str(chi_out),
            "inferential_tests_bh": str(inferential_out),
            "log_file": str(dataset_out / "step6.log"),
        },
    }
    metadata_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Saved cluster summary: %s", summary_out)
    logger.info("Saved train labeled table: %s", train_labeled_out)
    logger.info("Saved Kruskal tests: %s", kruskal_out)
    logger.info("Saved chi-square tests: %s", chi_out)
    logger.info("Saved BH-adjusted inferential table: %s", inferential_out)
    logger.info("Saved metadata: %s", metadata_out)
    logger.info("Step 6 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 6: characterize train-set clusters using survival summaries, statistical tests, "
            "and high-risk cluster identification."
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
        help="Directory where Step 6 outputs are written.",
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
            output_dir=args.output_dir,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()