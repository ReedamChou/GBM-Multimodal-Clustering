from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from cluster_validation_utils import (
    assign_to_nearest_centroid,
    build_cluster_proportion_comparison,
    build_event_observed,
    build_high_risk_validation,
    compute_cluster_summary,
    first_present,
)

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
