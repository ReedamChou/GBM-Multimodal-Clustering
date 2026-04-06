from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

from step4_split_train_test import preprocess_split_with_train_only_rules, split_with_best_stratification
from step5_spectral_clustering import build_centroids, clean_feature_matrix, evaluate_k_values, fit_spectral_labels
from step6_characterize_clusters import compute_cluster_summary, pick_high_risk_cluster
from step7_validate_test_set import assign_to_nearest_centroid, build_cluster_proportion_comparison, build_high_risk_validation
from pipeline_preprocessing import map_mgmt_to_binary

DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "nested-validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def select_best_k(scores_df: pd.DataFrame) -> pd.Series:
    valid_scores = scores_df[scores_df["status"] == "ok"].copy()
    if valid_scores.empty:
        raise RuntimeError("No valid k candidates were produced during inner selection.")
    return valid_scores.sort_values(
        ["selection_score", "survival_consistency_score", "stability_norm", "silhouette_norm"],
        ascending=[False, False, False, False],
    ).iloc[0]


def write_preprocessing_artifact(
    output_path: Path,
    dataset: str,
    preprocessing_meta: dict,
    id_col: str,
    random_state: int,
    test_size: float,
) -> None:
    scaler = preprocessing_meta["scaler"]
    continuous_cols = preprocessing_meta["continuous_columns"]
    artifact = {
        "preprocessing_version": "nested_outer_train_only_step10_v1",
        "dataset": dataset,
        "id_column": id_col,
        "outcome_columns": preprocessing_meta["outcome_columns"],
        "random_state": random_state,
        "test_size": test_size,
        "feature_columns": preprocessing_meta["feature_columns"],
        "continuous_columns": continuous_cols,
        "binary_columns": preprocessing_meta["binary_columns"],
        "discrete_numeric_columns": preprocessing_meta["discrete_numeric_columns"],
        "one_hot_column_order": preprocessing_meta["one_hot_column_order"],
        "binary_encoding": preprocessing_meta["binary_encoding"],
        "dominant_lobe_encoding": preprocessing_meta["dominant_lobe_encoding"],
        "categorical_fill": preprocessing_meta["categorical_fill"],
        "continuous_median_imputation": preprocessing_meta["continuous_median_imputation"],
        "discrete_numeric_fill": preprocessing_meta["discrete_numeric_fill"],
        "dropped_feature_rules": preprocessing_meta["dropped_feature_rules"],
        "dropped_feature_columns": preprocessing_meta["dropped_feature_columns"],
        "scaler": scaler,
        "scaler_mean_by_column": {
            col: float(val) for col, val in zip(continuous_cols, getattr(scaler, "mean_", []))
        },
        "scaler_scale_by_column": {
            col: float(val) for col, val in zip(continuous_cols, getattr(scaler, "scale_", []))
        },
    }
    joblib.dump(artifact, output_path)


def process_dataset_seed(
    dataset: str,
    seed: int,
    step3_dir: Path,
    output_dir: Path,
    outer_test_size: float,
    inner_test_size: float,
    k_values: list[int],
    n_neighbors: int,
    gap_refs: int,
    stability_resamples: int,
    stability_sample_fraction: float,
) -> dict:
    dataset_step3_dir = step3_dir / dataset
    master_path = dataset_step3_dir / f"{dataset}_master_table_step3.csv"
    meta_path = dataset_step3_dir / f"{dataset}_step3_metadata.json"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing Step 3 master table: {master_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Step 3 metadata: {meta_path}")

    df = pd.read_csv(master_path)
    meta = load_json(meta_path)
    resolved = meta.get("resolved_columns", {})
    id_col = resolved.get("id") or "ID"
    os_col = resolved.get("os")
    mgmt_source_col = resolved.get("mgmt")

    if id_col not in df.columns:
        raise ValueError(f"{dataset}: missing ID column {id_col} in Step 3 master table.")

    mgmt_strata_col = None
    if mgmt_source_col is not None and mgmt_source_col in df.columns:
        mgmt_strata_col = "__mgmt_strata_bin__"
        df[mgmt_strata_col] = map_mgmt_to_binary(df[mgmt_source_col])

    outer_dev_df, outer_test_raw_df, outer_strat = split_with_best_stratification(
        df=df,
        id_col=id_col,
        mgmt_col=mgmt_strata_col,
        os_col=os_col,
        test_size=outer_test_size,
        random_state=seed,
        logger=__import__("logging").getLogger(f"step10.outer.{dataset}.{seed}"),
    )
    outer_dev_df = outer_dev_df.drop(columns=[mgmt_strata_col], errors="ignore")
    outer_test_raw_df = outer_test_raw_df.drop(columns=[mgmt_strata_col], errors="ignore")

    inner_dev_df = outer_dev_df.copy()
    if mgmt_source_col is not None and mgmt_source_col in inner_dev_df.columns:
        inner_dev_df["__mgmt_strata_bin__"] = map_mgmt_to_binary(inner_dev_df[mgmt_source_col])
        inner_mgmt_col = "__mgmt_strata_bin__"
    else:
        inner_mgmt_col = None

    inner_train_raw_df, inner_val_raw_df, inner_strat = split_with_best_stratification(
        df=inner_dev_df,
        id_col=id_col,
        mgmt_col=inner_mgmt_col,
        os_col=os_col,
        test_size=inner_test_size,
        random_state=seed + 1000,
        logger=__import__("logging").getLogger(f"step10.inner.{dataset}.{seed}"),
    )
    inner_train_raw_df = inner_train_raw_df.drop(columns=["__mgmt_strata_bin__"], errors="ignore")
    inner_val_raw_df = inner_val_raw_df.drop(columns=["__mgmt_strata_bin__"], errors="ignore")

    inner_train_df, inner_val_df, inner_preprocessing_meta = preprocess_split_with_train_only_rules(inner_train_raw_df, inner_val_raw_df, meta)
    inner_feature_cols = inner_preprocessing_meta["feature_columns"]
    inner_x_train, _ = clean_feature_matrix(inner_train_df, inner_feature_cols)

    scores_df, _, selection_config = evaluate_k_values(
        dataset=dataset,
        x_train=inner_x_train,
        train_master_df=inner_train_df,
        test_master_df=inner_val_df,
        test_features_df=inner_val_df[[id_col] + inner_feature_cols].copy(),
        id_col=id_col,
        feature_cols=inner_feature_cols,
        k_values=sorted(set(k_values)),
        random_state=seed + 2000,
        n_neighbors=n_neighbors,
        gap_refs=gap_refs,
        stability_resamples=stability_resamples,
        stability_sample_fraction=stability_sample_fraction,
        logger=__import__("logging").getLogger(f"step10.select.{dataset}.{seed}"),
        step4_meta={"os_column": os_col},
    )
    best_row = select_best_k(scores_df)
    best_k = int(best_row["k"])

    outer_dev_df_processed, outer_test_df_processed, outer_preprocessing_meta = preprocess_split_with_train_only_rules(
        outer_dev_df,
        outer_test_raw_df,
        meta,
    )
    outer_feature_cols = outer_preprocessing_meta["feature_columns"]
    outer_x_train, outer_fill_medians = clean_feature_matrix(outer_dev_df_processed, outer_feature_cols)
    outer_labels = fit_spectral_labels(
        outer_x_train,
        k=best_k,
        random_state=seed + 3000,
        n_neighbors=n_neighbors,
    )

    labels_df = pd.DataFrame({id_col: outer_dev_df_processed[id_col].values, "cluster_label": outer_labels.astype(int)})
    outer_dev_labeled = outer_dev_df_processed.merge(labels_df, on=id_col, how="inner")

    train_summary = compute_cluster_summary(
        df=outer_dev_labeled,
        cluster_col="cluster_label",
        os_col=os_col,
        mgmt_col="mgmt_bin" if "mgmt_bin" in outer_dev_labeled.columns else None,
        idh_col="idh_bin" if "idh_bin" in outer_dev_labeled.columns else None,
        nc_en_col="global_nc_en_ratio" if "global_nc_en_ratio" in outer_dev_labeled.columns else None,
        ed_en_col="global_ed_en_ratio" if "global_ed_en_ratio" in outer_dev_labeled.columns else None,
        tbi_col="tumor_burden_index" if "tumor_burden_index" in outer_dev_labeled.columns else None,
        age_col="Age at MRI" if "Age at MRI" in outer_dev_labeled.columns else ("Age_at_scan_years" if "Age_at_scan_years" in outer_dev_labeled.columns else None),
        lobe_col="dominant_lobe_clean" if "dominant_lobe_clean" in outer_dev_labeled.columns else None,
    )
    high_risk_cluster, high_risk_label = pick_high_risk_cluster(train_summary)

    centroids = build_centroids(outer_x_train, outer_labels, outer_feature_cols)
    outer_x_test = outer_test_df_processed[outer_feature_cols].copy()
    for col in outer_feature_cols:
        outer_x_test[col] = pd.to_numeric(outer_x_test[col], errors="coerce").fillna(float(outer_fill_medians.get(col, 0.0)))
    assigned_labels, distances = assign_to_nearest_centroid(outer_x_test, centroids, outer_feature_cols)
    assigned_df = pd.DataFrame(
        {
            id_col: outer_test_df_processed[id_col].values,
            "cluster_label": assigned_labels.astype(int),
            "nearest_centroid_distance": distances,
        }
    )
    outer_test_labeled = outer_test_df_processed.merge(assigned_df, on=id_col, how="inner")

    test_summary = compute_cluster_summary(
        df=outer_test_labeled,
        cluster_col="cluster_label",
        os_col=os_col,
        mgmt_col="mgmt_bin" if "mgmt_bin" in outer_test_labeled.columns else None,
        idh_col="idh_bin" if "idh_bin" in outer_test_labeled.columns else None,
        nc_en_col="global_nc_en_ratio" if "global_nc_en_ratio" in outer_test_labeled.columns else None,
        ed_en_col="global_ed_en_ratio" if "global_ed_en_ratio" in outer_test_labeled.columns else None,
        tbi_col="tumor_burden_index" if "tumor_burden_index" in outer_test_labeled.columns else None,
        age_col="Age at MRI" if "Age at MRI" in outer_test_labeled.columns else ("Age_at_scan_years" if "Age_at_scan_years" in outer_test_labeled.columns else None),
        lobe_col="dominant_lobe_clean" if "dominant_lobe_clean" in outer_test_labeled.columns else None,
    )
    proportions_df = build_cluster_proportion_comparison(
        train_labels_df=labels_df,
        test_labels_df=assigned_df,
        cluster_col="cluster_label",
    )
    high_risk_validation = build_high_risk_validation(
        train_summary_df=train_summary,
        test_summary_df=test_summary,
        proportions_df=proportions_df,
        high_risk_cluster=high_risk_cluster,
        high_risk_label=high_risk_label,
    )

    from step5_spectral_clustering import compute_logrank_p

    train_logrank_p = compute_logrank_p(
        df=outer_dev_labeled,
        cluster_col="cluster_label",
        os_col=os_col,
        censor_col=resolved.get("censor"),
        survival_status_col=resolved.get("survival_status"),
    )
    test_logrank_p = compute_logrank_p(
        df=outer_test_labeled,
        cluster_col="cluster_label",
        os_col=os_col,
        censor_col=resolved.get("censor"),
        survival_status_col=resolved.get("survival_status"),
    )

    run_dataset_dir = output_dir / "runs" / f"seed_{seed:04d}" / dataset
    run_dataset_dir.mkdir(parents=True, exist_ok=True)

    inner_eval_out = run_dataset_dir / f"{dataset}_inner_k_evaluation.csv"
    outer_dev_labels_out = run_dataset_dir / f"{dataset}_outer_dev_cluster_labels.csv"
    outer_dev_summary_out = run_dataset_dir / f"{dataset}_outer_dev_cluster_summary.csv"
    outer_test_assigned_out = run_dataset_dir / f"{dataset}_outer_test_assigned_clusters.csv"
    outer_test_summary_out = run_dataset_dir / f"{dataset}_outer_test_cluster_summary.csv"
    outer_prop_out = run_dataset_dir / f"{dataset}_outer_test_cluster_proportions.csv"
    preprocessing_out = run_dataset_dir / f"{dataset}_outer_train_preprocessing.joblib"
    meta_out = run_dataset_dir / f"{dataset}_step10_nested_metadata.json"

    scores_df.to_csv(inner_eval_out, index=False)
    labels_df.to_csv(outer_dev_labels_out, index=False)
    train_summary.to_csv(outer_dev_summary_out, index=False)
    outer_test_labeled.to_csv(outer_test_assigned_out, index=False)
    test_summary.to_csv(outer_test_summary_out, index=False)
    proportions_df.to_csv(outer_prop_out, index=False)
    write_preprocessing_artifact(
        output_path=preprocessing_out,
        dataset=dataset,
        preprocessing_meta=outer_preprocessing_meta,
        id_col=id_col,
        random_state=seed,
        test_size=outer_test_size,
    )

    meta_payload = {
        "dataset": dataset,
        "seed": seed,
        "outer_split": {
            "train_rows": int(len(outer_dev_df_processed)),
            "test_rows": int(len(outer_test_df_processed)),
            "test_size": outer_test_size,
            "stratification": outer_strat,
        },
        "inner_split": {
            "train_rows": int(len(inner_train_df)),
            "validation_rows": int(len(inner_val_df)),
            "validation_size_within_outer_train": inner_test_size,
            "stratification": inner_strat,
        },
        "selected_k": best_k,
        "selected_k_row": best_row.to_dict(),
        "selection_config": selection_config,
        "outer_train_logrank_p": train_logrank_p,
        "outer_test_logrank_p": test_logrank_p,
        "high_risk_cluster": {
            "cluster_label": int(high_risk_cluster),
            "label": high_risk_label,
        },
        "high_risk_cluster_validation": high_risk_validation,
        "outputs": {
            "inner_k_evaluation": str(inner_eval_out),
            "outer_dev_cluster_labels": str(outer_dev_labels_out),
            "outer_dev_cluster_summary": str(outer_dev_summary_out),
            "outer_test_assigned_clusters": str(outer_test_assigned_out),
            "outer_test_cluster_summary": str(outer_test_summary_out),
            "outer_test_cluster_proportions": str(outer_prop_out),
            "outer_train_preprocessing": str(preprocessing_out),
        },
    }
    meta_out.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    return {
        "dataset": dataset,
        "seed": seed,
        "selected_k": best_k,
        "outer_train_logrank_p": train_logrank_p,
        "outer_test_logrank_p": test_logrank_p,
        "outer_train_significant": bool((train_logrank_p or 1.0) < 0.05),
        "outer_test_significant": bool((test_logrank_p or 1.0) < 0.05),
        "outer_train_survival_order": survival_order_from_summary(train_summary),
        "outer_test_survival_order": survival_order_from_summary(test_summary),
        "high_risk_cluster": int(high_risk_cluster),
        "high_risk_label": high_risk_label,
        "train_pct": (high_risk_validation.get("cluster_proportion_check") or {}).get("train_pct"),
        "test_pct": (high_risk_validation.get("cluster_proportion_check") or {}).get("test_pct"),
        "pct_point_diff": (high_risk_validation.get("cluster_proportion_check") or {}).get("pct_point_diff"),
        "within_10_pct_points": (high_risk_validation.get("cluster_proportion_check") or {}).get("within_10_pct_points"),
        "dominant_lobe_matches_train": high_risk_validation.get("dominant_lobe_matches_train"),
        "temporal_replication_expected": high_risk_validation.get("temporal_replication_expected"),
        "temporal_replication_observed": high_risk_validation.get("temporal_replication_observed"),
        "test_high_risk_nc_en_rank": (high_risk_validation.get("test_signature") or {}).get("nc_en_rank_among_test_clusters"),
        "run_dir": str(run_dataset_dir),
    }


def build_dataset_report(dataset: str, df: pd.DataFrame) -> str:
    df = df.sort_values("seed").reset_index(drop=True)
    return "\n".join(
        [
            f"{dataset.upper()}",
            f"- Outer seeds: {len(df)}",
            f"- Selected k counts: {', '.join(f'k={int(k)}: {int(v)}' for k, v in df['selected_k'].value_counts().sort_index().items())}",
            f"- Outer train survival significant in {float(df['outer_train_significant'].mean() * 100):.1f}% of runs",
            f"- Outer untouched test survival significant in {float(df['outer_test_significant'].mean() * 100):.1f}% of runs",
            f"- Median outer train log-rank p-value: {float(pd.to_numeric(df['outer_train_logrank_p'], errors='coerce').median()):.4g}",
            f"- Median outer untouched test log-rank p-value: {float(pd.to_numeric(df['outer_test_logrank_p'], errors='coerce').median()):.4g}",
            f"- High-risk cluster proportion stayed within 10 percentage points in {float(df['within_10_pct_points'].fillna(False).astype(bool).mean() * 100):.1f}% of runs",
            f"- Dominant lobe matched train in {float(df['dominant_lobe_matches_train'].fillna(False).astype(bool).mean() * 100):.1f}% of runs",
            f"- High-risk NC/EN was rank 1 or 2 on untouched test in {float((pd.to_numeric(df['test_high_risk_nc_en_rank'], errors='coerce') <= 2).mean() * 100):.1f}% of runs",
            f"- Most common outer untouched test survival ordering: {df['outer_test_survival_order'].mode().iloc[0] if not df['outer_test_survival_order'].mode().empty else ''}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 10: nested validation with an untouched outer test split and an inner split used only for k selection."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["ucsf", "upenn"], choices=["ucsf", "upenn"])
    parser.add_argument("--step3-dir", type=Path, default=DEFAULT_STEP3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-seeds", nargs="+", type=int, help="Explicit outer seeds to run.")
    parser.add_argument("--n-outer-runs", type=int, default=10)
    parser.add_argument("--outer-seed-start", type=int, default=1)
    parser.add_argument("--outer-test-size", type=float, default=0.20)
    parser.add_argument("--inner-test-size", type=float, default=0.25)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--gap-refs", type=int, default=5)
    parser.add_argument("--stability-resamples", type=int, default=8)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.outer_seeds if args.outer_seeds else list(range(args.outer_seed_start, args.outer_seed_start + args.n_outer_runs))

    output_dir = args.output_dir
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for seed in seeds:
        print(f"[outer seed {seed}] nested evaluation")
        for dataset in args.datasets:
            row = process_dataset_seed(
                dataset=dataset,
                seed=seed,
                step3_dir=args.step3_dir,
                output_dir=output_dir,
                outer_test_size=args.outer_test_size,
                inner_test_size=args.inner_test_size,
                k_values=args.k_values,
                n_neighbors=args.n_neighbors,
                gap_refs=args.gap_refs,
                stability_resamples=args.stability_resamples,
                stability_sample_fraction=args.stability_sample_fraction,
            )
            rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    summary_csv = output_dir / "summary" / "nested_validation_run_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    report_lines = [
        "Nested Validation Summary",
        f"Outer seeds run: {', '.join(str(s) for s in seeds)}",
        f"Datasets: {', '.join(args.datasets)}",
        "",
    ]
    for dataset in args.datasets:
        report_lines.append(build_dataset_report(dataset, summary_df[summary_df["dataset"] == dataset].copy()))
        report_lines.append("")

    summary_txt = output_dir / "summary" / "nested_validation_summary.txt"
    summary_txt.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    manifest = {
        "datasets": args.datasets,
        "outer_seeds": seeds,
        "step3_dir": str(args.step3_dir),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "summary_txt": str(summary_txt),
        "parameters": {
            "outer_test_size": args.outer_test_size,
            "inner_test_size": args.inner_test_size,
            "k_values": args.k_values,
            "n_neighbors": args.n_neighbors,
            "gap_refs": args.gap_refs,
            "stability_resamples": args.stability_resamples,
            "stability_sample_fraction": args.stability_sample_fraction,
        },
    }
    (output_dir / "summary" / "nested_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved nested metrics: {summary_csv}")
    print(f"Saved nested report: {summary_txt}")


if __name__ == "__main__":
    main()
