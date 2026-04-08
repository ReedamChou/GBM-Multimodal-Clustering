from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from cluster_validation_utils import survival_order_from_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT, env=env)


def run_pipeline_for_seed(
    *,
    datasets: list[str],
    seed: int,
    step3_dir: Path,
    run_dir: Path,
    split_test_size: float,
    inner_test_size: float,
    k_values: list[int],
    n_neighbors: int,
    gap_refs: int,
    stability_resamples: int,
    stability_sample_fraction: float,
    log_level: str,
    max_missing_feature_frac: float,
    corr_prune_threshold: float,
    winsorize_lower_quantile: float,
    winsorize_upper_quantile: float,
    scaler_type: str,
    power_transform: str,
) -> None:
    step4_dir = run_dir / "step4"
    step5_dir = run_dir / "step5"
    step6_dir = run_dir / "step6"
    step7_dir = run_dir / "step7"

    print(f"[seed {seed}] running Step 4")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "step4_split_train_test.py"),
            "--datasets",
            *datasets,
            "--step3-dir",
            str(step3_dir),
            "--output-dir",
            str(step4_dir),
            "--test-size",
            str(split_test_size),
            "--random-state",
            str(seed),
            "--log-level",
            log_level,
            "--max-missing-feature-frac",
            str(max_missing_feature_frac),
            "--corr-prune-threshold",
            str(corr_prune_threshold),
            "--winsorize-lower-quantile",
            str(winsorize_lower_quantile),
            "--winsorize-upper-quantile",
            str(winsorize_upper_quantile),
            "--scaler-type",
            scaler_type,
            "--power-transform",
            power_transform,
        ]
    )

    print(f"[seed {seed}] running Step 5")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "step5_spectral_clustering.py"),
            "--datasets",
            *datasets,
            "--step4-dir",
            str(step4_dir),
            "--output-dir",
            str(step5_dir),
            "--k-values",
            *[str(v) for v in k_values],
            "--inner-test-size",
            str(inner_test_size),
            "--n-neighbors",
            str(n_neighbors),
            "--gap-refs",
            str(gap_refs),
            "--stability-resamples",
            str(stability_resamples),
            "--stability-sample-fraction",
            str(stability_sample_fraction),
            "--random-state",
            str(seed),
            "--log-level",
            log_level,
        ]
    )

    print(f"[seed {seed}] running Step 6")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "step6_characterize_clusters.py"),
            "--datasets",
            *datasets,
            "--step4-dir",
            str(step4_dir),
            "--step5-dir",
            str(step5_dir),
            "--output-dir",
            str(step6_dir),
            "--log-level",
            log_level,
        ]
    )

    print(f"[seed {seed}] running Step 7")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "step7_validate_test_set.py"),
            "--datasets",
            *datasets,
            "--step4-dir",
            str(step4_dir),
            "--step5-dir",
            str(step5_dir),
            "--step6-dir",
            str(step6_dir),
            "--output-dir",
            str(step7_dir),
            "--log-level",
            log_level,
        ]
    )


def collect_run_context(dataset: str, seed: int, run_dir: Path) -> dict:
    step4_meta_path = run_dir / "step4" / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = run_dir / "step5" / dataset / f"{dataset}_step5_selection.json"
    step6_meta_path = run_dir / "step6" / dataset / f"{dataset}_step6_metadata.json"
    step7_meta_path = run_dir / "step7" / dataset / f"{dataset}_step7_metadata.json"

    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)
    step6_meta = load_json(step6_meta_path)
    step7_meta = load_json(step7_meta_path)

    train_summary = pd.read_csv(resolve_path(step6_meta["outputs"]["cluster_summary"]))
    test_summary = pd.read_csv(resolve_path(step7_meta["outputs"]["cluster_summary_test"]))
    proportions = pd.read_csv(resolve_path(step7_meta["outputs"]["cluster_proportions_comparison"]))

    high_risk = step7_meta.get("high_risk_cluster_validation", {})
    train_sig = high_risk.get("train_signature", {}) or {}
    test_sig = high_risk.get("test_signature", {}) or {}
    proportion_check = high_risk.get("cluster_proportion_check", {}) or {}

    return {
        "dataset": dataset,
        "seed": seed,
        "run_dir": run_dir,
        "step4_meta_path": step4_meta_path,
        "step5_selection_path": step5_selection_path,
        "step6_meta_path": step6_meta_path,
        "step7_meta_path": step7_meta_path,
        "step4_meta": step4_meta,
        "step5_selection": step5_selection,
        "step6_meta": step6_meta,
        "step7_meta": step7_meta,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "proportions": proportions,
        "high_risk": high_risk,
        "train_sig": train_sig,
        "test_sig": test_sig,
        "proportion_check": proportion_check,
    }


def summarize_repeated_run(dataset: str, seed: int, run_dir: Path) -> tuple[dict, pd.DataFrame]:
    context = collect_run_context(dataset, seed, run_dir)
    row = {
        "dataset": dataset,
        "seed": seed,
        "best_k": int(context["step5_selection"].get("best_k")),
        "best_silhouette": float(context["step5_selection"].get("best_silhouette")),
        "high_risk_cluster": int(context["high_risk"].get("cluster_label")),
        "high_risk_label": context["high_risk"].get("label"),
        "train_logrank_p": context["step6_meta"].get("survival_test", {}).get("p_value_raw"),
        "test_logrank_p": context["step7_meta"].get("survival_validation", {}).get("p_value_raw"),
        "train_survival_significant": bool((context["step6_meta"].get("survival_test", {}).get("p_value_raw") or 1.0) < 0.05),
        "test_survival_significant": bool((context["step7_meta"].get("survival_validation", {}).get("p_value_raw") or 1.0) < 0.05),
        "train_survival_order": survival_order_from_summary(context["train_summary"]),
        "test_survival_order": survival_order_from_summary(context["test_summary"]),
        "train_high_risk_n": context["train_sig"].get("n"),
        "test_high_risk_n": context["test_sig"].get("n"),
        "train_high_risk_pct": context["proportion_check"].get("train_pct"),
        "test_high_risk_pct": context["proportion_check"].get("test_pct"),
        "high_risk_pct_point_diff": context["proportion_check"].get("pct_point_diff"),
        "high_risk_within_10_pct_points": context["proportion_check"].get("within_10_pct_points"),
        "train_high_risk_median_os": context["train_sig"].get("median_os"),
        "test_high_risk_median_os": context["test_sig"].get("median_os"),
        "train_high_risk_mgmt_pct": context["train_sig"].get("mgmt_methylated_pct"),
        "test_high_risk_mgmt_pct": context["test_sig"].get("mgmt_methylated_pct"),
        "train_high_risk_nc_en": context["train_sig"].get("mean_global_nc_en_ratio"),
        "test_high_risk_nc_en": context["test_sig"].get("mean_global_nc_en_ratio"),
        "test_high_risk_nc_en_rank": context["test_sig"].get("nc_en_rank_among_test_clusters"),
        "train_high_risk_dominant_lobe": context["train_sig"].get("dominant_lobe_mode"),
        "test_high_risk_dominant_lobe": context["test_sig"].get("dominant_lobe_mode"),
        "dominant_lobe_matches_train": context["high_risk"].get("dominant_lobe_matches_train"),
        "temporal_replication_expected": context["high_risk"].get("temporal_replication_expected"),
        "temporal_replication_observed": context["high_risk"].get("temporal_replication_observed"),
        "train_high_risk_lowest_os": (
            survival_order_from_summary(context["train_summary"]).split(">")[0] == str(int(context["high_risk"].get("cluster_label")))
            if context["high_risk"].get("cluster_label") is not None
            else False
        ),
        "test_high_risk_lowest_os": (
            survival_order_from_summary(context["test_summary"]).split(">")[0] == str(int(context["high_risk"].get("cluster_label")))
            if context["high_risk"].get("cluster_label") is not None
            else False
        ),
        "run_dir": str(run_dir),
    }

    proportions_df = context["proportions"].copy()
    proportions_df.insert(0, "seed", seed)
    proportions_df.insert(0, "dataset", dataset)
    return row, proportions_df


def build_repeated_validation_report(dataset: str, dataset_df: pd.DataFrame) -> str:
    dataset_df = dataset_df.sort_values("seed").reset_index(drop=True)

    k_counts = dataset_df["best_k"].value_counts().sort_index()
    hr_counts = dataset_df["high_risk_cluster"].value_counts().sort_index()
    train_sig_rate = float(dataset_df["train_survival_significant"].mean() * 100.0)
    test_sig_rate = float(dataset_df["test_survival_significant"].mean() * 100.0)
    lobe_match_rate = float(dataset_df["dominant_lobe_matches_train"].fillna(False).astype(bool).mean() * 100.0)
    within_10_rate = float(dataset_df["high_risk_within_10_pct_points"].fillna(False).astype(bool).mean() * 100.0)
    nc_rank1_rate = float((pd.to_numeric(dataset_df["test_high_risk_nc_en_rank"], errors="coerce") == 1).mean() * 100.0)
    nc_rank2_or_better_rate = float((pd.to_numeric(dataset_df["test_high_risk_nc_en_rank"], errors="coerce") <= 2).mean() * 100.0)

    train_p = pd.to_numeric(dataset_df["train_logrank_p"], errors="coerce")
    test_p = pd.to_numeric(dataset_df["test_logrank_p"], errors="coerce")
    pct_diff = pd.to_numeric(dataset_df["high_risk_pct_point_diff"], errors="coerce")

    lines = [
        f"{dataset.upper()}",
        f"- Repeated splits: {len(dataset_df)}",
        f"- Chosen k counts: {', '.join(f'k={int(k)}: {int(v)}' for k, v in k_counts.items())}",
        f"- High-risk cluster label counts: {', '.join(f'cluster {int(k)}: {int(v)}' for k, v in hr_counts.items())}",
        f"- Train survival significant in {train_sig_rate:.1f}% of runs",
        f"- Test survival significant in {test_sig_rate:.1f}% of runs",
        f"- Median train log-rank p-value: {train_p.median():.4g}",
        f"- Median test log-rank p-value: {test_p.median():.4g}",
        f"- High-risk cluster proportion stayed within 10 percentage points in {within_10_rate:.1f}% of runs",
        f"- Median high-risk proportion shift: {pct_diff.median():.2f} percentage points",
        f"- Dominant lobe matched train in {lobe_match_rate:.1f}% of runs",
        f"- High-risk NC/EN was rank 1 on test in {nc_rank1_rate:.1f}% of runs",
        f"- High-risk NC/EN was rank 1 or 2 on test in {nc_rank2_or_better_rate:.1f}% of runs",
        f"- Most common train survival ordering: {dataset_df['train_survival_order'].mode().iloc[0] if not dataset_df['train_survival_order'].mode().empty else ''}",
        f"- Most common test survival ordering: {dataset_df['test_survival_order'].mode().iloc[0] if not dataset_df['test_survival_order'].mode().empty else ''}",
    ]

    temporal_mask = dataset_df["temporal_replication_expected"].fillna(False).astype(bool)
    if temporal_mask.any():
        temporal_obs = dataset_df.loc[temporal_mask, "temporal_replication_observed"].astype("boolean").fillna(False).astype(bool)
        lines.append(f"- Temporal-lobe replication observed in {float(temporal_obs.mean() * 100.0):.1f}% of runs where it was expected")

    return "\n".join(lines)


def run_repeated_validation(
    *,
    datasets: list[str],
    seeds: list[int],
    step3_dir: Path,
    output_dir: Path,
    test_size: float,
    inner_test_size: float,
    k_values: list[int],
    n_neighbors: int,
    gap_refs: int,
    stability_resamples: int,
    stability_sample_fraction: float,
    log_level: str,
    max_missing_feature_frac: float,
    corr_prune_threshold: float,
    winsorize_lower_quantile: float,
    winsorize_upper_quantile: float,
    scaler_type: str,
    power_transform: str,
) -> None:
    runs_dir = output_dir / "runs"
    summary_dir = output_dir / "summary"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    proportions_rows: list[pd.DataFrame] = []

    for seed in seeds:
        run_dir = runs_dir / f"seed_{seed:04d}"
        run_pipeline_for_seed(
            datasets=datasets,
            seed=seed,
            step3_dir=step3_dir,
            run_dir=run_dir,
            split_test_size=test_size,
            inner_test_size=inner_test_size,
            k_values=k_values,
            n_neighbors=n_neighbors,
            gap_refs=gap_refs,
            stability_resamples=stability_resamples,
            stability_sample_fraction=stability_sample_fraction,
            log_level=log_level,
            max_missing_feature_frac=max_missing_feature_frac,
            corr_prune_threshold=corr_prune_threshold,
            winsorize_lower_quantile=winsorize_lower_quantile,
            winsorize_upper_quantile=winsorize_upper_quantile,
            scaler_type=scaler_type,
            power_transform=power_transform,
        )

        for dataset in datasets:
            row, proportions_df = summarize_repeated_run(dataset=dataset, seed=seed, run_dir=run_dir)
            summary_rows.append(row)
            proportions_rows.append(proportions_df)

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "seed"]).reset_index(drop=True)
    proportions_df = pd.concat(proportions_rows, ignore_index=True) if proportions_rows else pd.DataFrame()

    summary_csv = summary_dir / "repeated_validation_run_metrics.csv"
    proportions_csv = summary_dir / "repeated_validation_cluster_proportions_long.csv"
    summary_df.to_csv(summary_csv, index=False)
    if not proportions_df.empty:
        proportions_df.to_csv(proportions_csv, index=False)

    report_lines = [
        "Repeated Validation Summary",
        f"Seeds run: {', '.join(str(s) for s in seeds)}",
        f"Datasets: {', '.join(datasets)}",
        "",
    ]
    for dataset in datasets:
        report_lines.append(build_repeated_validation_report(dataset, summary_df[summary_df["dataset"] == dataset].copy()))
        report_lines.append("")

    report_txt = summary_dir / "repeated_validation_summary.txt"
    report_txt.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    manifest = {
        "datasets": datasets,
        "seeds": seeds,
        "step3_dir": str(step3_dir),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "proportions_csv": str(proportions_csv) if not proportions_df.empty else None,
        "summary_report": str(report_txt),
        "parameters": {
            "test_size": test_size,
            "inner_test_size": inner_test_size,
            "k_values": k_values,
            "n_neighbors": n_neighbors,
            "gap_refs": gap_refs,
            "stability_resamples": stability_resamples,
            "stability_sample_fraction": stability_sample_fraction,
            "log_level": log_level,
            "max_missing_feature_frac": max_missing_feature_frac,
            "corr_prune_threshold": corr_prune_threshold,
            "winsorize_lower_quantile": winsorize_lower_quantile,
            "winsorize_upper_quantile": winsorize_upper_quantile,
            "scaler_type": scaler_type,
            "power_transform": power_transform,
        },
    }
    (summary_dir / "repeated_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved run metrics: {summary_csv}")
    if not proportions_df.empty:
        print(f"Saved cluster proportions: {proportions_csv}")
    print(f"Saved summary report: {report_txt}")
