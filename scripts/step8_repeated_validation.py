from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "repeated-validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    mpl_cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache_dir)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT, env=env)


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


def summarize_dataset_run(dataset: str, seed: int, run_dir: Path) -> tuple[dict, pd.DataFrame]:
    step5_selection = load_json(run_dir / "step5" / dataset / f"{dataset}_step5_selection.json")
    step6_meta = load_json(run_dir / "step6" / dataset / f"{dataset}_step6_metadata.json")
    step7_meta = load_json(run_dir / "step7" / dataset / f"{dataset}_step7_metadata.json")

    train_summary = pd.read_csv(resolve_path(step6_meta["outputs"]["cluster_summary"]))
    test_summary = pd.read_csv(resolve_path(step7_meta["outputs"]["cluster_summary_test"]))
    proportions = pd.read_csv(resolve_path(step7_meta["outputs"]["cluster_proportions_comparison"]))

    high_risk = step7_meta.get("high_risk_cluster_validation", {})
    train_sig = high_risk.get("train_signature", {})
    test_sig = high_risk.get("test_signature", {})
    proportion_check = high_risk.get("cluster_proportion_check", {})

    summary_row = {
        "dataset": dataset,
        "seed": seed,
        "best_k": int(step5_selection.get("best_k")),
        "best_silhouette": float(step5_selection.get("best_silhouette")),
        "high_risk_cluster": int(high_risk.get("cluster_label")),
        "high_risk_label": high_risk.get("label"),
        "train_logrank_p": step6_meta.get("survival_test", {}).get("p_value_raw"),
        "test_logrank_p": step7_meta.get("survival_validation", {}).get("p_value_raw"),
        "train_survival_significant": bool((step6_meta.get("survival_test", {}).get("p_value_raw") or 1.0) < 0.05),
        "test_survival_significant": bool((step7_meta.get("survival_validation", {}).get("p_value_raw") or 1.0) < 0.05),
        "train_survival_order": survival_order_from_summary(train_summary),
        "test_survival_order": survival_order_from_summary(test_summary),
        "train_high_risk_n": train_sig.get("n"),
        "test_high_risk_n": test_sig.get("n"),
        "train_high_risk_pct": proportion_check.get("train_pct"),
        "test_high_risk_pct": proportion_check.get("test_pct"),
        "high_risk_pct_point_diff": proportion_check.get("pct_point_diff"),
        "high_risk_within_10_pct_points": proportion_check.get("within_10_pct_points"),
        "train_high_risk_median_os": train_sig.get("median_os"),
        "test_high_risk_median_os": test_sig.get("median_os"),
        "train_high_risk_mgmt_pct": train_sig.get("mgmt_methylated_pct"),
        "test_high_risk_mgmt_pct": test_sig.get("mgmt_methylated_pct"),
        "train_high_risk_nc_en": train_sig.get("mean_global_nc_en_ratio"),
        "test_high_risk_nc_en": test_sig.get("mean_global_nc_en_ratio"),
        "test_high_risk_nc_en_rank": test_sig.get("nc_en_rank_among_test_clusters"),
        "train_high_risk_dominant_lobe": train_sig.get("dominant_lobe_mode"),
        "test_high_risk_dominant_lobe": test_sig.get("dominant_lobe_mode"),
        "dominant_lobe_matches_train": high_risk.get("dominant_lobe_matches_train"),
        "temporal_replication_expected": high_risk.get("temporal_replication_expected"),
        "temporal_replication_observed": high_risk.get("temporal_replication_observed"),
        "run_dir": str(run_dir),
    }

    proportions_long = proportions.copy()
    proportions_long.insert(0, "seed", seed)
    proportions_long.insert(0, "dataset", dataset)
    return summary_row, proportions_long


def build_dataset_report(dataset: str, dataset_df: pd.DataFrame) -> str:
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

    if "temporal_replication_observed" in dataset_df.columns:
        temporal_mask = dataset_df["temporal_replication_expected"].fillna(False).astype(bool)
        if temporal_mask.any():
            temporal_obs = dataset_df.loc[temporal_mask, "temporal_replication_observed"].astype("boolean").fillna(False).astype(bool)
            observed_rate = float(temporal_obs.mean() * 100.0)
            lines.append(f"- Temporal-lobe replication observed in {observed_rate:.1f}% of runs where it was expected")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 8: repeated validation over many train/test splits using the leakage-safe pipeline. "
            "For each seed, rerun Steps 4 through 7, store the outputs, and aggregate stability metrics."
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
        "--step3-dir",
        type=Path,
        default=DEFAULT_STEP3_DIR,
        help="Directory containing leakage-safe Step 3 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where repeated-validation outputs will be written.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit random seeds to run. If omitted, --n-runs and --seed-start are used.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Number of repeated runs when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="First seed to use when --seeds is not provided.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Test split fraction passed to Step 4.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="k values passed to Step 5.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Nearest-neighbor count passed to Step 5.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level used by repeated-validation child steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.n_runs))

    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    summary_dir = output_dir / "summary"
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    proportions_rows: list[pd.DataFrame] = []

    for seed in seeds:
        run_dir = runs_dir / f"seed_{seed:04d}"
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
                *args.datasets,
                "--step3-dir",
                str(args.step3_dir),
                "--output-dir",
                str(step4_dir),
                "--test-size",
                str(args.test_size),
                "--random-state",
                str(seed),
                "--log-level",
                args.log_level,
            ]
        )

        print(f"[seed {seed}] running Step 5")
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "step5_spectral_clustering.py"),
                "--datasets",
                *args.datasets,
                "--step4-dir",
                str(step4_dir),
                "--output-dir",
                str(step5_dir),
                "--k-values",
                *[str(v) for v in args.k_values],
                "--n-neighbors",
                str(args.n_neighbors),
                "--random-state",
                str(seed),
                "--log-level",
                args.log_level,
            ]
        )

        print(f"[seed {seed}] running Step 6")
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "step6_characterize_clusters.py"),
                "--datasets",
                *args.datasets,
                "--step4-dir",
                str(step4_dir),
                "--step5-dir",
                str(step5_dir),
                "--output-dir",
                str(step6_dir),
                "--log-level",
                args.log_level,
            ]
        )

        print(f"[seed {seed}] running Step 7")
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "step7_validate_test_set.py"),
                "--datasets",
                *args.datasets,
                "--step4-dir",
                str(step4_dir),
                "--step5-dir",
                str(step5_dir),
                "--step6-dir",
                str(step6_dir),
                "--output-dir",
                str(step7_dir),
                "--log-level",
                args.log_level,
            ]
        )

        for dataset in args.datasets:
            row, proportions_df = summarize_dataset_run(dataset=dataset, seed=seed, run_dir=run_dir)
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
        f"Datasets: {', '.join(args.datasets)}",
        "",
    ]
    for dataset in args.datasets:
        dataset_df = summary_df[summary_df["dataset"] == dataset].copy()
        report_lines.append(build_dataset_report(dataset, dataset_df))
        report_lines.append("")

    report_txt = summary_dir / "repeated_validation_summary.txt"
    report_txt.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    manifest = {
        "datasets": args.datasets,
        "seeds": seeds,
        "step3_dir": str(args.step3_dir),
        "output_dir": str(output_dir),
        "summary_csv": str(summary_csv),
        "proportions_csv": str(proportions_csv) if proportions_df is not None else None,
        "summary_report": str(report_txt),
        "parameters": {
            "test_size": args.test_size,
            "k_values": args.k_values,
            "n_neighbors": args.n_neighbors,
            "log_level": args.log_level,
        },
    }
    (summary_dir / "repeated_validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved run metrics: {summary_csv}")
    print(f"Saved cluster proportions: {proportions_csv}")
    print(f"Saved summary report: {report_txt}")


if __name__ == "__main__":
    main()
