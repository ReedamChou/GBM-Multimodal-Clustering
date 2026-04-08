from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from iteration_paths import next_iteration_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def pipeline_python() -> str:
    preferred = PROJECT_ROOT / "venv-gbm" / "bin" / "python"
    if preferred.exists():
        return str(preferred)
    return sys.executable


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the maintained GBM clustering pipeline into the next available results/iteration-N directory, "
            "then rebuild the project summary artifacts from that iteration."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["ucsf", "upenn"], choices=["ucsf", "upenn"])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--inner-test-size", type=float, default=0.25)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--gap-refs", type=int, default=5)
    parser.add_argument("--stability-resamples", type=int, default=8)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.80)
    parser.add_argument("--repeated-runs", type=int, default=30)
    parser.add_argument("--repeated-seed-start", type=int, default=1)
    parser.add_argument("--step9-bootstraps", type=int, default=100)
    parser.add_argument("--step9-bootstrap-fraction", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-missing-feature-frac", type=float, default=1.0)
    parser.add_argument("--corr-prune-threshold", type=float, default=1.01)
    parser.add_argument("--winsorize-lower-quantile", type=float, default=0.0)
    parser.add_argument("--winsorize-upper-quantile", type=float, default=1.0)
    parser.add_argument("--scaler-type", choices=["standard", "robust"], default="standard")
    parser.add_argument("--power-transform", choices=["none", "yeo-johnson"], default="none")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iteration_number, iteration_dir = next_iteration_info(args.results_root)
    iteration_dir.mkdir(parents=True, exist_ok=False)

    step3_dir = iteration_dir / "step3"
    step4_dir = iteration_dir / "step4"
    step5_dir = iteration_dir / "step5"
    step6_dir = iteration_dir / "step6"
    step7_dir = iteration_dir / "step7"
    repeated_dir = iteration_dir / "repeated-validation"
    step9_dir = iteration_dir / "step9"
    step11_dir = iteration_dir / "step11"
    reports_dir = iteration_dir / "reports"
    report_data_path = reports_dir / "project_report_data.json"
    iteration_summary_path = reports_dir / "summarised-results.txt"
    iteration_presentation_path = reports_dir / "presentation-results.txt"
    iteration_tex_path = reports_dir / "professor_results_summary.tex"

    print(f"Running maintained pipeline into {iteration_dir}")

    python_bin = pipeline_python()

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step3_build_master_feature_table.py"),
            "--output-dir",
            str(step3_dir),
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step4_split_train_test.py"),
            "--datasets",
            *args.datasets,
            "--step3-dir",
            str(step3_dir),
            "--output-dir",
            str(step4_dir),
            "--test-size",
            str(args.test_size),
            "--random-state",
            str(args.random_state),
            "--log-level",
            args.log_level,
            "--max-missing-feature-frac",
            str(args.max_missing_feature_frac),
            "--corr-prune-threshold",
            str(args.corr_prune_threshold),
            "--winsorize-lower-quantile",
            str(args.winsorize_lower_quantile),
            "--winsorize-upper-quantile",
            str(args.winsorize_upper_quantile),
            "--scaler-type",
            args.scaler_type,
            "--power-transform",
            args.power_transform,
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step5_spectral_clustering.py"),
            "--datasets",
            *args.datasets,
            "--step4-dir",
            str(step4_dir),
            "--output-dir",
            str(step5_dir),
            "--k-values",
            *[str(v) for v in args.k_values],
            "--inner-test-size",
            str(args.inner_test_size),
            "--n-neighbors",
            str(args.n_neighbors),
            "--gap-refs",
            str(args.gap_refs),
            "--stability-resamples",
            str(args.stability_resamples),
            "--stability-sample-fraction",
            str(args.stability_sample_fraction),
            "--random-state",
            str(args.random_state),
            "--log-level",
            args.log_level,
        ]
    )

    run_command(
        [
            python_bin,
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

    run_command(
        [
            python_bin,
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

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step8_repeated_validation.py"),
            "--datasets",
            *args.datasets,
            "--step3-dir",
            str(step3_dir),
            "--output-dir",
            str(repeated_dir),
            "--n-runs",
            str(args.repeated_runs),
            "--seed-start",
            str(args.repeated_seed_start),
            "--test-size",
            str(args.test_size),
            "--inner-test-size",
            str(args.inner_test_size),
            "--k-values",
            *[str(v) for v in args.k_values],
            "--n-neighbors",
            str(args.n_neighbors),
            "--gap-refs",
            str(args.gap_refs),
            "--stability-resamples",
            str(args.stability_resamples),
            "--stability-sample-fraction",
            str(args.stability_sample_fraction),
            "--max-missing-feature-frac",
            str(args.max_missing_feature_frac),
            "--corr-prune-threshold",
            str(args.corr_prune_threshold),
            "--winsorize-lower-quantile",
            str(args.winsorize_lower_quantile),
            "--winsorize-upper-quantile",
            str(args.winsorize_upper_quantile),
            "--scaler-type",
            args.scaler_type,
            "--power-transform",
            args.power_transform,
            "--log-level",
            args.log_level,
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step9_cluster_stability.py"),
            "--datasets",
            *args.datasets,
            "--step4-dir",
            str(step4_dir),
            "--step5-dir",
            str(step5_dir),
            "--output-dir",
            str(step9_dir),
            "--n-bootstraps",
            str(args.step9_bootstraps),
            "--bootstrap-fraction",
            str(args.step9_bootstrap_fraction),
            "--random-state",
            str(args.random_state),
            "--log-level",
            args.log_level,
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "step11_cohort_shift_analysis.py"),
            "--step3-dir",
            str(step3_dir),
            "--output-dir",
            str(step11_dir),
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "build_project_report_data.py"),
            "--outputs-dir",
            str(iteration_dir),
            "--repeated-summary-dir",
            str(repeated_dir / "summary"),
            "--output-json",
            str(report_data_path),
        ]
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "render_project_reports.py"),
            "--report-data",
            str(report_data_path),
            "--summary-out",
            str(PROJECT_ROOT / "summarised-results.txt"),
            "--presentation-out",
            str(PROJECT_ROOT / "presentation-results.txt"),
            "--tex-out",
            str(PROJECT_ROOT / "professor_results_summary.tex"),
        ]
    )

    run_command(
        [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "render_project_reports.py"),
            "--report-data",
            str(report_data_path),
            "--summary-out",
            str(iteration_summary_path),
            "--presentation-out",
            str(iteration_presentation_path),
            "--tex-out",
            str(iteration_tex_path),
        ]
    )

    print(f"Completed iteration-{iteration_number}: {iteration_dir}")


if __name__ == "__main__":
    main()
