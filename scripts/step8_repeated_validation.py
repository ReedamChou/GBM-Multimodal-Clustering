from __future__ import annotations

import argparse
from pathlib import Path

from repeated_evaluation_runner import PROJECT_ROOT, run_repeated_validation


DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "repeated-validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 8: repeated validation over many outer train/test splits using the leakage-safe pipeline. "
            "This is now a thin wrapper over the shared repeated-evaluation runner used by Step 10 as well."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=["ucsf", "upenn"], choices=["ucsf", "upenn"])
    parser.add_argument("--step3-dir", type=Path, default=DEFAULT_STEP3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit random seeds to run.")
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--inner-test-size", type=float, default=0.25)
    parser.add_argument("--k-values", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--gap-refs", type=int, default=5)
    parser.add_argument("--stability-resamples", type=int, default=8)
    parser.add_argument("--stability-sample-fraction", type=float, default=0.80)
    parser.add_argument("--max-missing-feature-frac", type=float, default=1.0)
    parser.add_argument("--corr-prune-threshold", type=float, default=1.01)
    parser.add_argument("--winsorize-lower-quantile", type=float, default=0.0)
    parser.add_argument("--winsorize-upper-quantile", type=float, default=1.0)
    parser.add_argument("--scaler-type", choices=["standard", "robust"], default="standard")
    parser.add_argument("--power-transform", choices=["none", "yeo-johnson"], default="none")
    parser.add_argument("--log-level", type=str, default="WARNING")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.n_runs))
    run_repeated_validation(
        datasets=args.datasets,
        seeds=seeds,
        step3_dir=args.step3_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        inner_test_size=args.inner_test_size,
        k_values=args.k_values,
        n_neighbors=args.n_neighbors,
        gap_refs=args.gap_refs,
        stability_resamples=args.stability_resamples,
        stability_sample_fraction=args.stability_sample_fraction,
        log_level=args.log_level,
        max_missing_feature_frac=args.max_missing_feature_frac,
        corr_prune_threshold=args.corr_prune_threshold,
        winsorize_lower_quantile=args.winsorize_lower_quantile,
        winsorize_upper_quantile=args.winsorize_upper_quantile,
        scaler_type=args.scaler_type,
        power_transform=args.power_transform,
    )


if __name__ == "__main__":
    main()
