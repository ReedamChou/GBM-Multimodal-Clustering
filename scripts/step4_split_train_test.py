from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step4.{log_file.parent.name}")
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


def find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the expected files were found: {paths}")


def build_paths(step3_dir: Path, dataset: str) -> tuple[Path, Path]:
    dataset_dir = step3_dir / dataset
    master_candidates = [
        dataset_dir / f"{dataset}_master_table_step3.csv",
    ]
    metadata_candidates = [
        dataset_dir / f"{dataset}_step3_metadata.json",
    ]
    return find_first_existing(master_candidates), find_first_existing(metadata_candidates)


def detect_already_scaled(df: pd.DataFrame, continuous_cols: list[str]) -> bool:
    if not continuous_cols:
        return False

    checks = []
    for col in continuous_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        mean_close = abs(float(s.mean())) < 0.2
        std_close = 0.8 < float(s.std(ddof=0)) < 1.2
        checks.append(mean_close and std_close)

    if not checks:
        return False
    return (sum(checks) / len(checks)) >= 0.8


def build_os_bins(os_series: pd.Series, bins: int) -> pd.Series:
    os_numeric = pd.to_numeric(os_series, errors="coerce")
    valid = os_numeric.notna().sum()
    if valid < 10:
        return pd.Series(["os_missing"] * len(os_series), index=os_series.index, dtype="string")

    # Create quantile bins only from non-missing values, then map back to full index.
    tmp = pd.Series(pd.NA, index=os_series.index, dtype="string")
    try:
        binned = pd.qcut(os_numeric.dropna(), q=bins, duplicates="drop")
    except ValueError:
        return pd.Series(["os_missing"] * len(os_series), index=os_series.index, dtype="string")

    tmp.loc[binned.index] = binned.astype("string")
    tmp = tmp.fillna("os_missing")
    return tmp


def build_strata_labels(df: pd.DataFrame, mgmt_col: str | None, os_col: str | None, bins: int) -> pd.Series:
    if mgmt_col is None and os_col is None:
        return pd.Series(["all"] * len(df), index=df.index, dtype="string")

    mgmt_label = None
    if mgmt_col is not None and mgmt_col in df.columns:
        mgmt_numeric = pd.to_numeric(df[mgmt_col], errors="coerce")
        mgmt_label = mgmt_numeric.fillna(-1).astype(int).astype(str).radd("mgmt_")

    os_label = None
    if os_col is not None and os_col in df.columns:
        os_label = build_os_bins(df[os_col], bins).radd("os_")

    if mgmt_label is not None and os_label is not None:
        return (mgmt_label + "|" + os_label).astype("string")
    if mgmt_label is not None:
        return mgmt_label.astype("string")
    return os_label.astype("string")


def split_with_best_stratification(
    df: pd.DataFrame,
    id_col: str,
    mgmt_col: str | None,
    os_col: str | None,
    test_size: float,
    random_state: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    n = len(df)
    n_test = int(math.ceil(test_size * n))

    strategies: list[tuple[str, str | None, str | None]] = [
        ("mgmt_plus_os", mgmt_col, os_col),
        ("mgmt_only", mgmt_col, None),
        ("os_only", None, os_col),
    ]

    for strategy_name, strategy_mgmt, strategy_os in strategies:
        if strategy_mgmt is None and strategy_os is None:
            continue

        for bins in [10, 8, 6, 5, 4, 3, 2]:
            strata = build_strata_labels(df, strategy_mgmt, strategy_os, bins)
            counts = strata.value_counts(dropna=False)

            if counts.empty:
                continue
            if int(counts.min()) < 2:
                continue

            n_classes = int(counts.shape[0])
            if n_classes > n_test or n_classes > (n - n_test):
                continue

            try:
                train_df, test_df = train_test_split(
                    df,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=strata,
                )
                info = {
                    "strategy": strategy_name,
                    "os_bins": bins,
                    "n_classes": n_classes,
                }
                logger.info(
                    "Using stratification strategy=%s os_bins=%s classes=%s",
                    strategy_name,
                    bins,
                    n_classes,
                )
                return train_df, test_df, info
            except ValueError:
                continue

    logger.warning("Could not satisfy robust stratification constraints; using random split.")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )
    return train_df, test_df, {"strategy": "random", "os_bins": None, "n_classes": None}


def process_dataset(
    dataset: str,
    step3_dir: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step4.log", log_level)
    logger.info("Starting Step 4 for dataset=%s", dataset)

    master_path, metadata_path = build_paths(step3_dir, dataset)
    logger.info("Reading master table: %s", master_path)
    logger.info("Reading metadata: %s", metadata_path)

    df = pd.read_csv(master_path)
    meta = load_json(metadata_path)

    resolved = meta.get("resolved_columns", {})
    id_col = resolved.get("id") or "ID"
    os_col = resolved.get("os")
    mgmt_col = "mgmt_bin" if "mgmt_bin" in df.columns else None

    if id_col not in df.columns:
        raise ValueError(f"{dataset}: ID column not found in master table: {id_col}")

    feature_cols = [
        c
        for c in meta.get("clustering_feature_columns", [])
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    continuous_cols = [
        c
        for c in meta.get("continuous_columns", [])
        if c in feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError(f"{dataset}: no usable numeric clustering feature columns found.")

    logger.info("Rows=%s | feature_cols=%s | continuous_cols=%s", len(df), len(feature_cols), len(continuous_cols))

    if detect_already_scaled(df, continuous_cols):
        logger.warning(
            "Many continuous columns already look standardized. "
            "Step 4 will still re-fit a train-only scaler to enforce leakage-safe train/test transformation."
        )

    train_df, test_df, strat_info = split_with_best_stratification(
        df=df,
        id_col=id_col,
        mgmt_col=mgmt_col,
        os_col=os_col,
        test_size=test_size,
        random_state=random_state,
        logger=logger,
    )

    scaler = StandardScaler()
    if continuous_cols:
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df.loc[:, continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
        test_df.loc[:, continuous_cols] = scaler.transform(test_df[continuous_cols])
        logger.info("Applied train-only scaling to %s continuous columns.", len(continuous_cols))

    train_features = train_df[[id_col] + feature_cols].copy()
    test_features = test_df[[id_col] + feature_cols].copy()

    dataset_out.mkdir(parents=True, exist_ok=True)
    train_master_out = dataset_out / f"{dataset}_train_master_table_step4.csv"
    test_master_out = dataset_out / f"{dataset}_test_master_table_step4.csv"
    train_features_out = dataset_out / f"{dataset}_train_clustering_features_step4.csv"
    test_features_out = dataset_out / f"{dataset}_test_clustering_features_step4.csv"
    scaler_out = dataset_out / f"{dataset}_train_scaler_step4.joblib"
    split_meta_out = dataset_out / f"{dataset}_step4_split_metadata.json"

    train_df.to_csv(train_master_out, index=False)
    test_df.to_csv(test_master_out, index=False)
    train_features.to_csv(train_features_out, index=False)
    test_features.to_csv(test_features_out, index=False)

    joblib.dump(
        {
            "scaler": scaler,
            "continuous_columns": continuous_cols,
            "feature_columns": feature_cols,
            "id_column": id_col,
            "dataset": dataset,
            "random_state": random_state,
            "test_size": test_size,
        },
        scaler_out,
    )

    split_meta = {
        "dataset": dataset,
        "source_master_table": str(master_path),
        "source_step3_metadata": str(metadata_path),
        "id_column": id_col,
        "os_column": os_col,
        "mgmt_column_for_stratification": mgmt_col,
        "split": {
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "test_size": test_size,
            "random_state": random_state,
            "stratification": strat_info,
        },
        "feature_columns": feature_cols,
        "continuous_columns": continuous_cols,
        "outputs": {
            "train_master": str(train_master_out),
            "test_master": str(test_master_out),
            "train_features": str(train_features_out),
            "test_features": str(test_features_out),
            "train_scaler": str(scaler_out),
            "log_file": str(dataset_out / "step4.log"),
        },
    }
    split_meta_out.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")

    if mgmt_col and mgmt_col in train_df.columns and mgmt_col in test_df.columns:
        train_mgmt_rate = float(pd.to_numeric(train_df[mgmt_col], errors="coerce").mean())
        test_mgmt_rate = float(pd.to_numeric(test_df[mgmt_col], errors="coerce").mean())
        logger.info("MGMT bin mean train=%.4f | test=%.4f", train_mgmt_rate, test_mgmt_rate)

    if os_col and os_col in train_df.columns and os_col in test_df.columns:
        train_os = pd.to_numeric(train_df[os_col], errors="coerce")
        test_os = pd.to_numeric(test_df[os_col], errors="coerce")
        logger.info(
            "OS median train=%.2f | test=%.2f",
            float(train_os.median(skipna=True)),
            float(test_os.median(skipna=True)),
        )

    logger.info("Saved outputs to %s", dataset_out)
    logger.info("Step 4 complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: split Step 3 master table into train/test with stratification and "
            "fit train-only scaler for continuous clustering features."
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
        help="Directory containing Step 3 outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Step 4 outputs will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Test split fraction, default 0.30.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
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

    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")

    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            step3_dir=args.step3_dir,
            output_dir=args.output_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()