from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4"


def normalize_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def safe_mode(series: pd.Series, default: Any = "unknown") -> Any:
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return default
    return mode_values.iloc[0]


def map_sex_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.isin(["m", "male"])] = 1
    out[s.isin(["f", "female"])] = 0
    return out


def map_mgmt_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("positive", na=False) | s.str.contains("methylated", na=False)] = 1
    out[s.str.contains("negative", na=False) | s.str.contains("unmethylated", na=False)] = 0
    return out


def map_idh_to_binary(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out[s.str.contains("wild", na=False)] = 0
    out[s.str.contains("mutat", na=False) | s.str.contains("idh1 p\\.", na=False) | s.str.contains("idh2 p\\.", na=False)] = 1
    return out


def canonicalize_lobe(series: pd.Series) -> pd.Series:
    s = normalize_string_series(series)
    out = pd.Series(pd.NA, index=series.index, dtype="string")
    out[s.str.contains("frontal", na=False)] = "frontal"
    out[s.str.contains("temporal", na=False)] = "temporal"
    out[s.str.contains("parietal", na=False)] = "parietal"
    out[s.str.contains("occipital", na=False)] = "occipital"
    return out


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
    master_candidates = [dataset_dir / f"{dataset}_master_table_step3.csv"]
    metadata_candidates = [dataset_dir / f"{dataset}_step3_metadata.json"]
    return find_first_existing(master_candidates), find_first_existing(metadata_candidates)


def build_os_bins(os_series: pd.Series, bins: int) -> pd.Series:
    os_numeric = pd.to_numeric(os_series, errors="coerce")
    valid = os_numeric.notna().sum()
    if valid < 10:
        return pd.Series(["os_missing"] * len(os_series), index=os_series.index, dtype="string")

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


def encode_binary_with_train_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_col: str,
    output_col: str,
    mapper,
    unknown_threshold: float = 0.10,
) -> tuple[str, str | None, dict]:
    train_mapped = mapper(train_df[source_col])
    test_mapped = mapper(test_df[source_col])

    missing_rate = float(train_mapped.isna().mean())
    unknown_col = None
    if missing_rate > unknown_threshold:
        unknown_col = f"{output_col}_unknown"
        train_df[unknown_col] = train_mapped.isna().astype(int)
        test_df[unknown_col] = test_mapped.isna().astype(int)

    fill_value = int(safe_mode(train_mapped, default=0))
    train_df[output_col] = train_mapped.fillna(fill_value).astype(int)
    test_df[output_col] = test_mapped.fillna(fill_value).astype(int)

    return output_col, unknown_col, {
        "source_column": source_col,
        "fill_value": fill_value,
        "train_missing_rate": missing_rate,
        "unknown_indicator_added": unknown_col is not None,
    }


def encode_lobe_with_train_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    source_col: str,
    unknown_threshold: float = 0.10,
) -> tuple[list[str], dict]:
    train_lobe = canonicalize_lobe(train_df[source_col])
    test_lobe = canonicalize_lobe(test_df[source_col])

    missing_rate = float(train_lobe.isna().mean())
    fill_value = "unknown" if missing_rate > unknown_threshold else str(safe_mode(train_lobe, default="unknown"))

    train_lobe = train_lobe.fillna(fill_value)
    test_lobe = test_lobe.fillna(fill_value)

    train_df["dominant_lobe_clean"] = train_lobe
    test_df["dominant_lobe_clean"] = test_lobe

    created_columns = []
    for lobe_name in ["frontal", "temporal", "parietal", "occipital"]:
        col_name = f"dominant_lobe_{lobe_name}"
        train_df[col_name] = (train_lobe == lobe_name).astype(int)
        test_df[col_name] = (test_lobe == lobe_name).astype(int)
        created_columns.append(col_name)

    if fill_value == "unknown":
        train_df["dominant_lobe_unknown"] = (train_lobe == "unknown").astype(int)
        test_df["dominant_lobe_unknown"] = (test_lobe == "unknown").astype(int)
        created_columns.append("dominant_lobe_unknown")

    return created_columns, {
        "source_column": source_col,
        "fill_value": fill_value,
        "train_missing_rate": missing_rate,
    }


def fill_string_columns_with_train_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    outcome_cols: set[str],
) -> dict[str, dict]:
    fill_metadata: dict[str, dict] = {}
    for col in train_df.columns:
        if col == id_col or col in outcome_cols:
            continue
        if pd.api.types.is_object_dtype(train_df[col]) or pd.api.types.is_string_dtype(train_df[col]):
            train_missing_rate = float(train_df[col].isna().mean())
            test_missing_rate = float(test_df[col].isna().mean())
            if train_missing_rate == 0 and test_missing_rate == 0:
                continue
            fill_value = "unknown" if train_missing_rate > 0.10 else str(safe_mode(train_df[col], default="unknown"))
            train_df[col] = train_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)
            fill_metadata[col] = {
                "fill_value": fill_value,
                "train_missing_rate": train_missing_rate,
                "test_missing_rate_before_fill": test_missing_rate,
            }
    return fill_metadata


def preprocess_split_with_train_only_rules(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    meta: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    resolved = meta.get("resolved_columns", {})
    id_col = resolved.get("id") or "ID"
    outcome_cols = {
        c
        for c in [resolved.get("os"), resolved.get("censor"), resolved.get("survival_status")]
        if c is not None
    }

    binary_columns: list[str] = []
    binary_encoding_meta: dict[str, dict] = {}
    for key, mapper, out_col in [
        ("sex", map_sex_to_binary, "sex_bin"),
        ("mgmt", map_mgmt_to_binary, "mgmt_bin"),
        ("idh", map_idh_to_binary, "idh_bin"),
    ]:
        src = resolved.get(key)
        if src is None or src not in train_df.columns or src not in test_df.columns:
            continue
        encoded_col, unknown_col, enc_meta = encode_binary_with_train_rules(
            train_df=train_df,
            test_df=test_df,
            source_col=src,
            output_col=out_col,
            mapper=mapper,
        )
        binary_columns.append(encoded_col)
        if unknown_col is not None:
            binary_columns.append(unknown_col)
        binary_encoding_meta[out_col] = enc_meta

    dominant_lobe_meta = None
    dominant_lobe_col = resolved.get("dominant_lobe")
    if dominant_lobe_col is not None and dominant_lobe_col in train_df.columns and dominant_lobe_col in test_df.columns:
        created_cols, dominant_lobe_meta = encode_lobe_with_train_rules(train_df, test_df, dominant_lobe_col)
        binary_columns.extend(created_cols)

    categorical_fill_meta = fill_string_columns_with_train_rules(train_df, test_df, id_col, outcome_cols)

    numeric_feature_cols = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c != id_col and c not in outcome_cols
    ]
    continuous_cols = [
        c
        for c in numeric_feature_cols
        if c not in binary_columns and train_df[c].nunique(dropna=True) > 2
    ]
    discrete_numeric_cols = [c for c in numeric_feature_cols if c not in continuous_cols]

    continuous_fill_values: dict[str, float] = {}
    for col in continuous_cols:
        train_numeric = pd.to_numeric(train_df[col], errors="coerce")
        test_numeric = pd.to_numeric(test_df[col], errors="coerce")
        fill_value = float(train_numeric.median(skipna=True))
        if np.isnan(fill_value):
            fill_value = 0.0
        train_df[col] = train_numeric.fillna(fill_value)
        test_df[col] = test_numeric.fillna(fill_value)
        continuous_fill_values[col] = fill_value

    discrete_fill_values: dict[str, float | int] = {}
    for col in discrete_numeric_cols:
        train_numeric = pd.to_numeric(train_df[col], errors="coerce")
        test_numeric = pd.to_numeric(test_df[col], errors="coerce")
        fill_value = safe_mode(train_numeric, default=0)
        if pd.isna(fill_value):
            fill_value = 0
        train_df[col] = train_numeric.fillna(fill_value)
        test_df[col] = test_numeric.fillna(fill_value)
        discrete_fill_values[col] = int(fill_value) if float(fill_value).is_integer() else float(fill_value)

    scaler = StandardScaler()
    if continuous_cols:
        train_df = train_df.astype({col: "float64" for col in continuous_cols}, copy=False)
        test_df = test_df.astype({col: "float64" for col in continuous_cols}, copy=False)
        train_df.loc[:, continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
        test_df.loc[:, continuous_cols] = scaler.transform(test_df[continuous_cols])

    feature_cols = [
        c
        for c in train_df.columns
        if c != id_col and c not in outcome_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    preprocessing_meta = {
        "id_column": id_col,
        "outcome_columns": sorted(outcome_cols),
        "binary_columns": sorted(binary_columns),
        "continuous_columns": sorted(continuous_cols),
        "discrete_numeric_columns": sorted(discrete_numeric_cols),
        "feature_columns": feature_cols,
        "binary_encoding": binary_encoding_meta,
        "dominant_lobe_encoding": dominant_lobe_meta,
        "categorical_fill": categorical_fill_meta,
        "continuous_median_imputation": continuous_fill_values,
        "discrete_numeric_fill": discrete_fill_values,
        "scaler": scaler,
    }
    return train_df, test_df, preprocessing_meta


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
    mgmt_source_col = resolved.get("mgmt")
    mgmt_strata_col = None

    if id_col not in df.columns:
        raise ValueError(f"{dataset}: ID column not found in master table: {id_col}")

    if mgmt_source_col is not None and mgmt_source_col in df.columns:
        mgmt_strata_col = "__mgmt_strata_bin__"
        df[mgmt_strata_col] = map_mgmt_to_binary(df[mgmt_source_col])

    logger.info("Rows=%s | columns=%s", len(df), len(df.columns))

    train_df, test_df, strat_info = split_with_best_stratification(
        df=df,
        id_col=id_col,
        mgmt_col=mgmt_strata_col,
        os_col=os_col,
        test_size=test_size,
        random_state=random_state,
        logger=logger,
    )

    if mgmt_strata_col is not None:
        train_df = train_df.drop(columns=[mgmt_strata_col], errors="ignore")
        test_df = test_df.drop(columns=[mgmt_strata_col], errors="ignore")

    train_df, test_df, preprocessing_meta = preprocess_split_with_train_only_rules(train_df, test_df, meta)

    feature_cols = preprocessing_meta["feature_columns"]
    continuous_cols = preprocessing_meta["continuous_columns"]
    scaler = preprocessing_meta["scaler"]
    logger.info("Prepared leakage-safe features=%s | continuous_cols=%s", len(feature_cols), len(continuous_cols))

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
            "preprocessing_version": "train_only_step4_v2",
            "scaler": scaler,
            "continuous_columns": continuous_cols,
            "feature_columns": feature_cols,
            "binary_columns": preprocessing_meta["binary_columns"],
            "discrete_numeric_columns": preprocessing_meta["discrete_numeric_columns"],
            "binary_encoding": preprocessing_meta["binary_encoding"],
            "dominant_lobe_encoding": preprocessing_meta["dominant_lobe_encoding"],
            "categorical_fill": preprocessing_meta["categorical_fill"],
            "continuous_median_imputation": preprocessing_meta["continuous_median_imputation"],
            "discrete_numeric_fill": preprocessing_meta["discrete_numeric_fill"],
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
        "mgmt_column_for_stratification": mgmt_source_col,
        "split": {
            "train_rows": int(train_df.shape[0]),
            "test_rows": int(test_df.shape[0]),
            "test_size": test_size,
            "random_state": random_state,
            "stratification": strat_info,
        },
        "feature_columns": feature_cols,
        "continuous_columns": continuous_cols,
        "binary_columns": preprocessing_meta["binary_columns"],
        "discrete_numeric_columns": preprocessing_meta["discrete_numeric_columns"],
        "preprocessing": {
            "fit_scope": "train_only",
            "binary_encoding": preprocessing_meta["binary_encoding"],
            "dominant_lobe_encoding": preprocessing_meta["dominant_lobe_encoding"],
            "categorical_fill": preprocessing_meta["categorical_fill"],
            "continuous_median_imputation": preprocessing_meta["continuous_median_imputation"],
            "discrete_numeric_fill": preprocessing_meta["discrete_numeric_fill"],
        },
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

    if "mgmt_bin" in train_df.columns and "mgmt_bin" in test_df.columns:
        train_mgmt_rate = float(pd.to_numeric(train_df["mgmt_bin"], errors="coerce").mean())
        test_mgmt_rate = float(pd.to_numeric(test_df["mgmt_bin"], errors="coerce").mean())
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
            "fit train-only imputers, encoders, and scaler before exporting clustering features."
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