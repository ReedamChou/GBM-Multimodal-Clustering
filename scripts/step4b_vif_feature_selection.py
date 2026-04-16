"""Step 4b: VIF-based feature selection.

Runs after Step 4 and before Step 5.  Reads the Step 4 train/test
clustering-feature CSVs, computes VIF on the *continuous* train columns,
iteratively drops the feature with the highest VIF until every remaining
feature satisfies VIF <= threshold, and writes filtered CSVs + a report.

Binary / one-hot columns (sex_bin, mgmt_bin, idh_bin, dominant_lobe_*)
are excluded from VIF computation but kept in the final feature set.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP4_DIR = PROJECT_ROOT / "outputs" / "step4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step4b"


def configure_logger(log_file: Path, level: str) -> logging.Logger:
    logger = logging.getLogger(f"step4b.{log_file.parent.name}")
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


# Columns that are binary / one-hot and should be excluded from VIF.
BINARY_COLUMN_PREFIXES = (
    "sex_bin",
    "mgmt_bin",
    "idh_bin",
    "dominant_lobe_",
    "lobe_assignment_reliable",
    "mgmt_bin_unknown",
)


def is_binary_column(col: str) -> bool:
    """Return True if *col* is a binary / one-hot column that should be
    excluded from VIF computation."""
    for prefix in BINARY_COLUMN_PREFIXES:
        if col == prefix or col.startswith(prefix):
            return True
    return False


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns [feature, vif]."""
    arr = df.values.astype(float)
    records = []
    for i, col in enumerate(df.columns):
        try:
            vif_val = variance_inflation_factor(arr, i)
        except Exception:
            vif_val = np.inf
        records.append({"feature": col, "vif": float(vif_val)})
    return pd.DataFrame(records)


def iterative_vif_filter(
    df: pd.DataFrame,
    threshold: float,
    logger: logging.Logger,
) -> tuple[list[str], pd.DataFrame]:
    """Iteratively drop the feature with the highest VIF > threshold.

    Returns (kept_columns, full_vif_report_df).
    """
    candidates = list(df.columns)
    report_rows: list[dict] = []
    iteration = 0

    while True:
        if len(candidates) < 2:
            break

        vif_df = compute_vif(df[candidates])
        max_row = vif_df.loc[vif_df["vif"].idxmax()]
        max_vif = float(max_row["vif"])
        max_feature = str(max_row["feature"])

        if max_vif <= threshold:
            # All features pass — record their final VIF values.
            for _, row in vif_df.iterrows():
                report_rows.append(
                    {
                        "feature": row["feature"],
                        "vif_final": float(row["vif"]),
                        "dropped": False,
                        "drop_iteration": None,
                        "vif_at_drop": None,
                    }
                )
            break

        # Drop the worst feature.
        iteration += 1
        logger.info(
            "VIF iter %d: dropping '%s' (VIF=%.2f, threshold=%.1f, remaining=%d)",
            iteration,
            max_feature,
            max_vif,
            threshold,
            len(candidates) - 1,
        )
        report_rows.append(
            {
                "feature": max_feature,
                "vif_final": None,
                "dropped": True,
                "drop_iteration": iteration,
                "vif_at_drop": max_vif,
            }
        )
        candidates.remove(max_feature)

    report_df = pd.DataFrame(report_rows)
    kept = [c for c in candidates if c in df.columns]
    return kept, report_df


def process_dataset(
    dataset: str,
    step4_dir: Path,
    output_dir: Path,
    vif_threshold: float,
    log_level: str,
) -> None:
    dataset_out = output_dir / dataset
    logger = configure_logger(dataset_out / "step4b.log", log_level)
    logger.info("Starting Step 4b VIF feature selection for dataset=%s", dataset)

    # --- Load Step 4 metadata and feature files. ---
    step4_meta_path = step4_dir / dataset / f"{dataset}_step4_split_metadata.json"
    if not step4_meta_path.exists():
        raise FileNotFoundError(f"Step 4 metadata not found: {step4_meta_path}")

    step4_meta = load_json(step4_meta_path)
    id_col = step4_meta.get("id_column", "patient_id")
    feature_cols = step4_meta.get("feature_columns", [])

    train_path = step4_dir / dataset / f"{dataset}_train_clustering_features_step4.csv"
    test_path = step4_dir / dataset / f"{dataset}_test_clustering_features_step4.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Step 4 train features not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Step 4 test features not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate binary columns from continuous candidates.
    binary_cols = [c for c in feature_cols if is_binary_column(c) and c in train_df.columns]
    continuous_cols = [c for c in feature_cols if not is_binary_column(c) and c in train_df.columns]

    logger.info(
        "Total features=%d | binary (excluded from VIF)=%d | continuous (VIF candidates)=%d",
        len(feature_cols),
        len(binary_cols),
        len(continuous_cols),
    )

    # --- Ensure continuous columns are numeric and clean. ---
    train_continuous = train_df[continuous_cols].copy()
    for col in continuous_cols:
        train_continuous[col] = pd.to_numeric(train_continuous[col], errors="coerce")
    # Drop columns that are all NaN or zero variance (VIF would fail).
    valid_cols = [
        c
        for c in continuous_cols
        if train_continuous[c].notna().sum() > 2 and train_continuous[c].std(skipna=True) > 1e-12
    ]
    train_continuous = train_continuous[valid_cols].fillna(0.0)

    logger.info("Valid continuous features for VIF: %d", len(valid_cols))

    # --- Iterative VIF filtering. ---
    kept_continuous, vif_report = iterative_vif_filter(
        train_continuous,
        threshold=vif_threshold,
        logger=logger,
    )

    # Final selected features = kept continuous + all binary.
    selected_features = kept_continuous + binary_cols
    logger.info(
        "VIF selection complete: continuous kept=%d, binary kept=%d, total=%d (from original %d)",
        len(kept_continuous),
        len(binary_cols),
        len(selected_features),
        len(feature_cols),
    )

    # --- Write outputs. ---
    dataset_out.mkdir(parents=True, exist_ok=True)

    # Filtered feature CSVs.
    train_out = dataset_out / f"{dataset}_train_clustering_features_step4b.csv"
    test_out = dataset_out / f"{dataset}_test_clustering_features_step4b.csv"
    train_df[[id_col] + selected_features].to_csv(train_out, index=False)
    test_df[[id_col] + selected_features].to_csv(test_out, index=False)

    # VIF report.
    vif_report_out = dataset_out / f"{dataset}_step4b_vif_report.csv"
    vif_report.to_csv(vif_report_out, index=False)

    # Metadata JSON.
    metadata = {
        "dataset": dataset,
        "step4_metadata": str(step4_meta_path),
        "vif_threshold": vif_threshold,
        "original_feature_count": len(feature_cols),
        "continuous_candidates": len(valid_cols),
        "continuous_kept": len(kept_continuous),
        "binary_kept": len(binary_cols),
        "total_selected_features": len(selected_features),
        "selected_feature_columns": selected_features,
        "binary_columns_excluded_from_vif": binary_cols,
        "id_column": id_col,
        "outputs": {
            "train_features": str(train_out),
            "test_features": str(test_out),
            "vif_report": str(vif_report_out),
            "log_file": str(dataset_out / "step4b.log"),
        },
    }
    metadata_out = dataset_out / f"{dataset}_step4b_metadata.json"
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved filtered train features: %s", train_out)
    logger.info("Saved filtered test features: %s", test_out)
    logger.info("Saved VIF report: %s", vif_report_out)
    logger.info("Saved metadata: %s", metadata_out)
    logger.info("Step 4b complete for dataset=%s", dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4b: VIF-based feature selection.  Iteratively drops "
            "correlated continuous features until all VIF <= threshold."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ucsf"],
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Step 4b outputs will be written.",
    )
    parser.add_argument(
        "--vif-threshold",
        type=float,
        default=5.0,
        help="VIF threshold. Features with VIF > threshold are dropped iteratively. Default 5.",
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
            output_dir=args.output_dir,
            vif_threshold=args.vif_threshold,
            log_level=args.log_level,
        )


if __name__ == "__main__":
    main()
