from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "training"
OUTPUT_ROOT = TRAINING_ROOT / "outputs"
MODELS_ROOT = TRAINING_ROOT / "models"
CACHE_ROOT = TRAINING_ROOT / "cache"
MPL_CACHE_ROOT = CACHE_ROOT / "matplotlib"

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_ROOT))

INPUT_DEFAULT = PROJECT_ROOT / "outputs" / "new" / "step3_ucsf_preprocessed_features.csv"

print("INPUT_DEFAULT= ", INPUT_DEFAULT)

IMAGING_FEATURES = [
    "global_nc_en_ratio",
    "global_ed_en_ratio",
    "global_ed_total_ratio",
    "tumor_burden_index",
    "frontal_ed_ratio",
    "frontal_en_ratio",
    "frontal_nc_ratio",
    "temporal_ed_ratio",
    "temporal_nc_ratio",
    "parietal_ed_ratio",
    "parietal_nc_ratio",
    "occipital_ed_ratio",
    "occipital_nc_ratio",
    "dominant_brain_lobe_frontal",
    "dominant_brain_lobe_occipital",
    "dominant_brain_lobe_parietal",
    "dominant_brain_lobe_temporal",
]
TARGET = "risk_cluster_label"
LABEL_MAP = {
    "high_risk": 0,
    "low_risk": 1,
}
LABEL_NAMES = ["high_risk", "low_risk"]

def ensure_directories() -> dict[str, Path]:
    paths = {
        "root": TRAINING_ROOT,
        "configs": TRAINING_ROOT / "configs",
        "scripts": TRAINING_ROOT / "scripts",
        "models": MODELS_ROOT,
        "outputs": OUTPUT_ROOT,
        "figures": OUTPUT_ROOT / "figures",
        "metrics": OUTPUT_ROOT / "metrics",
        "reports": OUTPUT_ROOT / "reports",
        "tables": OUTPUT_ROOT / "tables",
        "logs": OUTPUT_ROOT / "logs",
        "cache": CACHE_ROOT,
        "mpl_cache": MPL_CACHE_ROOT,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, object]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    dropped_null_labels = int(df[TARGET].isna().sum())
    clean_df = df.dropna(subset=[TARGET]).copy()

    missing_features = [feature for feature in IMAGING_FEATURES if feature not in clean_df.columns]
    if missing_features:
        raise ValueError(f"Missing required imaging features: {missing_features}")

    X = clean_df[IMAGING_FEATURES].copy()
    y = clean_df[TARGET].map(LABEL_MAP)
    if y.isna().any():
        raise ValueError("Found labels outside the expected label map.")

    summary = {
        "input_rows": int(len(df)),
        "rows_after_null_label_drop": int(len(clean_df)),
        "dropped_null_labels": dropped_null_labels,
        "n_features": len(IMAGING_FEATURES),
        "class_distribution": clean_df[TARGET].value_counts().reindex(LABEL_NAMES).fillna(0).astype(int).to_dict(),
        "total_feature_nulls": int(X.isna().sum().sum()),
        "feature_range_min": float(X.min().min()),
        "feature_range_max": float(X.max().max()),
    }
    return clean_df, X, y.astype(int), summary


def save_json(payload: dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_markdown(text: str, output_path: Path) -> None:
    output_path.write_text(text, encoding="utf-8")


def detect_xgboost_device(random_state: int = 42) -> dict[str, object]:
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is not None:
        nvidia_check = subprocess.run(
            [nvidia_smi_path, "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if nvidia_check.returncode != 0:
            return {
                "requested_device": "cuda",
                "selected_device": "cpu",
                "gpu_available": False,
                "reason": "nvidia-smi could not communicate with the NVIDIA driver.",
                "stderr": nvidia_check.stderr.strip(),
            }

    X_probe = np.random.default_rng(random_state).random((32, 4))
    y_probe = np.random.default_rng(random_state + 1).integers(0, 3, size=32)
    probe_messages: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = xgb.XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=5,
                max_depth=2,
                learning_rate=0.1,
                tree_method="hist",
                device="cuda",
                eval_metric="mlogloss",
                random_state=random_state,
                verbosity=0,
            )
            model.fit(X_probe, y_probe)
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "requested_device": "cuda",
                "selected_device": "cpu",
                "gpu_available": False,
                "reason": f"XGBoost CUDA probe failed: {exc!r}",
            }

    for warning in caught:
        probe_messages.append(str(warning.message))

    gpu_warning_tokens = [
        "No visible GPU is found",
        "Device is changed from GPU to CPU",
    ]
    if any(token in message for token in gpu_warning_tokens for message in probe_messages):
        return {
            "requested_device": "cuda",
            "selected_device": "cpu",
            "gpu_available": False,
            "reason": "XGBoost did not find a visible GPU and fell back to CPU.",
            "warnings": probe_messages,
        }

    return {
        "requested_device": "cuda",
        "selected_device": "cuda",
        "gpu_available": True,
        "reason": "XGBoost CUDA probe succeeded.",
        "warnings": probe_messages,
    }


def xgb_base_params(
    device: str,
    random_state: int,
    n_jobs: int,
    early_stopping_rounds: int | None = None,
) -> dict[str, object]:
    params: dict[str, object] = {
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    return params
