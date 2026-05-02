from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


TRAINING_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = TRAINING_ROOT / "models"


def load_artifacts() -> tuple[object, list[str], dict[int, str]]:
    model = joblib.load(MODELS_ROOT / "best_xgb_risk_classifier.pkl")
    with (MODELS_ROOT / "feature_list.json").open("r", encoding="utf-8") as handle:
        feature_list = json.load(handle)["features"]
    with (MODELS_ROOT / "label_map.json").open("r", encoding="utf-8") as handle:
        raw_map = json.load(handle)["map"]
    inverse_map = {int(value): key for key, value in raw_map.items()}
    return model, feature_list, inverse_map


def predict_risk(input_csv: Path) -> pd.DataFrame:
    model, features, inverse_map = load_artifacts()
    df = pd.read_csv(input_csv)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    preds = model.predict(df[features])
    result = df.copy()
    result["predicted_risk_label"] = [inverse_map[int(pred)] for pred in preds]
    return result


if __name__ == "__main__":
    sample_input = Path("outputs/new/step3_ucsf_preprocessed_features.csv")
    output = predict_risk(sample_input)
    print(output[["patient_id", "predicted_risk_label"]].head().to_string(index=False))
