"""One-off helper: create a Step 4 metadata JSON that points to VIF-filtered features."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

step4_path = PROJECT_ROOT / "outputs" / "step4" / "ucsf" / "ucsf_step4_split_metadata.json"
step4b_path = PROJECT_ROOT / "outputs" / "step4b" / "ucsf" / "ucsf_step4b_metadata.json"

step4_meta = json.loads(step4_path.read_text(encoding="utf-8"))
step4b_meta = json.loads(step4b_path.read_text(encoding="utf-8"))

# Override feature columns and paths with VIF-filtered versions.
step4_meta["feature_columns"] = step4b_meta["selected_feature_columns"]
step4_meta["outputs"]["train_features"] = step4b_meta["outputs"]["train_features"]
step4_meta["outputs"]["test_features"] = step4b_meta["outputs"]["test_features"]
step4_meta["vif_filtering"] = {
    "applied": True,
    "vif_threshold": step4b_meta["vif_threshold"],
    "original_feature_count": step4b_meta["original_feature_count"],
    "filtered_feature_count": step4b_meta["total_selected_features"],
}

out_path = PROJECT_ROOT / "outputs" / "step4" / "ucsf" / "ucsf_step4_split_metadata_vif.json"
out_path.write_text(json.dumps(step4_meta, indent=2), encoding="utf-8")
print("Saved:", out_path)
print("Features:", len(step4_meta["feature_columns"]))
