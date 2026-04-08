from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from iteration_paths import latest_iteration_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_REPORT_DATA = PROJECT_ROOT / "outputs" / "reports" / "project_report_data.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


def default_outputs_dir() -> Path:
    latest = latest_iteration_dir(DEFAULT_RESULTS_ROOT)
    return DEFAULT_OUTPUTS_DIR if latest is None else latest


def default_repeated_summary_dir() -> Path:
    latest = latest_iteration_dir(DEFAULT_RESULTS_ROOT)
    if latest is None:
        return DEFAULT_RESULTS_ROOT / "repeated-validation" / "summary"
    return latest / "repeated-validation" / "summary"


def default_report_json() -> Path:
    latest = latest_iteration_dir(DEFAULT_RESULTS_ROOT)
    if latest is None:
        return DEFAULT_REPORT_DATA
    return latest / "reports" / "project_report_data.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    parsed = parse_float(value)
    return None if parsed is None else int(parsed)


def parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(int(key)): int(counter[key]) for key in sorted(counter)}


def summarize_repeated_metrics(rows: list[dict[str, str]]) -> dict:
    best_k_counter = Counter()
    high_risk_counter = Counter()
    train_sig = 0
    test_sig = 0
    within_10 = 0
    lobe_match = 0
    nc_rank_1 = 0
    nc_rank_1_or_2 = 0
    train_lowest = 0
    test_lowest = 0
    temporal_expected_rows = 0
    temporal_observed_rows = 0
    train_p_values: list[float] = []
    test_p_values: list[float] = []
    pct_diffs: list[float] = []
    train_orders: list[str] = []
    test_orders: list[str] = []

    for row in rows:
        best_k = parse_int(row.get("best_k"))
        high_risk_cluster = parse_int(row.get("high_risk_cluster"))
        test_high_risk_nc_en_rank = parse_int(row.get("test_high_risk_nc_en_rank"))

        if best_k is not None:
            best_k_counter[best_k] += 1
        if high_risk_cluster is not None:
            high_risk_counter[high_risk_cluster] += 1
        if parse_bool(row.get("train_survival_significant")):
            train_sig += 1
        if parse_bool(row.get("test_survival_significant")):
            test_sig += 1
        if parse_bool(row.get("high_risk_within_10_pct_points")):
            within_10 += 1
        if parse_bool(row.get("dominant_lobe_matches_train")):
            lobe_match += 1
        if parse_bool(row.get("train_high_risk_lowest_os")):
            train_lowest += 1
        if parse_bool(row.get("test_high_risk_lowest_os")):
            test_lowest += 1
        if test_high_risk_nc_en_rank == 1:
            nc_rank_1 += 1
        if test_high_risk_nc_en_rank is not None and test_high_risk_nc_en_rank <= 2:
            nc_rank_1_or_2 += 1

        train_p = parse_float(row.get("train_logrank_p"))
        test_p = parse_float(row.get("test_logrank_p"))
        pct_diff = parse_float(row.get("high_risk_pct_point_diff"))
        if train_p is not None:
            train_p_values.append(train_p)
        if test_p is not None:
            test_p_values.append(test_p)
        if pct_diff is not None:
            pct_diffs.append(pct_diff)

        train_order = str(row.get("train_survival_order") or "")
        test_order = str(row.get("test_survival_order") or "")
        if train_order:
            train_orders.append(train_order)
        if test_order:
            test_orders.append(test_order)

        if parse_bool(row.get("temporal_replication_expected")):
            temporal_expected_rows += 1
            if parse_bool(row.get("temporal_replication_observed")):
                temporal_observed_rows += 1

    n_runs = len(rows)
    return {
        "n_runs": n_runs,
        "chosen_k_counts": format_counter(best_k_counter),
        "high_risk_cluster_counts": format_counter(high_risk_counter),
        "train_survival_significant_runs": train_sig,
        "test_survival_significant_runs": test_sig,
        "median_train_logrank_p": median(train_p_values) if train_p_values else None,
        "median_test_logrank_p": median(test_p_values) if test_p_values else None,
        "high_risk_within_10_pct_runs": within_10,
        "median_high_risk_pct_point_diff": median(pct_diffs) if pct_diffs else None,
        "dominant_lobe_matches_train_runs": lobe_match,
        "test_high_risk_nc_en_rank_1_runs": nc_rank_1,
        "test_high_risk_nc_en_rank_1_or_2_runs": nc_rank_1_or_2,
        "train_high_risk_lowest_os_runs": train_lowest,
        "test_high_risk_lowest_os_runs": test_lowest,
        "most_common_train_survival_order": Counter(train_orders).most_common(1)[0][0] if train_orders else "",
        "most_common_test_survival_order": Counter(test_orders).most_common(1)[0][0] if test_orders else "",
        "temporal_replication_expected_runs": temporal_expected_rows,
        "temporal_replication_observed_runs": temporal_observed_rows,
    }


def load_single_split_dataset(dataset: str, outputs_dir: Path) -> dict:
    step3_master_path = outputs_dir / "step3" / dataset / f"{dataset}_master_table_step3.csv"
    step4_meta_path = outputs_dir / "step4" / dataset / f"{dataset}_step4_split_metadata.json"
    step5_selection_path = outputs_dir / "step5" / dataset / f"{dataset}_step5_selection.json"
    step6_meta_path = outputs_dir / "step6" / dataset / f"{dataset}_step6_metadata.json"
    step7_meta_path = outputs_dir / "step7" / dataset / f"{dataset}_step7_metadata.json"
    step9_summary_path = outputs_dir / "step9" / dataset / f"{dataset}_step9_summary.json"

    step3_rows = read_csv_rows(step3_master_path)
    step4_meta = load_json(step4_meta_path)
    step5_selection = load_json(step5_selection_path)
    step6_meta = load_json(step6_meta_path)
    step7_meta = load_json(step7_meta_path)
    step9_summary = load_json(step9_summary_path)
    high_risk = step7_meta.get("high_risk_cluster_validation", {})
    train_sig = high_risk.get("train_signature", {}) or {}
    test_sig = high_risk.get("test_signature", {}) or {}
    proportion_check = high_risk.get("cluster_proportion_check", {}) or {}

    return {
        "step3_shape": {
            "rows": len(step3_rows),
            "columns": len(step3_rows[0]) if step3_rows else 0,
        },
        "single_split": {
            "train_rows": int(step4_meta["split"]["train_rows"]),
            "test_rows": int(step4_meta["split"]["test_rows"]),
            "selected_k": int(step5_selection["best_k"]),
            "train_logrank_p": step6_meta.get("survival_test", {}).get("p_value_raw"),
            "test_logrank_p": step7_meta.get("survival_validation", {}).get("p_value_raw"),
            "high_risk_cluster": int(high_risk.get("cluster_label")),
            "high_risk_label": high_risk.get("label"),
            "high_risk_train_pct": proportion_check.get("train_pct"),
            "high_risk_test_pct": proportion_check.get("test_pct"),
            "high_risk_train_lobe": train_sig.get("dominant_lobe_mode"),
            "high_risk_test_lobe": test_sig.get("dominant_lobe_mode"),
            "test_high_risk_nc_en_rank": test_sig.get("nc_en_rank_among_test_clusters"),
        },
        "stability": step9_summary,
    }


def load_cohort_shift(outputs_dir: Path) -> dict:
    summary = load_json(outputs_dir / "step11" / "step11_summary.json")
    feature_rows = read_csv_rows(outputs_dir / "step11" / "step11_feature_distribution_comparison.csv")
    missingness_rows = read_csv_rows(outputs_dir / "step11" / "step11_missingness_comparison.csv")
    lobe_rows = read_csv_rows(outputs_dir / "step11" / "step11_lobe_distribution_comparison.csv")

    feature_lookup = {row["feature"]: row for row in feature_rows}
    missingness_lookup = {row["column"]: row for row in missingness_rows}
    lobe_lookup = {row["lobe"]: row for row in lobe_rows}

    return {
        "summary": summary,
        "age": feature_lookup["age"],
        "mgmt": feature_lookup["mgmt_methylated_binary"],
        "idh": feature_lookup["idh_mutant_binary"],
        "global_nc_en_ratio": feature_lookup["global_nc_en_ratio"],
        "global_ed_en_ratio": feature_lookup["global_ed_en_ratio"],
        "tumor_burden_index": feature_lookup["tumor_burden_index"],
        "missingness": {
            "occipital_ed_ratio": missingness_lookup["occipital_ed_ratio"],
            "temporal_ed_ratio": missingness_lookup["temporal_ed_ratio"],
            "global_nc_en_ratio": missingness_lookup["global_nc_en_ratio"],
            "global_ed_en_ratio": missingness_lookup["global_ed_en_ratio"],
        },
        "lobe_distribution": {
            "summary": summary["lobe_distribution"],
            "frontal": lobe_lookup["frontal"],
            "temporal": lobe_lookup["temporal"],
        },
    }


def build_report_data(outputs_dir: Path, repeated_summary_dir: Path) -> dict:
    repeated_rows = read_csv_rows(repeated_summary_dir / "repeated_validation_run_metrics.csv")

    datasets: dict[str, dict] = {}
    for dataset in ["ucsf", "upenn"]:
        repeated_dataset_rows = [row for row in repeated_rows if row["dataset"] == dataset]
        datasets[dataset] = {
            **load_single_split_dataset(dataset, outputs_dir),
            "repeated_validation": summarize_repeated_metrics(repeated_dataset_rows),
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "outputs_dir": str(outputs_dir),
            "repeated_summary_dir": str(repeated_summary_dir),
        },
        "pipeline_status": {
            "implemented_steps": [3, 4, 5, 6, 7, 8, 9, 11],
            "single_source_of_truth_reports": True,
        },
        "datasets": datasets,
        "cohort_shift": load_cohort_shift(outputs_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a structured JSON artifact with project-wide results used by the text and LaTeX report renderers."
    )
    parser.add_argument("--outputs-dir", type=Path, default=default_outputs_dir())
    parser.add_argument("--repeated-summary-dir", type=Path, default=default_repeated_summary_dir())
    parser.add_argument("--output-json", type=Path, default=default_report_json())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_data = build_report_data(
        outputs_dir=args.outputs_dir,
        repeated_summary_dir=args.repeated_summary_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Saved project report data: {args.output_json}")


if __name__ == "__main__":
    main()
