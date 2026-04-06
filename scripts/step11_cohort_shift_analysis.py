from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pipeline_preprocessing import canonicalize_lobe, map_idh_to_binary, map_mgmt_to_binary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STEP3_DIR = PROJECT_ROOT / "outputs" / "step3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "step11"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_paths(step3_dir: Path, dataset: str) -> tuple[Path, Path]:
    dataset_dir = step3_dir / dataset
    master_path = dataset_dir / f"{dataset}_master_table_step3.csv"
    meta_path = dataset_dir / f"{dataset}_step3_metadata.json"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing Step 3 master table: {master_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Step 3 metadata: {meta_path}")
    return master_path, meta_path


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float | None:
    a_num = pd.to_numeric(a, errors="coerce").dropna()
    b_num = pd.to_numeric(b, errors="coerce").dropna()
    if a_num.empty or b_num.empty:
        return None

    mean_a = float(a_num.mean())
    mean_b = float(b_num.mean())
    std_a = float(a_num.std(ddof=1)) if len(a_num) > 1 else 0.0
    std_b = float(b_num.std(ddof=1)) if len(b_num) > 1 else 0.0
    pooled = np.sqrt((std_a**2 + std_b**2) / 2.0)
    if pooled == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled)


def summarize_continuous(name: str, ucsf: pd.Series, upenn: pd.Series) -> dict:
    u = pd.to_numeric(ucsf, errors="coerce").dropna()
    p = pd.to_numeric(upenn, errors="coerce").dropna()

    mannwhitney_p = None
    ks_p = None
    if not u.empty and not p.empty:
        try:
            mannwhitney_p = float(stats.mannwhitneyu(u, p, alternative="two-sided").pvalue)
        except Exception:
            mannwhitney_p = None
        try:
            ks_p = float(stats.ks_2samp(u, p).pvalue)
        except Exception:
            ks_p = None

    return {
        "feature": name,
        "feature_type": "continuous",
        "ucsf_n_non_missing": int(u.shape[0]),
        "upenn_n_non_missing": int(p.shape[0]),
        "ucsf_missing_rate": float(1.0 - (u.shape[0] / max(len(ucsf), 1))),
        "upenn_missing_rate": float(1.0 - (p.shape[0] / max(len(upenn), 1))),
        "ucsf_mean": float(u.mean()) if not u.empty else None,
        "upenn_mean": float(p.mean()) if not p.empty else None,
        "ucsf_median": float(u.median()) if not u.empty else None,
        "upenn_median": float(p.median()) if not p.empty else None,
        "ucsf_std": float(u.std(ddof=1)) if len(u) > 1 else None,
        "upenn_std": float(p.std(ddof=1)) if len(p) > 1 else None,
        "ucsf_q1": float(u.quantile(0.25)) if not u.empty else None,
        "ucsf_q3": float(u.quantile(0.75)) if not u.empty else None,
        "upenn_q1": float(p.quantile(0.25)) if not p.empty else None,
        "upenn_q3": float(p.quantile(0.75)) if not p.empty else None,
        "standardized_mean_difference": standardized_mean_difference(u, p),
        "mannwhitney_p": mannwhitney_p,
        "ks_p": ks_p,
    }


def cramers_v(table: pd.DataFrame) -> float | None:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, _, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2 / denom))


def summarize_binary(name: str, ucsf: pd.Series, upenn: pd.Series) -> dict:
    u = pd.to_numeric(ucsf, errors="coerce")
    p = pd.to_numeric(upenn, errors="coerce")
    combined = pd.DataFrame(
        {
            "dataset": (["ucsf"] * len(u)) + (["upenn"] * len(p)),
            "value": pd.concat([u, p], ignore_index=True),
        }
    ).dropna()

    chi2_p = None
    v = None
    if not combined.empty:
        table = pd.crosstab(combined["dataset"], combined["value"].astype(int))
        if table.shape[0] >= 2 and table.shape[1] >= 2:
            try:
                chi2_p = float(stats.chi2_contingency(table)[1])
                v = cramers_v(table)
            except Exception:
                chi2_p = None
                v = None

    return {
        "feature": name,
        "feature_type": "binary",
        "ucsf_n_non_missing": int(u.notna().sum()),
        "upenn_n_non_missing": int(p.notna().sum()),
        "ucsf_missing_rate": float(u.isna().mean()),
        "upenn_missing_rate": float(p.isna().mean()),
        "ucsf_positive_pct": float(u.mean(skipna=True) * 100.0) if u.notna().any() else None,
        "upenn_positive_pct": float(p.mean(skipna=True) * 100.0) if p.notna().any() else None,
        "absolute_pct_point_difference": float(abs((u.mean(skipna=True) - p.mean(skipna=True)) * 100.0)) if u.notna().any() and p.notna().any() else None,
        "chi_square_p": chi2_p,
        "cramers_v": v,
    }


def summarize_lobe_distribution(ucsf_lobe: pd.Series, upenn_lobe: pd.Series) -> tuple[pd.DataFrame, dict]:
    u = canonicalize_lobe(ucsf_lobe).fillna("unknown")
    p = canonicalize_lobe(upenn_lobe).fillna("unknown")
    categories = ["frontal", "temporal", "parietal", "occipital", "unknown"]

    rows: list[dict] = []
    for lobe in categories:
        u_pct = float((u == lobe).mean() * 100.0)
        p_pct = float((p == lobe).mean() * 100.0)
        rows.append(
            {
                "lobe": lobe,
                "ucsf_n": int((u == lobe).sum()),
                "ucsf_pct": u_pct,
                "upenn_n": int((p == lobe).sum()),
                "upenn_pct": p_pct,
                "pct_point_diff": p_pct - u_pct,
            }
        )

    combined = pd.DataFrame(
        {
            "dataset": (["ucsf"] * len(u)) + (["upenn"] * len(p)),
            "lobe": pd.concat([u, p], ignore_index=True),
        }
    )
    table = pd.crosstab(combined["dataset"], combined["lobe"])
    chi2_p = float(stats.chi2_contingency(table)[1]) if table.shape[0] >= 2 and table.shape[1] >= 2 else None
    summary = {
        "ucsf_mode": str(u.mode(dropna=True).iloc[0]) if not u.mode(dropna=True).empty else None,
        "upenn_mode": str(p.mode(dropna=True).iloc[0]) if not p.mode(dropna=True).empty else None,
        "chi_square_p": chi2_p,
        "cramers_v": cramers_v(table),
    }
    return pd.DataFrame(rows), summary


def build_missingness_table(ucsf_df: pd.DataFrame, upenn_df: pd.DataFrame) -> pd.DataFrame:
    all_columns = sorted(set(ucsf_df.columns.tolist()) | set(upenn_df.columns.tolist()))
    rows: list[dict] = []
    for col in all_columns:
        in_ucsf = col in ucsf_df.columns
        in_upenn = col in upenn_df.columns
        u_missing = float(ucsf_df[col].isna().mean()) if in_ucsf else None
        p_missing = float(upenn_df[col].isna().mean()) if in_upenn else None
        abs_diff = None
        if u_missing is not None and p_missing is not None:
            abs_diff = abs(u_missing - p_missing)
        rows.append(
            {
                "column": col,
                "present_in_ucsf": in_ucsf,
                "present_in_upenn": in_upenn,
                "ucsf_missing_rate": u_missing,
                "upenn_missing_rate": p_missing,
                "absolute_missing_rate_diff": abs_diff,
            }
        )
    return pd.DataFrame(rows).sort_values(["absolute_missing_rate_diff", "column"], ascending=[False, True]).reset_index(drop=True)


def build_endpoint_definition(meta_ucsf: dict, meta_upenn: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "ucsf",
                "survival_time_column": meta_ucsf["resolved_columns"].get("os"),
                "event_column": meta_ucsf["resolved_columns"].get("censor"),
                "status_column": meta_ucsf["resolved_columns"].get("survival_status"),
                "time_origin_note": "User provided: OS is overall survival in days from initial diagnosis to last clinical follow-up.",
            },
            {
                "dataset": "upenn",
                "survival_time_column": meta_upenn["resolved_columns"].get("os"),
                "event_column": meta_upenn["resolved_columns"].get("censor"),
                "status_column": meta_upenn["resolved_columns"].get("survival_status"),
                "time_origin_note": "Column name indicates survival time from surgery rather than diagnosis.",
            },
        ]
    )


def classify_shift_strength(value: float | None, threshold_moderate: float, threshold_high: float) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    abs_value = abs(float(value))
    if abs_value >= threshold_high:
        return "high"
    if abs_value >= threshold_moderate:
        return "moderate"
    return "low"


def build_report(
    feature_df: pd.DataFrame,
    lobe_summary: dict,
    missingness_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
) -> str:
    def row_for(name: str) -> pd.Series:
        return feature_df.loc[feature_df["feature"] == name].iloc[0]

    age = row_for("age")
    mgmt = row_for("mgmt_methylated_binary")
    idh = row_for("idh_mutant_binary")
    nc_en = row_for("global_nc_en_ratio")
    ed_en = row_for("global_ed_en_ratio")
    tbi = row_for("tumor_burden_index")

    top_missing = missingness_df[
        missingness_df["present_in_ucsf"] & missingness_df["present_in_upenn"]
    ].head(8)

    lines = [
        "Cohort Shift Summary",
        "",
        "Key distribution comparisons before clustering:",
        f"- Age: UCSF median {age['ucsf_median']:.2f} vs UPenn median {age['upenn_median']:.2f}; SMD={age['standardized_mean_difference']:.3f}; Mann-Whitney p={age['mannwhitney_p']:.3g}",
        f"- MGMT methylated rate: UCSF {mgmt['ucsf_positive_pct']:.2f}% vs UPenn {mgmt['upenn_positive_pct']:.2f}% ; absolute gap={mgmt['absolute_pct_point_difference']:.2f} points; chi-square p={mgmt['chi_square_p']:.3g}",
        f"- IDH mutant rate: UCSF {idh['ucsf_positive_pct']:.2f}% vs UPenn {idh['upenn_positive_pct']:.2f}% ; absolute gap={idh['absolute_pct_point_difference']:.2f} points; chi-square p={idh['chi_square_p']:.3g}",
        f"- Dominant lobe mode: UCSF {lobe_summary['ucsf_mode']} vs UPenn {lobe_summary['upenn_mode']}; chi-square p={lobe_summary['chi_square_p']:.3g}; Cramer's V={lobe_summary['cramers_v']:.3f}",
        f"- NC/EN: UCSF median {nc_en['ucsf_median']:.3f} vs UPenn median {nc_en['upenn_median']:.3f}; SMD={nc_en['standardized_mean_difference']:.3f}; Mann-Whitney p={nc_en['mannwhitney_p']:.3g}",
        f"- ED/EN: UCSF median {ed_en['ucsf_median']:.3f} vs UPenn median {ed_en['upenn_median']:.3f}; SMD={ed_en['standardized_mean_difference']:.3f}; Mann-Whitney p={ed_en['mannwhitney_p']:.3g}",
        f"- TBI: UCSF median {tbi['ucsf_median']:.3f} vs UPenn median {tbi['upenn_median']:.3f}; SMD={tbi['standardized_mean_difference']:.3f}; Mann-Whitney p={tbi['mannwhitney_p']:.3g}",
        "",
        "Largest shared-column missingness differences:",
    ]

    for _, row in top_missing.iterrows():
        lines.append(
            f"- {row['column']}: UCSF missing {row['ucsf_missing_rate']:.3f}, UPenn missing {row['upenn_missing_rate']:.3f}, abs diff {row['absolute_missing_rate_diff']:.3f}"
        )

    lines.extend(
        [
            "",
            "Survival endpoint definitions:",
            f"- UCSF: {endpoint_df.iloc[0]['time_origin_note']}",
            f"- UPenn: {endpoint_df.iloc[1]['time_origin_note']}",
            "",
            "Interpretation:",
            (
                "- The largest cohort shifts are in MGMT, IDH, lobe distribution, and the survival endpoint definition. "
                "Those differences are strong enough that UPenn likely needs harmonization or cohort-aware interpretation rather than more tuning alone."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 11: compare UCSF and UPenn for cohort shift before clustering."
    )
    parser.add_argument("--step3-dir", type=Path, default=DEFAULT_STEP3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ucsf_master, ucsf_meta_path = resolve_dataset_paths(args.step3_dir, "ucsf")
    upenn_master, upenn_meta_path = resolve_dataset_paths(args.step3_dir, "upenn")

    ucsf_df = pd.read_csv(ucsf_master)
    upenn_df = pd.read_csv(upenn_master)
    ucsf_meta = load_json(ucsf_meta_path)
    upenn_meta = load_json(upenn_meta_path)

    ucsf_resolved = ucsf_meta["resolved_columns"]
    upenn_resolved = upenn_meta["resolved_columns"]

    feature_rows = [
        summarize_continuous("age", ucsf_df["Age at MRI"], upenn_df["Age_at_scan_years"]),
        summarize_binary("mgmt_methylated_binary", map_mgmt_to_binary(ucsf_df[ucsf_resolved["mgmt"]]), map_mgmt_to_binary(upenn_df[upenn_resolved["mgmt"]])),
        summarize_binary("idh_mutant_binary", map_idh_to_binary(ucsf_df[ucsf_resolved["idh"]]), map_idh_to_binary(upenn_df[upenn_resolved["idh"]])),
        summarize_continuous("global_nc_en_ratio", ucsf_df["global_nc_en_ratio"], upenn_df["global_nc_en_ratio"]),
        summarize_continuous("global_ed_en_ratio", ucsf_df["global_ed_en_ratio"], upenn_df["global_ed_en_ratio"]),
        summarize_continuous("tumor_burden_index", ucsf_df["tumor_burden_index"], upenn_df["tumor_burden_index"]),
    ]
    feature_df = pd.DataFrame(feature_rows)
    feature_df["shift_strength"] = None
    for idx, row in feature_df.iterrows():
        if row["feature_type"] == "continuous":
            feature_df.loc[idx, "shift_strength"] = classify_shift_strength(row["standardized_mean_difference"], 0.2, 0.5)
        else:
            feature_df.loc[idx, "shift_strength"] = classify_shift_strength(row["absolute_pct_point_difference"], 10.0, 20.0)

    lobe_df, lobe_summary = summarize_lobe_distribution(
        ucsf_df[ucsf_resolved["dominant_lobe"]],
        upenn_df[upenn_resolved["dominant_lobe"]],
    )
    missingness_df = build_missingness_table(ucsf_df, upenn_df)
    endpoint_df = build_endpoint_definition(ucsf_meta, upenn_meta)

    feature_out = args.output_dir / "step11_feature_distribution_comparison.csv"
    lobe_out = args.output_dir / "step11_lobe_distribution_comparison.csv"
    missingness_out = args.output_dir / "step11_missingness_comparison.csv"
    endpoint_out = args.output_dir / "step11_survival_endpoint_definition.csv"
    summary_json_out = args.output_dir / "step11_summary.json"
    report_out = args.output_dir / "step11_cohort_shift_report.txt"

    feature_df.to_csv(feature_out, index=False)
    lobe_df.to_csv(lobe_out, index=False)
    missingness_df.to_csv(missingness_out, index=False)
    endpoint_df.to_csv(endpoint_out, index=False)

    summary_payload = {
        "ucsf_rows": int(len(ucsf_df)),
        "upenn_rows": int(len(upenn_df)),
        "major_continuous_shifts": feature_df.loc[
            (feature_df["feature_type"] == "continuous") & (feature_df["shift_strength"].isin(["moderate", "high"])),
            ["feature", "standardized_mean_difference", "mannwhitney_p", "shift_strength"],
        ].to_dict(orient="records"),
        "major_binary_shifts": feature_df.loc[
            (feature_df["feature_type"] == "binary") & (feature_df["shift_strength"].isin(["moderate", "high"])),
            ["feature", "absolute_pct_point_difference", "chi_square_p", "shift_strength"],
        ].to_dict(orient="records"),
        "lobe_distribution": lobe_summary,
        "top_missingness_differences": missingness_df[
            missingness_df["present_in_ucsf"] & missingness_df["present_in_upenn"]
        ].head(10).to_dict(orient="records"),
        "survival_endpoint_definition": endpoint_df.to_dict(orient="records"),
    }
    summary_json_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    report_out.write_text(build_report(feature_df, lobe_summary, missingness_df, endpoint_df), encoding="utf-8")

    print(f"Saved feature comparison: {feature_out}")
    print(f"Saved lobe comparison: {lobe_out}")
    print(f"Saved missingness comparison: {missingness_out}")
    print(f"Saved endpoint definition: {endpoint_out}")
    print(f"Saved summary JSON: {summary_json_out}")
    print(f"Saved text report: {report_out}")


if __name__ == "__main__":
    main()
