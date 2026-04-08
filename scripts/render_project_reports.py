from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from iteration_paths import latest_iteration_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DATA = PROJECT_ROOT / "outputs" / "reports" / "project_report_data.json"
DEFAULT_SUMMARY_OUT = PROJECT_ROOT / "summarised-results.txt"
DEFAULT_PRESENTATION_OUT = PROJECT_ROOT / "presentation-results.txt"
DEFAULT_TEX_OUT = PROJECT_ROOT / "professor_results_summary.tex"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


def default_report_data_path() -> Path:
    latest = latest_iteration_dir(DEFAULT_RESULTS_ROOT)
    if latest is None:
        return DEFAULT_REPORT_DATA
    return latest / "reports" / "project_report_data.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}g}"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}%"


def fmt_fraction_as_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:.{digits}f}%"


def fmt_ratio(numerator: int | None, denominator: int | None) -> str:
    if numerator is None or denominator is None:
        return "NA"
    return f"{int(numerator)}/{int(denominator)}"


def format_k_counts(counts: dict[str, int], connector: str = " in ") -> str:
    parts = []
    for key in sorted(counts, key=lambda item: int(item)):
        parts.append(f"k={int(key)}{connector}{int(counts[key])} runs" if connector == " in " else f"k={int(key)}: {int(counts[key])}")
    return ", ".join(parts)


def format_cluster_counts(counts: dict[str, int]) -> str:
    parts = []
    for key in sorted(counts, key=lambda item: int(item)):
        parts.append(f"cluster {int(key)}: {int(counts[key])}")
    return ", ".join(parts)


def tex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def build_summary_text(data: dict) -> str:
    ucsf = data["datasets"]["ucsf"]
    upenn = data["datasets"]["upenn"]
    cohort_shift = data["cohort_shift"]

    lines = [
        "Project: GBM multimodal clustering pipeline across UCSF and UPenn",
        "",
        "Current pipeline status",
        "- Implemented steps: Step 3 to Step 9, plus Step 11",
        "- Step 3: raw cleaning only",
        "- Step 4: leakage-safe train/test split, train-only preprocessing, and explicit split membership export",
        "- Step 5: nested-safe spectral clustering with multi-criteria k selection using the Step 4 train split only",
        "- Step 6: train-set cluster characterization via shared summary utilities",
        "- Step 7: held-out test assignment by nearest train centroid via shared validation utilities",
        "- Step 8: repeated validation over many random splits using the shared evaluation runner",
        "- Step 9: train-set cluster stability analysis",
        "- Step 11: cohort-shift analysis between UCSF and UPenn",
        "- Project summaries are now rendered from a structured report-data artifact instead of hand-maintained metrics",
        "",
        "Data sizes",
        f"- UCSF Step 3 master table: n={ucsf['step3_shape']['rows']} rows, {ucsf['step3_shape']['columns']} columns",
        f"- UPenn Step 3 master table: n={upenn['step3_shape']['rows']} rows, {upenn['step3_shape']['columns']} columns",
        "",
        "Current default single-split pipeline results after leakage fix",
        f"- Step 4 split:",
        f"  - UCSF train/test = {ucsf['single_split']['train_rows']} / {ucsf['single_split']['test_rows']}",
        f"  - UPenn train/test = {upenn['single_split']['train_rows']} / {upenn['single_split']['test_rows']}",
        f"- Step 5 selected k:",
        f"  - UCSF: k={ucsf['single_split']['selected_k']}",
        f"  - UPenn: k={upenn['single_split']['selected_k']}",
        "",
        "Current single-split train/test findings",
        f"- UCSF Step 6 train log-rank p = {fmt_float(ucsf['single_split']['train_logrank_p'])}",
        f"- UCSF Step 7 test log-rank p = {fmt_float(ucsf['single_split']['test_logrank_p'])}",
        f"- UCSF high-risk cluster = cluster {ucsf['single_split']['high_risk_cluster']}",
        f"- UCSF high-risk train/test proportion = {fmt_pct(ucsf['single_split']['high_risk_train_pct'])} / {fmt_pct(ucsf['single_split']['high_risk_test_pct'])}",
        f"- UCSF high-risk dominant lobe = {ucsf['single_split']['high_risk_train_lobe']} in train and {ucsf['single_split']['high_risk_test_lobe']} in test",
        f"- UCSF test high-risk NC/EN rank = {ucsf['single_split']['test_high_risk_nc_en_rank']} among test clusters",
        "",
        f"- UPenn Step 6 train log-rank p = {fmt_float(upenn['single_split']['train_logrank_p'])}",
        f"- UPenn Step 7 test log-rank p = {fmt_float(upenn['single_split']['test_logrank_p'])}",
        f"- UPenn high-risk cluster = cluster {upenn['single_split']['high_risk_cluster']}",
        f"- UPenn high-risk train/test proportion = {fmt_pct(upenn['single_split']['high_risk_train_pct'])} / {fmt_pct(upenn['single_split']['high_risk_test_pct'])}",
        f"- UPenn high-risk dominant lobe = {upenn['single_split']['high_risk_train_lobe']} in train, {upenn['single_split']['high_risk_test_lobe']} in test",
        f"- UPenn test high-risk NC/EN rank = {upenn['single_split']['test_high_risk_nc_en_rank']} among test clusters",
        "",
        f"Repeated validation across {ucsf['repeated_validation']['n_runs']} random splits",
        "UCSF",
        f"- Chosen k counts: {format_k_counts(ucsf['repeated_validation']['chosen_k_counts'])}",
        f"- Train survival significant in {fmt_ratio(ucsf['repeated_validation']['train_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        f"- Test survival significant in {fmt_ratio(ucsf['repeated_validation']['test_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        f"- Median train log-rank p = {fmt_float(ucsf['repeated_validation']['median_train_logrank_p'])}",
        f"- Median test log-rank p = {fmt_float(ucsf['repeated_validation']['median_test_logrank_p'])}",
        f"- High-risk cluster proportion within 10 percentage points in {fmt_ratio(ucsf['repeated_validation']['high_risk_within_10_pct_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        f"- Dominant lobe matched train in {fmt_ratio(ucsf['repeated_validation']['dominant_lobe_matches_train_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        f"- High-risk NC/EN rank 1 on test in {fmt_ratio(ucsf['repeated_validation']['test_high_risk_nc_en_rank_1_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        f"- High-risk NC/EN rank 1 or 2 on test in {fmt_ratio(ucsf['repeated_validation']['test_high_risk_nc_en_rank_1_or_2_runs'], ucsf['repeated_validation']['n_runs'])} runs",
        "",
        "UPenn",
        f"- Chosen k counts: {format_k_counts(upenn['repeated_validation']['chosen_k_counts'])}",
        f"- Train survival significant in {fmt_ratio(upenn['repeated_validation']['train_survival_significant_runs'], upenn['repeated_validation']['n_runs'])} runs",
        f"- Test survival significant in {fmt_ratio(upenn['repeated_validation']['test_survival_significant_runs'], upenn['repeated_validation']['n_runs'])} runs",
        f"- Median train log-rank p = {fmt_float(upenn['repeated_validation']['median_train_logrank_p'])}",
        f"- Median test log-rank p = {fmt_float(upenn['repeated_validation']['median_test_logrank_p'])}",
        f"- High-risk cluster proportion within 10 percentage points in {fmt_ratio(upenn['repeated_validation']['high_risk_within_10_pct_runs'], upenn['repeated_validation']['n_runs'])} runs",
        f"- Dominant lobe matched train in {fmt_ratio(upenn['repeated_validation']['dominant_lobe_matches_train_runs'], upenn['repeated_validation']['n_runs'])} runs",
        f"- Temporal-lobe replication observed in {fmt_ratio(upenn['repeated_validation']['temporal_replication_observed_runs'], upenn['repeated_validation']['temporal_replication_expected_runs'])} runs where expected",
        f"- High-risk NC/EN rank 1 on test in {fmt_ratio(upenn['repeated_validation']['test_high_risk_nc_en_rank_1_runs'], upenn['repeated_validation']['n_runs'])} runs",
        f"- High-risk NC/EN rank 1 or 2 on test in {fmt_ratio(upenn['repeated_validation']['test_high_risk_nc_en_rank_1_or_2_runs'], upenn['repeated_validation']['n_runs'])} runs",
        "",
        "Interpretation from repeated validation",
        "- UCSF is robust under repeated splitting",
        "- UPenn shows partial phenotype stability but weak prognostic replication",
        "",
        "Cluster stability analysis on current train split, 100 bootstraps",
        "UCSF",
        f"- best_k_from_step5 = {ucsf['stability']['best_k_from_step5']}",
        f"- baseline silhouette = {fmt_float(ucsf['stability']['baseline_silhouette'])}",
        f"- consensus vs original train clustering ARI = {fmt_float(ucsf['stability']['consensus_vs_full_train_ari'])}",
        f"- bootstrap ARI vs full-train clustering:",
        f"  - mean = {fmt_float(ucsf['stability']['bootstrap_ari_vs_full_train']['mean'])}",
        f"  - median = {fmt_float(ucsf['stability']['bootstrap_ari_vs_full_train']['median'])}",
        f"  - 5th percentile = {fmt_float(ucsf['stability']['bootstrap_ari_vs_full_train']['p05'])}",
        f"  - 95th percentile = {fmt_float(ucsf['stability']['bootstrap_ari_vs_full_train']['p95'])}",
        f"- pairwise bootstrap ARI:",
        f"  - mean = {fmt_float(ucsf['stability']['pairwise_bootstrap_ari']['mean'])}",
        f"  - median = {fmt_float(ucsf['stability']['pairwise_bootstrap_ari']['median'])}",
        f"- consensus strength:",
        f"  - mean within-cluster co-clustering = {fmt_float(ucsf['stability']['consensus_strength']['mean_within_cluster'])}",
        f"  - mean between-cluster co-clustering = {fmt_float(ucsf['stability']['consensus_strength']['mean_between_cluster'])}",
        f"  - mean stability gap = {fmt_float(ucsf['stability']['consensus_strength']['mean_stability_gap'])}",
        "",
        "UPenn",
        f"- best_k_from_step5 = {upenn['stability']['best_k_from_step5']}",
        f"- baseline silhouette = {fmt_float(upenn['stability']['baseline_silhouette'])}",
        f"- consensus vs original train clustering ARI = {fmt_float(upenn['stability']['consensus_vs_full_train_ari'])}",
        f"- bootstrap ARI vs full-train clustering:",
        f"  - mean = {fmt_float(upenn['stability']['bootstrap_ari_vs_full_train']['mean'])}",
        f"  - median = {fmt_float(upenn['stability']['bootstrap_ari_vs_full_train']['median'])}",
        f"  - 5th percentile = {fmt_float(upenn['stability']['bootstrap_ari_vs_full_train']['p05'])}",
        f"  - 95th percentile = {fmt_float(upenn['stability']['bootstrap_ari_vs_full_train']['p95'])}",
        f"- pairwise bootstrap ARI:",
        f"  - mean = {fmt_float(upenn['stability']['pairwise_bootstrap_ari']['mean'])}",
        f"  - median = {fmt_float(upenn['stability']['pairwise_bootstrap_ari']['median'])}",
        f"- consensus strength:",
        f"  - mean within-cluster co-clustering = {fmt_float(upenn['stability']['consensus_strength']['mean_within_cluster'])}",
        f"  - mean between-cluster co-clustering = {fmt_float(upenn['stability']['consensus_strength']['mean_between_cluster'])}",
        f"  - mean stability gap = {fmt_float(upenn['stability']['consensus_strength']['mean_stability_gap'])}",
        "",
        "Interpretation from cluster stability",
        "- UPenn clustering structure is not random or collapsing",
        "- The main weakness in UPenn is prognostic generalization, not total absence of cluster structure",
        "",
        "Cohort shift analysis between UCSF and UPenn before clustering",
        "Age",
        f"- UCSF median age = {cohort_shift['age']['ucsf_median']}",
        f"- UPenn median age = {cohort_shift['age']['upenn_median']}",
        f"- SMD = {fmt_float(float(cohort_shift['age']['standardized_mean_difference']), 3)}",
        f"- Mann-Whitney p = {fmt_float(float(cohort_shift['age']['mannwhitney_p']))}",
        "",
        "MGMT",
        f"- UCSF methylated rate = {fmt_pct(float(cohort_shift['mgmt']['ucsf_positive_pct']))}",
        f"- UPenn methylated rate = {fmt_pct(float(cohort_shift['mgmt']['upenn_positive_pct']))}",
        f"- Absolute gap = {float(cohort_shift['mgmt']['absolute_pct_point_difference']):.2f} percentage points",
        f"- Chi-square p = {fmt_float(float(cohort_shift['mgmt']['chi_square_p']))}",
        f"- MGMT missingness:",
        f"  - UCSF = {fmt_fraction_as_pct(float(cohort_shift['mgmt']['ucsf_missing_rate']))}",
        f"  - UPenn = {fmt_fraction_as_pct(float(cohort_shift['mgmt']['upenn_missing_rate']))}",
        "",
        "IDH",
        f"- UCSF mutant rate = {fmt_pct(float(cohort_shift['idh']['ucsf_positive_pct']))}",
        f"- UPenn mutant rate = {fmt_pct(float(cohort_shift['idh']['upenn_positive_pct']))}",
        f"- Absolute gap = {float(cohort_shift['idh']['absolute_pct_point_difference']):.2f} percentage points",
        f"- Chi-square p = {fmt_float(float(cohort_shift['idh']['chi_square_p']))}",
        f"- IDH missingness:",
        f"  - UCSF = {fmt_fraction_as_pct(float(cohort_shift['idh']['ucsf_missing_rate']))}",
        f"  - UPenn = {fmt_fraction_as_pct(float(cohort_shift['idh']['upenn_missing_rate']))}",
        "",
        "Lobe distribution",
        f"- UCSF dominant lobe mode = {cohort_shift['summary']['lobe_distribution']['ucsf_mode']}",
        f"- UPenn dominant lobe mode = {cohort_shift['summary']['lobe_distribution']['upenn_mode']}",
        f"- Chi-square p = {fmt_float(float(cohort_shift['summary']['lobe_distribution']['chi_square_p']))}",
        f"- Cramer's V = {fmt_float(float(cohort_shift['summary']['lobe_distribution']['cramers_v']), 3)}",
        f"- Frontal dominance:",
        f"  - UCSF = {fmt_pct(float(cohort_shift['lobe_distribution']['frontal']['ucsf_pct']))}",
        f"  - UPenn = {fmt_pct(float(cohort_shift['lobe_distribution']['frontal']['upenn_pct']))}",
        f"- Temporal dominance:",
        f"  - UCSF = {fmt_pct(float(cohort_shift['lobe_distribution']['temporal']['ucsf_pct']))}",
        f"  - UPenn = {fmt_pct(float(cohort_shift['lobe_distribution']['temporal']['upenn_pct']))}",
        "",
        "Global imaging feature distributions",
        f"- NC/EN median:",
        f"  - UCSF = {fmt_float(float(cohort_shift['global_nc_en_ratio']['ucsf_median']), 3)}",
        f"  - UPenn = {fmt_float(float(cohort_shift['global_nc_en_ratio']['upenn_median']), 3)}",
        f"  - SMD = {fmt_float(float(cohort_shift['global_nc_en_ratio']['standardized_mean_difference']), 3)}",
        f"  - Mann-Whitney p = {fmt_float(float(cohort_shift['global_nc_en_ratio']['mannwhitney_p']))}",
        f"- ED/EN median:",
        f"  - UCSF = {fmt_float(float(cohort_shift['global_ed_en_ratio']['ucsf_median']), 3)}",
        f"  - UPenn = {fmt_float(float(cohort_shift['global_ed_en_ratio']['upenn_median']), 3)}",
        f"  - SMD = {fmt_float(float(cohort_shift['global_ed_en_ratio']['standardized_mean_difference']), 3)}",
        f"  - Mann-Whitney p = {fmt_float(float(cohort_shift['global_ed_en_ratio']['mannwhitney_p']))}",
        f"- TBI median:",
        f"  - UCSF = {fmt_float(float(cohort_shift['tumor_burden_index']['ucsf_median']), 3)}",
        f"  - UPenn = {fmt_float(float(cohort_shift['tumor_burden_index']['upenn_median']), 3)}",
        f"  - SMD = {fmt_float(float(cohort_shift['tumor_burden_index']['standardized_mean_difference']), 3)}",
        f"  - Mann-Whitney p = {fmt_float(float(cohort_shift['tumor_burden_index']['mannwhitney_p']))}",
        "",
        "Shared-column missingness differences",
        f"- occipital_*_ratio missingness difference ≈ {fmt_float(float(cohort_shift['missingness']['occipital_ed_ratio']['absolute_missing_rate_diff']), 3)}",
        f"- temporal_*_ratio missingness difference ≈ {fmt_float(float(cohort_shift['missingness']['temporal_ed_ratio']['absolute_missing_rate_diff']), 3)}",
        f"- global_nc_en_ratio and global_ed_en_ratio missingness difference ≈ {fmt_float(float(cohort_shift['missingness']['global_nc_en_ratio']['absolute_missing_rate_diff']), 3)}",
        "",
        "Survival endpoint mismatch",
        "- UCSF OS = days from initial diagnosis to last clinical follow-up",
        "- UPenn survival = days from surgery",
        "- Therefore absolute survival is not directly comparable across cohorts",
        "",
        "Overall interpretation",
        "- UCSF and UPenn are meaningfully shifted cohorts",
        "- The biggest shifts are in MGMT, IDH, lobe distribution, missingness patterns, age, and survival endpoint definition",
        "- UPenn likely needs harmonization or cohort-aware analysis, not just more tuning",
        "- The weakness in UPenn is not that clustering is entirely random; phenotype structure is moderately stable, but prognostic replication is weak",
        "- Best current paper framing:",
        "  - UCSF: robust internally validated survival-relevant subtype structure",
        "  - UPenn: partial phenotype replication, limited prognostic replication",
    ]
    return "\n".join(lines) + "\n"


def build_presentation_text(data: dict) -> str:
    ucsf = data["datasets"]["ucsf"]
    upenn = data["datasets"]["upenn"]
    return "\n".join(
        [
            "GBM Multimodal Clustering: Slide Summary",
            "",
            "Pipeline changes",
            "- Step 4 now writes explicit split membership so downstream steps no longer reconstruct the outer split.",
            "- Steps 5, 6, and 7 share one cluster validation utility layer for survival summaries, high-risk labeling, centroid assignment, and train/test proportion checks.",
            "- Step 10 has been retired; Step 8 is the only repeated-validation layer kept in the maintained pipeline.",
            "- Text and LaTeX summaries are generated from a structured report-data JSON.",
            "",
            "Main result",
            f"- UCSF remains strong: single-split test p={fmt_float(ucsf['single_split']['test_logrank_p'])}, repeated validation test significance {fmt_ratio(ucsf['repeated_validation']['test_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])}.",
            f"- UPenn remains weaker prognostically: single-split test p={fmt_float(upenn['single_split']['test_logrank_p'])}, repeated validation test significance {fmt_ratio(upenn['repeated_validation']['test_survival_significant_runs'], upenn['repeated_validation']['n_runs'])}.",
            "",
            "Interpretation",
            "- UCSF supports a robust survival-relevant subtype structure.",
            "- UPenn shows phenotype replication more than survival replication.",
            "- Cohort shift remains substantial in MGMT, IDH, lobe distribution, age, missingness, and survival endpoint definition.",
        ]
    ) + "\n"


def build_professor_tex(data: dict) -> str:
    ucsf = data["datasets"]["ucsf"]
    upenn = data["datasets"]["upenn"]
    cohort_shift = data["cohort_shift"]
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{array}}

\begin{{document}}

\section*{{Pipeline Overview}}
A multi-step pipeline was developed to identify and validate multimodal subtypes in Glioblastoma (GBM) cohorts from UCSF and UPenn. The current implementation exports explicit split membership from Step 4, shares one validation utility layer across Steps 5--7, keeps Step 8 as the single repeated-validation layer, and renders the narrative summaries from structured report data instead of hand-maintained metrics.

\begin{{itemize}}
    \item[\textbf{{Step 3:}}] \textbf{{Master Table Creation.}} Initial data cleaning resolves column aliases and missing-value markers, creating a consistent master table for each cohort.
    \item[\textbf{{Step 4:}}] \textbf{{Leakage-Safe Preprocessing.}} The data is split into training and testing sets. All preprocessing steps are learned only on the training data, then applied to the test set. Step 4 also writes an explicit split-membership file.
    \item[\textbf{{Step 5:}}] \textbf{{Nested-Safe Spectral Clustering.}} Spectral clustering is performed on the training data. The optimal number of clusters ($k$) is selected inside a train-only inner split using silhouette score, gap statistic, resample stability, survival separation, and biological interpretability.
    \item[\textbf{{Step 6:}}] \textbf{{Cluster Characterization.}} The resulting training clusters are analyzed clinically and biologically, and a high-risk cluster is identified.
    \item[\textbf{{Step 7:}}] \textbf{{Held-out Validation.}} Patients in the test set are assigned to the nearest train-derived centroid to validate prognostic and phenotypic generalization.
    \item[\textbf{{Step 8:}}] \textbf{{Repeated Evaluation.}} The full workflow from Step 4 through Step 7 is rerun across many random splits to measure robustness.
    \item[\textbf{{Step 9:}}] \textbf{{Stability Analysis.}} Bootstrap resampling and consensus clustering measure train-set reproducibility.
    \item[\textbf{{Step 11:}}] \textbf{{Cohort Shift Analysis.}} Pre-clustering differences between UCSF and UPenn are quantified.
\end{{itemize}}

\begin{{table}}[ht]
\centering
\caption{{Compact summary of clustering, validation, and cohort-shift results for UCSF and UPenn.}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l l c c@{{}}}}
\toprule
\textbf{{Section}} & \textbf{{Metric}} & \textbf{{UCSF}} & \textbf{{UPenn}} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Clustering and Validation}}}} \\
 & Cohort size & {ucsf['step3_shape']['rows']} & {upenn['step3_shape']['rows']} \\
 & Current selected $k$ & {ucsf['single_split']['selected_k']} & {upenn['single_split']['selected_k']} \\
 & Current train log-rank $p$ & {fmt_float(ucsf['single_split']['train_logrank_p'])} & {fmt_float(upenn['single_split']['train_logrank_p'])} \\
 & Current holdout log-rank $p$ & {fmt_float(ucsf['single_split']['test_logrank_p'])} & {fmt_float(upenn['single_split']['test_logrank_p'])} \\
 & Repeated-split test significance & {fmt_ratio(ucsf['repeated_validation']['test_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])} & {fmt_ratio(upenn['repeated_validation']['test_survival_significant_runs'], upenn['repeated_validation']['n_runs'])} \\
 & Consensus vs baseline ARI & {fmt_float(ucsf['stability']['consensus_vs_full_train_ari'], 3)} & {fmt_float(upenn['stability']['consensus_vs_full_train_ari'], 3)} \\
 & Pairwise bootstrap ARI median & {fmt_float(ucsf['stability']['pairwise_bootstrap_ari']['median'], 3)} & {fmt_float(upenn['stability']['pairwise_bootstrap_ari']['median'], 3)} \\
 & Main message & Robust clustering and survival replication & Cluster structure exists, but prognostic replication is weak \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Pre-clustering Cohort Differences}}}} \\
 & Median age & {cohort_shift['age']['ucsf_median']} & {cohort_shift['age']['upenn_median']} \\
 & MGMT methylated rate & {fmt_pct(float(cohort_shift['mgmt']['ucsf_positive_pct']))} & {fmt_pct(float(cohort_shift['mgmt']['upenn_positive_pct']))} \\
 & IDH mutant rate & {fmt_pct(float(cohort_shift['idh']['ucsf_positive_pct']))} & {fmt_pct(float(cohort_shift['idh']['upenn_positive_pct']))} \\
 & Dominant lobe mode & {tex_escape(cohort_shift['summary']['lobe_distribution']['ucsf_mode'].title())} & {tex_escape(cohort_shift['summary']['lobe_distribution']['upenn_mode'].title())} \\
 & Survival time origin & From diagnosis & From surgery \\
\bottomrule
\end{{tabularx}}
\end{{table}}

\begin{{table}}[ht]
\centering
\caption{{More detailed validation summary for discussion with faculty.}}
\renewcommand{{\arraystretch}}{{1.18}}
\begin{{tabularx}}{{\textwidth}}{{@{{}}l l c c@{{}}}}
\toprule
\textbf{{Layer}} & \textbf{{Metric}} & \textbf{{UCSF}} & \textbf{{UPenn}} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Current Single-Split Pipeline}}}} \\
 & Train / test size & {ucsf['single_split']['train_rows']} / {ucsf['single_split']['test_rows']} & {upenn['single_split']['train_rows']} / {upenn['single_split']['test_rows']} \\
 & Selected $k$ & {ucsf['single_split']['selected_k']} & {upenn['single_split']['selected_k']} \\
 & High-risk cluster label & {ucsf['single_split']['high_risk_cluster']} & {upenn['single_split']['high_risk_cluster']} \\
 & Train log-rank $p$ & {fmt_float(ucsf['single_split']['train_logrank_p'])} & {fmt_float(upenn['single_split']['train_logrank_p'])} \\
 & Test log-rank $p$ & {fmt_float(ucsf['single_split']['test_logrank_p'])} & {fmt_float(upenn['single_split']['test_logrank_p'])} \\
 & High-risk train / test proportion & {fmt_pct(ucsf['single_split']['high_risk_train_pct'])} / {fmt_pct(ucsf['single_split']['high_risk_test_pct'])} & {fmt_pct(upenn['single_split']['high_risk_train_pct'])} / {fmt_pct(upenn['single_split']['high_risk_test_pct'])} \\
 & High-risk dominant lobe, train / test & {tex_escape(str(ucsf['single_split']['high_risk_train_lobe']).title())} / {tex_escape(str(ucsf['single_split']['high_risk_test_lobe']).title())} & {tex_escape(str(upenn['single_split']['high_risk_train_lobe']).title())} / {tex_escape(str(upenn['single_split']['high_risk_test_lobe']).title())} \\
 & High-risk NC/EN rank on test & {ucsf['single_split']['test_high_risk_nc_en_rank']} & {upenn['single_split']['test_high_risk_nc_en_rank']} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Repeated Validation}}}} \\
 & Runs & {ucsf['repeated_validation']['n_runs']} & {upenn['repeated_validation']['n_runs']} \\
 & Most common selected $k$ & {Counter(ucsf['repeated_validation']['chosen_k_counts']).most_common(1)[0][0] if ucsf['repeated_validation']['chosen_k_counts'] else ''} & {Counter(upenn['repeated_validation']['chosen_k_counts']).most_common(1)[0][0] if upenn['repeated_validation']['chosen_k_counts'] else ''} \\
 & Train survival significant & {fmt_ratio(ucsf['repeated_validation']['train_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])} & {fmt_ratio(upenn['repeated_validation']['train_survival_significant_runs'], upenn['repeated_validation']['n_runs'])} \\
 & Test survival significant & {fmt_ratio(ucsf['repeated_validation']['test_survival_significant_runs'], ucsf['repeated_validation']['n_runs'])} & {fmt_ratio(upenn['repeated_validation']['test_survival_significant_runs'], upenn['repeated_validation']['n_runs'])} \\
 & Median train log-rank $p$ & {fmt_float(ucsf['repeated_validation']['median_train_logrank_p'])} & {fmt_float(upenn['repeated_validation']['median_train_logrank_p'])} \\
 & Median test log-rank $p$ & {fmt_float(ucsf['repeated_validation']['median_test_logrank_p'])} & {fmt_float(upenn['repeated_validation']['median_test_logrank_p'])} \\
 & Lobe matched train on test & {fmt_ratio(ucsf['repeated_validation']['dominant_lobe_matches_train_runs'], ucsf['repeated_validation']['n_runs'])} & {fmt_ratio(upenn['repeated_validation']['dominant_lobe_matches_train_runs'], upenn['repeated_validation']['n_runs'])} \\
 & High-risk NC/EN rank 1 or 2 on test & {fmt_ratio(ucsf['repeated_validation']['test_high_risk_nc_en_rank_1_or_2_runs'], ucsf['repeated_validation']['n_runs'])} & {fmt_ratio(upenn['repeated_validation']['test_high_risk_nc_en_rank_1_or_2_runs'], upenn['repeated_validation']['n_runs'])} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Cluster Stability}}}} \\
 & Consensus vs baseline ARI & {fmt_float(ucsf['stability']['consensus_vs_full_train_ari'], 3)} & {fmt_float(upenn['stability']['consensus_vs_full_train_ari'], 3)} \\
 & Bootstrap ARI vs baseline, median & {fmt_float(ucsf['stability']['bootstrap_ari_vs_full_train']['median'], 3)} & {fmt_float(upenn['stability']['bootstrap_ari_vs_full_train']['median'], 3)} \\
 & Pairwise bootstrap ARI, median & {fmt_float(ucsf['stability']['pairwise_bootstrap_ari']['median'], 3)} & {fmt_float(upenn['stability']['pairwise_bootstrap_ari']['median'], 3)} \\
 & Mean within-cluster consensus & {fmt_float(ucsf['stability']['consensus_strength']['mean_within_cluster'], 3)} & {fmt_float(upenn['stability']['consensus_strength']['mean_within_cluster'], 3)} \\
 & Mean between-cluster consensus & {fmt_float(ucsf['stability']['consensus_strength']['mean_between_cluster'], 3)} & {fmt_float(upenn['stability']['consensus_strength']['mean_between_cluster'], 3)} \\
\midrule
\multicolumn{{4}}{{@{{}}l}}{{\textbf{{Pre-clustering Cohort Shift}}}} \\
 & Median age & {cohort_shift['age']['ucsf_median']} & {cohort_shift['age']['upenn_median']} \\
 & MGMT methylated rate & {fmt_pct(float(cohort_shift['mgmt']['ucsf_positive_pct']))} & {fmt_pct(float(cohort_shift['mgmt']['upenn_positive_pct']))} \\
 & IDH mutant rate & {fmt_pct(float(cohort_shift['idh']['ucsf_positive_pct']))} & {fmt_pct(float(cohort_shift['idh']['upenn_positive_pct']))} \\
 & Dominant lobe mode & {tex_escape(cohort_shift['summary']['lobe_distribution']['ucsf_mode'].title())} & {tex_escape(cohort_shift['summary']['lobe_distribution']['upenn_mode'].title())} \\
 & Survival endpoint & From diagnosis & From surgery \\
\bottomrule
\end{{tabularx}}
\end{{table}}

\noindent\textbf{{Interpretation.}}
UCSF remains strong across the single split, repeated validation, and cluster-stability layers. UPenn shows reasonably stable cluster structure, but survival separation is not consistently reproduced. The large cohort differences in MGMT, IDH, lobe distribution, age, and survival endpoint suggest that UPenn likely needs harmonization or cohort-aware analysis rather than more tuning alone.

\end{{document}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render project summary artifacts from the structured report-data JSON."
    )
    parser.add_argument("--report-data", type=Path, default=default_report_data_path())
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--presentation-out", type=Path, default=DEFAULT_PRESENTATION_OUT)
    parser.add_argument("--tex-out", type=Path, default=DEFAULT_TEX_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.report_data)
    args.summary_out.write_text(build_summary_text(data), encoding="utf-8")
    args.presentation_out.write_text(build_presentation_text(data), encoding="utf-8")
    args.tex_out.write_text(build_professor_tex(data), encoding="utf-8")
    print(f"Rendered summary: {args.summary_out}")
    print(f"Rendered presentation notes: {args.presentation_out}")
    print(f"Rendered LaTeX summary: {args.tex_out}")


if __name__ == "__main__":
    main()
