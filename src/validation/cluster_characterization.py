"""
Cluster characterization — statistical profiling of each cluster via SHAP
feature importance, clinical enrichment tests, and per-cluster summary
statistics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.ensemble import GradientBoostingClassifier

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()

try:
    import shap
except ImportError:
    shap = None


# ══════════════════════════════════════════════════════════════════════════════
#  Per-cluster summary statistics
# ══════════════════════════════════════════════════════════════════════════════
def cluster_summary_table(
    meta: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Produce a per-cluster summary of key clinical variables:
    age (mean ± SD), sex ratio, IDH mutant %, MGMT methylated %,
    median OS, etc.
    """
    df = meta.copy()
    df["cluster"] = labels

    rows = []
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        n = len(sub)
        rec = {"cluster": cl, "n": n}

        # Age
        rec["age_mean"] = sub["age"].mean()
        rec["age_std"] = sub["age"].std()

        # Sex (coded 0=M, 1=F)
        if "sex" in sub.columns:
            rec["pct_female"] = sub["sex"].mean() * 100

        # Molecular
        for col in ["idh", "mgmt"]:
            if col in sub.columns:
                rec[f"pct_{col}_pos"] = sub[col].mean() * 100

        # WHO grade
        if "who_grade" in sub.columns:
            rec["pct_grade4"] = (sub["who_grade"] == 4).mean() * 100

        # Survival
        if "os_days" in sub.columns:
            rec["median_os_days"] = sub["os_days"].median()

        rows.append(rec)

    summary = pd.DataFrame(rows)
    logger.info("Cluster summary table (%d clusters)", len(summary))
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  Enrichment / association tests
# ══════════════════════════════════════════════════════════════════════════════
def enrichment_tests(
    meta: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    For each clinical variable test whether distribution differs across
    clusters (Kruskal-Wallis for continuous, chi-squared for categorical).
    """
    df = meta.copy()
    df["cluster"] = labels

    continuous = ["age", "os_days"]
    categorical = ["sex", "idh", "mgmt", "who_grade", "eor"]

    results = []
    for col in continuous:
        if col not in df.columns:
            continue
        groups = [g[col].dropna().values for _, g in df.groupby("cluster")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        results.append({"variable": col, "test": "Kruskal-Wallis", "statistic": stat, "p_value": p})

    for col in categorical:
        if col not in df.columns:
            continue
        ct = pd.crosstab(df["cluster"], df[col].dropna())
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2, p, _, _ = chi2_contingency(ct)
        results.append({"variable": col, "test": "Chi-squared", "statistic": chi2, "p_value": p})

    enrichment = pd.DataFrame(results)
    logger.info("Enrichment tests: %d variables tested", len(enrichment))
    return enrichment


# ══════════════════════════════════════════════════════════════════════════════
#  SHAP-based feature importance
# ══════════════════════════════════════════════════════════════════════════════
def shap_feature_importance(
    X: pd.DataFrame,
    labels: np.ndarray,
    top_n: int = 20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Train a GBM classifier to predict cluster labels and extract SHAP values.

    Returns
    -------
    importance_df : DataFrame
        Feature importance ranking (mean |SHAP|).
    shap_values : shap.Explanation or None
        Raw SHAP values for downstream plotting (None if shap unavailable).
    """
    if shap is None:
        logger.warning("shap not installed — skipping SHAP feature importance")
        return pd.DataFrame(), None

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=seed,
    )
    clf.fit(X, labels)
    acc = clf.score(X, labels)
    logger.info("  GBM classifier accuracy (train): %.2f%%", acc * 100)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)

    # For multi-class: mean absolute SHAP across classes
    if isinstance(shap_values.values, list):
        vals = np.mean([np.abs(sv) for sv in shap_values.values], axis=0)
    elif shap_values.values.ndim == 3:
        vals = np.abs(shap_values.values).mean(axis=2)  # (n, features)
    else:
        vals = np.abs(shap_values.values)

    mean_shap = vals.mean(axis=0)
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("  Top 5 SHAP features: %s",
                imp_df.head(5)["feature"].tolist())

    return imp_df.head(top_n), shap_values


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════
def characterize_clusters(
    meta: pd.DataFrame,
    X: pd.DataFrame,
    labels: np.ndarray,
    cfg: dict | None = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run full cluster characterization and save tables."""
    if cfg is None:
        cfg = load_config()

    summary = cluster_summary_table(meta, labels)
    enrichment = enrichment_tests(meta, labels)
    importance, _ = shap_feature_importance(X, labels, seed=cfg["project"]["seed"])

    if save:
        tbl_dir = resolve_path(cfg["paths"]["output"]["tables_dir"])
        tbl_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(tbl_dir / "cluster_summary.csv", index=False)
        enrichment.to_csv(tbl_dir / "enrichment_tests.csv", index=False)
        if not importance.empty:
            importance.to_csv(tbl_dir / "shap_importance.csv", index=False)
        logger.info("  Characterization tables saved → %s", tbl_dir)

    return {
        "summary": summary,
        "enrichment": enrichment,
        "shap_importance": importance,
    }
