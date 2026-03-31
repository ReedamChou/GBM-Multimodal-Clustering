"""
Survival analysis — Kaplan-Meier curves, log-rank tests between clusters,
and multivariate Cox proportional-hazards model.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()


# ══════════════════════════════════════════════════════════════════════════════
#  Kaplan-Meier
# ══════════════════════════════════════════════════════════════════════════════
def fit_kaplan_meier(
    meta: pd.DataFrame,
    labels: np.ndarray,
) -> Dict[int, KaplanMeierFitter]:
    """
    Fit a KaplanMeierFitter for each cluster.

    Parameters
    ----------
    meta : DataFrame
        Must contain ``os_days`` and ``os_event`` columns.
    labels : array
        Cluster assignment per patient.

    Returns
    -------
    km_dict : dict
        cluster_id → fitted KaplanMeierFitter.
    """
    valid = meta[["os_days", "os_event"]].notna().all(axis=1)
    meta_v = meta.loc[valid].copy()
    labels_v = labels[valid.values]

    km_dict: Dict[int, KaplanMeierFitter] = {}
    for cl in sorted(np.unique(labels_v)):
        mask = labels_v == cl
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=meta_v.loc[mask, "os_days"],
            event_observed=meta_v.loc[mask, "os_event"],
            label=f"Cluster {cl}",
        )
        km_dict[cl] = kmf
        median = kmf.median_survival_time_
        logger.info("  Cluster %d  n=%d  median OS=%.0f days", cl, mask.sum(), median)

    return km_dict


# ══════════════════════════════════════════════════════════════════════════════
#  Log-rank tests
# ══════════════════════════════════════════════════════════════════════════════
def logrank_between_clusters(
    meta: pd.DataFrame,
    labels: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Perform pairwise log-rank tests (and a global multivariate log-rank)
    between clusters.

    Returns a DataFrame of pairwise results.
    """
    valid = meta[["os_days", "os_event"]].notna().all(axis=1)
    meta_v = meta.loc[valid].copy()
    labels_v = labels[valid.values]
    clusters = sorted(np.unique(labels_v))

    # Global test
    mv = multivariate_logrank_test(
        meta_v["os_days"], labels_v, meta_v["os_event"]
    )
    logger.info("  Global log-rank  chi²=%.2f  p=%.4g", mv.test_statistic, mv.p_value)

    # Pairwise
    rows = []
    for a, b in combinations(clusters, 2):
        mask_a = labels_v == a
        mask_b = labels_v == b
        res = logrank_test(
            meta_v.loc[mask_a, "os_days"],
            meta_v.loc[mask_b, "os_days"],
            meta_v.loc[mask_a, "os_event"],
            meta_v.loc[mask_b, "os_event"],
        )
        rows.append({
            "cluster_a": a,
            "cluster_b": b,
            "chi2": res.test_statistic,
            "p_value": res.p_value,
            "significant": res.p_value < alpha,
        })
    pairwise = pd.DataFrame(rows)
    return pairwise


# ══════════════════════════════════════════════════════════════════════════════
#  Cox PH model
# ══════════════════════════════════════════════════════════════════════════════
def fit_cox_model(
    meta: pd.DataFrame,
    labels: np.ndarray,
    covariates: list[str] | None = None,
    cfg: dict | None = None,
) -> CoxPHFitter:
    """
    Fit a multivariate Cox PH model with cluster assignment and optional
    clinical covariates.

    Parameters
    ----------
    meta : DataFrame
        Clinical metadata (must have os_days, os_event).
    labels : array
        Cluster assignment.
    covariates : list of str, optional
        Additional columns from meta to include in the model.
    cfg : dict, optional
        Config (for penalizer value).

    Returns
    -------
    cph : CoxPHFitter
        Fitted model.
    """
    if cfg is None:
        cfg = load_config()
    if covariates is None:
        covariates = ["age", "sex", "idh", "mgmt", "eor"]

    df = meta.copy()
    df["cluster"] = labels

    # Keep only available covariates
    keep = ["os_days", "os_event", "cluster"]
    for c in covariates:
        if c in df.columns:
            keep.append(c)

    df = df[keep].dropna()
    logger.info("  Cox PH model — %d patients, covariates: %s", len(df), keep[2:])

    cph = CoxPHFitter(penalizer=cfg["survival"]["cox_penalizer"])
    cph.fit(df, duration_col="os_days", event_col="os_event")
    cph.print_summary()

    return cph


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience runner
# ══════════════════════════════════════════════════════════════════════════════
def run_survival_analysis(
    meta: pd.DataFrame,
    labels: np.ndarray,
    cfg: dict | None = None,
    save: bool = True,
) -> Tuple[Dict[int, KaplanMeierFitter], pd.DataFrame, CoxPHFitter]:
    """End-to-end survival analysis for the chosen partition."""
    if cfg is None:
        cfg = load_config()

    logger.info("Running survival analysis …")
    km_dict = fit_kaplan_meier(meta, labels)
    pairwise = logrank_between_clusters(meta, labels, cfg["survival"]["alpha"])
    cph = fit_cox_model(meta, labels, cfg=cfg)

    if save:
        tbl_dir = resolve_path(cfg["paths"]["output"]["tables_dir"])
        tbl_dir.mkdir(parents=True, exist_ok=True)
        pairwise.to_csv(tbl_dir / "logrank_pairwise.csv", index=False)
        cph.summary.to_csv(tbl_dir / "cox_summary.csv")
        logger.info("  Tables saved → %s", tbl_dir)

    return km_dict, pairwise, cph
