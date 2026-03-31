"""
Clinical data harmonization — load UCSF-PDGM and UPENN-GBM clinical CSVs,
standardise column names, encode categorical variables, and merge into a
single, analysis-ready clinical DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.helpers import load_config, resolve_path, setup_logging

logger = setup_logging()


# ══════════════════════════════════════════════════════════════════════════════
#  UCSF-PDGM
# ══════════════════════════════════════════════════════════════════════════════
def _load_ucsf(cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(resolve_path(cfg["paths"]["ucsf"]["clinical_csv"]))

    df = pd.DataFrame()
    df["patient_id"] = raw.iloc[:, 0].astype(str).str.strip()
    df["cohort"] = "UCSF"

    # Demographics
    df["sex"] = raw["Sex"].map({"M": 0, "F": 1})
    df["age"] = pd.to_numeric(raw["Age at MRI"], errors="coerce")

    # Molecular markers
    df["idh"] = raw["IDH"].map({"wildtype": 0, "mutant": 1})
    df["mgmt"] = raw["MGMT status"].map(
        {"unmethylated": 0, "methylated": 1, "indeterminate": np.nan}
    )
    df["codel_1p19q"] = raw["1p/19q"].map(
        {"non-codel": 0, "codel": 1, "not tested": np.nan}
    )

    # Grade
    df["who_grade"] = pd.to_numeric(raw["WHO CNS Grade"], errors="coerce")

    # Survival
    df["os_days"] = pd.to_numeric(raw["OS"], errors="coerce")
    df["os_event"] = raw["1-dead 0-alive"].map({1: 1, 0: 0})

    # Extent of resection
    df["eor"] = raw["EOR"].map(
        {"GTR": 2, "STR": 1, "Biopsy": 0}
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  UPENN-GBM
# ══════════════════════════════════════════════════════════════════════════════
def _load_upenn(cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(resolve_path(cfg["paths"]["upenn"]["clinical_csv"]))

    df = pd.DataFrame()
    df["patient_id"] = raw.iloc[:, 0].astype(str).str.strip()
    df["cohort"] = "UPenn"

    # Demographics
    df["sex"] = raw["Gender"].map({"M": 0, "F": 1})
    df["age"] = pd.to_numeric(raw["Age_at_scan_years"], errors="coerce")

    # Molecular markers
    # UPenn IDH1 column: 0 / 1 / "Not Available"
    idh_raw = pd.to_numeric(raw["IDH1"], errors="coerce")
    df["idh"] = idh_raw.where(idh_raw.isin([0, 1]))

    mgmt_raw = pd.to_numeric(raw["MGMT"], errors="coerce")
    df["mgmt"] = mgmt_raw.where(mgmt_raw.isin([0, 1]))

    df["codel_1p19q"] = np.nan  # not available in UPenn

    df["who_grade"] = np.nan  # not available in UPenn (all GBM → grade 4)

    # Survival
    df["os_days"] = pd.to_numeric(
        raw["Survival_from_surgery_days_UPDATED"], errors="coerce"
    )
    # Survival_Status: 1 = dead, 0 = alive  (use Survival_Censor as backup)
    status = pd.to_numeric(raw["Survival_Status"], errors="coerce")
    df["os_event"] = status.where(status.isin([0, 1]))

    # Extent of resection
    gtr = pd.to_numeric(raw["GTR_over90percent"], errors="coerce")
    df["eor"] = gtr.map({1: 2, 0: 1})  # 1 → GTR, 0 → STR; no biopsy flag

    # KPS (extra clinical variable)
    df["kps"] = pd.to_numeric(raw["KPS"], errors="coerce")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════
def harmonize_clinical(cfg: dict | None = None, save: bool = True) -> pd.DataFrame:
    """
    Load and harmonize clinical data from both cohorts.

    Returns a combined DataFrame with standardised columns, ready to merge
    with radiomics features.
    """
    if cfg is None:
        cfg = load_config()

    logger.info("Loading UCSF clinical data …")
    ucsf = _load_ucsf(cfg)
    logger.info("  %d UCSF patients", len(ucsf))

    logger.info("Loading UPenn clinical data …")
    upenn = _load_upenn(cfg)
    logger.info("  %d UPenn patients", len(upenn))

    combined = pd.concat([ucsf, upenn], ignore_index=True)

    # UPenn is all GBM (WHO grade 4) — fill in where missing
    combined.loc[
        (combined["cohort"] == "UPenn") & combined["who_grade"].isna(),
        "who_grade",
    ] = 4

    logger.info("Harmonised clinical data: %d patients, %d columns",
                *combined.shape)

    if save:
        out = resolve_path(cfg["paths"]["output"]["data_dir"]) / "clinical_harmonized.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False)
        logger.info("  Saved → %s", out)

    return combined


if __name__ == "__main__":
    harmonize_clinical()
