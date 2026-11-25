"""Shared downstream data, folds, paths and numerical utilities."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BIOMARKER_ROOT = PROJECT_ROOT / "biomarker"
DEFAULT_STORAGE_ROOT = Path("/jhcnas5/zq/data/PathSeg2/biomarker/results")
DEFAULT_LABEL_ROOT = BIOMARKER_ROOT / "clinical_data" / "downstream_csv"


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


STORAGE_ROOT = _env_path("STORAGE_ROOT", DEFAULT_STORAGE_ROOT)
RESULTS_ROOT = _env_path("RESULTS_ROOT", STORAGE_ROOT)
SUMMARY_ROOT = _env_path("SUMMARY_ROOT", BIOMARKER_ROOT)
LABEL_ROOT = _env_path("LABEL_ROOT", _env_path("CSV_ROOT", DEFAULT_LABEL_ROOT))

FINAL_ROOT = RESULTS_ROOT / "final"
READOUT_ROOT = _env_path("READOUT_ROOT", Path("/jhcnas3/MRI_jhcnas3/biomarker/tme_readout"))
