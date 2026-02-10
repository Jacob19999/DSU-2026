"""
Pipeline A: Global GBDT — Configuration & constants.

Single source of truth for paths, validation folds, feature engineering
parameters, and LightGBM hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # DSU-2026/
PIPELINE_DIR = Path(__file__).resolve().parent               # Pipelines/Pipeline A/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET = DATA_SOURCE_DIR / "master_block_history.parquet"
_DATA_SOURCE_CSV = DATA_SOURCE_DIR / "DSU-Dataset.csv"
_DATASET_CSV = PROJECT_ROOT / "Dataset" / "DSU-Dataset.csv"
RAW_VISITS_CSV = _DATA_SOURCE_CSV if _DATA_SOURCE_CSV.exists() else _DATASET_CSV

OUTPUT_DIR = PIPELINE_DIR / "output"
MODEL_DIR = PIPELINE_DIR / "models"

# ── Sites & Blocks ───────────────────────────────────────────────────────────

SITES = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]

# ── Validation Folds (from eval.md) ─────────────────────────────────────────

FOLDS = [
    {"id": 1, "train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-02-28"},
    {"id": 2, "train_end": "2025-02-28", "val_start": "2025-03-01", "val_end": "2025-04-30"},
    {"id": 3, "train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-06-30"},
    {"id": 4, "train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2025-08-31"},
]

# ── COVID Era ────────────────────────────────────────────────────────────────

COVID_START = "2020-03-01"
COVID_END = "2021-06-30"
COVID_SAMPLE_WEIGHT = 0.1   # Downweight factor for COVID-era rows

# ── Feature Engineering ──────────────────────────────────────────────────────

MAX_HORIZON = 63                         # Max forecast horizon (days)
LAG_DAYS = [63, 70, 77, 91, 182, 364]   # All >= MAX_HORIZON (no leakage)
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
ROLLING_SHIFT = 63                       # Shift rolling calcs by >= MAX_HORIZON
TOP_N_SHARE_REASONS = 8                  # Top reason categories for share features

# Approximate school-year start dates (Fargo / Sioux Falls)
SCHOOL_STARTS = [
    "2018-08-20", "2019-08-19", "2020-09-08",   # 2020: COVID late start
    "2021-08-23", "2022-08-22", "2023-08-21",
    "2024-08-19", "2025-08-18",
]

# ── LightGBM Defaults (pre-Optuna) ──────────────────────────────────────────

LGBM_DEFAULT_A1 = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

LGBM_DEFAULT_A2 = {
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

# ── Optuna ───────────────────────────────────────────────────────────────────

OPTUNA_N_TRIALS_A1 = 100
OPTUNA_N_TRIALS_A2 = 50


def ensure_dirs() -> None:
    """Create output and model directories if they don't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
