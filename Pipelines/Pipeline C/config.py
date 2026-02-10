"""
Pipeline C: Hierarchical Reconciliation (Daily → Block) — Configuration.

Single source of truth for paths, validation folds, feature lists,
share-model configuration, and LightGBM hyperparameters.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # DSU-2026/
PIPELINE_DIR = Path(__file__).resolve().parent               # Pipelines/Pipeline C/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET = DATA_SOURCE_DIR / "master_block_history.parquet"
RAW_VISITS_CSV = DATA_SOURCE_DIR / "DSU-Dataset.csv"

OUTPUT_DIR = PIPELINE_DIR / "output"
MODEL_DIR = PIPELINE_DIR / "models"
DATA_DIR = PIPELINE_DIR / "data"

# ── Sites & Blocks ───────────────────────────────────────────────────────────

SITES = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]
N_BLOCKS = 4

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
COVID_SAMPLE_WEIGHT = 0.1

# ── Daily Model Feature Engineering ─────────────────────────────────────────

MAX_HORIZON = 63
LAG_DAYS_DAILY = [63, 70, 77, 91, 182, 364]
ROLLING_WINDOWS_DAILY = [7, 14, 28, 56, 91]
ROLLING_SHIFT_DAILY = 63

# Approximate school-year start dates (Fargo / Sioux Falls)
SCHOOL_STARTS = [
    "2018-08-20", "2019-08-19", "2020-09-08",   # 2020: COVID late start
    "2021-08-23", "2022-08-22", "2023-08-21",
    "2024-08-19", "2025-08-18",
]

# ── Share Model Configuration ────────────────────────────────────────────────

SHARE_MODEL_TYPE = "softmax_gbdt"  # "softmax_gbdt", "climatology"
CLIMATOLOGY_KEYS = ["site", "dow", "month"]
LAG_DAYS_SHARES = [63, 70, 77, 91, 182, 364]
ROLLING_SHIFT_SHARES = 63

# ── LightGBM Defaults (Daily Total — Tweedie) ───────────────────────────────

LGBM_DAILY_TOTAL = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "verbosity": -1,
}

# ── LightGBM Defaults (Daily Admit Rate — MSE) ──────────────────────────────

LGBM_DAILY_RATE = {
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "verbosity": -1,
}

# ── LightGBM Defaults (Softmax Block Share — multiclass) ────────────────────

LGBM_SHARE = {
    "objective": "multiclass",
    "num_class": 4,
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

# ── Optuna ───────────────────────────────────────────────────────────────────

OPTUNA_N_TRIALS_DAILY = 100
OPTUNA_N_TRIALS_SHARE = 50


def ensure_dirs() -> None:
    """Create output, model, and data directories if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
