"""
Pipeline B Configuration — all constants and search spaces in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DSU-2026/
DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET = DATA_SOURCE_DIR / "master_block_history.parquet"
MASTER_CSV = DATA_SOURCE_DIR / "master_block_history.csv"
RAW_DATASET = PROJECT_ROOT / "Dataset" / "DSU-Dataset.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
LOGS_DIR = OUTPUT_DIR / "logs"

# ── Sites & Blocks ──────────────────────────────────────────────────────────
SITES = ("A", "B", "C", "D")
BLOCKS = (0, 1, 2, 3)

# ── Validation Folds (eval.md §2.1) ─────────────────────────────────────────
@dataclass(frozen=True)
class Fold:
    period_id: int
    train_end: str       # inclusive upper bound for training data
    test_start: str
    test_end: str

FOLDS: List[Fold] = [
    Fold(1, "2024-12-31", "2025-01-01", "2025-02-28"),
    Fold(2, "2025-02-28", "2025-03-01", "2025-04-30"),
    Fold(3, "2025-04-30", "2025-05-01", "2025-06-30"),
    Fold(4, "2025-06-30", "2025-07-01", "2025-08-31"),
]

# Final submission window
FINAL_TRAIN_END = "2025-08-31"
FINAL_TEST_START = "2025-09-01"
FINAL_TEST_END = "2025-10-31"

# ── COVID Policy (§3.0 — Option 3: downweight) ──────────────────────────────
COVID_START = "2020-03-01"
COVID_END = "2021-06-30"
COVID_WEIGHT_FACTOR = 0.1      # multiply sample weight by this during COVID era
MIN_SAMPLE_WEIGHT = 1.0        # floor to avoid zero-weight rows

# ── Horizon Buckets (Option B1 — PREFERRED) ─────────────────────────────────
@dataclass(frozen=True)
class HorizonBucket:
    bucket_id: int
    horizon_min: int             # inclusive
    horizon_max: int             # inclusive
    min_lag: int                 # minimum safe lag to avoid leakage
    lags: Tuple[int, ...]        # specific lag days to use
    rolling_shift: int           # shift applied before rolling window computation

HORIZON_BUCKETS: List[HorizonBucket] = [
    HorizonBucket(
        bucket_id=1,
        horizon_min=1, horizon_max=15,
        min_lag=16,
        lags=(16, 21, 28, 56, 91, 182, 364),
        rolling_shift=16,
    ),
    HorizonBucket(
        bucket_id=2,
        horizon_min=16, horizon_max=30,
        min_lag=31,
        lags=(31, 35, 42, 56, 91, 182, 364),
        rolling_shift=31,
    ),
    HorizonBucket(
        bucket_id=3,
        horizon_min=31, horizon_max=61,
        min_lag=62,
        lags=(63, 70, 77, 91, 182, 364),
        rolling_shift=63,
    ),
]

# ── Rolling Windows ──────────────────────────────────────────────────────────
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
ROLLING_STATS = ["mean", "std", "min", "max"]

# ── Horizon Sub-Sampling (to reduce training data size) ──────────────────────
# Representative horizons per bucket instead of every day
HORIZON_SAMPLES: Dict[int, List[int]] = {
    1: [1, 4, 7, 10, 13, 15],       # 6 horizons from Bucket 1
    2: [16, 20, 24, 28, 30],          # 5 horizons from Bucket 2
    3: [31, 37, 43, 49, 55, 61],      # 6 horizons from Bucket 3
}

# ── Feature Lists ────────────────────────────────────────────────────────────
CALENDAR_FEATURES = [
    "dow", "month", "day", "week_of_year", "quarter",
    "day_of_year", "is_weekend", "days_since_epoch", "year_frac",
]

CYCLICAL_FEATURES = [
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
]

EVENT_FEATURES = [
    "is_holiday", "is_halloween", "event_count",
]

HOLIDAY_PROXIMITY_FEATURES = [
    "days_since_xmas", "days_until_thanksgiving", "days_since_july4",
    "days_since_school_start",
]

WEATHER_FEATURES = [
    "temp_min", "temp_max", "precip", "snowfall", "temp_range",
]

INTERACTION_FEATURES = [
    "holiday_x_block", "weekend_x_block",
]

CATEGORICAL_FEATURES = ["site", "block"]

# ── LightGBM Defaults ───────────────────────────────────────────────────────
LGBM_FIXED_PARAMS = {
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "early_stopping_round": 50,
}

# ── Optuna Search Space ─────────────────────────────────────────────────────
OPTUNA_N_TRIALS = 100
OPTUNA_SEARCH_SPACE = {
    "n_estimators":     (800, 3000),
    "max_depth":        (4, 8),
    "learning_rate":    (0.01, 0.05),
    "subsample":        (0.7, 0.95),
    "colsample_bytree": (0.6, 0.9),
    "reg_lambda":       (1.0, 10.0),
    "min_child_weight": (1, 10),
    "num_leaves":       (31, 255),
}

# ── Targets ──────────────────────────────────────────────────────────────────
TARGET_TOTAL = "total_enc"
TARGET_RATE = "admit_rate"
TARGET_ADMITTED = "admitted_enc"

TOTAL_OBJECTIVE = "tweedie"       # or "poisson"
TOTAL_TWEEDIE_POWER = 1.5
RATE_OBJECTIVE = "regression"

# ── Random Seed ──────────────────────────────────────────────────────────────
SEED = 42

# ── Internal Validation (early stopping split) ──────────────────────────────
EARLY_STOP_VAL_DAYS = 30  # last N days of training set used for early stopping


def print_config_summary() -> None:
    """Print a human-readable config summary for verification."""
    print("=" * 60)
    print("PIPELINE B — Direct Multi-Step GBDT — Config Summary")
    print("=" * 60)
    print(f"  Master data:     {MASTER_PARQUET}")
    print(f"  Output dir:      {OUTPUT_DIR}")
    print(f"  Folds:           {len(FOLDS)}")
    for f in FOLDS:
        print(f"    Fold {f.period_id}: train≤{f.train_end} → test {f.test_start}..{f.test_end}")
    print(f"  COVID policy:    downweight ×{COVID_WEIGHT_FACTOR} ({COVID_START} to {COVID_END})")
    print(f"  Horizon buckets: {len(HORIZON_BUCKETS)}")
    for b in HORIZON_BUCKETS:
        print(f"    Bucket {b.bucket_id}: days {b.horizon_min}-{b.horizon_max}, "
              f"min_lag={b.min_lag}, lags={b.lags}")
    print(f"  Optuna trials:   {OPTUNA_N_TRIALS}")
    print(f"  Seed:            {SEED}")
    print("=" * 60)


# Print on import for manual review
if os.environ.get("PIPELINE_B_QUIET") != "1":
    print_config_summary()
