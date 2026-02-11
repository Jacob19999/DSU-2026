"""
Pipeline B: Direct Multi-Step GBDT — Configuration & constants.

Horizon-aware bucket models: short-horizon predictions exploit recent lags.
3 buckets × 2 targets = 6 LightGBM models.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # DSU-2026/
PIPELINE_DIR = Path(__file__).resolve().parent               # Pipelines/Pipeline B/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET = DATA_SOURCE_DIR / "master_block_history.parquet"
RAW_VISITS_CSV = DATA_SOURCE_DIR / "DSU-Dataset.csv"

OUTPUT_DIR  = PIPELINE_DIR / "output"
MODEL_DIR   = OUTPUT_DIR / "models"
PRED_DIR    = OUTPUT_DIR / "predictions"
EVAL_DIR    = OUTPUT_DIR / "evaluation"
FI_DIR      = OUTPUT_DIR / "feature_importance"
LOG_DIR     = OUTPUT_DIR / "logs"

# ── Sites & Blocks ───────────────────────────────────────────────────────────

SITES  = ["A", "B", "C", "D"]
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
COVID_END   = "2021-06-30"
COVID_WEIGHT = 0.1     # Downweight factor for COVID-era rows

# ── Horizon Buckets (Option B1 — preferred) ─────────────────────────────────

BUCKETS = {
    1: {"h_min": 1,  "h_max": 15, "min_lag": 16},
    2: {"h_min": 16, "h_max": 30, "min_lag": 31},
    3: {"h_min": 31, "h_max": 62, "min_lag": 63},
}

# Sub-sampled horizons per bucket — reduces 61× expansion to ~13×
BUCKET_HORIZONS = {
    1: [1, 4, 7, 10, 13],
    2: [16, 20, 24, 28],
    3: [31, 40, 50, 61],
}

# Lag sets per bucket (distance from as-of date)
BUCKET_LAGS = {
    1: [16, 21, 28, 56, 91, 182, 364],
    2: [31, 35, 42, 56, 91, 182, 364],
    3: [63, 70, 77, 91, 182, 364],
}

ROLLING_WINDOWS = [7, 14, 28, 56, 91]

# ── School-year start dates (Fargo / Sioux Falls) ───────────────────────────

SCHOOL_STARTS = [
    "2018-08-20", "2019-08-19", "2020-09-08",   # 2020: COVID late start
    "2021-08-23", "2022-08-22", "2023-08-21",
    "2024-08-19", "2025-08-18",
]

# ── LightGBM Defaults (pre-Optuna) ──────────────────────────────────────────

LGBM_DEFAULT_TOTAL = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

LGBM_DEFAULT_RATE = {
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

# ── Optuna ───────────────────────────────────────────────────────────────────

OPTUNA_N_TRIALS_TOTAL = 100   # Per bucket, total_enc model
OPTUNA_N_TRIALS_RATE  = 50    # Per bucket, admit_rate model

# ── Random Seed ──────────────────────────────────────────────────────────────

SEED = 42

# ── Early Stopping ───────────────────────────────────────────────────────────

ES_PATIENCE  = 50
ES_HOLD_DAYS = 30     # Last N days of training used for early-stopping validation


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all output directories."""
    for d in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, EVAL_DIR, FI_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def fold_model_dir(fold_id: int) -> Path:
    """Return per-fold model directory, creating it if needed."""
    p = MODEL_DIR / f"fold_{fold_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def print_config_summary() -> None:
    """Print configuration for manual review."""
    print("Pipeline B: Direct Multi-Step GBDT — Config Summary")
    print(f"  Master data : {MASTER_PARQUET}")
    print(f"  Buckets     : {len(BUCKETS)}")
    for bid, bcfg in BUCKETS.items():
        lags = BUCKET_LAGS[bid]
        hrzs = BUCKET_HORIZONS[bid]
        print(f"    Bucket {bid}: h=[{bcfg['h_min']},{bcfg['h_max']}]  "
              f"min_lag={bcfg['min_lag']}  lags={lags}  horizons={hrzs}")
    print(f"  Rolling     : {ROLLING_WINDOWS}")
    print(f"  Folds       : {len(FOLDS)}")
    for f in FOLDS:
        print(f"    Fold {f['id']}: train<={f['train_end']}  val={f['val_start']}..{f['val_end']}")
    print(f"  Optuna      : {OPTUNA_N_TRIALS_TOTAL} total / {OPTUNA_N_TRIALS_RATE} rate per bucket")
    print(f"  Seed        : {SEED}")


# ── Safety checks on import ──────────────────────────────────────────────────

for _bid, _bcfg in BUCKETS.items():
    _min_lag_used = min(BUCKET_LAGS[_bid])
    assert _min_lag_used >= _bcfg["min_lag"], (
        f"LEAKAGE: Bucket {_bid} min lag {_min_lag_used} < required {_bcfg['min_lag']}"
    )
