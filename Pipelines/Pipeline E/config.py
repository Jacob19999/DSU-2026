"""
Pipeline E: Reason-Mix Latent Factor Model — Configuration & constants.

Compress visit-reason composition into latent factors (PCA/NMF),
forecast factors forward via GBDT, use predicted factors + momentum
as extra regressors in a final GBDT that predicts total_enc and admit_rate.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # DSU-2026/
PIPELINE_DIR = Path(__file__).resolve().parent               # Pipelines/Pipeline E/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET = DATA_SOURCE_DIR / "master_block_history.parquet"
RAW_VISITS_CSV = DATA_SOURCE_DIR / "DSU-Dataset.csv"

OUTPUT_DIR = PIPELINE_DIR / "output"
MODEL_DIR  = OUTPUT_DIR / "models"
PRED_DIR   = OUTPUT_DIR / "predictions"
EVAL_DIR   = OUTPUT_DIR / "evaluation"
FI_DIR     = OUTPUT_DIR / "feature_importance"
DATA_DIR   = PIPELINE_DIR / "data"

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
COVID_WEIGHT = 0.1

# ── Factor Extraction ───────────────────────────────────────────────────────

N_FACTORS           = 5         # Latent factors (test 3-7 via ablation)
FACTOR_METHOD       = "pca"    # "pca" or "nmf"
TOP_N_REASONS       = 20       # Match data_source.md §2.2
MIN_CATEGORY_VOLUME = 100      # Drop categories below this total volume
SHARE_SMOOTH_WINDOW = 7        # Rolling mean window for share stabilisation

# ── Factor Forecasting (safe-lag v1: all lags >= 63) ────────────────────────

FACTOR_LAG_DAYS        = [63, 70, 77, 91, 182, 364]
FACTOR_ROLLING_WINDOWS = [7, 14, 28]
FACTOR_ROLLING_SHIFT   = 63

# ── Final Model ─────────────────────────────────────────────────────────────

MAX_HORIZON     = 63
LAG_DAYS        = [63, 70, 77, 91, 182, 364]
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
ROLLING_SHIFT   = 63

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

LGBM_FACTOR_FORECAST = {
    "objective": "regression",
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "min_child_weight": 3,
    "verbosity": -1,
}

# ── Optuna ───────────────────────────────────────────────────────────────────

OPTUNA_N_TRIALS_FACTOR = 30    # Stage 1: factor extraction + forecast config
OPTUNA_N_TRIALS_TOTAL  = 100   # Stage 2: final total_enc model
OPTUNA_N_TRIALS_RATE   = 50    # Stage 2: final admit_rate model

# ── Early Stopping & Seeds ──────────────────────────────────────────────────

ES_PATIENCE  = 50
ES_HOLD_DAYS = 30
SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────────


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, EVAL_DIR, FI_DIR, DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def fold_model_dir(fold_id: int) -> Path:
    """Return per-fold model directory, creating it if needed."""
    p = MODEL_DIR / f"fold_{fold_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def print_config_summary() -> None:
    """Print configuration for manual review."""
    print("Pipeline E: Reason-Mix Latent Factor Model — Config Summary")
    print(f"  Master data  : {MASTER_PARQUET}")
    print(f"  N factors    : {N_FACTORS} ({FACTOR_METHOD})")
    print(f"  Top reasons  : {TOP_N_REASONS}")
    print(f"  Share smooth : {SHARE_SMOOTH_WINDOW}-day rolling mean")
    print(f"  Max horizon  : {MAX_HORIZON}")
    print(f"  Target lags  : {LAG_DAYS}")
    print(f"  Factor lags  : {FACTOR_LAG_DAYS}")
    print(f"  Rolling      : {ROLLING_WINDOWS}")
    print(f"  Folds        : {len(FOLDS)}")
    for f in FOLDS:
        print(f"    Fold {f['id']}: train<={f['train_end']}  "
              f"val={f['val_start']}..{f['val_end']}")
    print(f"  Seed         : {SEED}")
