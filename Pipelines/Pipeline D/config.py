"""
Pipeline D: GLM/GAM with Fourier Seasonality — Configuration & constants.

Per-(Site, Block) Poisson GLM with explicit Fourier seasonal decomposition.
16 total_enc models + 16 admit_rate models = 32 models per fold.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]           # DSU-2026/
PIPELINE_DIR = Path(__file__).resolve().parent                # Pipelines/Pipeline D/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
MASTER_PARQUET  = DATA_SOURCE_DIR / "master_block_history.parquet"
RAW_VISITS_CSV  = DATA_SOURCE_DIR / "DSU-Dataset.csv"

OUTPUT_DIR  = PIPELINE_DIR / "output"
MODEL_DIR   = OUTPUT_DIR / "models"
PRED_DIR    = OUTPUT_DIR / "predictions"
EVAL_DIR    = OUTPUT_DIR / "evaluation"
DIAG_DIR    = OUTPUT_DIR / "diagnostics"
LOG_DIR     = OUTPUT_DIR / "logs"

# ── Sites & Blocks ───────────────────────────────────────────────────────────

SITES  = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]

# ── Validation Folds (from eval.md §2.1) ─────────────────────────────────────

FOLDS = [
    {"id": 1, "train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-02-28"},
    {"id": 2, "train_end": "2025-02-28", "val_start": "2025-03-01", "val_end": "2025-04-30"},
    {"id": 3, "train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-06-30"},
    {"id": 4, "train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2025-08-31"},
]

# ── COVID Era ────────────────────────────────────────────────────────────────

COVID_START  = "2020-03-01"
COVID_END    = "2021-06-30"
COVID_WEIGHT = 0.1   # freq_weights factor for COVID-era rows (Policy 3)

# ── Fourier Specification ────────────────────────────────────────────────────
# §3.4: sin/cos at periods 7, 365.25 (order 3 and 10 respectively)

FOURIER_TERMS = [
    {"period": 7,      "order": 3},    # Weekly (6 features)
    {"period": 365.25, "order": 10},   # Annual (20 features)
]

# ── GLM Model Configuration ─────────────────────────────────────────────────

GLM_ALPHA   = 0.1     # L2 regularization strength
GLM_L1_WT   = 0.0     # 0 = pure Ridge; 1 = pure Lasso
GLM_MAXITER = 200

# ── Tuning Search Space ─────────────────────────────────────────────────────

FOURIER_ORDER_SEARCH = {
    "weekly_order":  [1, 2, 3, 4, 5],
    "annual_order":  [3, 5, 7, 10, 12, 15],
}
ALPHA_SEARCH = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# ── School-year start dates (Fargo / Sioux Falls) ───────────────────────────

SCHOOL_STARTS = [
    "2018-08-20", "2019-08-19", "2020-09-08",
    "2021-08-23", "2022-08-22", "2023-08-21",
    "2024-08-19", "2025-08-18",
]

# ── Post-processing ─────────────────────────────────────────────────────────

CLIP_TOTAL_MIN    = 0
ADMIT_RATE_CLIP   = (0.0, 1.0)
MAX_TOTAL_FACTOR  = 1.5   # Safety rail: cap predictions at factor × historical max


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create all output directories."""
    for d in [OUTPUT_DIR, MODEL_DIR, PRED_DIR, EVAL_DIR, DIAG_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def fold_model_dir(fold_id: int) -> Path:
    """Return per-fold model directory, creating it if needed."""
    p = MODEL_DIR / f"fold_{fold_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def print_config_summary() -> None:
    """Print configuration for manual review."""
    print("Pipeline D: GLM/GAM with Fourier Seasonality - Config Summary")
    print(f"  Master data  : {MASTER_PARQUET}")
    print(f"  Fourier terms:")
    for ft in FOURIER_TERMS:
        n_feat = 2 * ft["order"]
        print(f"    period={ft['period']:>7}  order={ft['order']:>2}  -> {n_feat} features")
    print(f"  GLM alpha    : {GLM_ALPHA}")
    print(f"  COVID weight : {COVID_WEIGHT}")
    print(f"  Folds        : {len(FOLDS)}")
    for f in FOLDS:
        print(f"    Fold {f['id']}: train<={f['train_end']}  val={f['val_start']}..{f['val_end']}")
