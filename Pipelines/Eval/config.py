"""
Eval: Centralized Pipeline Evaluation — Configuration.

Pipeline-agnostic scoring per eval.md contract.
Auto-discovers prediction CSVs from all pipelines (A–E).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]           # DSU-2026/
EVAL_DIR     = Path(__file__).resolve().parent                # Pipelines/Eval/

DATA_SOURCE_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
_DATA_SOURCE_CSV = DATA_SOURCE_DIR / "DSU-Dataset.csv"
_DATASET_CSV     = PROJECT_ROOT / "Dataset" / "DSU-Dataset.csv"
RAW_VISITS_CSV   = _DATA_SOURCE_CSV if _DATA_SOURCE_CSV.exists() else _DATASET_CSV

OUTPUT_DIR = EVAL_DIR / "output"

# ── Sites & Blocks (must match eval.md) ──────────────────────────────────────

SITES  = ("A", "B", "C", "D")
BLOCKS = (0, 1, 2, 3)
COLS   = ("Site", "Date", "Block", "ED Enc", "ED Enc Admitted")

# ── Validation Folds (from eval.md §2) ──────────────────────────────────────


@dataclass(frozen=True)
class Fold:
    period_id:   int
    train_end:   str
    test_start:  str
    test_end:    str
    description: str = ""


FOLDS: List[Fold] = [
    Fold(1, "2024-12-31", "2025-01-01", "2025-02-28", "Train≤Dec2024 → Valid Jan-Feb2025"),
    Fold(2, "2025-02-28", "2025-03-01", "2025-04-30", "Train≤Feb2025 → Valid Mar-Apr2025"),
    Fold(3, "2025-04-30", "2025-05-01", "2025-06-30", "Train≤Apr2025 → Valid May-Jun2025"),
    Fold(4, "2025-06-30", "2025-07-01", "2025-08-31", "Train≤Jun2025 → Valid Jul-Aug2025"),
]

# ── Pipeline Registry ────────────────────────────────────────────────────────
# Maps pipeline name → list of candidate directories where fold CSVs live.
# The evaluator tries each dir in order and picks the first that has files.

PIPELINES_DIR = PROJECT_ROOT / "Pipelines"

PIPELINE_PRED_DIRS: Dict[str, List[Path]] = {
    "A": [
        PIPELINES_DIR / "Pipeline A" / "output",
    ],
    "B": [
        PIPELINES_DIR / "Pipeline B" / "output" / "predictions",
    ],
    "C": [
        PIPELINES_DIR / "Pipeline C" / "output",
        PIPELINES_DIR / "Pipeline C" / "output" / "predictions",
    ],
    "D": [
        PIPELINES_DIR / "Pipeline D" / "output" / "predictions",
    ],
    "E": [
        PIPELINES_DIR / "Pipeline E" / "output" / "predictions",
        PIPELINES_DIR / "Pipeline E" / "output",
    ],
}

# Prediction file naming: fold_{id}_predictions.csv (universal across pipelines)
PRED_FILE_TEMPLATE = "fold_{fold_id}_predictions.csv"


# ── Discovery helpers ────────────────────────────────────────────────────────

def _resolve_pred_dir(pipeline: str) -> Optional[Path]:
    """Return the first candidate dir that contains at least one fold CSV."""
    for d in PIPELINE_PRED_DIRS.get(pipeline, []):
        if d.exists() and any(d.glob("fold_*_predictions.csv")):
            return d
    return None


def get_fold_csv_path(pipeline: str, fold_id: int) -> Optional[Path]:
    """Return path to a specific fold's prediction CSV, or None if missing."""
    pred_dir = _resolve_pred_dir(pipeline)
    if pred_dir is None:
        return None
    p = pred_dir / PRED_FILE_TEMPLATE.format(fold_id=fold_id)
    return p if p.exists() else None


def discover_pipelines() -> Dict[str, Path]:
    """Return {pipeline_name: pred_dir} for all pipelines with predictions."""
    found = {}
    for name in PIPELINE_PRED_DIRS:
        d = _resolve_pred_dir(name)
        if d is not None:
            found[name] = d
    return found


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    """Create output directory tree."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
