"""
Data Source configuration — paths and constants for ingestion.

This is intentionally self-contained inside `Pipelines` and uses
`Pipelines/Data Source/Data/DSU-Dataset.csv` as the only raw input.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DSU-2026/

# Original transactional visit-level dataset (single canonical source)
DATA_DIR = PROJECT_ROOT / "Pipelines" / "Data Source" / "Data"
RAW_VISITS = DATA_DIR / "DSU-Dataset.csv"

# Unified block-level history produced by this layer
MASTER_PARQUET = DATA_DIR / "master_block_history.parquet"
MASTER_CSV = DATA_DIR / "master_block_history.csv"


# ── Grid Definition ────────────────────────────────────────────────────────────

# Sites and blocks are fixed by the competition contract
SITES = ("A", "B", "C", "D")
BLOCKS = (0, 1, 2, 3)

# Full history span for the unified grid (eval.md / strategy doc)
GRID_START = "2018-01-01"
GRID_END = "2025-10-31"

# Reference epoch for trend-style features
EPOCH_START = "2018-01-01"


# ── COVID Era (used only for flagging in this layer) ───────────────────────────

COVID_START = "2020-03-01"
COVID_END = "2021-06-30"


# ── Case-Mix Configuration ─────────────────────────────────────────────────────

# Number of top REASON_VISIT_NAME categories to keep as explicit columns
TOP_N_REASONS = 20


@dataclass(frozen=True)
class DataSourceConfig:
    """Lightweight configuration passed into the ingestion runner."""

    raw_visits: Path = RAW_VISITS
    master_parquet: Path = MASTER_PARQUET
    master_csv: Path = MASTER_CSV
    grid_start: str = GRID_START
    grid_end: str = GRID_END
    top_n_reasons: int = TOP_N_REASONS


