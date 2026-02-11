"""
Data Source ingestion — build the unified raw block-level dataset.

Implements the plan in `Strategies/Data/data_source.md` using only the
visit-level CSV at `Pipelines/Data Source/Data/DSU-Dataset.csv`.

This layer:
  - Aggregates visit-level data to (site, date, block)
  - Builds a full skeleton grid across all dates/sites/blocks
  - Adds raw case-mix counts for the top-N REASON_VISIT_NAME categories
  - Adds deterministic calendar features and COVID-era flags
  - Stubs external features (events/weather/school/optional) as missing

NO IMPUTATION is performed except:
  - Targets (`total_enc`, `admitted_enc`) → 0 when no records exist
  - Case-mix counts → 0 when no visits for that reason/block
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import (
    BLOCKS,
    COVID_END,
    COVID_START,
    DATA_DIR,
    DataSourceConfig,
    EPOCH_START,
    SITES,
)
from .external_data import add_external_features
from .embedding import EmbeddingConfig, add_embedding_features


logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class IngestionArtifacts:
    """Optional side artifacts produced during ingestion."""

    top_reasons: List[str]
    reason_slug_map: Dict[str, str]
    reason_summary_path: Path | None


def run_data_ingestion(
    config: DataSourceConfig | None = None,
    save: bool = True,
    save_reason_summary: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate the full ingestion flow and optionally persist outputs.

    Returns:
        The unified block-level DataFrame with schema matching the strategy doc.
    """
    config = config or DataSourceConfig()

    logger.info("Starting Data Source ingestion")
    logger.info("Raw visits: %s", config.raw_visits)

    # 1. Load and standardize visit-level data
    visits = load_visits(config.raw_visits)
    logger.info("Loaded %d visit rows", len(visits))

    # 2. Build full skeleton grid (sites × dates × blocks)
    grid = create_block_grid(
        start_date=config.grid_start,
        end_date=config.grid_end,
        sites=SITES,
        blocks=BLOCKS,
    )
    logger.info("Skeleton grid size: %d rows", len(grid))

    # 3. Core targets (total_enc / admitted_enc)
    core = aggregate_core_metrics(visits)
    master = (
        grid.merge(core, on=["site", "date", "block"], how="left")
        .assign(
            total_enc=lambda df: df["total_enc"].fillna(0).astype(int),
            admitted_enc=lambda df: df["admitted_enc"].fillna(0).astype(int),
        )
    )

    # 4. Case mix: top-N reasons + other bucket
    reason_summary_path: Path | None = None
    (
        reasons,
        top_reasons,
        reason_slug_map,
    ) = aggregate_reasons(visits, top_n=config.top_n_reasons)

    master = master.merge(reasons, on=["site", "date", "block"], how="left")

    # Fill any missing count_reason_* columns with 0
    reason_cols = [c for c in master.columns if c.startswith("count_reason_")]
    if reason_cols:
        master[reason_cols] = master[reason_cols].fillna(0).astype(int)

    # Optional: save a human-readable summary of reason categories
    if save and save_reason_summary:
        reason_summary_path = _save_reason_summary(
            visits=visits,
            top_reasons=top_reasons,
            slug_map=reason_slug_map,
            output_dir=DATA_DIR,
        )

    # 5. Optional: embedding features derived from reason mix
    if getattr(config, "use_reason_embeddings", False):
        logger.info("Adding reason-embedding features (SapBERT → MiniLM → TF-IDF+SVD).")
        embed_cfg = EmbeddingConfig(cache_dir=config.external_cache_dir)
        master = add_embedding_features(
            block_df=master,
            visits=visits,
            data_config=config,
            embed_config=embed_cfg,
        )

    # 6. Deterministic calendar and COVID/halloween flags
    master = add_calendar_features(master)

    # 7. External data (events, weather, school calendar, CDC ILI, AQI)
    master = add_external_features(
        df=master,
        sites=list(SITES),
        start_date=config.grid_start,
        end_date=config.grid_end,
        cache_dir=config.external_cache_dir,
        fetch_apis=config.fetch_apis,
    )

    # Final sort for downstream lag/rolling ops
    master = master.sort_values(["site", "block", "date"]).reset_index(drop=True)

    logger.info(
        "Final master_block_history shape: %d rows × %d columns",
        len(master),
        master.shape[1],
    )

    # 8. Persist parquet/csv if requested
    if save:
        _save_master_dataset(master, config)

    # Log any schema anomalies that might bite downstream
    _log_basic_checks(master)

    # Expose artifacts via logger (artifacts object returned only if needed later)
    artifacts = IngestionArtifacts(
        top_reasons=top_reasons,
        reason_slug_map=reason_slug_map,
        reason_summary_path=reason_summary_path,
    )
    logger.info("Top %d reasons: %s", len(artifacts.top_reasons), artifacts.top_reasons)

    return master


# ── Step 1: Load / Standardize Visits ─────────────────────────────────────────

EXPECTED_VISIT_COLUMNS = {
    "Site",
    "Date",
    "Hour",
    "REASON_VISIT_NAME",
    "ED Enc",
    "ED Enc Admitted",
}


def load_visits(path: Path) -> pd.DataFrame:
    """
    Load the raw visit-level CSV and standardise column names / types.

    - Ensures expected columns are present
    - Normalises date to midnight
    - Casts hour to int and derives block = hour // 6
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw visits CSV not found: {path}")

    df = pd.read_csv(path)

    missing = EXPECTED_VISIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw visits missing expected columns: {missing}")

    df = df.rename(
        columns={
            "Site": "site",
            "Date": "date",
            "Hour": "hour",
            "REASON_VISIT_NAME": "reason_visit_name",
            "ED Enc": "ed_enc",
            "ED Enc Admitted": "ed_enc_admitted",
        }
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["hour"] = df["hour"].astype(int)
    df["block"] = (df["hour"] // 6).astype(int)

    # Safety: block should be 0-3 by construction; warn if anything else shows up
    bad_blocks = sorted(set(df["block"].unique()) - set(BLOCKS))
    if bad_blocks:
        logger.warning("Unexpected block values encountered: %s", bad_blocks)

    return df


# ── Step 2: Skeleton Grid ─────────────────────────────────────────────────────

def create_block_grid(
    start_date: str,
    end_date: str,
    sites: Iterable[str],
    blocks: Iterable[int],
) -> pd.DataFrame:
    """
    Create the full Cartesian product:
        (all sites) × (all dates) × (blocks 0-3)
    """
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    idx = pd.MultiIndex.from_product(
        [list(sites), date_index, list(blocks)],
        names=["site", "date", "block"],
    )
    grid = idx.to_frame(index=False)
    return grid


# ── Step 3: Core Targets ──────────────────────────────────────────────────────

def aggregate_core_metrics(visits: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate visit-level rows to (site, date, block) targets.

    - total_enc: sum of ED Enc
    - admitted_enc: sum of ED Enc Admitted
    """
    grouped = (
        visits.groupby(["site", "date", "block"], as_index=False)[
            ["ed_enc", "ed_enc_admitted"]
        ]
        .sum()
        .rename(
            columns={
                "ed_enc": "total_enc",
                "ed_enc_admitted": "admitted_enc",
            }
        )
    )
    return grouped


# ── Step 4: Case Mix Aggregation ──────────────────────────────────────────────

def _slugify_reason(reason: str) -> str:
    """Convert a REASON_VISIT_NAME into a safe column suffix."""
    reason = reason.lower().strip()
    reason = re.sub(r"[^0-9a-z]+", "_", reason)
    reason = re.sub(r"_+", "_", reason).strip("_")
    return reason or "unknown"


def aggregate_reasons(
    visits: pd.DataFrame,
    top_n: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Build raw count_reason_* columns for the top-N categories.

    Returns:
        reasons_df: (site, date, block, count_reason_*) wide frame
        top_reasons: list of top-N REASON_VISIT_NAME values (by volume)
        slug_map: mapping REASON_VISIT_NAME → slug used in column names
    """
    volume_by_reason = (
        visits.groupby("reason_visit_name", as_index=False)["ed_enc"]
        .sum()
        .sort_values("ed_enc", ascending=False)
    )
    top_reasons = volume_by_reason["reason_visit_name"].head(top_n).tolist()
    top_set = set(top_reasons)

    # Aggregated counts per (site, date, block, reason)
    grouped = (
        visits.groupby(
            ["site", "date", "block", "reason_visit_name"],
            as_index=False,
        )["ed_enc"].sum()
    )

    def map_reason_key(name: str) -> str:
        return name if name in top_set else "__OTHER__"

    grouped["reason_key"] = grouped["reason_visit_name"].map(map_reason_key)

    # Build a deterministic slug for each top reason; "other" bucket uses fixed name
    unique_keys = sorted(grouped["reason_key"].unique())
    slug_map: Dict[str, str] = {}
    for key in unique_keys:
        if key == "__OTHER__":
            slug_map[key] = "other"
        else:
            slug_map[key] = _slugify_reason(key)

    grouped["col_name"] = grouped["reason_key"].map(
        lambda k: f"count_reason_{slug_map[k]}"
    )

    # Collapse to wide format: one row per (site, date, block)
    pivot = (
        grouped.pivot_table(
            index=["site", "date", "block"],
            columns="col_name",
            values="ed_enc",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    # Ensure deterministic column order: index cols first, then sorted counts
    count_cols = sorted([c for c in pivot.columns if c.startswith("count_reason_")])
    pivot = pivot[["site", "date", "block", *count_cols]]

    # Map back to full REASON_VISIT_NAME → slug (excluding the synthetic other-key)
    full_slug_map = {reason: _slugify_reason(reason) for reason in top_reasons}

    return pivot, top_reasons, full_slug_map


def _save_reason_summary(
    visits: pd.DataFrame,
    top_reasons: List[str],
    slug_map: Dict[str, str],
    output_dir: Path,
) -> Path:
    """
    Save a summary CSV describing which reasons were selected as "top N".

    This makes it easy to inspect case-mix choices without digging into code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_by_reason = (
        visits.groupby("reason_visit_name", as_index=False)["ed_enc"]
        .sum()
        .rename(columns={"ed_enc": "total_ed_enc"})
        .sort_values("total_ed_enc", ascending=False)
    )
    volume_by_reason["rank"] = (
        volume_by_reason["total_ed_enc"].rank(
            method="first", ascending=False
        ).astype(int)
    )
    volume_by_reason["is_top_n"] = volume_by_reason["reason_visit_name"].isin(
        top_reasons
    )
    volume_by_reason["slug"] = volume_by_reason["reason_visit_name"].map(
        lambda r: slug_map.get(r, _slugify_reason(r))
    )

    path = output_dir / "reason_category_summary.csv"
    volume_by_reason.to_csv(path, index=False)
    logger.info("Saved reason category summary: %s", path)
    return path


# ── Step 5: Calendar & Temporal Features ──────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar features and COVID / halloween flags."""
    df = df.copy()

    df["dow"] = df["date"].dt.weekday  # 0=Mon
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)

    iso_week = df["date"].dt.isocalendar().week
    df["week_of_year"] = iso_week.astype(int)

    df["is_weekend"] = df["dow"].isin([5, 6])

    epoch = pd.Timestamp(EPOCH_START)
    df["days_since_epoch"] = (df["date"] - epoch).dt.days.astype(int)

    df["is_covid_era"] = df["date"].between(
        pd.Timestamp(COVID_START),
        pd.Timestamp(COVID_END),
    )

    df["is_halloween"] = (df["month"] == 10) & (df["day"] == 31)

    return df


# ── Step 6: External Data Stubs ───────────────────────────────────────────────

def add_stub_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED — replaced by ``external_data.add_external_features()``.

    Retained as fallback reference.  The real implementation fetches data
    from US-holidays, Open-Meteo, CDC FluView, and EPA AQI sources.
    """
    df = df.copy()

    # Event / holiday fields
    if "is_holiday" not in df.columns:
        df["is_holiday"] = False
    if "event_name" not in df.columns:
        df["event_name"] = pd.NA
    if "event_type" not in df.columns:
        df["event_type"] = pd.NA
    if "event_count" not in df.columns:
        df["event_count"] = 0

    # Weather fields (kept as NaN for now)
    for col in ("temp_min", "temp_max", "precip", "snowfall"):
        if col not in df.columns:
            df[col] = np.nan

    # School calendar
    if "school_in_session" not in df.columns:
        df["school_in_session"] = np.nan

    # Optional external enrichments
    for col in ("cdc_ili_rate", "aqi"):
        if col not in df.columns:
            df[col] = np.nan

    return df


# ── Step 7: Persistence & Checks ──────────────────────────────────────────────

def _save_master_dataset(df: pd.DataFrame, config: DataSourceConfig) -> None:
    """Persist master_block_history.{parquet,csv} side by side."""
    out_dir = config.master_parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet (preferred) + CSV mirror for easy inspection
    try:
        df.to_parquet(config.master_parquet, index=False)
        logger.info("Saved master parquet: %s", config.master_parquet)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.error("Failed to write parquet (%s). CSV will still be written.", exc)

    df.to_csv(config.master_csv, index=False)
    logger.info("Saved master CSV: %s", config.master_csv)


def _log_basic_checks(df: pd.DataFrame) -> None:
    """Log a few sanity checks on the resulting dataset."""
    n_sites = df["site"].nunique()
    n_blocks = df["block"].nunique()
    n_dates = df["date"].nunique()
    expected_rows = n_sites * n_blocks * n_dates

    logger.info(
        "Grid check: sites=%d, blocks=%d, dates=%d -> expected_rows=%d, actual=%d",
        n_sites,
        n_blocks,
        n_dates,
        expected_rows,
        len(df),
    )
    if len(df) != expected_rows:
        logger.warning("Row-count mismatch: grid is not fully dense")

    for col in ("total_enc", "admitted_enc"):
        n_nan = df[col].isna().sum()
        if n_nan:
            logger.error("%s has %d NaN values (expected 0)", col, n_nan)

