"""
Embedding hooks for REASON_VISIT_NAME case-mix.

This module is intentionally a lightweight, no-op placeholder so that we can
wire an "embedding features" path all the way through the data source layer
without committing to a specific embedding implementation yet.

Intended future design (not implemented yet):
  - Learn a dense embedding for each REASON_VISIT_NAME using visit-level data
    (e.g. frequency-based, co-occurrence-based, or a small neural model).
  - Aggregate per-visit embeddings up to (site, date, block) using a pooling
    strategy (mean pooling, attention-style weighting, etc.).
  - Emit one or more block-level embedding feature columns that can be
    consumed by downstream models (LightGBM, etc.) alongside existing
    case-mix and calendar features.

For now, `add_embedding_features` is a pure no-op that simply returns the
input DataFrame unchanged. This safely enables the CLI / config wiring and
gives us a stable surface area for later development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import DataSourceConfig


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Placeholder configuration for future embedding implementations.

    This is kept separate from `DataSourceConfig` so that we can evolve the
    embedding-specific hyperparameters (dimension, pooling method, training
    window, etc.) without touching the core ingestion contract.
    """

    dim: int = 16  # target embedding dimension (future use)
    pooling: str = "mean"  # or "attention", etc. (future use)


def add_embedding_features(
    block_df: pd.DataFrame,
    visits: pd.DataFrame,
    *,
    data_config: Optional[DataSourceConfig] = None,
    embed_config: Optional[EmbeddingConfig] = None,
) -> pd.DataFrame:
    """
    Hook for adding block-level embedding features derived from reason mix.

    Parameters
    ----------
    block_df:
        Current block-level master DataFrame, after core targets and case-mix
        counts have been added. One row per (site, date, block).
    visits:
        Raw visit-level DataFrame as returned by `load_visits`. This gives
        access to full REASON_VISIT_NAME and any other per-visit attributes we
        may want to use when learning embeddings.
    data_config:
        Optional `DataSourceConfig` instance used for paths / metadata.
    embed_config:
        Optional `EmbeddingConfig` instance controlling embedding behaviour
        (dimension, pooling method, etc.).

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same index/rows as `block_df`, potentially with
        additional embedding-derived feature columns.

    Notes
    -----
    - CURRENTLY A NO-OP: this function simply returns `block_df` unchanged.
    - When implemented, this function should:
        * Be deterministic for a fixed random seed.
        * Avoid data leakage across evaluation folds (fit on training spans
          only, then apply to validation / future spans).
        * Preserve the existing master schema and only append new columns.
    """

    # TODO: implement embedding pipeline:
    #   - learn embeddings for REASON_VISIT_NAME
    #   - aggregate to (site, date, block) via mean/attention pool
    #   - append embedding feature columns to `block_df`
    _ = (visits, data_config, embed_config)  # silence unused-argument warnings
    return block_df

