"""
Data Source layer package.

Implements the unified raw dataset described in
`Strategies/Data/data_source.md` using only the block-level visits
CSV in `Pipelines/Data Source/Data/DSU-Dataset.csv`.
"""

from __future__ import annotations

from . import config, ingestion  # re-export for convenience

__all__ = ["config", "ingestion"]

