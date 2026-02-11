"""
Data Source entry point.

Usage (from project root):

    python -m Pipelines.data_source.run_data_source

This will:
  - Load `Pipelines/Data Source/Data/DSU-Dataset.csv`
  - Build the unified block-level dataset
  - Write:
        master_block_history.parquet
        master_block_history.csv
        reason_category_summary.csv
    into `Pipelines/Data Source/Data/`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DataSourceConfig
from .ingestion import run_data_ingestion


def _setup_logging() -> None:
    """Configure basic logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Source ingestion — build master_block_history dataset",
    )
    parser.add_argument(
        "--top-n-reasons",
        type=int,
        default=DataSourceConfig().top_n_reasons,
        help="Number of top REASON_VISIT_NAME categories to keep as explicit "
        "count_reason_* columns (default: %(default)s).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run ingestion but do not write parquet/csv outputs.",
    )
    parser.add_argument(
        "--no-reason-summary",
        action="store_true",
        help="Skip writing reason_category_summary.csv.",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip network API calls for external data (weather, CDC ILI, AQI). "
        "Deterministic features (events, school calendar) are always generated. "
        "Cached API data is still loaded if available.",
    )
    parser.add_argument(
        "--use-reason-embeddings",
        action="store_true",
        help=(
            "Enable experimental reason-embedding features in the Data Source. "
            "This currently wires a no-op hook (`embedding.add_embedding_features`) "
            "that can later be extended to emit block-level embedding vectors "
            "derived from REASON_VISIT_NAME."
        ),
    )

    args = parser.parse_args()
    _setup_logging()

    config = DataSourceConfig(
        top_n_reasons=args.top_n_reasons,
        fetch_apis=not args.no_fetch,
        use_reason_embeddings=args.use_reason_embeddings,
    )
    logging.getLogger(__name__).info("Using config: %s", config)

    df = run_data_ingestion(
        config=config,
        save=not args.no_save,
        save_reason_summary=not args.no_reason_summary,
    )

    logging.getLogger(__name__).info(
        "Ingestion complete - final shape: %d rows x %d columns",
        len(df),
        df.shape[1],
    )


if __name__ == "__main__":
    main()

