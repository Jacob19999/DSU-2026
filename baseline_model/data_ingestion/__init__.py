"""Data ingestion module for baseline model pipelines."""

from .loader import load_dataset, create_validation_splits

__all__ = ['load_dataset', 'create_validation_splits']
