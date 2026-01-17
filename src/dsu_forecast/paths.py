from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # This file is src/dsu_forecast/paths.py → repo root is 3 levels up.
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return repo_root() / "config"


def cache_dir() -> Path:
    return repo_root() / "data_cache"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def outputs_dir() -> Path:
    return repo_root() / "outputs"

