from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from dsu_forecast import paths


@dataclass(frozen=True)
class SiteMeta:
    site: str
    name: str
    latitude: float | None
    longitude: float | None
    timezone: str


def load_sites_config(path: Path | None = None) -> dict[str, SiteMeta]:
    p = path or (paths.config_dir() / "sites.json")
    raw: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, SiteMeta] = {}
    for site, v in raw.items():
        out[site] = SiteMeta(
            site=site,
            name=str(v.get("name", f"Site {site}")),
            latitude=v.get("latitude", None),
            longitude=v.get("longitude", None),
            timezone=str(v.get("timezone", "America/Chicago")),
        )
    return out


def load_events_config(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or (paths.config_dir() / "events.yaml")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return list(raw.get("events", []))

