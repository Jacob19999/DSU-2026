from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _key_to_filename(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.json"


def read_json_cache(cache_dir: Path, key: str) -> dict[str, Any] | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / _key_to_filename(key)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_json_cache(cache_dir: Path, key: str, payload: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / _key_to_filename(key)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p

