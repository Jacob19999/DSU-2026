from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running without installing the package (repo-local `src/` layout)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsu_forecast.llm.openai_compat import OpenAICompatClient, chat_json
from dsu_forecast.paths import artifacts_dir, cache_dir


SYSTEM = """You convert National Weather Service alert text into structured, numeric features for ED forecasting.
Return JSON only.
Be conservative: if unsure, set low confidence and small impacts.
"""


def _alert_to_prompt(props: dict[str, Any]) -> str:
    # Keep it short to control token cost.
    fields = {
        "event": props.get("event"),
        "severity": props.get("severity"),
        "certainty": props.get("certainty"),
        "urgency": props.get("urgency"),
        "headline": props.get("headline"),
        "description": (props.get("description") or "")[:800],
        "instruction": (props.get("instruction") or "")[:400],
        "effective": props.get("effective") or props.get("onset"),
        "expires": props.get("expires") or props.get("ends"),
    }
    return f"""Alert:\n{json.dumps(fields, ensure_ascii=False)}\n\nReturn JSON with keys:\n- hazard: one of [heat,cold,snow,ice,wind,storm,flood,air_quality,fire,other]\n- severity_level: 0..3 (0 none, 3 extreme)\n- confidence: 0..1\n- expected_volume_uplift: -1..1 (negative means volume down)\n- expected_admit_uplift: -1..1\n- likely_blocks: array subset of [0,1,2,3] (6-hour blocks)\n- notes: short string\n"""


def build_llm_alert_features(cache_root: Path, out_path: Path) -> None:
    nws_dir = cache_root / "nws"
    if not nws_dir.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "Date",
                "llm_alert_count",
                "llm_severity_index",
                "llm_confidence_mean",
                "llm_expected_volume_uplift",
                "llm_expected_admit_uplift",
            ]
        ).to_parquet(out_path, index=False)
        print(f"[OK] no NWS cache found at {nws_dir}; wrote empty {out_path}")
        return

    client = OpenAICompatClient.from_env()

    rows: list[dict[str, Any]] = []
    for p in sorted(nws_dir.glob("*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        for f in payload.get("features", []) or []:
            props = f.get("properties", {}) or {}
            onset = props.get("onset") or props.get("effective")
            if not onset:
                continue
            try:
                onset_dt = pd.to_datetime(onset)
            except Exception:
                continue

            resp = chat_json(client, system=SYSTEM, user=_alert_to_prompt(props), temperature=0.0)

            rows.append(
                {
                    "Date": pd.to_datetime(onset_dt.date()),
                    "hazard": resp.get("hazard"),
                    "severity_level": resp.get("severity_level"),
                    "confidence": resp.get("confidence"),
                    "expected_volume_uplift": resp.get("expected_volume_uplift"),
                    "expected_admit_uplift": resp.get("expected_admit_uplift"),
                    "likely_blocks": json.dumps(resp.get("likely_blocks", [])),
                }
            )

    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "Date",
                "llm_alert_count",
                "llm_severity_index",
                "llm_confidence_mean",
                "llm_expected_volume_uplift",
                "llm_expected_admit_uplift",
            ]
        ).to_parquet(out_path, index=False)
        print(f"[OK] no alerts parsed; wrote empty {out_path}")
        return

    df = pd.DataFrame(rows)
    # Aggregate to daily hazard indices
    df["severity_level"] = pd.to_numeric(df["severity_level"], errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["expected_volume_uplift"] = pd.to_numeric(df["expected_volume_uplift"], errors="coerce").fillna(0.0)
    df["expected_admit_uplift"] = pd.to_numeric(df["expected_admit_uplift"], errors="coerce").fillna(0.0)

    daily = (
        df.groupby(["Date"], as_index=False)
        .agg(
            llm_alert_count=("hazard", "count"),
            llm_severity_index=("severity_level", "sum"),
            llm_confidence_mean=("confidence", "mean"),
            llm_expected_volume_uplift=("expected_volume_uplift", "sum"),
            llm_expected_admit_uplift=("expected_admit_uplift", "sum"),
        )
        .sort_values("Date")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(daily):,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", default=str(cache_dir()))
    ap.add_argument("--out", default=str(artifacts_dir() / "llm_nws_daily.parquet"))
    args = ap.parse_args()

    build_llm_alert_features(Path(args.cache_root), Path(args.out))


if __name__ == "__main__":
    main()

