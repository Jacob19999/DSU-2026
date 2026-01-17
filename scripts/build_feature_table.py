from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running without installing the package (repo-local `src/` layout)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsu_forecast.config import load_sites_config
from dsu_forecast.data.aggregate import aggregate_targets_to_block, make_forecast_grid
from dsu_forecast.features.calendar import add_calendar_features
from dsu_forecast.features.case_mix import attach_case_mix_features
from dsu_forecast.features.external_covariates import join_external_covariates
from dsu_forecast.paths import outputs_dir, repo_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_end", default="2025-08-31")
    ap.add_argument("--forecast_start", default="2025-09-01")
    ap.add_argument("--forecast_end", default="2025-10-31")
    ap.add_argument("--out", default=str(outputs_dir() / "feature_table.parquet"))
    ap.add_argument("--use_cache", action="store_true", default=True)
    args = ap.parse_args()

    raw_path = repo_root() / "Dataset" / "DSU-Dataset.csv"
    df_raw = pd.read_csv(raw_path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    # Training targets up to train_end
    train_raw = df_raw[df_raw["Date"] <= pd.to_datetime(args.train_end)].copy()
    y = aggregate_targets_to_block(train_raw)

    sites = sorted(df_raw["Site"].dropna().unique().tolist())
    horizon = make_forecast_grid(sites=sites, start_date=args.forecast_start, end_date=args.forecast_end)

    base = pd.concat([y[["Site", "Date", "Block", "total_enc", "admitted_enc"]], horizon], ignore_index=True, sort=False)
    base = base.drop_duplicates(subset=["Site", "Date", "Block"], keep="first")

    base = add_calendar_features(base, date_col="Date")
    base = attach_case_mix_features(base, train_end=args.train_end)
    sites_meta = load_sites_config()
    base = join_external_covariates(base, site_meta=sites_meta, train_end=args.train_end, use_cache=args.use_cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(base):,} cols={len(base.columns)}")


if __name__ == "__main__":
    main()

