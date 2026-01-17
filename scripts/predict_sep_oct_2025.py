from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# Allow running without installing the package (repo-local `src/` layout)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsu_forecast.modeling.features import add_lag_features, add_rolling_mean_features
from dsu_forecast.paths import artifacts_dir, outputs_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(outputs_dir() / "feature_table.parquet"))
    ap.add_argument("--artifacts", default=str(artifacts_dir()))
    ap.add_argument("--forecast_start", default="2025-09-01")
    ap.add_argument("--forecast_end", default="2025-10-31")
    ap.add_argument("--out", default=str(outputs_dir() / "forecast_sep_oct_2025.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.features))
    df["Date"] = pd.to_datetime(df["Date"])

    # Recreate lag features the same way as training (but only needed for prediction rows)
    df_hist = df[df["total_enc"].notna() & df["admitted_enc"].notna()].copy()
    df_hist["admit_rate"] = (df_hist["admitted_enc"] / df_hist["total_enc"].clip(lower=1.0)).clip(0, 1)

    df_all = pd.concat([df_hist, df[~(df["total_enc"].notna() & df["admitted_enc"].notna())].copy()], ignore_index=True, sort=False)
    df_all = df_all.sort_values(["Site", "Block", "Date"]).reset_index(drop=True)

    df_all = add_lag_features(
        df_all,
        group_cols=["Site", "Block"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        lags=[1, 7, 14, 28],
    )
    df_all = add_rolling_mean_features(
        df_all,
        group_cols=["Site", "Block"],
        sort_cols=["Date"],
        target_cols=["total_enc", "admitted_enc", "admit_rate"],
        windows=[7, 28],
    )

    art = Path(args.artifacts)
    feature_cols = json.loads((art / "feature_cols.json").read_text(encoding="utf-8"))

    def _load_booster(primary: str, fallback: str) -> xgb.Booster:
        b = xgb.Booster()
        p1 = art / primary
        p2 = art / fallback
        if p1.exists():
            b.load_model(str(p1))
            return b
        b.load_model(str(p2))
        return b

    booster_total = _load_booster("booster_total.json", "xgb_total.json")
    booster_rate = _load_booster("booster_admit_rate.json", "xgb_admit_rate.json")

    horizon = df_all[(df_all["Date"] >= pd.to_datetime(args.forecast_start)) & (df_all["Date"] <= pd.to_datetime(args.forecast_end))].copy()
    X = horizon[feature_cols].fillna(0.0)

    dm = xgb.DMatrix(X, feature_names=feature_cols)
    pred_total = np.clip(booster_total.predict(dm), 0, None)
    pred_rate = np.clip(booster_rate.predict(dm), 0, 1)
    pred_adm = np.minimum(pred_total, pred_total * pred_rate)

    out = horizon[["Site", "Date", "Block"]].copy()
    out["ED Enc"] = np.round(pred_total).astype(int)
    out["ED Enc Admitted"] = np.round(pred_adm).astype(int)
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out):,}")


if __name__ == "__main__":
    main()

