## DSU-2026 Forecasting Pipeline (LLM+ML hybrid)

### Install

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### Configure site locations (for weather/alerts features)

Edit `config/sites.json` and fill in `latitude`/`longitude` for each site `A..D`.

If you leave them as `null`, the pipeline still runs but external covariates are blank (you’ll lose the main novelty/accuracy boost).

### Build the feature table

```bash
python scripts/build_feature_table.py --out outputs/feature_table.parquet
```

This writes a single table at (Site, Date, Block) containing:\n- targets for training rows (`total_enc`, `admitted_enc`)\n- calendar features\n- event + weather + NWS alert features\n+
This writes a single table at (Site, Date, Block) containing:
- targets for training rows (`total_enc`, `admitted_enc`)
- calendar features
- event + weather + NWS alert features
