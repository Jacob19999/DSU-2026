import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class PlotConfig:
    facets_per_figure: int = 6  # 2 columns x 3 rows, matching existing injury faceted style
    ncols: int = 2
    cmap_counts: str = "YlOrRd"
    cmap_rate: str = "YlGnBu"
    dpi: int = 300


MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv("Dataset/DSU-Dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    if not os.path.exists("reason_categories.csv"):
        raise FileNotFoundError(
            "Missing 'reason_categories.csv'. Run `python categorize_reasons.py` first to generate it."
        )

    reason_to_category = pd.read_csv("reason_categories.csv")
    mapping = dict(zip(reason_to_category["REASON_VISIT_NAME"], reason_to_category["Category"]))
    df["Reason_Category"] = df["REASON_VISIT_NAME"].map(mapping)

    return df


def prepare_cardiovascular(df: pd.DataFrame) -> pd.DataFrame:
    cardio = df[df["Reason_Category"] == "Cardiovascular"].copy()
    cardio["Year"] = cardio["Date"].dt.year
    cardio["Month"] = cardio["Date"].dt.month
    cardio["YearMonth"] = cardio["Date"].dt.to_period("M").astype(str)
    return cardio


def pivot_year_month(
    df: pd.DataFrame,
    value_col: str,
    years: list[int],
) -> pd.DataFrame:
    pivot = (
        df.pivot(index="Year", columns="Month", values=value_col)
        .reindex(index=years, fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )
    pivot.columns = MONTH_NAMES
    return pivot


def create_faceted_year_month_heatmaps(
    *,
    cardio_agg: pd.DataFrame,
    sites: list[str],
    years: list[int],
    value_col: str,
    title: str,
    filename_prefix: str,
    cmap: str,
    annot_fmt: str,
    cbar_label: str,
    cfg: PlotConfig,
) -> None:
    os.makedirs("Visualizations", exist_ok=True)

    for chunk_idx, site_chunk in enumerate(chunked(sites, cfg.facets_per_figure), start=1):
        n = len(site_chunk)
        rows = max(1, (n + cfg.ncols - 1) // cfg.ncols)

        fig, axes = plt.subplots(rows, cfg.ncols, figsize=(18, 4 * rows + 2))
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for ax_idx, site in enumerate(site_chunk):
            ax = axes[ax_idx]
            site_data = cardio_agg[cardio_agg["Site"] == site]

            pivot = pivot_year_month(site_data, value_col=value_col, years=years)

            if value_col == "Admission Rate %":
                total_enc = float(site_data["ED Enc"].sum())
                total_adm = float(site_data["ED Enc Admitted"].sum())
                total_rate = (total_adm / total_enc * 100.0) if total_enc > 0 else float("nan")
                subtitle = f"(Total Enc: {total_enc:,.0f}, Admit%: {total_rate:.1f})"
            else:
                total = float(pivot.to_numpy().sum())
                subtitle = f"(Total: {total:,.0f})"

            sns.heatmap(
                pivot,
                cmap=cmap,
                ax=ax,
                cbar_kws={"label": cbar_label},
                annot=True,
                fmt=annot_fmt,
                annot_kws={"size": 7},
                linewidths=0.5,
                cbar=True,
                square=False,
            )

            ax.set_title(f"{site}\n{subtitle}", fontweight="bold", pad=12, fontsize=11)
            ax.set_xlabel("Month", fontweight="bold")
            ax.set_ylabel("Year", fontweight="bold")

        for ax_idx in range(n, len(axes)):
            axes[ax_idx].set_visible(False)

        plt.tight_layout()
        out = f"Visualizations/{filename_prefix}_{chunk_idx:02d}.png"
        plt.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)


def create_site_correlation_heatmap(cardio: pd.DataFrame, cfg: PlotConfig) -> None:
    """
    Correlation of cardiovascular *monthly volume* across sites.
    Each site's time series is the YearMonth total ED Enc.
    """
    os.makedirs("Visualizations", exist_ok=True)

    by_site_month = (
        cardio.groupby(["Site", "YearMonth"], as_index=False)["ED Enc"].sum().pivot(
            index="YearMonth", columns="Site", values="ED Enc"
        )
    ).fillna(0)

    # If there is only one site, correlation isn't meaningful.
    if by_site_month.shape[1] < 2:
        print("Skipping correlation heatmap (need at least 2 sites).")
        return

    corr = by_site_month.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(
        "Cardiovascular Encounters: Site-to-Site Correlation (Monthly Volume)",
        fontweight="bold",
        pad=14,
    )
    sns.heatmap(
        corr,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    plt.tight_layout()
    out = "Visualizations/cardiovascular_site_monthly_correlation.png"
    plt.savefig(out, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 10

    cfg = PlotConfig()

    print("Loading dataset...")
    df = load_dataset()

    cardio = prepare_cardiovascular(df)
    total_cardio = float(cardio["ED Enc"].sum())
    print(f"Total cardiovascular encounters: {total_cardio:,.0f}")

    if cardio.empty:
        print("No cardiovascular records found (Reason_Category == 'Cardiovascular').")
        return

    years = sorted(cardio["Year"].unique().tolist())

    cardio_agg = (
        cardio.groupby(["Site", "Year", "Month"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
        .assign(**{"Admission Rate %": lambda x: np.where(x["ED Enc"] > 0, x["ED Enc Admitted"] / x["ED Enc"] * 100.0, np.nan)})
    )

    site_totals = cardio_agg.groupby("Site", as_index=False)["ED Enc"].sum().sort_values("ED Enc", ascending=False)
    sites = site_totals["Site"].tolist()
    print(f"Sites with cardiovascular volume: {len(sites)}")

    create_faceted_year_month_heatmaps(
        cardio_agg=cardio_agg,
        sites=sites,
        years=years,
        value_col="ED Enc",
        title="Cardiovascular Encounters by Year and Month (Faceted by Site)",
        filename_prefix="cardiovascular_year_month_by_site_faceted_counts",
        cmap=cfg.cmap_counts,
        annot_fmt=".0f",
        cbar_label="Encounters",
        cfg=cfg,
    )

    create_faceted_year_month_heatmaps(
        cardio_agg=cardio_agg,
        sites=sites,
        years=years,
        value_col="Admission Rate %",
        title="Cardiovascular Admission Rate (%) by Year and Month (Faceted by Site)",
        filename_prefix="cardiovascular_year_month_by_site_faceted_admission_rate",
        cmap=cfg.cmap_rate,
        annot_fmt=".1f",
        cbar_label="Admission Rate (%)",
        cfg=cfg,
    )

    create_site_correlation_heatmap(cardio, cfg)

    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

