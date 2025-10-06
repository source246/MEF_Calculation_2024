from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns


BBH_COLORS = {
    "primary": "#DC0C23",
    "petrol": "#518E9F",
    "darkblue": "#00374B",
    "umbra": "#A08269",
    "darkred": "#9B0028",
    "grey": "#5F5E5E",
    "good": "#007D55",
    "warning": "#D2B900",
    "bad": "#F5644B",
    "bg_plot": "#E6F0F7",
    "bg_slide": "#F3F0E7",
    "text": "#000000",
    "grid": "#5F5E5E",
}

BBH_SERIES_COLORS = [
    BBH_COLORS["petrol"],
    BBH_COLORS["darkblue"],
    BBH_COLORS["umbra"],
    BBH_COLORS["darkred"],
    BBH_COLORS["grey"],
]


def setup_bbh_style() -> None:
    """Configure matplotlib defaults to match BBH corporate styling."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BBH_COLORS["bg_plot"],
            "axes.facecolor": BBH_COLORS["bg_plot"],
            "axes.edgecolor": BBH_COLORS["text"],
            "axes.labelcolor": BBH_COLORS["text"],
            "xtick.color": BBH_COLORS["text"],
            "ytick.color": BBH_COLORS["text"],
            "text.color": BBH_COLORS["text"],
            "grid.color": BBH_COLORS["grid"],
            "grid.alpha": 0.3,
            "axes.grid": True,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def set_bbh_style() -> None:
    """Legacy convenience wrapper."""
    setup_bbh_style()


def _to_rgb(hex_code: str) -> tuple[float, float, float]:
    return mcolors.to_rgb(hex_code)


def _blend(c1: str, c2: str, t: float = 0.3) -> tuple[float, float, float]:
    r1, g1, b1 = _to_rgb(c1)
    r2, g2, b2 = _to_rgb(c2)
    return (r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t)


def lighten(hex_code: str, t: float = 0.25) -> str:
    r, g, b = _blend(hex_code, "#FFFFFF", t)
    return mcolors.to_hex((r, g, b))


def darken(hex_code: str, t: float = 0.2) -> str:
    r, g, b = _blend(hex_code, "#000000", t)
    return mcolors.to_hex((r, g, b))


GLOBAL = {
    "primary": "#00374B",
    "secondary": "#1B6E8C",
    "tertiary": "#2F4B7C",
    "accent": "#9B0028",
    "warning": "#D1495B",
    "success": "#0F8C4A",
    "neutral": "#444444",
    "muted": "#8D99A6",
    "background": "#F4EFE8",
    "outline": "#8D7B5A",
}

PALETTE = {
    "DE": GLOBAL["primary"],
    "IMPORT": GLOBAL["secondary"],
    "price": GLOBAL["muted"],
    "bg": GLOBAL["background"],
    "bg_edge": GLOBAL["outline"],
}

DEFAULT_TZ = "Europe/Berlin"

TECH_COLORS = {
    "Braunkohle": darken(GLOBAL["neutral"], 0.15),
    "Steinkohle": "#A08268",
    "Erdgas": GLOBAL["secondary"],
    "EE": GLOBAL["success"],
    "Solar": lighten(GLOBAL["success"], 0.35),
    "Wind Onshore": GLOBAL["success"],
    "Wind Offshore": darken(GLOBAL["success"], 0.15),
    "Reservoir Hydro": darken(GLOBAL["secondary"], 0.10),
    "Hydro Pumped Storage": lighten(GLOBAL["secondary"], 0.45),
    "MustrunMix": "#8063A7",
    "Heizöl schwer": GLOBAL["warning"],
    "Nuclear": "#6B3FA0",
    "Waste": "#8A8D35",
    "Biomasse": "#6E8B3D",
    "Biomass": "#6E8B3D",
}

SYNONYMS = {
    "Wind": "Wind Onshore",
    "PV": "Solar",
    "Renewables": "EE",
    "RES": "EE",
    "Pumped Storage": "Hydro Pumped Storage",
    "Hydro PumpedStorage": "Hydro Pumped Storage",
    "Reservoir": "Reservoir Hydro",
    "Hard coal": "Steinkohle",
    "Lignite": "Braunkohle",
    "Oil": "Heizöl schwer",
}


def color_for(label: str) -> str:
    if not isinstance(label, str):
        return GLOBAL["muted"]
    key = SYNONYMS.get(label.strip(), label.strip())
    return TECH_COLORS.get(key, GLOBAL["muted"])


def _legend_bbh(ax, *args, **kwargs):
    """Legend helper that applies BBH framing."""
    legend = ax.legend(*args, **kwargs)
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor(PALETTE["bg"])
        frame.set_edgecolor(PALETTE["bg_edge"])
        frame.set_alpha(0.95)
        frame.set_linewidth(0.8)
    return legend


def _ts_ax(ax, tzlabel: str = "") -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m', tz=None))
    ax.tick_params(axis='x', rotation=45)
    if tzlabel:
        ax.set_xlabel(f"Zeit ({tzlabel})")


def _safe_sum(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    return df.loc[:, list(cols)].sum(axis=1, min_count=1)


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def create_load_coverage_chart(outdir: Path, df_res: pd.DataFrame, de_gen: pd.DataFrame, de_load: pd.Series, args) -> bool:
    """Create stacked load coverage chart summarising monthly averages."""
    try:
        load_dir = outdir / "analysis" / "load_coverage"
        load_dir.mkdir(parents=True, exist_ok=True)

        if de_gen.empty or len(de_gen.columns) == 0:
            print("[WARNING] Keine Generation-Daten verfügbar für Load Coverage Chart")
            return False

        tech_colors = {
            'Nuclear': BBH_COLORS['darkred'],
            'Hydro Pumped Storage': BBH_COLORS['petrol'],
            'Hydro Run-of-river and poundage': BBH_COLORS['good'],
            'Biomass': BBH_COLORS['warning'],
            'Waste': BBH_COLORS['umbra'],
            'Gas': '#FB9A99',
            'Oil': '#A6CEE3',
            'Coal': BBH_COLORS['grey'],
            'Solar': '#FDBF6F',
            'Wind Onshore': '#CAB2D6',
            'Wind Offshore': '#FFFF99',
            'Other': '#B15928',
        }

        df_load_monthly = pd.DataFrame()
        df_load_monthly['timestamp'] = pd.date_range('2024-01-01', '2024-12-31', freq='M')

        common_idx = de_load.index.intersection(de_gen.index)
        if len(common_idx) == 0:
            print("[WARNING] Keine gemeinsamen Zeitstempel zwischen Load und Generation")
            return False

        load_aligned = de_load.reindex(common_idx)
        gen_aligned = de_gen.reindex(common_idx)
        df_load_monthly['load_total'] = [load_aligned[load_aligned.index.month == m].mean() for m in range(1, 13)]

        total_gen_check = 0
        for tech in tech_colors.keys():
            if tech in gen_aligned.columns:
                monthly_values = [gen_aligned[tech][gen_aligned.index.month == m].mean() for m in range(1, 13)]
                df_load_monthly[tech] = monthly_values
                total_gen_check += sum(monthly_values)
            else:
                df_load_monthly[tech] = [0.0] * 12

        if total_gen_check < 1000:
            print(f"[WARNING] Generation data appears to be dummy/empty (total: {total_gen_check:.0f} MW)")
            df_load_monthly['Domestic Generation'] = [v * 0.8 for v in df_load_monthly['load_total']]
            df_load_monthly['Net Import'] = [v * 0.2 for v in df_load_monthly['load_total']]
            for tech in tech_colors.keys():
                df_load_monthly[tech] = [0.0] * 12
        else:
            if 'net_import_total_MW' in df_res.columns:
                import_aligned = df_res.set_index('timestamp')['net_import_total_MW'].reindex(common_idx)
                df_load_monthly['Net Import'] = [import_aligned[import_aligned.index.month == m].mean() for m in range(1, 13)]
            else:
                df_load_monthly['Net Import'] = [0.0] * 12

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(df_load_monthly['timestamp'], df_load_monthly['load_total'], color='black', linewidth=3, label='Load Target', zorder=10)
        bottom = np.zeros(len(df_load_monthly))

        if 'Domestic Generation' in df_load_monthly.columns and df_load_monthly['Domestic Generation'][0] > 0:
            values = df_load_monthly['Domestic Generation']
            ax.fill_between(df_load_monthly['timestamp'], bottom, bottom + values, color=BBH_COLORS['petrol'], alpha=0.8, label='Domestic Generation')
            bottom += values
        else:
            for tech in tech_colors.keys():
                if tech in df_load_monthly.columns:
                    values = df_load_monthly[tech]
                    if sum(values) > 0:
                        ax.fill_between(df_load_monthly['timestamp'], bottom, bottom + values, color=tech_colors[tech], alpha=0.8, label=tech)
                        bottom += values

        if 'Net Import' in df_load_monthly.columns:
            import_values = df_load_monthly['Net Import']
            if sum(abs(v) for v in import_values) > 0:
                ax.fill_between(df_load_monthly['timestamp'], bottom, np.array(bottom) + np.array(import_values), color=BBH_COLORS['bad'], alpha=0.6, label='Net Import', hatch='///')

        ax.set_xlabel('Monat 2024')
        ax.set_ylabel('Leistung [MW]')
        ax.set_title('Load Coverage by Technology (Global Income Distribution Style)\nMonthly Average 2024')
        ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        fig.tight_layout()
        fig.savefig(load_dir / "load_coverage_monthly.png", dpi=300, bbox_inches='tight')
        fig.savefig(load_dir / "load_coverage_monthly.pdf", bbox_inches='tight')
        plt.close(fig)

        print("[OK] Load Coverage Chart erstellt")
        return True

    except Exception as exc:  # pragma: no cover - plotting robustness
        print(f"[ERROR] create_load_coverage_chart failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


# (Additional plotting helpers appended below)

def generate_enhanced_plots(
    outdir: Path,
    df_res: pd.DataFrame,
    df_dbg: pd.DataFrame,
    df_val: Optional[pd.DataFrame],
    fuel_prices: pd.DataFrame,
    de_gen: pd.DataFrame,
    flows: pd.DataFrame,
    args,
) -> bool:
    """Create a suite of enhanced diagnostic plots."""
    try:
        setup_bbh_style()
        plots_dir = outdir / "analysis" / "enhanced_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        if 'timestamp' in df_res.columns:
            df_res_ts = df_res.copy()
        else:
            df_res_ts = df_res.reset_index()
        if 'timestamp' not in df_res_ts.columns and len(df_res_ts.columns) > 0:
            first_col = df_res_ts.columns[0]
            df_res_ts = df_res_ts.rename(columns={first_col: 'timestamp'})
        if 'timestamp' in df_res_ts.columns:
            df_res_ts['timestamp'] = pd.to_datetime(df_res_ts['timestamp'], errors='coerce')

        success_count = 0
        total_plots = 5

        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            mask = (
                df_res['price_DE'].notna()
                & df_res['marginal_srmc_eur_per_mwh'].notna()
                & (df_res['price_DE'] >= 0)
                & (df_res['price_DE'] <= 300)
            )
            df_plot = df_res[mask].copy()
            for side in df_plot['marginal_side'].dropna().unique():
                subset = df_plot[df_plot['marginal_side'] == side]
                color = BBH_COLORS['petrol'] if side == 'DE' else BBH_COLORS['darkred']
                ax.scatter(
                    subset['marginal_srmc_eur_per_mwh'],
                    subset['price_DE'],
                    alpha=0.6,
                    s=8,
                    color=color,
                    label=f'Marginal: {side}',
                )
            max_val = min(300, float(df_plot[['price_DE', 'marginal_srmc_eur_per_mwh']].max().max()))
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Correlation')
            ax.set_xlabel('Marginal SRMC [EUR/MWh]')
            ax.set_ylabel('Price DE [EUR/MWh]')
            ax.set_title('Price vs. Marginal SRMC by Setting Technology')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
            fig.tight_layout()
            fig.savefig(plots_dir / "price_srmc_scatter_enhanced.png", dpi=300, bbox_inches='tight')
            fig.savefig(plots_dir / "price_srmc_scatter_enhanced.pdf", bbox_inches='tight')
            plt.close(fig)
            success_count += 1
        except Exception as exc:  # pragma: no cover - plotting robustness
            print(f"[WARNING] Price-SRMC scatter failed: {exc}")

        try:
            fig, ax = plt.subplots(figsize=(14, 4.8))
            df_monthly = df_res_ts.copy()
            df_monthly['month'] = df_monthly['timestamp'].dt.month
            fuel_counts = df_monthly.groupby(['month', 'marginal_fuel']).size().unstack(fill_value=0)
            fuel_shares = fuel_counts.div(fuel_counts.sum(axis=1), axis=0) * 100
            fuel_colors: Dict[str, str] = {
                'Gas': BBH_COLORS['petrol'],
                'Coal': BBH_COLORS['grey'],
                'Nuclear': BBH_COLORS['darkred'],
                'Hydro Pumped Storage': BBH_COLORS['good'],
                'Solar': BBH_COLORS['warning'],
                'Wind Onshore': '#CAB2D6',
                'Oil': BBH_COLORS['bad'],
                'mix': BBH_COLORS['umbra'],
            }
            bottom = np.zeros(len(fuel_shares))
            for fuel in fuel_shares.columns:
                color = fuel_colors.get(fuel, BBH_COLORS['darkblue'])
                ax.bar(fuel_shares.index, fuel_shares[fuel], bottom=bottom, label=fuel, color=color, alpha=0.8)
                bottom += fuel_shares[fuel]
            ax.set_xlabel('Monat 2024')
            ax.set_ylabel('Anteil Marginal Fuel [%]')
            ax.set_title('Monthly Marginal Fuel Mix Distribution')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
            fig.tight_layout()
            fig.savefig(plots_dir / "marginal_fuel_stacked.png", dpi=300, bbox_inches='tight')
            fig.savefig(plots_dir / "marginal_fuel_stacked.pdf", bbox_inches='tight')
            plt.close(fig)
            success_count += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Marginal fuel mix plot failed: {exc}")

        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_plot = df_res_ts[['timestamp', 'residual_domestic_fossil_MW', 'price_DE']].dropna().copy()
            ax.plot(df_plot['timestamp'], df_plot['residual_domestic_fossil_MW'], color=BBH_COLORS['petrol'], label='Residual Load')
            ax.set_ylabel('Residual Load [MW]')
            ax2 = ax.twinx()
            ax2.plot(df_plot['timestamp'], df_plot['price_DE'], color=BBH_COLORS['darkred'], label='Price DE')
            ax2.set_ylabel('Price DE [EUR/MWh]')
            ax.set_title('Residual Load vs. Price (Sample)')
            ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
            _legend_bbh(ax)
            fig.tight_layout()
            fig.savefig(plots_dir / "residual_load_price.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            success_count += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Residual load plot failed: {exc}")

        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            hourly = flows.copy()
            hourly.index = pd.to_datetime(hourly.index)
            hourly['month'] = hourly.index.month
            hourly['hour'] = hourly.index.hour
            if 'net_import_total' in hourly.columns:
                pivot = hourly.pivot_table(values='net_import_total', index='hour', columns='month', aggfunc='mean')
            else:
                pivot = pd.DataFrame()
            sns.heatmap(pivot, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Average Net Imports by Hour/Month')
            ax.set_xlabel('Monat')
            ax.set_ylabel('Stunde')
            fig.tight_layout()
            fig.savefig(plots_dir / "net_import_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            success_count += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] Net import heatmap failed: {exc}")

        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            mef_data = df_res['mef_g_per_kwh'].dropna()
            axes[0].hist(mef_data, bins=50, alpha=0.7, color=BBH_COLORS['petrol'], edgecolor='black')
            axes[0].set_title('MEF Distribution')
            axes[0].set_xlabel('MEF [g/kWh]')
            axes[0].set_ylabel('Stunden')
            axes[0].grid(True, alpha=0.3, color=BBH_COLORS['grid'])
            if 'marginal_fuel' in df_res.columns:
                fuel_mef = df_res.groupby('marginal_fuel')['mef_g_per_kwh'].median().dropna().sort_values(ascending=False)
                colors = [BBH_COLORS['petrol'] if v >= 0 else BBH_COLORS['bad'] for v in fuel_mef]
                axes[1].bar(range(len(fuel_mef)), fuel_mef.values, color=colors, alpha=0.8)
                axes[1].set_xticks(range(len(fuel_mef)))
                axes[1].set_xticklabels(fuel_mef.index, rotation=45, ha='right')
                axes[1].set_ylabel('Median MEF [g CO₂/kWh]')
                axes[1].set_title('Median MEF by Marginal Fuel Type')
                axes[1].grid(True, alpha=0.3, color=BBH_COLORS['grid'])
            fig.tight_layout()
            fig.savefig(plots_dir / "mef_analysis.png", dpi=300, bbox_inches='tight')
            fig.savefig(plots_dir / "mef_analysis.pdf", bbox_inches='tight')
            plt.close(fig)
            success_count += 1
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] MEF analysis failed: {exc}")

        print(f"[INFO] Enhanced Plots: {success_count}/{total_plots} erfolgreich erstellt")
        return success_count >= 3

    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] generate_enhanced_plots failed: {exc}")
        return False

def _plot_residuallast_decomposition_per_month(
    outdir: Path,
    month: int,
    de_load: pd.Series,
    de_gen: pd.DataFrame,
    de_min_total: pd.Series,
    flows: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    mask = de_load.index.month == month
    if not mask.any():
        return
    idx = de_load.index[mask]

    L = de_load.reindex(idx).astype(float)
    FEE = _safe_sum(de_gen.reindex(idx), ["Solar", "Wind Onshore", "Wind Offshore"])
    WB = _safe_sum(de_gen.reindex(idx), ["Waste", "Biomasse", "Biomass"])
    MU = de_min_total.reindex(idx).fillna(0.0).astype(float)
    NI = (
        flows.reindex(idx)["net_import_total"].fillna(0.0).astype(float)
        if "net_import_total" in flows.columns
        else pd.Series(0.0, index=idx)
    )
    PSP = pd.to_numeric(
        de_gen.reindex(idx).get("Hydro Pumped Storage", pd.Series(0.0, index=idx)),
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)
    P = (
        prices.reindex(idx)["price_DE_LU"].astype(float)
        if "price_DE_LU" in prices.columns
        else None
    )

    RL1 = (L - FEE).rename("RL1")
    RL2 = (RL1 - WB).rename("RL2")
    RL3 = (RL2 - MU).rename("RL3")
    RL3_PSP = (RL3 - PSP).rename("RL3 minus PSP")
    RL4 = (RL3 - NI).rename("RL4 (NI last)")
    RL4_psp_first = (RL3_PSP - NI).rename("RL4 (PSP→NI)")

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(L.index, L.values, label="Last", lw=1.8, color=GLOBAL["primary"])
    ax.plot(RL1.index, RL1.values, label="RL1 = Last − FEE", lw=1.6, color=GLOBAL["secondary"])
    ax.plot(RL2.index, RL2.values, label="RL2 − (Waste+Biomasse)", lw=1.6, color=GLOBAL["tertiary"])
    ax.plot(RL3.index, RL3.values, label="RL3 − foss. Mindestlast", lw=1.7, color=GLOBAL["accent"])
    ax.plot(RL3_PSP.index, RL3_PSP.values, label="RL3 − PSP", lw=1.6, color=darken(GLOBAL["secondary"], 0.15))
    ax.plot(RL4.index, RL4.values, label="RL4 − nach Handel (NI)", lw=1.6, color=PALETTE["bg_edge"], alpha=0.9)

    ax.plot(FEE.index, FEE.values, label="FEE", lw=1.0, alpha=0.35, color=color_for("EE"))
    ax.plot(WB.index, WB.values, label="Waste+Biomasse", lw=1.0, alpha=0.35, color=lighten(TECH_COLORS["Waste"], 0.35))
    ax.plot(MU.index, MU.values, label="Fossile Mindestlast", lw=1.0, alpha=0.35, color=lighten(GLOBAL["accent"], 0.55))
    ax.plot(PSP.index, PSP.values, label="PSP Gen", lw=1.0, alpha=0.35, color=lighten(GLOBAL["secondary"], 0.55))

    _ts_ax(ax, tzlabel=DEFAULT_TZ)
    ax.set_ylabel("Leistung [MW]")

    if P is not None:
        ax2 = ax.twinx()
        ax2.plot(P.index, P.values, ls="--", lw=1.2, color=PALETTE["price"], label="Preis (DE/LU)")
        ax2.set_ylabel("Preis [EUR/MWh]")
        handles, labels = [], []
        for axis in (ax, ax2):
            h_i, l_i = axis.get_legend_handles_labels()
            handles += h_i
            labels += l_i
        _legend_bbh(ax, handles=handles, labels=labels, loc="upper left")
    else:
        _legend_bbh(ax, loc="upper left")

    ax.set_title(f"Residuallast-Zerlegung (RL1…RL4) – Monat {month:02d} (mit PSP-Stufe)")
    fig.tight_layout()
    fig.savefig(pdir / f"rl_decomposition_month_{month:02d}.png", dpi=160)
    plt.close(fig)

    out_csv = outdir / "analysis" / f"residuallast_decomp_month_{month:02d}.csv"
    pd.DataFrame(
        {
            "L": L,
            "FEE": FEE,
            "WB": WB,
            "MU": MU,
            "PSP_gen": PSP,
            "NI": NI,
            "RL1": RL1,
            "RL2": RL2,
            "RL3": RL3,
            "RL3_minus_PSP": RL3_PSP,
            "RL4_original(NI_last)": RL4,
            "RL4_psp_first": RL4_psp_first,
        }
    ).to_csv(out_csv, index=True)


def plot_rl_ladder_week(
    outdir: Path,
    week_start: pd.Timestamp,
    de_load: pd.Series,
    de_gen: pd.DataFrame,
    de_min_total: pd.Series,
    flows: pd.DataFrame,
) -> None:
    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    idx = pd.date_range(week_start, week_start + pd.Timedelta(days=7), freq="h", inclusive="left")
    L = de_load.reindex(idx).astype(float)
    FEE = _safe_sum(de_gen.reindex(idx), ["Solar", "Wind Onshore", "Wind Offshore"])
    WB = _safe_sum(de_gen.reindex(idx), ["Waste", "Biomasse", "Biomass"])
    MU = de_min_total.reindex(idx).fillna(0.0).astype(float)
    NI = (
        flows.reindex(idx)["net_import_total"].fillna(0.0).astype(float)
        if "net_import_total" in flows.columns
        else pd.Series(0.0, index=idx)
    )

    RL1 = (L - FEE).clip(lower=0)
    RL2 = (RL1 - WB).clip(lower=0)
    RL3 = (RL2 - MU).clip(lower=0)
    RL4 = RL3 - NI

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.fill_between(idx, 0, L, facecolor=lighten(GLOBAL["primary"], 0.75), label="Last")
    ax.fill_between(idx, L - FEE, L, facecolor=lighten(color_for("EE"), 0.35), label="FEE")
    ax.fill_between(idx, L - (FEE + WB), L - FEE, facecolor=lighten(TECH_COLORS["Waste"], 0.35), label="Waste+Biomasse")
    ax.fill_between(idx, L - (FEE + WB + MU), L - (FEE + WB), facecolor=lighten(GLOBAL["accent"], 0.55), label="foss. Mindestlast")
    ax.plot(idx, RL3, lw=1.5, color=GLOBAL["accent"], label="RL3")
    ax.plot(idx, RL4, lw=1.2, color=PALETTE["bg_edge"], label="RL4")
    ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
    _legend_bbh(ax, loc="upper left")
    ax.set_ylabel("Leistung [MW]")
    ax.set_title(f"Residuallast-Leiter – Woche ab {week_start:%Y-%m-%d}")
    fig.tight_layout()
    fig.savefig(pdir / f"rl_ladder_week_{week_start:%Y%m%d}.png", dpi=160)
    plt.close(fig)


def plot_rl_duration_curves(outdir: Path, df: pd.DataFrame) -> None:
    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    cols = [
        c
        for c in [
            "RL_after_FEE_MW",
            "RL_after_NDext_MW",
            "residual_domestic_fossil_MW",
            "residual_after_trade_MW",
        ]
        if c in df.columns
    ]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [
        GLOBAL["secondary"],
        GLOBAL["tertiary"],
        GLOBAL["accent"],
        PALETTE["bg_edge"],
    ]
    for c, color in zip(cols, colors):
        series = df[c].dropna().sort_values(ascending=False).reset_index(drop=True)
        ax.plot(series.index, series.values, label=c, color=color)
    ax.set_xlabel('Stunden (sortiert)')
    ax.set_ylabel('MW')
    ax.set_title('Residuallast-Dauerlinien (RL1…RL4)')
    ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
    _legend_bbh(ax, loc='upper right')
    fig.tight_layout()
    fig.savefig(pdir / "rl_duration_curves.png", dpi=160)
    plt.close(fig)


def plot_monthly_contributions(outdir: Path) -> None:
    csv_path = outdir / "analysis" / "residuallast_decomp_month_01.csv"
    if not csv_path.exists():
        return
    frames = []
    for m in range(1, 13):
        path = outdir / "analysis" / f"residuallast_decomp_month_{m:02d}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df['month'] = m
        frames.append(df)
    if not frames:
        return
    data = pd.concat(frames, axis=0)
    rows = []
    for month, d in data.groupby('month'):
        rows.append(
            {
                'month': month,
                'FEE': d['FEE'].mean(),
                'Waste+Biomasse': d['WB'].mean(),
                'foss. Mindestlast': d['MU'].mean(),
                'Nettoimporte': d['NI'].mean(),
            }
        )
    summary = pd.DataFrame(rows).set_index('month').sort_index()

    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(summary))
    for label, color in [
        ('FEE', color_for('EE')),
        ('Waste+Biomasse', lighten(TECH_COLORS['Waste'], 0.35)),
        ('foss. Mindestlast', lighten(GLOBAL['accent'], 0.55)),
        ('Nettoimporte', PALETTE['bg_edge']),
    ]:
        ax.bar(summary.index, summary[label], bottom=bottom, label=label, edgecolor='none', color=color)
        bottom += summary[label].values
    ax.set_xlabel('Monat')
    ax.set_ylabel('mittlere Reduktion [MW]')
    ax.set_title('Durchschnittliche Beiträge zur RL-Reduktion je Monat')
    ax.grid(axis='y', alpha=0.3)
    _legend_bbh(ax, ncol=2, loc='upper left')
    fig.tight_layout()
    fig.savefig(pdir / "rl_contributions_by_month.png", dpi=160)
    plt.close(fig)


def plot_rl_vs_price(outdir: Path, df_res: pd.DataFrame) -> None:
    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    cols = [
        c
        for c in [
            "RL_after_FEE_MW",
            "RL_after_NDext_MW",
            "residual_domestic_fossil_MW",
            "residual_after_trade_MW",
        ]
        if c in df_res.columns
    ]
    if not cols:
        return

    price = df_res['price_DE']
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = [GLOBAL["secondary"], GLOBAL["tertiary"], GLOBAL["accent"], PALETTE["bg_edge"]]
    for c, color in zip(cols, palette):
        ax.scatter(df_res[c], price, s=8, alpha=0.25, label=c, color=color)
    ax.set_xlabel('MW')
    ax.set_ylabel('Preis [EUR/MWh]')
    ax.set_title('Preis vs. RL-Stufe')
    ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
    _legend_bbh(ax, loc='best')
    fig.tight_layout()
    fig.savefig(pdir / "scatter_price_vs_rl_stages.png", dpi=160)
    plt.close(fig)


def _plot_mef_hourly_mean_per_month(outdir: Path, df_res: pd.DataFrame) -> None:
    pdir = outdir / "analysis" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    s = df_res['mef_g_per_kwh'].dropna()
    if s.empty:
        return
    df = s.to_frame('mef').copy()
    df['month'] = df.index.month
    df['hour'] = df.index.hour
    piv = df.pivot_table(index='month', columns='hour', values='mef', aggfunc='mean')

    for month in piv.index:
        vals = piv.loc[month].astype(float).values
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(range(24), vals, marker='o', linewidth=1.6)
        ax.set_xlabel('Stunde')
        ax.set_ylabel('Ø MEF [g/kWh]')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3, color=BBH_COLORS['grid'])
        ax.set_title(f'Ø MEF je Stunde – Monat {int(month):02d}')
        fig.tight_layout()
        fig.savefig(pdir / f"mef_hourly_mean_month_{int(month):02d}.png", dpi=160)
        plt.close(fig)
def make_validation_plots(outdir, df_res, df_dbg, df_val, nei_prices, de_gen=None, 
                         de_min_total=None, de_load=None, flows=None, args=None):
    """Erstelle umfassende Validierungsdiagramme mit BBH-Farbpalette"""
    setup_bbh_style()
    val_dir = outdir / "analysis" / "validation_plots"
    val_dir.mkdir(exist_ok=True, parents=True)
    
    # Merge fÃƒÂ¼r vollstÃƒÂ¤ndige Daten (both dataframes have timestamp as index)
    df_full = df_res.merge(df_val, left_index=True, right_index=True, how="left", suffixes=("", "_val"))
    df_full = df_full.reset_index()  # Convert timestamp index to column
    df_full["month"] = pd.to_datetime(df_full["timestamp"]).dt.month
    df_full["hour"] = pd.to_datetime(df_full["timestamp"]).dt.hour
    df_full["abs_error"] = np.abs(df_full["price_DE"] - df_full["marginal_srmc_eur_per_mwh"])
    
    print(f"[VALIDATION] Erstelle Validierungsdiagramme in {val_dir}...")
    
    # 1) Monatliche untertÃƒÂ¤gige Struktur: Grenzkosten vs Preis
    _plot_monthly_intraday_price_srmc(df_full, val_dir)
    
    # 2) Preis vs. SRMC Zeitreihe 
    _plot_price_srmc_timeseries(df_full, val_dir)
    
    # 3) Scatter Preis vs. SRMC nach Seite mit 45Ã‚Â°-Linie
    _plot_price_srmc_scatter(df_full, val_dir)
    
    # 4) Histogramm und ECDF der Fehler
    _plot_error_distribution(df_full, val_dir)
    
    # 5) Merit Order Lastdeckung nach Monat
    _plot_monthly_residual_load_decomposition(df_full, val_dir)
    
    # 6) SensitivitÃƒÂ¤tsanalyse (wenn Parameter verfÃƒÂ¼gbar)
    if args:
        _plot_sensitivity_analysis(df_full, val_dir, args)
    
    # 7) Export Rohdaten
    _export_validation_raw_data(df_full, val_dir)
    
    print(f"[OK] Validierungsdiagramme in {val_dir} erstellt.")


def _plot_monthly_intraday_price_srmc(df_full, val_dir):
    """Monatliche untertÃƒÂ¤gige Struktur: Grenzkosten vs Preis"""
    months = sorted(df_full["month"].unique())
    
    for month in months:
        if pd.isna(month):
            continue
            
        df_month = df_full[df_full["month"] == month].copy()
        if len(df_month) == 0:
            continue
            
        # StÃƒÂ¼ndliche Mittelwerte
        hourly_stats = df_month.groupby("hour").agg({
            "price_DE": ["mean", "std", "median"],
            "marginal_srmc_eur_per_mwh": ["mean", "std", "median"],
            "mef_g_per_kwh": ["mean", "std", "median"]
        }).round(2)
        
        # Reindex to ensure all 24 hours are present
        all_hours = pd.Index(range(24), name='hour')
        hourly_stats = hourly_stats.reindex(all_hours)
        
        # Export Rohdaten
        hourly_stats.columns = [f"{col[1]}_{col[0]}" for col in hourly_stats.columns]
        hourly_stats.to_csv(val_dir / f"monthly_intraday_data_month_{month:02d}.csv")
        
        # Check if we have sufficient data for plotting
        valid_hours = hourly_stats.dropna(subset=['mean_price_DE']).index
        if len(valid_hours) < 2:
            print(f"[VALIDATION] Skipping month {month} - insufficient data ({len(valid_hours)} hours)")
            continue
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        hours = range(24)
        
        # Oberes Panel: Preis vs SRMC (handle NaN values)
        price_mask = ~pd.isna(hourly_stats["mean_price_DE"])
        srmc_mask = ~pd.isna(hourly_stats["mean_marginal_srmc_eur_per_mwh"])
        
        if price_mask.any():
            ax1.plot([h for h, m in zip(hours, price_mask) if m], 
                    hourly_stats["mean_price_DE"].dropna(), 'o-', color=BBH_COLORS['petrol'], 
                    linewidth=2, markersize=4, label='Preis (Mittel)')
            
            # Fill between only for valid data points
            valid_price_hours = [h for h, m in zip(hours, price_mask) if m]
            valid_price_mean = hourly_stats["mean_price_DE"].dropna()
            valid_price_std = hourly_stats["std_price_DE"].dropna()
            ax1.fill_between(valid_price_hours, 
                            valid_price_mean - valid_price_std,
                            valid_price_mean + valid_price_std,
                            alpha=0.2, color=BBH_COLORS['petrol'])
        
        if srmc_mask.any():
            ax1.plot([h for h, m in zip(hours, srmc_mask) if m], 
                    hourly_stats["mean_marginal_srmc_eur_per_mwh"].dropna(), 's-', 
                    color=BBH_COLORS['darkblue'], linewidth=2, markersize=4, label='SRMC (Mittel)')
            
            # Fill between only for valid data points  
            valid_srmc_hours = [h for h, m in zip(hours, srmc_mask) if m]
            valid_srmc_mean = hourly_stats["mean_marginal_srmc_eur_per_mwh"].dropna()
            valid_srmc_std = hourly_stats["std_marginal_srmc_eur_per_mwh"].dropna()
            ax1.fill_between(valid_srmc_hours,
                            valid_srmc_mean - valid_srmc_std,
                            valid_srmc_mean + valid_srmc_std,
                            alpha=0.2, color=BBH_COLORS['darkblue'])
                        
        ax1.set_ylabel('EUR/MWh')
        ax1.set_title(f'Monat {month:02d}: UntertÃƒÂ¤gige Struktur Preis vs. SRMC', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Unteres Panel: MEF (handle NaN values)
        mef_mask = ~pd.isna(hourly_stats["mean_mef_g_per_kwh"])
        
        if mef_mask.any():
            ax2.plot([h for h, m in zip(hours, mef_mask) if m], 
                    hourly_stats["mean_mef_g_per_kwh"].dropna(), '^-', color=BBH_COLORS['umbra'], 
                    linewidth=2, markersize=4, label='MEF (Mittel)')
            
            # Fill between only for valid data points
            valid_mef_hours = [h for h, m in zip(hours, mef_mask) if m]
            valid_mef_mean = hourly_stats["mean_mef_g_per_kwh"].dropna()
            valid_mef_std = hourly_stats["std_mef_g_per_kwh"].dropna()
            ax2.fill_between(valid_mef_hours,
                            valid_mef_mean - valid_mef_std,
                            valid_mef_mean + valid_mef_std,
                            alpha=0.2, color=BBH_COLORS['umbra'])
        
        ax2.set_xlabel('Stunde')
        ax2.set_ylabel('g COÃ¢â€šâ€š/kWh')
        ax2.set_title('MEF Verlauf')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(val_dir / f"monthly_intraday_month_{month:02d}.png", dpi=160, bbox_inches='tight')
        plt.close()


def _plot_price_srmc_timeseries(df_full, val_dir):
    """Preis vs. SRMC Zeitreihe"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Sample fÃƒÂ¼r bessere Performance bei groÃƒÅ¸en DatensÃƒÂ¤tzen
    if len(df_full) > 2000:
        df_plot = df_full.sample(2000).sort_values('timestamp')
    else:
        df_plot = df_full.copy()
    
    timestamps = pd.to_datetime(df_plot["timestamp"])
    
    ax.plot(timestamps, df_plot["price_DE"], color=BBH_COLORS['petrol'], 
            linewidth=1.5, alpha=0.8, label='BÃƒÂ¶rsenpreis DE')
    ax.plot(timestamps, df_plot["marginal_srmc_eur_per_mwh"], color=BBH_COLORS['darkblue'], 
            linewidth=1.5, alpha=0.8, label='GewÃƒÂ¤hlte SRMC')
    
    # Hervorhebung groÃƒÅ¸er Abweichungen
    large_errors = df_plot["abs_error"] > 50
    if large_errors.any():
        ax.scatter(timestamps[large_errors], df_plot.loc[large_errors, "price_DE"], 
                  color=BBH_COLORS['primary'], s=30, alpha=0.7, zorder=5, label='GroÃƒÅ¸e Abweichungen (>50EUR)')
    
    ax.set_xlabel('Zeit')
    ax.set_ylabel('EUR/MWh')
    ax.set_title('Zeitreihe: BÃƒÂ¶rsenpreis vs. gewÃƒÂ¤hlte SRMC', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Formatierung x-Achse
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(val_dir / "price_srmc_timeseries.png", dpi=160, bbox_inches='tight')
    plt.close()


def _plot_price_srmc_scatter(df_full, val_dir):
    """Scatter Preis vs. SRMC nach Seite mit 45Ã‚Â°-Linie"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Daten nach Seite trennen
    df_de = df_full[df_full["marginal_side"] == "DE"]
    df_import = df_full[df_full["marginal_side"] == "IMPORT"]
    
    # Scatter plots
    if len(df_de) > 0:
        ax.scatter(df_de["marginal_srmc_eur_per_mwh"], df_de["price_DE"], 
                  color=BBH_COLORS['petrol'], s=20, alpha=0.6, label=f'DE ({len(df_de)} Stunden)')
    
    if len(df_import) > 0:
        ax.scatter(df_import["marginal_srmc_eur_per_mwh"], df_import["price_DE"], 
                  color=BBH_COLORS['darkblue'], s=20, alpha=0.6, label=f'Import ({len(df_import)} Stunden)')
    
    # 45Ã‚Â°-Linie
    max_val = max(df_full["price_DE"].max(), df_full["marginal_srmc_eur_per_mwh"].max())
    min_val = min(df_full["price_DE"].min(), df_full["marginal_srmc_eur_per_mwh"].min())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=BBH_COLORS['warning'], 
            linewidth=2, alpha=0.8, label='45Ã‚Â° Linie (perfekte Korrelation)')
    
    # Korrelationsstatistiken
    corr_all = df_full[["price_DE", "marginal_srmc_eur_per_mwh"]].corr().iloc[0,1]
    if len(df_de) > 1:
        corr_de = df_de[["price_DE", "marginal_srmc_eur_per_mwh"]].corr().iloc[0,1]
    else:
        corr_de = np.nan
    if len(df_import) > 1:
        corr_import = df_import[["price_DE", "marginal_srmc_eur_per_mwh"]].corr().iloc[0,1]
    else:
        corr_import = np.nan
    
    # Textbox mit Statistiken
    stats_text = f'Korrelation Gesamt: {corr_all:.3f}\n'
    if not np.isnan(corr_de):
        stats_text += f'Korrelation DE: {corr_de:.3f}\n'
    if not np.isnan(corr_import):
        stats_text += f'Korrelation Import: {corr_import:.3f}'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=BBH_COLORS['bg_plot'], alpha=0.8))
    
    ax.set_xlabel('SRMC (EUR/MWh)')
    ax.set_ylabel('BÃƒÂ¶rsenpreis DE (EUR/MWh)')
    ax.set_title('Scatter: BÃƒÂ¶rsenpreis vs. SRMC nach marginaler Seite', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(val_dir / "price_srmc_scatter.png", dpi=160, bbox_inches='tight')
    plt.close()


def _plot_error_distribution(df_full, val_dir):
    """Histogramm und ECDF der Fehler (Preis - SRMC)"""
    errors = df_full["price_DE"] - df_full["marginal_srmc_eur_per_mwh"]
    errors_clean = errors.dropna()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramm
    ax1.hist(errors_clean, bins=50, color=BBH_COLORS['petrol'], alpha=0.7, edgecolor='white')
    ax1.axvline(0, color=BBH_COLORS['warning'], linestyle='--', linewidth=2, label='Perfekte ÃƒÅ“bereinstimmung')
    ax1.axvline(errors_clean.median(), color=BBH_COLORS['primary'], linestyle='-', linewidth=2, 
               label=f'Median: {errors_clean.median():.1f}EUR')
    
    ax1.set_xlabel('Fehler: Preis - SRMC (EUR/MWh)')
    ax1.set_ylabel('HÃƒÂ¤ufigkeit')
    ax1.set_title('Verteilung der Preisfehler', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Statistiken als Text
    stats_text = f'Mittelwert: {errors_clean.mean():.1f}EUR\n'
    stats_text += f'Std.abw.: {errors_clean.std():.1f}EUR\n'
    stats_text += f'MAE: {np.abs(errors_clean).mean():.1f}EUR\n'
    stats_text += f'95%-Quantil: {np.abs(errors_clean).quantile(0.95):.1f}EUR'
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=BBH_COLORS['bg_plot'], alpha=0.8))
    
    # ECDF
    sorted_errors = np.sort(np.abs(errors_clean))
    ecdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    
    ax2.plot(sorted_errors, ecdf, color=BBH_COLORS['darkblue'], linewidth=2)
    ax2.axvline(10, color=BBH_COLORS['good'], linestyle='--', alpha=0.7, label='10EUR')
    ax2.axvline(25, color=BBH_COLORS['warning'], linestyle='--', alpha=0.7, label='25EUR')  
    ax2.axvline(50, color=BBH_COLORS['bad'], linestyle='--', alpha=0.7, label='50EUR')
    
    ax2.set_xlabel('Absoluter Fehler (EUR/MWh)')
    ax2.set_ylabel('Kumulative Wahrscheinlichkeit')
    ax2.set_title('ECDF: Absoluter Preisfehler', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(val_dir / "error_distribution.png", dpi=160, bbox_inches='tight')
    plt.close()


def _plot_monthly_residual_load_decomposition(df_full, val_dir):
    """Merit Order Lastdeckung nach Monat - zeigt wie verschiedene Erzeugungsarten die Last decken"""
    months = sorted(df_full["month"].unique())
    
    for month in months:
        if pd.isna(month):
            continue
            
        df_month = df_full[df_full["month"] == month].copy()
        if len(df_month) == 0:
            continue
        
        # StÃƒÂ¼ndliche Mittelwerte
        hourly_means = df_month.groupby("hour").agg({
            "FEE_MW": "mean",
            "RL_after_FEE_MW": "mean", 
            "RL_after_FOSSIL_MU_MW": "mean",
            "RL_after_PSP_MW": "mean",
            "net_import_total_MW": "mean",
            "price_DE": "mean"
        }).round(0)
        
        # Reindex to ensure all 24 hours are present
        all_hours = pd.Index(range(24), name='hour')
        hourly_means = hourly_means.reindex(all_hours)
        
        # Check if we have sufficient data for plotting
        valid_hours = hourly_means.dropna(subset=['FEE_MW']).index
        if len(valid_hours) < 2:
            print(f"[VALIDATION] Skipping residual load decomposition for month {month} - insufficient data ({len(valid_hours)} hours)")
            continue
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        hours = range(24)
        
        # Only plot for hours with valid data
        valid_hour_mask = ~hourly_means['FEE_MW'].isna()
        valid_hours_list = [h for h, valid in zip(hours, valid_hour_mask) if valid]
        
        if len(valid_hours_list) < 2:
            continue
            
        # Get valid data only
        fee_data = hourly_means["FEE_MW"].dropna()
        rl_fee_data = hourly_means["RL_after_FEE_MW"].dropna()
        rl_fossil_data = hourly_means["RL_after_FOSSIL_MU_MW"].dropna()
        rl_psp_data = hourly_means["RL_after_PSP_MW"].dropna()
        import_data = hourly_means["net_import_total_MW"].dropna()
        
        # Merit Order Stack: Must-Run -> FEE -> Must-Run Fossil -> Importe
        # Berechnung der kumulativen Werte fÃƒÂ¼r gestapelte Darstellung
        
        # Zuerst mÃƒÂ¼ssen wir die Must-Run KapazitÃƒÂ¤t aus den RL-Daten ableiten
        # Must-Run = Last - (RL nach FEE + FEE) 
        mustrun_data = []
        fee_adjusted = []
        mustrun_fossil_adjusted = []
        
        for i, hour in enumerate(valid_hours_list):
            hour_idx = hour
            # Must-Run ergibt sich aus der Differenz zwischen Gesamtlast und RL nach Must-Run
            # Wir approximieren das ÃƒÂ¼ber die verfÃƒÂ¼gbaren RL-Stufen
            total_load_approx = (hourly_means.loc[hour_idx, "FEE_MW"] + 
                               hourly_means.loc[hour_idx, "RL_after_FEE_MW"] if not pd.isna(hourly_means.loc[hour_idx, "RL_after_FEE_MW"]) else 0)
            
            mustrun_est = max(0, total_load_approx - hourly_means.loc[hour_idx, "FEE_MW"] - 
                             (hourly_means.loc[hour_idx, "RL_after_FOSSIL_MU_MW"] if not pd.isna(hourly_means.loc[hour_idx, "RL_after_FOSSIL_MU_MW"]) else 0))
            
            mustrun_data.append(mustrun_est)
            fee_adjusted.append(hourly_means.loc[hour_idx, "FEE_MW"])
            
            # Must-Run Fossil = Differenz zwischen RL nach FEE und RL nach Fossil Must-Run
            fossil_mu = (hourly_means.loc[hour_idx, "RL_after_FEE_MW"] - 
                        hourly_means.loc[hour_idx, "RL_after_FOSSIL_MU_MW"] 
                        if not pd.isna(hourly_means.loc[hour_idx, "RL_after_FEE_MW"]) and 
                           not pd.isna(hourly_means.loc[hour_idx, "RL_after_FOSSIL_MU_MW"]) else 0)
            mustrun_fossil_adjusted.append(max(0, fossil_mu))
        
        # Merit Order Stack plotten
        cumulative = [0] * len(valid_hours_list)
        
        # 1. Must-Run (Basis)
        if any(x > 0 for x in mustrun_data):
            ax1.fill_between(valid_hours_list, cumulative, mustrun_data, 
                            color=BBH_COLORS['umbra'], alpha=0.8, label='Must-Run')
            cumulative = mustrun_data.copy()
        
        # 2. FEE (Wind+Solar) on top
        if len(fee_adjusted) > 0:
            next_level = [cumulative[i] + fee_adjusted[i] for i in range(len(cumulative))]
            ax1.fill_between(valid_hours_list, cumulative, next_level, 
                            color=BBH_COLORS['good'], alpha=0.7, label='FEE (Wind+Solar)')
            cumulative = next_level.copy()
        
        # 3. Must-Run Fossil on top
        if any(x > 0 for x in mustrun_fossil_adjusted):
            next_level = [cumulative[i] + mustrun_fossil_adjusted[i] for i in range(len(cumulative))]
            ax1.fill_between(valid_hours_list, cumulative, next_level, 
                            color=BBH_COLORS['petrol'], alpha=0.7, label='Must-Run Fossil')
            cumulative = next_level.copy()
        
        # 4. Import als separate Linie (kann negativ sein)
        if len(import_data) > 0:
            ax1.plot(valid_hours_list, import_data, 'o-', 
                    color=BBH_COLORS['darkred'], linewidth=2, markersize=4, label='Netto-Import')
        
        ax1.set_xlabel('Stunde')
        ax1.set_ylabel('MW')
        ax1.set_title(f'Monat {month:02d}: Merit Order - Lastdeckung (untertÃƒÂ¤gig)', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Preis auf zweiter y-Achse
        ax2 = ax1.twinx()
        price_data = hourly_means["price_DE"].dropna()
        if len(price_data) > 0:
            ax2.plot(valid_hours_list, price_data, 's-', 
                    color=BBH_COLORS['primary'], linewidth=2, markersize=4, label='Preis')
        ax2.set_ylabel('EUR/MWh')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(val_dir / f"merit_order_load_coverage_month_{month:02d}.png", 
                   dpi=160, bbox_inches='tight')
        plt.close()


def _plot_sensitivity_analysis(df_full, val_dir, args):
    """SensitivitÃƒÂ¤tsanalyse - Vereinfachte Version mit verfÃƒÂ¼gbaren Parametern"""
    # Berechne Basis-Kennzahlen
    base_corr = df_full[["price_DE", "marginal_srmc_eur_per_mwh"]].corr().iloc[0,1]
    base_import_share = (df_full["marginal_side"] == "IMPORT").mean() * 100
    base_avg_mef = df_full["mef_g_per_kwh"].mean()
    
    # Sammle Parameter-Info
    params_info = {
        'epsilon': getattr(args, 'epsilon', None),
        'mu_cost_q': getattr(args, 'mu_cost_q', None),
        'mustrun_quantile': getattr(args, 'mustrun_quantile', None),
        'corr_cap_tol': getattr(args, 'corr_cap_tol', None)
    }
    
    # Erstelle Info-Plot mit aktuellen Parametern und Kennzahlen
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Parameter-Text
    param_text = "Aktuelle Parameter-Konfiguration:\n\n"
    for param, value in params_info.items():
        if value is not None:
            param_text += f"{param}: {value}\n"
    
    param_text += f"\nResultierende Kennzahlen:\n"
    param_text += f"Korrelation: {base_corr:.3f}\n"
    param_text += f"Import-Anteil: {base_import_share:.1f}%\n"  
    param_text += f"ÃƒËœ MEF: {base_avg_mef:.0f} g/kWh"
    
    ax.text(0.1, 0.9, param_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor=BBH_COLORS['bg_plot'], alpha=0.8))
    
    ax.set_title('Parameter-Konfiguration und Kennzahlen', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(val_dir / "sensitivity_current_config.png", dpi=160, bbox_inches='tight')
    plt.close()



def _export_validation_raw_data(df_full, val_dir):
    """Export Rohdaten fÃƒÂ¼r weitere Analyse"""
    # Haupt-Validierungsdaten
    validation_cols = [
        'timestamp', 'price_DE', 'marginal_srmc_eur_per_mwh', 'mef_g_per_kwh',
        'marginal_side', 'marginal_fuel', 'marginal_label', 'abs_error',
        'net_import_total_MW', 'cluster_zones', 'month', 'hour'
    ]
    
    available_cols = [col for col in validation_cols if col in df_full.columns]
    df_export = df_full[available_cols].copy()
    df_export.to_csv(val_dir / "validation_raw_data.csv", index=False)
    
    # Monatliche Zusammenfassung
    monthly_summary = df_full.groupby('month').agg({
        'price_DE': ['mean', 'std', 'min', 'max'],
        'marginal_srmc_eur_per_mwh': ['mean', 'std', 'min', 'max'], 
        'mef_g_per_kwh': ['mean', 'std', 'min', 'max'],
        'abs_error': ['mean', 'std', 'median', lambda x: np.percentile(x, 95)],
        'net_import_total_MW': 'mean'
    }).round(2)
    
    monthly_summary.columns = [f"{col[1]}_{col[0]}" if col[1] != '<lambda_0>' else f"p95_{col[0]}" 
                              for col in monthly_summary.columns]
    monthly_summary.to_csv(val_dir / "monthly_summary.csv")
    
    # Fuel-basierte Zusammenfassung
    fuel_summary = df_full.groupby('marginal_fuel').agg({
        'mef_g_per_kwh': ['mean', 'std', 'count'],
        'marginal_srmc_eur_per_mwh': ['mean', 'std'],
        'abs_error': ['mean', 'median']
    }).round(2)
    
    fuel_summary.columns = [f"{col[1]}_{col[0]}" for col in fuel_summary.columns]
    fuel_summary.to_csv(val_dir / "fuel_summary.csv")
    
    print(f"[OK] Rohdaten exportiert: validation_raw_data.csv, monthly_summary.csv, fuel_summary.csv")



def reservoir_qa_report(df: pd.DataFrame, zone: str, month: int, output_dir: str = "out") -> dict:
    """
    Quality Assurance Report fÃƒÂ¼r Enhanced Reservoir Hydro System
    
    Analysiert pro Zone & Monat:
    - Budget utilization und overruns
    - PSP/Reservoir overlap conflicts  
    - Price coupling correlation
    - Water value band effectiveness
    
    Args:
        df: Results DataFrame mit reservoir flags
        zone: Zone name (AT, CH, PL, SE4, etc.)
        month: Monat (1-12)
        output_dir: Output directory for plots
        
    Returns:
        dict: QA metrics
    """
    d = df[(df.get('zone', '') == zone) & (df['timestamp'].dt.month == month)]
    if len(d) == 0:
        return {"error": f"No data for {zone} month {month}"}
    
    # Required columns with fallbacks
    reservoir_flag = d.get('flag_reservoir_enhanced', pd.Series([False]*len(d), index=d.index))
    reservoir_dispatch = d.get('reservoir_dispatch_MWh', pd.Series([0.0]*len(d), index=d.index))
    reservoir_budget = d.get('reservoir_budget_MWh', pd.Series([0.0]*len(d), index=d.index))
    psp_flag = d.get('flag_psp_price_setting', pd.Series([False]*len(d), index=d.index))
    price_zone = d.get(f'price_{zone}', d.get('price_DE', pd.Series([0.0]*len(d), index=d.index)))
    water_value = d.get('water_value_eur_mwh', pd.Series([0.0]*len(d), index=d.index))
    is_coupled = abs(price_zone - d.get('price_DE', price_zone)) <= 7.5  # 7.5 cent coupling tolerance
    
    out = {}
    out["zone"] = zone
    out["month"] = month
    out["hours_total"] = len(d)
    out["hours_flagged"] = int(reservoir_flag.sum())
    out["energy_flagged_GWh"] = float((reservoir_flag * reservoir_dispatch).sum() / 1e3)
    out["budget_GWh"] = float(reservoir_budget.iloc[0] / 1e3) if len(reservoir_budget) > 0 else 0.0
    out["budget_util_%"] = 100.0 * out["energy_flagged_GWh"] / max(out["budget_GWh"], 1e-9)
    out["overrun_GWh"] = max(0.0, out["energy_flagged_GWh"] - out["budget_GWh"])
    out["overlap_psp_hours"] = int((reservoir_flag & psp_flag).sum())
    out["coupling_share_%"] = 100.0 * float((reservoir_flag & is_coupled).sum()) / max(out["hours_flagged"], 1)
    
    # Water value band analysis
    if out["hours_flagged"] > 0:
        price_water_diff = abs(price_zone[reservoir_flag] - water_value[reservoir_flag])
        out["price_water_dev_mean"] = float(price_water_diff.mean())
        out["price_water_dev_p95"] = float(price_water_diff.quantile(0.95))
        out["band_hit_rate_%"] = 100.0 * float((price_water_diff <= 5.0).sum()) / len(price_water_diff)
    else:
        out["price_water_dev_mean"] = 0.0
        out["price_water_dev_p95"] = 0.0
        out["band_hit_rate_%"] = 0.0
    
    return out

def create_reservoir_qa_plots(df: pd.DataFrame, zones: list, output_dir: str = "out"):
    """
    Erstellt QA-Plots fÃƒÂ¼r Enhanced Reservoir System
    
    1. ECDF von |Preis - Wasserwert| pro Zone/Monat
    2. Zeitreihe: Preis, Wasserwert, Flag + Budget-Rest
    3. Budget utilization bars pro Zone/Monat
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    qa_dir = Path(output_dir) / "analysis" / "reservoir_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: ECDF Price-Water Deviations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Enhanced Reservoir QA: Price-Water Value Deviations', fontsize=14)
    
    for i, zone in enumerate(zones[:4]):  # Top 4 zones
        ax = axes[i//2, i%2]
        zone_data = df[df.get('zone', '') == zone]
        if len(zone_data) == 0:
            continue
            
        reservoir_hours = zone_data[zone_data.get('flag_reservoir_enhanced', False)]
        if len(reservoir_hours) > 0:
            price_col = f'price_{zone}' if f'price_{zone}' in zone_data.columns else 'price_DE'
            water_col = 'water_value_eur_mwh' if 'water_value_eur_mwh' in zone_data.columns else price_col
            
            deviations = abs(reservoir_hours[price_col] - reservoir_hours[water_col])
            sorted_dev = np.sort(deviations)
            y = np.arange(1, len(sorted_dev) + 1) / len(sorted_dev)
            
            ax.plot(sorted_dev, y, label=f'{zone} (n={len(sorted_dev)})')
            ax.axvline(5.0, color='red', linestyle='--', alpha=0.7, label='Ã‚Â±5EUR Band')
            ax.set_xlabel('|Price - Water Value| [EUR/MWh]')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'{zone}: Price-Water Deviation ECDF')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(qa_dir / "price_water_deviation_ecdf.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Budget Utilization by Zone/Month
    budget_data = []
    for zone in zones:
        for month in range(1, 13):
            qa_result = reservoir_qa_report(df, zone, month)
            if "error" not in qa_result:
                budget_data.append(qa_result)
    
    if budget_data:
        import pandas as pd
        budget_df = pd.DataFrame(budget_data)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        zones_with_data = budget_df['zone'].unique()
        x_pos = np.arange(len(zones_with_data))
        
        for month in range(1, 13):
            month_data = budget_df[budget_df['month'] == month]
            if len(month_data) > 0:
                utilization = [month_data[month_data['zone'] == z]['budget_util_%'].iloc[0] 
                             if len(month_data[month_data['zone'] == z]) > 0 else 0 
                             for z in zones_with_data]
                ax.bar(x_pos + (month-6.5)*0.06, utilization, width=0.06, 
                      label=f'M{month}', alpha=0.8)
        
        ax.axhline(100, color='red', linestyle='--', label='100% Budget')
        ax.set_xlabel('Zone')
        ax.set_ylabel('Budget Utilization [%]')
        ax.set_title('Enhanced Reservoir: Monthly Budget Utilization by Zone')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(zones_with_data, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(qa_dir / "budget_utilization_by_zone_month.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[QA] Enhanced Reservoir QA plots saved to {qa_dir}")



