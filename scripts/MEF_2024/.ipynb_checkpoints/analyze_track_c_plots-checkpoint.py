#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse- & Plot-Skript für Track C Ergebnisse (2024)

Eingaben: mef_track_c_2024.csv (und optional zweiter Lauf)
Outputs:  PNG-Plots im angegebenen Outdir, mit Label im Dateinamen

Beispiele:
  # 1) Nur Basislauf (ohne Must-Run)
  python analyze_track_c_plots.py \
      --input out/track_c_base/mef_track_c_2024.csv \
      --label base \
      --outdir out/plots_track_c

  # 2) Vergleich: mit Must-Run 0.2 (zweiter Lauf)
  python analyze_track_c_plots.py \
      --input out/track_c_base/mef_track_c_2024.csv --label base \
      --input2 out/track_c_mustrun02/mef_track_c_2024.csv --label2 mustrun02 \
      --outdir out/plots_track_c

Farbschema ist fest definiert, aber unten leicht anpassbar.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="CSV mef_track_c_2024.csv (Run 1)")
p.add_argument("--label", required=True, help="Label für Run 1 (z.B. base)")
p.add_argument("--input2", default=None, help="CSV mef_track_c_2024.csv (Run 2, optional)")
p.add_argument("--label2", default=None, help="Label für Run 2 (z.B. mustrun02)")
p.add_argument("--outdir", required=True, help="Ausgabeverzeichnis für PNGs")
args = p.parse_args()

OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

# ---------- Farben ----------
# Einheitliche, deutliche Farben:
COLOR_MAP = {
    "Erdgas":            "#1f77b4",  # blue
    "Steinkohle":        "#2ca02c",  # green
    "Braunkohle":        "#8c564b",  # brown
    "Heizöl schwer":     "#bcbd22",  # olive
    "Heizöl leicht / Diesel": "#e377c2",  # pink
    "mix":               "#17becf",  # cyan
    None:                "#7f7f7f",  # gray
}
# Reihenfolge für Stacks/Legenden:
FUEL_ORDER = ["Braunkohle", "Steinkohle", "Erdgas", "Heizöl schwer", "Heizöl leicht / Diesel", "mix"]

def load_result(path: str) -> pd.DataFrame:
    import pandas as pd

    # robust: erst normal lesen
    df = pd.read_csv(path, low_memory=False)

    # mögliche Zeitspalten durchprobieren
    cand_cols = [
        "timestamp", "time", "datetime", "Datetime", "MTU (CET/CEST)", "MTU", df.columns[0]
    ]
    tcol = next((c for c in cand_cols if c in df.columns), None)
    if tcol is None:
        raise ValueError(f"Keine Zeitspalte in {path} gefunden.")

    # in tz-aware Europe/Berlin bringen (funktioniert auch mit '+01:00')
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=True)
    ts = ts.dt.tz_convert("Europe/Berlin")

    # Index setzen & aufräumen
    df = df.drop(columns=[tcol]).copy()
    df.index = ts
    df = df.sort_index()
    # (optional) NaNs der Zeit raus
    df = df[df.index.notna()]

    # marginal_fuel als string/None säubern
    if "marginal_fuel" in df.columns:
        df["marginal_fuel"] = df["marginal_fuel"].astype(str).replace({"nan": None})

    return df


def daily_mef_plot(df: pd.DataFrame, label: str):
    s = df["mef_g_per_kwh"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(s.index, s.values, lw=1.2)
    ax.set_ylabel("MEF (g/kWh)")
    ax.set_title(f"Täglicher Mittelwert MEF 2024 – {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"mef_daily_{label}.png", dpi=160)
    plt.close(fig)

def stacked_monthly_shares(df: pd.DataFrame, label: str):
    # Monatsanteile der wichtigsten fossilen Fuels (gemäß FUEL_ORDER)
    counts = (df["marginal_fuel"]
              .groupby([df.index.to_period("M")]).value_counts(normalize=True)
              .unstack().fillna(0.0))
    # Nur bekannte Brennstoffe zeigen
    cols = [c for c in FUEL_ORDER if c in counts.columns]
    m = counts[cols]
    # Plot
    fig, ax = plt.subplots(figsize=(12, 4.0))
    bottom = np.zeros(len(m))
    x = m.index.to_timestamp()
    for col in cols:
        ax.bar(x, m[col].values, bottom=bottom, width=24,  # ~monatliche Breite
               color=COLOR_MAP.get(col, "#999999"), label=col, align="center", edgecolor="none")
        bottom += m[col].values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Anteil")
    ax.set_title(f"Anteile marginaler Brennstoffe (monatlich, {label})")
    ax.legend(ncol=min(6, len(cols)))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"fuel_shares_monthly_{label}.png", dpi=160)
    plt.close(fig)

def heatmap_fuel_day_hour(df: pd.DataFrame, label: str):
    # Tag x Stunde – marginal_fuel (Mode je Zelle falls mehrfach)
    d = df.copy()
    d["day"] = d.index.dayofyear
    d["hour"] = d.index.hour

    # Kategoriereihenfolge = FUEL_ORDER (+ evtl. weitere, z. B. None/mix)
    fuels_present = list(dict.fromkeys([f for f in FUEL_ORDER if (d["marginal_fuel"] == f).any()]))
    # Ergänze weitere, die nicht in FUEL_ORDER stehen:
    extra = sorted(set(d["marginal_fuel"].dropna().unique()) - set(fuels_present))
    fuels_all = fuels_present + extra
    if None in d["marginal_fuel"].values:
        fuels_all = fuels_all + [None] if None not in fuels_all else fuels_all

    # Map Fuel -> Code
    code_map = {f: i for i, f in enumerate(fuels_all)}
    # 365 x 24 Matrix (einige Jahre haben 366)
    days = int(d.index.to_series().dt.dayofyear.max())
    mat = np.full((days, 24), np.nan)
    for (day, hour), grp in d.groupby(["day", "hour"]):
        # Modus des Brennstoffs in dieser (day,hour) – falls Lücken
        f = grp["marginal_fuel"].mode(dropna=True)
        if len(f) == 0:
            continue
        mat[day - 1, hour] = code_map.get(f.iloc[0], np.nan)

    # Colormap aus COLOR_MAP in der Reihenfolge fuels_all
    cmap_colors = [COLOR_MAP.get(f, "#7f7f7f") for f in fuels_all]
    cmap = ListedColormap(cmap_colors)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, origin="upper")
    ax.set_xlabel("Stunde")
    ax.set_ylabel("Tag im Jahr")
    ax.set_title(f"Heatmap marginaler Brennstoff (Tag × Stunde) – {label}")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(0, days, 14))
    ax.grid(False)
    # Farblegende (diskret)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLOR_MAP.get(f, "#7f7f7f"), label=str(f)) for f in fuels_all]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"fuel_heatmap_day_hour_{label}.png", dpi=160)
    plt.close(fig)

def comparison_overlay_daily(df1: pd.DataFrame, lab1: str, df2: pd.DataFrame, lab2: str):
    s1 = df1["mef_g_per_kwh"].resample("D").mean()
    s2 = df2["mef_g_per_kwh"].resample("D").mean()
    common = s1.index.intersection(s2.index)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(common, s1.reindex(common), lw=1.2, label=lab1)
    ax.plot(common, s2.reindex(common), lw=1.2, label=lab2)
    ax.set_ylabel("MEF (g/kWh)")
    ax.set_title(f"MEF Tagesmittel – Vergleich ({lab1} vs. {lab2})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"mef_daily_compare_{lab1}_vs_{lab2}.png", dpi=160)
    plt.close(fig)

def comparison_monthly_shares(df1: pd.DataFrame, lab1: str, df2: pd.DataFrame, lab2: str):
    def monthly(df):
        return (df["marginal_fuel"]
                .groupby([df.index.to_period("M")]).value_counts(normalize=True)
                .unstack().fillna(0.0))
    m1, m2 = monthly(df1), monthly(df2)
    cols = sorted(set(m1.columns).union(set(m2.columns)))
    m1, m2 = m1.reindex(columns=cols, fill_value=0.0), m2.reindex(columns=cols, fill_value=0.0)
    diff = (m2 - m1)  # positive = häufiger in Run2

    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = diff.index.to_timestamp()
    bottom = np.zeros(len(diff))
    for c in cols:
        if c not in COLOR_MAP: continue
        vals = diff[c].values
        pos = np.clip(vals, 0, None)
        neg = np.clip(vals, None, 0)
        # getrennte Bars für +/-
        ax.bar(x, pos, bottom=bottom, width=24, color=COLOR_MAP[c], edgecolor="none")
        ax.bar(x, neg, bottom=bottom, width=24, color=COLOR_MAP[c], edgecolor="none")
        bottom += pos  # reine Visualisierung; es ist eine "gestapelte Differenz"
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_ylabel(f"Δ Anteil ({lab2} − {lab1})")
    ax.set_title(f"Monatliche Differenz der Brennstoff-Anteile: {lab2} minus {lab1}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"fuel_share_diff_monthly_{lab1}_vs_{lab2}.png", dpi=160)
    plt.close(fig)

# ---------- Run 1 ----------
df1 = load_result(args.input)
daily_mef_plot(df1, args.label)
stacked_monthly_shares(df1, args.label)
heatmap_fuel_day_hour(df1, args.label)

# ---------- Run 2 (optional) ----------
if args.input2 and args.label2:
    df2 = load_result(args.input2)
    daily_mef_plot(df2, args.label2)
    stacked_monthly_shares(df2, args.label2)
    heatmap_fuel_day_hour(df2, args.label2)
    # Vergleiche
    comparison_overlay_daily(df1, args.label, df2, args.label2)
    comparison_monthly_shares(df1, args.label, df2, args.label2)
