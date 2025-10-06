#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse-Skript für Track C Ergebnisse:
- Eingabe: mef_track_c_2024.csv, _debug_hourly.csv (im --run_dir)
- Ausgabe: Plots + Statistiken unter <run_dir>/analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PALETTE = ["#518E9F", "#9B0028", "#00374B", "#E6F0F7", "#A08268"]
TZ = "Europe/Berlin"

# --- Robustes Zeit-Parsing ---
def _parse_ts(series: pd.Series) -> pd.DatetimeIndex:
    ser_utc = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.DatetimeIndex(ser_utc).tz_convert(TZ)

def _has_data(df: pd.DataFrame, cols) -> bool:
    cols = [cols] if isinstance(cols, str) else list(cols)
    for c in cols:
        if c not in df.columns:
            return False
        if df[c].dropna().shape[0] == 0:
            return False
    return True


def load_with_time(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Kandidaten für Zeitspalte
    cand = [c for c in df.columns if any(k in c.lower() for k in ("time", "stamp", "date"))]
    if "timestamp" in df.columns:
        tcol = "timestamp"
    elif cand:
        tcol = cand[0]
    else:
        tcol = df.columns[0]

    try:
        idx = _parse_ts(df[tcol])
        df = df.drop(columns=[tcol])
    except Exception:
        idx = _parse_ts(df.index.to_series())
    df.index = idx
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Konnte keinen DatetimeIndex herstellen.")
    return df

def main(run_dir: str):
    rundir = Path(run_dir)
    outdir = rundir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    mef_path = rundir / "mef_track_c_2024.csv"
    dbg_path = rundir / "_debug_hourly.csv"

    df = load_with_time(str(mef_path))
    dbg = load_with_time(str(dbg_path))

    # --- 1) Täglicher Durchschnitts-MEF pro Monat (12 Subplots) ---
    daily = df["mef_g_per_kwh"].resample("D").mean()
    monthly = daily.groupby(daily.index.month)

    fig, axes = plt.subplots(3, 4, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    for m in range(1, 13):
        ax = axes[m-1]
        if m in monthly.groups:
            monthly.get_group(m).plot(ax=ax, color=PALETTE[0])
        ax.set_title(f"Monat {m}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Täglicher Durchschnitts-MEF pro Monat", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "mef_daily_by_month.png", dpi=160)
    plt.close(fig)

    # --- 2) Dauerlinie MEF ---
    sorted_vals = df["mef_g_per_kwh"].sort_values().reset_index(drop=True)
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_vals.values, color=PALETTE[1])
    plt.title("Dauerlinie MEF 2024")
    plt.ylabel("g CO₂/kWh")
    plt.xlabel("Stunden (sortiert)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "mef_duration.png", dpi=160)
    plt.close()

    # --- 3) Fuel-Häufigkeiten (gesamt) ---
    if _has_data(df, "marginal_fuel"):
        fuels_clean = df["marginal_fuel"].astype(str).replace({"nan": np.nan})
        fuel_share = fuels_clean.value_counts(normalize=True, dropna=True).sort_values(ascending=False) * 100.0
        if fuel_share.shape[0] > 0:
            plt.figure(figsize=(8, 4))
            fuel_share.plot(kind="bar", color=PALETTE[2])
            plt.ylabel("Anteil [%]")
            plt.title("Häufigkeit marginaler Fuels (gesamt)")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "fuel_share_total.png", dpi=160)
            plt.close()
        else:
            (outdir / "notes.txt").write_text("fuel_share_total: keine nicht-leeren Werte in 'marginal_fuel'.\n", encoding="utf-8")
    else:
        (outdir / "notes.txt").write_text("fuel_share_total: Spalte 'marginal_fuel' fehlt oder nur NaN.\n", encoding="utf-8")


    # --- 4) Preis vs. SRMC (Scatter, Farbe = Seite) ---
    need_cols = ["marginal_srmc_eur_per_mwh", "price_DE", "marginal_side"]
    if all(c in df.columns for c in need_cols):
        tmp = df[need_cols].dropna()
        if not tmp.empty:
            colors = np.where(tmp["marginal_side"]=="IMPORT", PALETTE[1], PALETTE[0])
            plt.figure(figsize=(6.2, 6.2))
            plt.scatter(tmp["marginal_srmc_eur_per_mwh"], tmp["price_DE"], c=colors, alpha=0.6, s=10)
            lo = np.nanmin([tmp["marginal_srmc_eur_per_mwh"].min(), tmp["price_DE"].min()])
            hi = np.nanmax([tmp["marginal_srmc_eur_per_mwh"].max(), tmp["price_DE"].max()])
            plt.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=1)
            plt.xlabel("gewählter SRMC [€/MWh]"); plt.ylabel("Day-Ahead Preis [€/MWh]")
            plt.title("Preis vs. marginale SRMC")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "scatter_price_vs_srmc.png", dpi=160)
            plt.close()
        else:
            with open(outdir / "notes.txt", "a", encoding="utf-8") as f:
                f.write("scatter_price_vs_srmc: keine gemeinsamen nicht-NaN Werte.\n")
    else:
        with open(outdir / "notes.txt", "a", encoding="utf-8") as f:
            f.write("scatter_price_vs_srmc: benötigte Spalten fehlen.\n")


    # --- 5) Rolling Import-Anteil (7 Tage) ---
    imp_share = (df["marginal_side"]=="IMPORT").astype(float).rolling(24*7, min_periods=1).mean()
    plt.figure(figsize=(12, 4))
    plt.plot(imp_share.index, 100.0*imp_share.values, color=PALETTE[3])
    plt.ylabel("Import-Anteil [%]")
    plt.title("Import-Anteil (rollierendes 7-Tage-Mittel)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "import_share_rolling.png", dpi=160)
    plt.close()

    # --- 6) MEF nach marginalem Fuel (Boxplot) ---
    dfm = df[["marginal_fuel","mef_g_per_kwh"]].dropna()
    if not dfm.empty:
        order = dfm.groupby("marginal_fuel")["mef_g_per_kwh"].median().sort_values().index
        data = [dfm.loc[dfm["marginal_fuel"]==f, "mef_g_per_kwh"].values for f in order]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("MEF [g/kWh]")
        ax.set_title("MEF nach marginaler Technologie")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "box_mef_by_fuel.png", dpi=160)
        plt.close(fig)

    # --- 7) Preis-SRMС-Abweichung je Stunde×Monat (Heatmap Median) ---
    err = (df["price_DE"] - df["marginal_srmc_eur_per_mwh"]).dropna()
    if not err.empty:
        tmp = err.to_frame("err")
        tmp["month"] = tmp.index.month
        tmp["hour"] = tmp.index.hour
        piv = tmp.pivot_table(index="month", columns="hour", values="err", aggfunc="median")
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(piv.values, aspect="auto", origin="lower")
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
        ax.set_xlabel("Stunde"); ax.set_ylabel("Monat"); ax.set_title("Median(Preis − SRMC) [€/MWh]")
        fig.colorbar(im, ax=ax, shrink=0.9, label="€/MWh")
        fig.tight_layout()
        fig.savefig(outdir / "heatmap_median_err_month_hour.png", dpi=160)
        plt.close(fig)

    print("[OK] Analyseplots gespeichert in", outdir)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Ordner mit mef_track_c_2024.csv + _debug_hourly.csv")
    args = p.parse_args()
    main(args.run_dir)
