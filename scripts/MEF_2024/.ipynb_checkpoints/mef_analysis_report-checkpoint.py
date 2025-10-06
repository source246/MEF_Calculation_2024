#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse-Report für Track C (mef_dispatch Output).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("out/track_c/analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------- Load Data -------------------
df = pd.read_csv("out/track_c/mef_track_c_2024.csv", parse_dates=["timestamp"])
dbg = pd.read_csv("out/track_c/_debug_hourly.csv", parse_dates=["timestamp"])

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Berlin")
dbg["timestamp"] = pd.to_datetime(dbg["timestamp"], utc=True).dt.tz_convert("Europe/Berlin")

df = df.set_index("timestamp")
dbg = dbg.set_index("timestamp")

# ------------------- Kennzahlen -------------------
print("Anzahl Stunden:", len(df))
print("Ø MEF (g/kWh):", df["mef_g_per_kwh"].mean().round(1))
print("Quantile (g/kWh):")
print(df["mef_g_per_kwh"].quantile([0.1,0.5,0.9]).round(1))

print("\nAnteile marginal_fuel:")
print(df["marginal_fuel"].value_counts(normalize=True).round(3))

print("\nAnteile marginal_side:")
print(df["marginal_side"].value_counts(normalize=True).round(3))

# ------------------- Plots -------------------
plt.style.use("seaborn-v0_8-darkgrid")

# 1) Zeitreihe MEF
df["mef_g_per_kwh"].resample("D").mean().plot(figsize=(12,4))
plt.ylabel("MEF (g/kWh)")
plt.title("Täglicher Mittelwert MEF 2024")
plt.tight_layout()
plt.savefig(OUTDIR/"mef_timeseries_daily.png")
plt.close()

# 2) Boxplots je Monat
df["month"] = df.index.month
plt.figure(figsize=(10,5))
df.boxplot(column="mef_g_per_kwh", by="month", grid=False)
plt.suptitle("")
plt.title("Verteilung MEF nach Monat")
plt.xlabel("Monat")
plt.ylabel("MEF (g/kWh)")
plt.tight_layout()
plt.savefig(OUTDIR/"mef_boxplot_month.png")
plt.close()

# 3) Scatter Preis vs. MEF
plt.figure(figsize=(6,6))
plt.scatter(df["price_DE"], df["mef_g_per_kwh"], s=5, alpha=0.3)
plt.xlabel("Day-Ahead Preis (€/MWh)")
plt.ylabel("MEF (g/kWh)")
plt.title("Preis vs. MEF")
plt.tight_layout()
plt.savefig(OUTDIR/"mef_vs_price.png")
plt.close()

# 4) Stacked Bar Anteil marginal_fuel je Monat
fuel_month = (
    df.groupby(["month","marginal_fuel"]).size()
    .unstack(fill_value=0)
    .div(df.groupby("month").size(), axis=0)
)
fuel_month.plot(kind="bar", stacked=True, figsize=(12,5), colormap="tab20")
plt.ylabel("Anteil")
plt.title("Anteile marginal_fuel je Monat")
plt.tight_layout()
plt.savefig(OUTDIR/"fuel_share_month.png")
plt.close()

print(f"[OK] Analyse-Plots und Stats in {OUTDIR}")
