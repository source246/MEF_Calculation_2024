#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch Backcast mit Import-MEF (Nachbarländer)

(c) hasi + GPT-5, 2025
Lizenz: MIT

Funktion:
- Rekonstruiert stündlich die marginale Technologie in DE (dispatch backcast).
- Liest zusätzlich stündliche Erzeugung & Preise der Nachbarländer.
- Berechnet dort die mutmaßlich marginale Technologie (SRMC ≈ Preis).
- Erzeugt daraus einen Import-MEF je Zone und Stunde.
- Kombiniert diesen mit den DE-Importflüssen → gewichteter Import-MEF.
- Falls DE "non_thermal" und Import>0: ersetzt MEF durch Import-MEF.
- Ausgabe: stündliche CSV + Monats- und Validierungsplots.
"""

import argparse, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Standard Emissionsfaktoren (t/MWh_th)
EF_TH = {
    "gas": 0.202,        # Erdgas
    "hardcoal": 0.340,   # Steinkohle
    "lignite": 0.364,    # Braunkohle
    "oil": 0.267         # Heizöl
}

# Typische Wirkungsgrade η_el
ETA = {
    "gas": 0.52,         # CCGT ~52%
    "ocgt": 0.40,        # einfache GT ~40%
    "hardcoal": 0.33,
    "lignite": 0.35,
    "oil": 0.38
}

def fuel_to_label(col):
    """Erkennung der Spalte → fuel-key"""
    col_low = col.lower()
    if "gas" in col_low: return "gas"
    if "hard" in col_low and "coal" in col_low: return "hardcoal"
    if "brown" in col_low or "lignite" in col_low: return "lignite"
    if "oil" in col_low: return "oil"
    return None

def srmc(fuel, fuel_prices, co2_price):
    """berechnet SRMC für Fuel"""
    ef_th = EF_TH[fuel]
    eta = ETA[fuel]
    return fuel_prices[fuel] / eta + co2_price * ef_th / eta

# ---------------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Lade Timeseries DE
    df = pd.read_csv(args.timeseries, parse_dates=[args.timecol])
    df = df.set_index(args.timecol)

    # Lade Fuels (Brennstoffe + CO2)
    fuels = pd.read_csv(args.fuels, parse_dates=[args.fuel_timecol]).set_index(args.fuel_timecol)
    fuels = fuels.reindex(df.index, method="ffill")

    # Lade Flows
    flows = pd.read_csv(args.flows, parse_dates=[0]).set_index(df.index)
    # flows: Spalten je Grenze, positiv = Import nach DE

    # Lade Nachbarpreise
    prices_nb = pd.read_csv(args.neighbor_prices, parse_dates=[0]).set_index(df.index)

    # Lade Nachbar-Generationen
    gen_neighbors = {}
    for fn in os.listdir(args.neighbor_gen_dir):
        if not fn.endswith(".csv"): continue
        zone = fn.replace("actual_gen_","").replace("_2024.csv","").upper()
        gen_neighbors[zone] = pd.read_csv(os.path.join(args.neighbor_gen_dir, fn),
                                          parse_dates=[0]).set_index(df.index)

    # ---------------------------
    results = []
    for t in df.index:
        row = {}
        row["time"] = t
        price_de = df.loc[t, args.price_col]
        co2_price = fuels.loc[t, "co2_eur_t"]

        # --- DE Marginal bestimmen (nur grob: teuerste aktive Fossile ~ Preisband)
        marginal_label = "non_thermal"
        marginal_mef = 0.0
        max_srmc = -1

        for col in [args.gas_col, args.hardcoal_col, args.lignite_col, args.oil_col]:
            if col not in df.columns: continue
            val = df.loc[t, col]
            if val <= 1e-3: continue
            fuel = fuel_to_label(col)
            if fuel not in fuels.columns: continue
            cost = srmc(fuel, fuels.loc[t], co2_price)
            if cost > max_srmc and abs(cost - price_de) <= args.price_band:
                max_srmc = cost
                marginal_label = fuel
                marginal_mef = EF_TH[fuel] / ETA[fuel]

        # --- Import-MEF berechnen
        import_mef = 0.0
        import_sum = 0.0

        for zone, col in flows.items():
            pass  # Placeholder

        imp_val = flows.loc[t].clip(lower=0)  # nur Importe
        total_imp = imp_val.sum()
        if total_imp > 0:
            weighted = []
            for z in gen_neighbors:
                if z not in prices_nb.columns: continue
                if z not in flows.columns: continue
                imp_mw = flows.loc[t, z]
                if imp_mw <= 0: continue
                price_z = prices_nb.loc[t, args.zone_map.get(z, None)]
                co2_price_z = co2_price  # Proxy: gleicher CO2-Preis
                max_srmc_z = -1
                ef_z = 0.0
                gdf = gen_neighbors[z]
                for gc in gdf.columns:
                    fuel = fuel_to_label(gc)
                    if fuel is None: continue
                    if gdf.loc[t, gc] <= 1e-3: continue
                    cost_z = srmc(fuel, fuels.loc[t], co2_price_z)
                    if abs(cost_z - price_z) <= args.price_band and cost_z > max_srmc_z:
                        max_srmc_z = cost_z
                        ef_z = EF_TH[fuel] / ETA[fuel]
                if ef_z > 0:
                    weighted.append((imp_mw, ef_z))
            if weighted:
                import_mef = sum(mw*ef for mw,ef in weighted)/sum(mw for mw,ef in weighted)

        # --- Falls DE non_thermal und Import>0: ersetze
        if marginal_label == "non_thermal" and import_mef>0:
            marginal_mef = import_mef
            marginal_label = "IMPORT"

        row["price_da"] = price_de
        row["marginal_label"] = marginal_label
        row["mef_c_t_per_mwh"] = marginal_mef
        row["import_mef_t_per_mwh"] = import_mef
        row["import_sum_mw"] = total_imp
        results.append(row)

    out = pd.DataFrame(results).set_index("time")
    out.to_csv(os.path.join(args.outdir, "mef_c_timeseries.csv"))

    # Monatsmittel plot
    monthly = out.resample("M").mean()
    monthly[["mef_c_t_per_mwh","import_mef_t_per_mwh"]].plot()
    plt.ylabel("tCO₂/MWh")
    plt.title("Track C – Monatsmittel")
    plt.savefig(os.path.join(args.outdir, "mef_c_monthly.png"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--fuels", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--neighbor_gen_dir", required=True)
    ap.add_argument("--neighbor_prices", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--timecol", default="Datetime")
    ap.add_argument("--fuel_timecol", default="time")
    ap.add_argument("--price_col", default="price_da_eur_mwh")
    ap.add_argument("--gas_col", default="Fossil Gas")
    ap.add_argument("--hardcoal_col", default="Fossil Hard coal")
    ap.add_argument("--lignite_col", default="Fossil Brown coal")
    ap.add_argument("--oil_col", default="Fossil Oil")
    ap.add_argument("--price_band", type=float, default=8.0)
    ap.add_argument("--zone_map_json", type=str, default="{}")
    args = ap.parse_args()
    args.zone_map = json.loads(args.zone_map_json)
    main(args)
