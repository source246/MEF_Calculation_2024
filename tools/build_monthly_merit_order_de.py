#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build monthly merit orders for Germany (2024) with:
- color scheme fixed per fuel/technology
- variable O&M adders (EUR/MWh_el)
- unavailability (capacity derating)
- monthly SRMC per unit: ((fuel + CO2 * EF_th) / eta) + varOM

USAGE (example):
  python build_monthly_merit_order_de.py \
    --fleet "/path/Kraftwerke_eff_binned.csv" \
    --prices "/path/prices_2024.csv" \
    --outdir "./out/merit_orders_2024_DE"

If prices has hourly data, the script will aggregate monthly means per fuel and CO2.
"""

import argparse, os, re, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Config: emission factors (thermal basis) and adders
# -----------------------------
EF_TH = {  # tCO2/MWh_th
    "Kernenergie": 0.000,         # not used in SRMC, only for ordering completeness
    "Braunkohle": 0.383,
    "Steinkohle": 0.335,
    "GuD": 0.201,                  # CCGT -> Erdgas
    "Gasturbine": 0.201,           # OCGT -> Erdgas
    "Öl": 0.288,                   # Fuel oil (HFO/HSD ~0.266–0.288); we use conservative 0.288
    "Abfall": 0.000,               # typically handled as ND; keep 0 for ordering if present
    "Sonstige": 0.000,
}

VAR_OM_EUR_MWHEL = {  # from your table
    "Kernenergie": 1.20,
    "Braunkohle": 1.70,
    "Steinkohle": 1.30,
    "GuD": 1.50,
    "Gasturbine": 1.00,
    "Öl": 1.00,
    "Abfall": 1.00,
    "Sonstige": 1.00,
}

UNAVAIL = {  # as percent -> we will use (1 - x/100) to derate capacity
    "Kernenergie": 7.00,
    "Braunkohle": 13.00,
    "Steinkohle": 20.00,
    "GuD": 13.00,
    "Gasturbine": 13.00,
    "Öl": 15.00,
    "Abfall": 15.00,
    "Sonstige": 15.00,
}

# -----------------------------
# Color scheme (fix & consistent)
# -----------------------------
COLORS = {
    "Kernenergie":      "#d62728",  # red
    "Braunkohle":       "#ffdd57",  # yellow
    "Steinkohle":       "#7f7f7f",  # grey
    "GuD":              "#f5e663",  # pale yellow (CCGT)
    "Gasturbine":       "#ff7f0e",  # orange (OCGT/Peaker)
    "Öl":               "#bcbd22",  # olive
    "Abfall":           "#17becf",  # teal
    "Sonstige":         "#9467bd",  # purple
}

# Fallback prices (if columns missing) in EUR/MWh_th & EUR/tCO2
FALLBACK = {
    "gas": 27.0, "hardcoal": 13.0, "lignite": 4.0, "oil": 45.0, "co2": 80.0
}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def detect_cols_fleet(df: pd.DataFrame):
    cols = list(df.columns)
    # capacity
    cap_col = None
    for c in cols:
        cl = norm(c)
        if cl in ("nettonennleistungdereinheit","mwnettonennleistungdereinheit","leistungmw","nettonennleistung","capacitymw","leistung"):
            cap_col = c; break
        if ("leistung" in c.lower()) or (cl.endswith("mw")):
            cap_col = c
    # fuel
    fuel_col = None
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["energieträger","energietraeger","hauptbrennstoff","brennstoff","fuel","kraftwerkstyp"]):
            fuel_col = c; break
    # efficiency
    eta_col = None
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["wirkungsgrad","effizienz","eta"]):
            eta_col = c; break
    if not cap_col or not fuel_col or not eta_col:
        raise ValueError("Fleet: konnte zentrale Spalten nicht sicher erkennen. Bitte Spalten explizit prüfen.")
    return cap_col, fuel_col, eta_col

def map_fuel_for_mo(raw: str) -> str:
    t = str(raw or "").lower()
    if "kern" in t or "nuclear" in t:
        return "Kernenergie"
    if "lign" in t or "braunkoh" in t:
        return "Braunkohle"
    if "hard" in t or "steinkoh" in t or ("coal" in t and "lign" not in t):
        return "Steinkohle"
    if "gud" in t or "ccgt" in t or ("gas" in t and ("komb" in t or "combined" in t)):
        return "GuD"
    if "gasturbine" in t or "ocgt" in t or ("gas" in t and "turb" in t):
        return "Gasturbine"
    if "öl" in t or "oel" in t or "hfo" in t or "diesel" in t or ("oil" in t):
        return "Öl"
    if "abfall" in t or "waste" in t:
        return "Abfall"
    return "Sonstige"

def detect_cols_prices(df: pd.DataFrame):
    cols = list(df.columns)
    # time
    time_col = None
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["time","datum","datetime","mtu","timestamp"]):
            time_col = c; break
    # fuels (EUR/MWh_th)
    def find(*keys):
        for c in cols:
            cl = c.lower()
            if all(k in cl for k in keys):
                return c
        return None
    gas = find("gas")
    hardcoal = find("hard","coal") or find("steinkohle") or find("hardcoal")
    lignite = find("lignite") or find("braunkohle") or find("brown")
    oil = find("oil") or find("heiz") or find("hfo") or find("hso")
    co2 = None
    for c in cols:
        if "co2" in c.lower():
            co2 = c; break
    return time_col, gas, hardcoal, lignite, oil, co2

def monthly_price_table(prices_csv: str) -> pd.DataFrame:
    df = pd.read_csv(prices_csv, encoding="utf-8-sig", low_memory=False)
    time_col, gas, hardcoal, lignite, oil, co2 = detect_cols_prices(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df["month"] = df[time_col].dt.month
        grp = df.groupby("month")
        # average per month with fallbacks
        out = pd.DataFrame({"month": range(1,13)})
        out["gas"] = out["month"].map(lambda m: grp[gas].mean().get(m, np.nan) if gas else np.nan)
        out["hardcoal"] = out["month"].map(lambda m: grp[hardcoal].mean().get(m, np.nan) if hardcoal else np.nan)
        out["lignite"] = out["month"].map(lambda m: grp[lignite].mean().get(m, np.nan) if lignite else np.nan)
        out["oil"] = out["month"].map(lambda m: grp[oil].mean().get(m, np.nan) if oil else np.nan)
        out["co2"] = out["month"].map(lambda m: grp[co2].mean().get(m, np.nan) if co2 else np.nan)
    else:
        # assume already monthly rows with month column 1..12
        out = df.copy()
        if "month" not in out.columns:
            out["month"] = range(1, len(out)+1)
        out = out[["month","gas","hardcoal","lignite","oil","co2"]]
    # fill fallbacks
    for k in ["gas","hardcoal","lignite","oil","co2"]:
        if k not in out.columns: out[k] = np.nan
        out[k] = out[k].fillna(FALLBACK[k])
    return out

def build_month(fleet_df: pd.DataFrame, month_prices: dict, out_png: str, out_csv: str):
    df = fleet_df.copy()
    # derate capacity by unavailability
    df["cap_avail_mw"] = df.apply(lambda r: float(r["capacity_mw"]) * (1.0 - UNAVAIL.get(r["fuel_mo"], 15.0)/100.0), axis=1)
    # fuel price selector (EUR/MWh_th)
    pf_map = {
        "Kernenergie": 0.0,  # often not price-setting; include with tiny SRMC if needed
        "Braunkohle": month_prices["lignite"],
        "Steinkohle": month_prices["hardcoal"],
        "GuD": month_prices["gas"],
        "Gasturbine": month_prices["gas"],
        "Öl": month_prices["oil"],
        "Abfall": 0.0,
        "Sonstige": 0.0,
    }
    # SRMC
    def srmc_row(r):
        tech = r["fuel_mo"]
        eta  = r["eta"]
        pf   = pf_map.get(tech, 0.0)
        ef   = EF_TH.get(tech, 0.0)
        var  = VAR_OM_EUR_MWHEL.get(tech, 0.0)
        # protect against missing/low eta
        eta = float(np.clip(pd.to_numeric(eta, errors="coerce"), 0.20, 0.65))
        return (pf + month_prices["co2"]*ef)/eta + var
    df["srmc_eur_mwhel"] = df.apply(srmc_row, axis=1)
    # Merit order
    df = df.sort_values(["srmc_eur_mwhel","fuel_mo"]).reset_index(drop=True)
    df["cum_mw"] = df["cap_avail_mw"].cumsum()

    # Save CSV
    cols = ["fuel_mo","capacity_mw","cap_avail_mw","eta","srmc_eur_mwhel","cum_mw"]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Plot: colored bars along cumulative capacity
    plt.figure(figsize=(12,4))
    x_left = 0.0
    for _, r in df.iterrows():
        w = float(r["cap_avail_mw"])
        if w <= 0: continue
        h = float(r["srmc_eur_mwhel"])
        color = COLORS.get(r["fuel_mo"], "#cccccc")
        plt.bar(x_left, h, width=w, align="edge", color=color, edgecolor="none")
        x_left += w
    plt.xlabel("Kumulierte verfügbare Kapazität [MW]")
    plt.ylabel("Grenzkosten SRMC [€/MWh_el]")
    plt.title(f"Merit Order Deutschland – Monat {month_prices['month']:02d}/2024")
    # Legend
    handles = []
    from matplotlib.patches import Patch
    for tech, col in COLORS.items():
        if (df["fuel_mo"]==tech).any():
            handles.append(Patch(facecolor=col, edgecolor="none", label=tech))
    plt.legend(handles=handles, ncol=min(6,len(handles)), frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Fleet
    raw = pd.read_csv(args.fleet, sep=";", encoding="utf-8-sig", low_memory=False)
    cap_col, fuel_col, eta_col = detect_cols_fleet(raw)
    fleet = raw.rename(columns={cap_col:"capacity_mw", fuel_col:"fuel_raw", eta_col:"eta"})
    fleet["fuel_mo"] = fleet["fuel_raw"].map(map_fuel_for_mo)
    fleet["capacity_mw"] = pd.to_numeric(fleet["capacity_mw"], errors="coerce").fillna(0.0)
    fleet["eta"] = pd.to_numeric(fleet["eta"], errors="coerce").clip(0.20, 0.65)

    # Drop non-thermal ND technologies if they slipped in (Wind/Solar/RoR/Biomass)
    mask_keep = fleet["fuel_mo"].isin(list(COLORS.keys()))
    fleet = fleet[mask_keep].copy()

    # Prices → monthly means
    monthly = monthly_price_table(args.prices)

    index_rows = []
    for _, row in monthly.iterrows():
        m = int(row["month"])
        out_csv = os.path.join(args.outdir, f"merit_order_DE_2024_month_{m:02d}.csv")
        out_png = os.path.join(args.outdir, f"merit_order_DE_2024_month_{m:02d}.png")
        month_prices = {
            "month": m,
            "gas": float(row["gas"]) if not pd.isna(row["gas"]) else FALLBACK["gas"],
            "hardcoal": float(row["hardcoal"]) if not pd.isna(row["hardcoal"]) else FALLBACK["hardcoal"],
            "lignite": float(row["lignite"]) if not pd.isna(row["lignite"]) else FALLBACK["lignite"],
            "oil": float(row["oil"]) if not pd.isna(row["oil"]) else FALLBACK["oil"],
            "co2": float(row["co2"]) if not pd.isna(row["co2"]) else FALLBACK["co2"],
        }
        build_month(fleet, month_prices, out_png, out_csv)
        index_rows.append({"month": m, "csv": out_csv, "png": out_png})

    pd.DataFrame(index_rows).to_csv(os.path.join(args.outdir, "index_merit_orders_DE_2024.csv"),
                                    index=False, encoding="utf-8-sig")
    print("Done. Files in:", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
