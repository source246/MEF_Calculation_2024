#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skaliert die Anlagenliste auf offizielle 2024-Kapazitäten je Zone×Fuel
(Quelle: vom Nutzer bereitgestellte Werte in letzter Nachricht).

- Zonen/Fuels ohne Offizialwert bleiben unskaliert.
- DK gesamt wird proportional auf DK_1 und DK_2 verteilt (Baseline-Anteile).
- NO und SE werden als NO_2 bzw. SE_4 behandelt (Annahme).
- Fuels: gas, hardcoal, lignite, oil (alles in MW).

Outputs:
  - plants_scaled_official_2024.csv
  - scaling_factors_official_2024.csv
"""

import argparse, os
import pandas as pd
import numpy as np

FUELS = ["gas","hardcoal","lignite","oil"]

def to_num(s):
    return pd.to_numeric(pd.Series(s).astype(str).str.replace(",", ".", regex=False), errors="coerce")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants-imputed", required=True, help="CSV mit Spalten: zone,fuel,capacity_mw,...")
    ap.add_argument("--out-scaled", required=True, help="Output: skalierte Anlagenliste")
    ap.add_argument("--out-factors", required=True, help="Output: Skalierungsfaktoren je Zone×Fuel")
    args = ap.parse_args()

    # ---------- Offizielle Kapazitäten 2024 (GW) aus der Nutzer-Nachricht ----------
    # Werte in GW → unten direkt in MW umgerechnet.
    OFF_CAP_GW_COUNTRY = {
        # Deutschland → Zone DE_LU (vereinigt)
        "DE_LU": {"gas":35.63, "hardcoal":15.57, "lignite":15.07, "oil":3.89},
        # Schweiz
        "CH":    {"gas":0.00,  "hardcoal":0.00,  "lignite":0.00,  "oil":0.00},
        # Österreich
        "AT":    {"gas":4.22,  "hardcoal":0.00,  "lignite":0.00,  "oil":0.12},
        # Belgien
        "BE":    {"gas":8.35,  "hardcoal":0.00,  "lignite":0.00,  "oil":0.59},
        # Tschechien (Gas = Erdgas + Kohlegas)
        "CZ":    {"gas":1.24+0.38, "hardcoal":1.20, "lignite":7.24, "oil":0.00},
        # Dänemark (gesamt; wird auf DK_1/DK_2 verteilt)
        "DK":    {"gas":1.57,  "hardcoal":3.02,  "lignite":0.00,  "oil":0.96},
        # Frankreich
        "FR":    {"gas":13.11, "hardcoal":1.81,  "lignite":0.00,  "oil":3.04},
        # Norwegen → in deinem Setup nur NO_2
        "NO_2":  {"gas":0.48,  "hardcoal":0.00,  "lignite":0.00,  "oil":0.00},
        # Polen
        "PL":    {"gas":5.25,  "hardcoal":18.53, "lignite":6.95,  "oil":0.39},
        # Schweden → in deinem Setup nur SE_4 (Gas/Öl auf 0 laut Liste)
        "SE_4":  {"gas":0.00,  "hardcoal":0.00,  "lignite":0.00,  "oil":0.00},
        # Niederlande nicht in der Liste → kein Override
        # "NL": { ... }
    }

    # ---------- Anlagen einlesen ----------
    plants = pd.read_csv(args.plants_imputed)
    # Case-insensitive Normalisierung
    lc = {c.lower(): c for c in plants.columns}
    for need in ["zone","fuel","capacity_mw"]:
        if need not in lc:
            raise SystemExit(f"[ERROR] Column '{need}' missing in plants file.")
    plants = plants.rename(columns={lc["zone"]:"zone", lc["fuel"]:"fuel", lc["capacity_mw"]:"capacity_mw"})
    plants["zone"] = plants["zone"].astype(str).str.upper()
    plants["fuel"] = plants["fuel"].astype(str).str.lower()
    plants["capacity_mw"] = to_num(plants["capacity_mw"])

    # ---------- DK-Verteilung vorbereiten: Baseline-Anteile DK_1 / DK_2 ----------
    base = (plants[plants["fuel"].isin(FUELS)]
            .groupby(["zone","fuel"], as_index=False)["capacity_mw"].sum()
            .rename(columns={"capacity_mw":"baseline_mw"}))

    def dk_split(fuel, total_mw):
        b1 = float(base.query("zone=='DK_1' and fuel==@fuel")["baseline_mw"].sum() or 0.0)
        b2 = float(base.query("zone=='DK_2' and fuel==@fuel")["baseline_mw"].sum() or 0.0)
        denom = b1 + b2
        if denom > 0:
            return {"DK_1": total_mw * b1/denom, "DK_2": total_mw * b2/denom}
        # Fallback gleich teilen
        return {"DK_1": total_mw * 0.5, "DK_2": total_mw * 0.5}

    # ---------- Offizielle Kapazitäten (MW) je ZONE×FUEL ableiten ----------
    off_rows = []
    for key, fuels in OFF_CAP_GW_COUNTRY.items():
        if key == "DK":
            # Verteile auf DK_1 / DK_2
            for f in FUELS:
                tot_mw = float(fuels.get(f, 0.0)) * 1000.0
                parts = dk_split(f, tot_mw)
                for z, mw in parts.items():
                    off_rows.append({"zone": z, "fuel": f, "official_cap_mw": mw})
        else:
            for f in FUELS:
                mw = float(fuels.get(f, 0.0)) * 1000.0
                off_rows.append({"zone": key, "fuel": f, "official_cap_mw": mw})

    off = pd.DataFrame(off_rows)

    # ---------- Skalierungsfaktoren: official / baseline ----------
    fac = (base.merge(off, on=["zone","fuel"], how="outer")
               .fillna({"baseline_mw":0.0, "official_cap_mw":np.nan}))
    # Scale: nur wenn Baseline > 0 und offizieller Wert vorhanden
    fac["scale"] = np.where(
        (fac["baseline_mw"] > 0) & fac["official_cap_mw"].notna(),
        fac["official_cap_mw"] / fac["baseline_mw"],
        np.nan
    )

    # ---------- Auf Anlagen anwenden ----------
    out_cols = ["name","country","zone","fuel","tech","commissioned","capacity_mw","eta"]
    for c in out_cols:
        if c not in plants.columns: plants[c] = np.nan

    scaled = plants[out_cols].merge(fac[["zone","fuel","scale"]], on=["zone","fuel"], how="left")
    scaled["scaled_capacity_mw"] = np.where(
        scaled["scale"].notna(),
        scaled["capacity_mw"] * scaled["scale"],
        scaled["capacity_mw"]
    )
    scaled["scaled_capacity_mw"] = scaled["scaled_capacity_mw"].clip(lower=0)   # Pandas.Series.clip erlaubt `lower`
    

    # ---------- Schreiben ----------
    os.makedirs(os.path.dirname(args.out_scaled), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_factors), exist_ok=True)
    scaled.to_csv(args.out_scaled, index=False)
    fac.to_csv(args.out_factors, index=False)

    print(f"[OK] Scaled plants  → {args.out_scaled}")
    print(f"[OK] Factors        → {args.out_factors}")

if __name__ == "__main__":
    main()
