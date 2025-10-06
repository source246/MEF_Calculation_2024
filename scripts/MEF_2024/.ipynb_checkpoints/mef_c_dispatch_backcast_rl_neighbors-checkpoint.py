#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch-Backcast via Residuallast/Merit-Order (DE + Nachbarn)
Autor: Yannick Schönmeier

Kern:
- Für DE und direkte Nachbarn: stündlich marginale Technologie per Residuallast-Füllung
  (Must-Run je Monat per Quantil, flexible Erzeugung in SRMC-Reihenfolge).
- Kein Preisanker. Preise werden nur optional für Visual-Checks eingelesen.
- DE-Importe: gewichteter Import-MEF aus Nachbar-Marginalen (per Flüsse).
- Exporte: verbrauchsbasiertes Verhalten einstellbar (--export_rule).

CLI:
--timeseries            : DE-Zeitreihe für Zeitachse (mind. Zeitspalte).
--de_gen_file           : DE-Erzeugung je Technologie (z.B. actual_gen_DE_LU_2024.csv). Empfohlen!
--fuels                 : Fuelpreise (€/MWh_th) + CO2 (€/t): gas_eur_mwh_th, coal_eur_mwh_th, lignite_eur_mwh_th, oil_eur_mwh_th, co2_eur_t
--flows                 : stündliche Flüsse (Spalten z.B. AT->DE_LU, DE_LU->AT, FR->DE_LU, ...)
--neighbor_gen_dir      : Ordner mit actual_gen_XXX_2024.csv (AT/BE/CH/.../DK_1/DK_2/NO_2/SE_4 etc.)
--neighbor_prices       : optional (nur für Plot)
--outdir                : Ausgabeverzeichnis

Optionen:
--timecol,--fuel_timecol
--mustrun_q             : Must-Run-Quantil je Monat (default 0.10)
--export_rule           : zero | domestic | importer  (default zero)
--ef_th_json,--eta_json,--vom_json : zur Überschreibung der Defaults

Ausgaben:
- mef_c_timeseries.csv  : time, label_domestic, mef_domestic, mef_import, mef_final, import_mw, export_mw
- mef_c_monthly.csv     : Monatsmittel & Anteile
- optional: price_fit_2weeks.png   (nur Visualisierung)
"""

import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PALETTE = ["#518E9F","#9B0028","#00374B","#E6F0F7","#A08268"]

# Defaults
EF_TH_DEFAULT = {"gas":0.202, "hardcoal":0.340, "lignite":0.364, "oil":0.267}   # t/MWh_th
ETA_DEFAULT   = {"gas":0.55,  "hardcoal":0.40,  "lignite":0.35,  "oil":0.38}
VOM_DEFAULT   = {"gas":1.5,   "hardcoal":1.3,   "lignite":1.7,   "oil":1.0}     # €/MWh_el

FUEL_ALIASES = {
    "gas":      ["Fossil Gas","Gas","Natural gas","Erdgas","gas"],
    "hardcoal": ["Fossil Hard coal","Hard coal","Steinkohle","hard coal","coal"],
    "lignite":  ["Fossil Brown coal","Brown coal","Braunkohle","Lignite","lignite"],
    "oil":      ["Fossil Oil","Oil","Öl","oil"],
    # Erneuerbare (nur Info, werden nicht als marginal gesetzt)
    "wind_on":  ["Wind onshore","Onshore wind","Wind Onshore"],
    "wind_off": ["Wind offshore","Offshore wind","Wind Offshore"],
    "solar":    ["Solar","PV","Photovoltaic"],
    "ror":      ["Hydro run-of-river","Run-of-river","Hydro Run-of-river","Wasserkraft Laufwasser","Hydro Run-of-River"],
    "nuclear":  ["Nuclear","Kernenergie"]
}

NEIGHBOR_CODES = ["FR","NL","PL","CZ","AT","CH","DK_1","DK_2","NO_2","SE_4","SI","HU","BE"]

# ---------- Helfer ----------

def _read_csv_any(path: str) -> pd.DataFrame:
    # Robust: automatische Trennung + UTF-8(-BOM) + Fallbacks
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except Exception as e:
            raise RuntimeError(f"CSV nicht lesbar: {path}; {e}")

def _to_utc_naive(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce")
    try:
        if hasattr(t.dt, "tz") and t.dt.tz is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return t

def _alias(df: pd.DataFrame, keys: List[str]) -> str:
    # exakte
    for k in keys:
        if k in df.columns: return k
    # case-insensitive
    low = {c.lower():c for c in df.columns}
    for k in keys:
        if k.lower() in low: return low[k.lower()]
    return None

def _find_cols(df: pd.DataFrame, mapping: Dict[str,List[str]]) -> Dict[str,str]:
    out={}
    for k, lst in mapping.items():
        c = _alias(df, lst)
        if c: out[k]=c
    return out

def _ef_el(fuel: str, ef_th: Dict[str,float], eta: Dict[str,float]) -> float:
    return ef_th[fuel]/eta[fuel]

def _srmc_per_fuel(fuel_prices: Dict[str,float], co2: float,
                   ef_th: Dict[str,float], eta: Dict[str,float], vom: Dict[str,float]) -> Dict[str,float]:
    out={}
    for f in ["gas","hardcoal","lignite","oil"]:
        out[f] = fuel_prices[f]/eta[f] + co2*ef_th[f]/eta[f] + vom[f]
    return out

def _monthly_mustrun(df: pd.DataFrame, timecol: str, cols_by_fuel: Dict[str,str], q: float=0.10) -> Dict[Tuple[int,str], float]:
    """Must-Run je (Monat, Fuel) = Quantil der Leistung (Monatsverteilung)."""
    base={}
    tt = pd.to_datetime(df[timecol])
    mon = tt.dt.month
    for f, col in cols_by_fuel.items():
        if col not in df.columns: 
            for m in range(1,13): base[(m,f)] = 0.0
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        for m in range(1,13):
            vals = x[mon==m].dropna().values
            base[(m,f)] = float(np.quantile(vals, q)) if len(vals)>0 else 0.0
    return base

def _marginal_from_rl_fill(row_flex: Dict[str,float], srmc: Dict[str,float]) -> Tuple[str,float]:
    """Marginal = teuerste aktive flexible Fuel (höchste SRMC unter flex>0)."""
    active = [(f, srmc[f]) for f,v in row_flex.items() if v>1e-6 and f in srmc]
    if not active: return ("non_thermal", 0.0)
    f_star, c_star = max(active, key=lambda kv: kv[1])
    return (f_star, c_star)

def _prepare_fuels_series(fuels: pd.DataFrame, fuel_timecol: str, target_index: pd.Series) -> pd.DataFrame:
    fu = fuels.copy()
    fu[fuel_timecol] = _to_utc_naive(fu[fuel_timecol])
    fu = fu.sort_values(fuel_timecol).set_index(fuel_timecol).resample("h").ffill()
    fu = fu.reindex(pd.to_datetime(target_index)).ffill()
    fu = fu.reset_index().rename(columns={"index":fuel_timecol})
    return fu

def _neighbor_codes_from_flows(flow_cols: List[str]) -> List[str]:
    codes=set()
    for c in flow_cols:
        cc = c.upper()
        for z in NEIGHBOR_CODES:
            if z in cc:
                codes.add(z)
    # harmonisiere DK/NO/SE Unterzonen-Bezeichner wie DK_1 etc.
    return sorted(codes)

def _split_import_export(flow_row: pd.Series) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Erwartete Flows-Spalten: <ZONE>->DE_LU (Import), DE_LU-><ZONE> (Export)
    Liefert dicts import_mw[ZONE], export_mw[ZONE]
    """
    imp = {}
    exp = {}
    for col, val in flow_row.items():
        if not isinstance(col, str): continue
        if "->" not in col: continue
        try:
            src, dst = col.split("->", 1)
        except ValueError:
            continue
        v = float(val) if pd.notna(val) else 0.0
        src = src.strip().upper(); dst = dst.strip().upper()
        if dst in ("DE","DE_LU","DE-AT-LU","DE_AT_LU") and src != dst:
            # Import nach DE
            imp[src] = imp.get(src, 0.0) + max(0.0, v)
        elif src in ("DE","DE_LU","DE-AT-LU","DE_AT_LU") and dst != src:
            # Export aus DE
            exp[dst] = exp.get(dst, 0.0) + max(0.0, v)
    return imp, exp

# ---------- Hauptlauf ----------

def run(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ggf. Parameter-Overrides
    ef_th = json.loads(args.ef_th_json) if args.ef_th_json else EF_TH_DEFAULT
    eta   = json.loads(args.eta_json)   if args.eta_json   else ETA_DEFAULT
    vom   = json.loads(args.vom_json)   if args.vom_json   else VOM_DEFAULT

    # ---- DE Timeseries (Zeitachse)
    de_ts = _read_csv_any(args.timeseries)
    if args.timecol not in de_ts.columns:
        raise KeyError(f"Zeitspalte '{args.timecol}' fehlt in {args.timeseries}")
    de_ts[args.timecol] = _to_utc_naive(de_ts[args.timecol])
    de_ts = de_ts.sort_values(args.timecol).reset_index(drop=True)

    # ---- Fuels (auf Stunde ffill)
    fuels = _read_csv_any(args.fuels)
    if args.fuel_timecol not in fuels.columns:
        raise KeyError(f"Fuel-Zeitspalte '{args.fuel_timecol}' fehlt in {args.fuels}")
    fu = _prepare_fuels_series(fuels, args.fuel_timecol, de_ts[args.timecol])

    # ---- Flows
    flows = _read_csv_any(args.flows)
    if args.timecol not in flows.columns:
        # nimm erste Spalte als Zeit
        flows.rename(columns={flows.columns[0]:args.timecol}, inplace=True)
    flows[args.timecol] = _to_utc_naive(flows[args.timecol])
    flows = flows.sort_values(args.timecol).reset_index(drop=True)
    flow_cols = [c for c in flows.columns if c != args.timecol]

    # ---- DE-Generationen (empfohlen)
    degen = None; cols_de = {}
    if args.de_gen_file:
        degen = _read_csv_any(args.de_gen_file)
        if args.timecol not in degen.columns:
            cand = _alias(degen, [args.timecol,"time","Time","Datetime","MTU","mtu","datetime"])
            if cand: degen.rename(columns={cand:args.timecol}, inplace=True)
            else: raise KeyError(f"Zeitspalte in {args.de_gen_file} nicht gefunden.")
        degen[args.timecol] = _to_utc_naive(degen[args.timecol])
        degen = degen.sort_values(args.timecol).reset_index(drop=True)
        cols_de = _find_cols(degen, {
            "gas": FUEL_ALIASES["gas"],
            "hardcoal": FUEL_ALIASES["hardcoal"],
            "lignite": FUEL_ALIASES["lignite"],
            "oil": FUEL_ALIASES["oil"]
        })
        # Must-Run Baselines nach echter DE-Generation
        base_de = _monthly_mustrun(degen, args.timecol,
            {k:v for k,v in cols_de.items() if k in ["gas","hardcoal","lignite","oil"]},
            q=args.mustrun_q
        )
    else:
        # Fallback: versuche Fossilspalten direkt aus timeseries
        cols_de = _find_cols(de_ts, {
            "gas": FUEL_ALIASES["gas"],
            "hardcoal": FUEL_ALIASES["hardcoal"],
            "lignite": FUEL_ALIASES["lignite"],
            "oil": FUEL_ALIASES["oil"]
        })
        base_de = _monthly_mustrun(de_ts, args.timecol,
            {k:v for k,v in cols_de.items() if k in ["gas","hardcoal","lignite","oil"]},
            q=args.mustrun_q
        )

    # ---- Nachbarn laden (nur direkte aus Flows)
    zones_from_flows = _neighbor_codes_from_flows(flow_cols)
    gen_neighbors: Dict[str,pd.DataFrame] = {}
    cols_nb: Dict[str,Dict[str,str]] = {}
    gen_dir = Path(args.neighbor_gen_dir)
    for csv in gen_dir.glob("actual_gen_*_2024.csv"):
        # Zonen-Token aus Dateiname ziehen
        zraw = csv.stem.replace("actual_gen_","").replace("_2024","")
        z = zraw.upper()
        if z not in zones_from_flows:
            continue
        g = _read_csv_any(str(csv))
        # Zeitspalte
        if args.timecol not in g.columns:
            cand = _alias(g, [args.timecol,"time","Time","Datetime","MTU","mtu","datetime"])
            if cand: g.rename(columns={cand:args.timecol}, inplace=True)
            else: raise KeyError(f"Zeitspalte in {csv} nicht gefunden.")
        g[args.timecol] = _to_utc_naive(g[args.timecol])
        g = g.sort_values(args.timecol).reset_index(drop=True)
        gen_neighbors[z] = g
        cols_nb[z] = _find_cols(g, {
            "gas": FUEL_ALIASES["gas"],
            "hardcoal": FUEL_ALIASES["hardcoal"],
            "lignite": FUEL_ALIASES["lignite"],
            "oil": FUEL_ALIASES["oil"]
        })
    # Must-Run pro Nachbar/Monat/Fuel
    base_nb: Dict[str,Dict[Tuple[int,str],float]] = {}
    for z,g in gen_neighbors.items():
        base_nb[z] = _monthly_mustrun(g, args.timecol,
            {k:v for k,v in cols_nb[z].items() if k in ["gas","hardcoal","lignite","oil"]},
            q=args.mustrun_q
        )

    # ---- Zeitlich alignen (asof merge für Fuels & Flows)
    de2 = pd.merge_asof(de_ts, fu, left_on=args.timecol, right_on=args.fuel_timecol, direction="backward")
    de2 = pd.merge_asof(de2, flows, on=args.timecol, direction="nearest")

    def _flex_row(vals: Dict[str,float], month: int, baseline: Dict[Tuple[int,str],float]) -> Dict[str,float]:
        out={}
        for f in ["gas","hardcoal","lignite","oil"]:
            v = max(0.0, vals.get(f,0.0) - baseline.get((month,f),0.0))
            out[f]=v
        return out

    results=[]
    for _, row in de2.iterrows():
        ts = row[args.timecol]
        # Fuelpreise & CO2
        try:
            fp = {
                "gas": float(row.get("gas_eur_mwh_th", np.nan)),
                "hardcoal": float(row.get("coal_eur_mwh_th", np.nan)),
                "lignite": float(row.get("lignite_eur_mwh_th", np.nan)),
                "oil": float(row.get("oil_eur_mwh_th", np.nan))
            }
            co2 = float(row.get("co2_eur_t", np.nan))
        except Exception:
            raise RuntimeError("Fuel/CO2-Spalten fehlen (erwartet: gas_eur_mwh_th, coal_eur_mwh_th, lignite_eur_mwh_th, oil_eur_mwh_th, co2_eur_t).")

        srmc_de = _srmc_per_fuel(fp, co2, ef_th, eta, vom)
        m = pd.to_datetime(ts).month

        # DE-Generationen (Fossil) an dieser Stunde
        vals_de = {}
        if degen is not None:
            g2 = degen[degen[args.timecol] <= ts].tail(1)
            for f in ["gas","hardcoal","lignite","oil"]:
                c = cols_de.get(f, None)
                vals_de[f] = float(g2.iloc[0][c]) if c and c in g2.columns and len(g2)>0 else 0.0
        else:
            for f in ["gas","hardcoal","lignite","oil"]:
                c = cols_de.get(f, None)
                vals_de[f] = float(row.get(c, 0.0)) if c else 0.0

        flex_de = _flex_row(vals_de, m, base_de)
        f_star_de, _ = _marginal_from_rl_fill(flex_de, srmc_de)
        mef_domestic = _ef_el(f_star_de, ef_th, eta) if f_star_de in ["gas","hardcoal","lignite","oil"] else 0.0

        # Flüsse dieser Stunde splitten
        flow_row = row[[c for c in flow_cols if c in row.index]].fillna(0.0)
        imports_by_zone, exports_by_zone = _split_import_export(flow_row)
        imp_total = sum(imports_by_zone.values())
        exp_total = sum(exports_by_zone.values())

        # Nachbar-Marginale & Import-MEF (gewichtet)
        imp_weight_ef = 0.0
        for z, mw in imports_by_zone.items():
            if mw <= 0: continue
            if z not in gen_neighbors: 
                # Zone evtl. in Flows (z. B. DE-AT-LU Aggregation), aber nicht in gen_dir – dann überspringen
                continue
            g = gen_neighbors[z]
            g2 = g[g[args.timecol] <= ts].tail(1)
            vals_z = {}
            for f, col in cols_nb[z].items():
                vals_z[f] = float(g2.iloc[0][col]) if len(g2)>0 and col in g2.columns else 0.0
            flex_z = _flex_row(vals_z, m, base_nb[z])
            srmc_z = _srmc_per_fuel(fp, co2, ef_th, eta, vom)  # Proxy: gleiche Fuelpreise/CO2
            f_star_z, _ = _marginal_from_rl_fill(flex_z, srmc_z)
            ef_z = _ef_el(f_star_z, ef_th, eta) if f_star_z in ef_th else 0.0
            imp_weight_ef += mw * ef_z
        mef_import = (imp_weight_ef/imp_total) if imp_total>1e-6 else 0.0

        # Verbrauchsregel bei Export
        rule = args.export_rule.lower()
        if rule == "zero" and exp_total > 1e-6:
            mef_final = 0.0
            label = f"{f_star_de} (export buffer)"
        elif rule == "importer" and imp_total > 1e-6:
            mef_final = mef_import
            label = "IMPORT"
        else:
            if f_star_de in ["gas","hardcoal","lignite","oil"]:
                mef_final = mef_domestic
                label = f_star_de
            elif imp_total > 1e-6:
                mef_final = mef_import
                label = "IMPORT"
            else:
                mef_final = 0.0
                label = "non_thermal"

        results.append({
            "time": ts,
            "label_domestic": label,
            "mef_domestic_t_per_mwh": mef_domestic,
            "mef_import_t_per_mwh": mef_import,
            "mef_final_t_per_mwh": mef_final,
            "import_mw": imp_total,
            "export_mw": exp_total
        })

    out = pd.DataFrame(results).sort_values("time")
    out_ts = outdir/"mef_c_timeseries.csv"
    out.to_csv(out_ts, index=False)

    # Monatsmittel
    out["month"] = pd.to_datetime(out["time"]).dt.month
    month = out.groupby("month").agg(
        mef_final_mean=("mef_final_t_per_mwh","mean"),
        mef_domestic_mean=("mef_domestic_t_per_mwh","mean"),
        mef_import_mean=("mef_import_t_per_mwh","mean"),
        import_share_hours=("import_mw", lambda s: (s>0).mean()),
        export_share_hours=("export_mw", lambda s: (s>0).mean())
    ).reset_index()
    month.to_csv(outdir/"mef_c_monthly.csv", index=False)

    print(f"[OK] Track C via Residuallast → {out_ts}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--de_gen_file", required=False, help="Pfad zu actual_gen_DE_LU_2024.csv (empfohlen)")
    ap.add_argument("--fuels", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--neighbor_gen_dir", required=True)
    ap.add_argument("--neighbor_prices", default=None, help="optional; nur für Visual-Check")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--timecol", default="time")
    ap.add_argument("--fuel_timecol", default="time")
    ap.add_argument("--mustrun_q", type=float, default=0.10)
    ap.add_argument("--export_rule", default="zero", choices=["zero","domestic","importer"])

    ap.add_argument("--ef_th_json", default=None)
    ap.add_argument("--eta_json", default=None)
    ap.add_argument("--vom_json", default=None)

    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
