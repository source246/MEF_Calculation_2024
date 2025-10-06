#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mef_dispatch_plus.py — Track-C Backcast (DE/LU) mit sauberer Pumpspeicher-Behandlung
und verbesserter Import-Attribution (dominante Grenze / dominanter Fuel).

Features:
- Saubere Zeitachse (UTC → Europe/Berlin), DST robust
- Nicht-disponible Erzeugung (Nuclear, Wind, PV, RoR; optional Biomasse/Waste)
- Pumpspeicher: Gen abziehen; optional Pumpstrom zur Last addieren
- Unit-basierte Merit-Order (Fleet mit η), SRMC je Stunde (Fuelpreise+CO2)
- Import-Attribution via Preis-Cluster (ε) + Schex (imp_*) → dominante Grenze/Fuel statt "mix"
- Output: mef_track_c_2024.csv + _debug_hourly.csv (KPIs & Plausibilisierung)

Benötigte Dateien (wie in deiner Struktur):
- Fleet:           input\de\fleet\Kraftwerke_eff_binned.csv  (mit Spalte "Imputed_Effizienz_binned")
- Fuelpreise:      input\de\fuels\prices_2024.csv             (Spalten: time, gas_eur_mwh_th, coal_eur_mwh_th, lignite_eur_mwh_th, oil_eur_mwh_th, co2_eur_t)
- DE-Gen/PS-Last:  input\de\timeseries\dispatch_2024.csv      (ENTSO-E Clean, enthält Pumped Storage Gen + (pumping))
- DE-Last:         neighbors\out_load\2024\load_DE_LU_2024.csv (oder explizit --de_load_file)
- Schex (Netto):   flows\flows_scheduled_DE_LU_2024_net.csv   (Spalten: time, imp_AT,..., net_import_total)
- Nachbarpreise:   neighbors\prices\neighbor_prices_2024.csv  (timestamp, price_DE_LU, price_AT, ...)

Autor: hasi + gpt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import argparse, json, re, sys, math
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------- CLI ---------------------------

def build_parser():
    p = argparse.ArgumentParser()
    # Inputs
    p.add_argument("--fleet", required=True, help="CSV mit Einheiten (inkl. η)")
    p.add_argument("--eta_col", default="Imputed_Effizienz_binned", help="Spalte mit elektrischem Wirkungsgrad (0..1)")
    p.add_argument("--fuel_prices", required=True, help="prices_2024.csv")
    p.add_argument("--dispatch_de", default=None, help="dispatch_2024.csv (mit Gen & Pumping)")
    p.add_argument("--de_load_file", default=None, help="Override: explizite DE/LU-Last (Sonst neighbors/out_load/2024/load_DE_LU_2024.csv erwartet)")
    p.add_argument("--neighbor_gen_dir", required=True, help="Ordner mit actual_gen_ZONE_2024.csv")
    p.add_argument("--neighbor_load_dir", required=True, help="Ordner mit load_ZONE_2024.csv (nur falls gebraucht)")
    p.add_argument("--neighbor_prices", required=True, help="neighbor_prices_2024.csv")
    p.add_argument("--flows", required=True, help="flows_scheduled_DE_LU_2024_net.csv")
    p.add_argument("--outdir", required=True)

    # Options
    p.add_argument("--tz", default="Europe/Berlin")
    p.add_argument("--epsilon", type=float, default=0.01, help="€-Schwelle für Preiskopplung")
    p.add_argument("--therm_avail", type=float, default=0.90, help="thermische Verfügbarkeit [0..1]")
    p.add_argument("--varom_json", default=None, help='JSON { "Gas": 2.0, "Steinkohle": 3.0, ... }  €/MWh_el')
    p.add_argument("--ps_mode", choices=["ignore","subtract_gen","subtract_gen_add_pump"], default="subtract_gen",
                   help="Pumpspeicher-Handling")
    p.add_argument("--nd_include_bio", action="store_true", help="Biomasse/Waste als ND abziehen")
    p.add_argument("--mustrun_mode", choices=["off","capacity","gen_quantile"], default="gen_quantile")
    p.add_argument("--mustrun_quantile", type=float, default=0.20, help="q für gen_quantile (Peak/Offpeak, monatlich)")
    p.add_argument("--mustrun_peak_hours", default="08-20")
    p.add_argument("--mustrun_monthly", action="store_true")

    # Import-Attribution
    p.add_argument("--import_label_mode", choices=["mix","dominant_zone","dominant_fuel"], default="dominant_fuel")
    p.add_argument("--dominant_share_zone", type=float, default=0.60)
    p.add_argument("--dominant_share_fuel", type=float, default=0.40)

    # Emissionsfaktoren (t/MWh_th), Override via JSON
    p.add_argument("--ef_json", default=None, help='JSON { "Gas": 0.202, "Steinkohle": 0.340, "Braunkohle": 0.370, "Oel": 0.270 }')

    return p

# --------------------------- Helpers ---------------------------

FUEL_MAP = {
    "erdgas":"Gas", "gas":"Gas", "fossil gas":"Gas", "erdölgas":"Gas", "erdolgas":"Gas",
    "steinkohle":"Steinkohle", "hard coal":"Steinkohle", "fossil hard coal":"Steinkohle", "coal":"Steinkohle",
    "braunkohle":"Braunkohle", "lignite":"Braunkohle", "brown coal":"Braunkohle",
    "öl":"Oel", "oel":"Oel", "oil":"Oel", "fossil oil":"Oel", "heizöl":"Oel",
}

FOSSIL = {"Gas","Steinkohle","Braunkohle","Oel"}

EF_DEFAULT = {  # t CO2 / MWh_th
    "Gas":        0.202,
    "Steinkohle": 0.340,
    "Braunkohle": 0.370,
    "Oel":        0.270,
}

VAROM_DEFAULT = { # €/MWh_el, simple defaults
    "Gas":        2.0,
    "Steinkohle": 3.0,
    "Braunkohle": 3.0,
    "Oel":        2.5,
}

def norm(s: str) -> str:
    t = str(s or "").lower()
    t = re.sub(r"[^\w\s]+"," ", t)
    t = re.sub(r"\s+"," ", t).strip()
    return t

def map_fuel(raw: str) -> str|None:
    t = norm(raw)
    for k,v in FUEL_MAP.items():
        if k in t:
            return v
    return None

def parse_time_col(df: pd.DataFrame, cols: List[str]) -> pd.DatetimeIndex:
    for c in cols:
        if c in df.columns:
            ser_utc = pd.to_datetime(df[c], utc=True, errors="coerce")
            if ser_utc.notna().any():
                return ser_utc
    # fallback: first column as datetime
    c0 = df.columns[0]
    ser_utc = pd.to_datetime(df[c0], utc=True, errors="coerce")
    return ser_utc

def to_local(idx_like, tz: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx_like)  # akzeptiert Series/Index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(tz)

# in ensure_hourly():
start = loc.min().floor("h")
end   = loc.max().ceil("h")
full  = pd.date_range(start, end, freq="h", tz=tz)

def ensure_hourly(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Index: localized hourly range; reindex+interpolate (only minor gaps)."""
    idx = df.index
    if idx.tz is None:
        raise ValueError("DatetimeIndex must be tz-aware (UTC or local).")
    # Bring to local tz hourly
    loc = idx.tz_convert(tz)
    start = loc.min().floor("H")
    end   = loc.max().ceil("H")
    full  = pd.date_range(start, end, freq="H", tz=tz)
    out   = df.copy()
    out.index = loc
    out = out[~out.index.duplicated(keep="first")]
    out = out.reindex(full)
    # interpolate prices only; for loads/gen use forward fill minimal
    return out

def read_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    t = parse_time_col(df, ["time","timestamp","Datetime"])
    df = df.set_index(t).drop(columns=[c for c in df.columns if c in ("time","timestamp","Datetime")], errors="ignore")
    return df

def read_dispatch_de(path: Path, tz: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    t = pd.to_datetime(d.iloc[:,0], utc=True, errors="coerce")
    d = d.loc[t.notna()].copy()
    d.index = t[t.notna()].tz_convert(tz)
    d = d.drop(columns=d.columns[0]).apply(pd.to_numeric, errors="coerce")
    d = d.sort_index()
    d = d.groupby(d.index).mean()  # falls 15-min
    return d

def read_load_file(path: Path, tz: str) -> pd.Series:
    df = pd.read_csv(path)
    t = pd.to_datetime(df["time"] if "time" in df else df.iloc[:,0],
                       utc=True, errors="coerce")
    # nimm die erste numerische Lastspalte
    val = df.select_dtypes(include="number").iloc[:,0]
    s = pd.Series(val.values, index=t.tz_convert(tz))
    s = s.sort_index()
    s = s.groupby(s.index).mean()  # 15-min → stündlich
    return s

def read_neighbor_prices(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    t = pd.to_datetime(df["timestamp"] if "timestamp" in df else df.iloc[:,0],
                       utc=True, errors="coerce")
    df = df.drop(columns=[c for c in ("time","timestamp") if c in df], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = t.tz_convert(tz)
    df = df.sort_index()
    df = df.groupby(df.index).mean()  # falls 15-min
    return df

def read_flows(path: Path, tz: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Flows sind i.d.R. UTC → dann in lokale Zeit umrechnen
    t_utc = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.drop(columns=["time"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = t_utc.tz_convert(tz)
    df = df.sort_index()

    # <<< WICHTIG: 15-min Werte zu Stunden aggregieren >>>
    # MW-Größen ⇒ für Stundenmittel den Mittelwert bilden
    df = df.groupby(df.index).mean()

    # Falls Index doch noch doppelt (DST etc.), erneut mitteln
    if df.index.has_duplicates:
        df = df.groupby(df.index).mean()

    return df


def load_de_gen_neighbor_dir(neighbor_gen_dir: Path, tz: str) -> pd.DataFrame:
    p = neighbor_gen_dir / "actual_gen_DE_LU_2024.csv"
    df = pd.read_csv(p)
    t = parse_time_col(df, ["MTU (CET/CEST)","time","timestamp","Datetime"])
    df = df.set_index(to_local(t, tz)).sort_index()
    # keep numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def build_nd_series(de_gen: pd.DataFrame, tz: str, include_bio: bool, ps_mode: str, de_load: pd.Series|None) -> Tuple[pd.Series, pd.Series]:
    cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    if include_bio:
        for extra in ["Biomass","Waste"]:
            if extra in de_gen.columns:
                cols.append(extra)
    nd = de_gen.reindex(columns=[c for c in cols if c in de_gen.columns]).sum(axis=1).fillna(0.0)

    # Pumpspeicher
    if ps_mode in ("subtract_gen","subtract_gen_add_pump"):
        ps_gen_cols = [c for c in de_gen.columns if c.startswith("Hydro Pumped Storage") and "(pumping)" not in c.lower()]
        if ps_gen_cols:
            nd = nd.add(de_gen[ps_gen_cols].sum(axis=1), fill_value=0.0)
    if ps_mode == "subtract_gen_add_pump" and de_load is not None:
        pump_cols = [c for c in de_gen.columns if "pumping" in c.lower()]
        if pump_cols:
            de_load = de_load.add(de_gen[pump_cols].sum(axis=1), fill_value=0.0)
    return nd, de_load

def pick_capacity_col(fleet: pd.DataFrame) -> str:
    for c in ["MW Nettonennleistung der Einheit","Nettonennleistung der Einheit","Nettonennleistung"]:
        if c in fleet.columns:
            return c
    for c in fleet.columns:
        if "leistung" in c.lower():
            return c
    raise KeyError("Keine Kapazitätsspalte in Fleet gefunden.")

# --------------------------- Merit & MEF ---------------------------

def compute_srmc_unitwise(units: pd.DataFrame, prices: pd.DataFrame, ef_by_fuel: Dict[str,float], varom: Dict[str,float]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    units: columns ['fuel','eta','mw']
    prices: index=hourly; columns gas_eur_mwh_th, coal_eur_mwh_th, lignite_eur_mwh_th, oil_eur_mwh_th, co2_eur_t
    returns: srmc_per_fuel (hourly), and a Series residual that’s convenient for mapping
    """
    # Build per-fuel SrMC (per unit differs only via eta)
    colmap = {
        "Gas":        "gas_eur_mwh_th",
        "Steinkohle": "coal_eur_mwh_th",
        "Braunkohle": "lignite_eur_mwh_th",
        "Oel":        "oil_eur_mwh_th",
    }
    need = list(colmap.values()) + ["co2_eur_t"]
    miss = [c for c in need if c not in prices.columns]
    if miss:
        raise KeyError(f"Fehlende Spalten in fuel_prices: {miss}")

    # Precompute per-fuel thermal price + CO2 * EF_th
    pf = {}
    for fuel, pcol in colmap.items():
        pf[fuel] = prices[pcol].astype(float) + prices["co2_eur_t"].astype(float) * float(ef_by_fuel[fuel])

    # For each unit: SRMC = (p_fuel + ...)/eta + varom[fuel]
    # We will *not* build a giant HxN matrix; instead, we compute a "stack" per hour from unit list.
    # (H loop is fine for 8784 hours with vector ops inside.)
    hours = prices.index
    # attach varom to unit
    u = units.copy()
    u["varom"] = u["fuel"].map(varom).fillna(0.0).astype(float)

    # Pre-group units by fuel to speed up
    by_fuel = {f: u.loc[u["fuel"]==f, ["eta","mw","varom"]].reset_index(drop=True) for f in FOSSIL}

    # Prepare outputs
    marginal_records = []  # dict per hour

    for ts in hours:
        # build merit list for this hour
        merit = []
        for f, tbl in by_fuel.items():
            if tbl.empty: 
                continue
            price_th = float(pf[f].loc[ts])
            # unit-level SRMC depends on eta
            srmc_units = price_th / tbl["eta"].values + tbl["varom"].values
            # stack by SRMC
            order = np.argsort(srmc_units)
            s_sorted = srmc_units[order]
            cap_sorted = tbl["mw"].values[order]
            fuel_sorted = np.array([f]*len(order))
            merit.append( (s_sorted, cap_sorted, fuel_sorted) )
        if not merit:
            marginal_records.append({"time": ts, "DE_srmc": np.nan, "DE_fuel": None})
            continue
        srmc = np.concatenate([m[0] for m in merit])
        cap  = np.concatenate([m[1] for m in merit])
        fuels= np.concatenate([m[2] for m in merit])
        order = np.argsort(srmc)
        srmc = srmc[order]; cap = cap[order]; fuels = fuels[order]
        marginal_records.append({"time": ts, "merit_srmc": srmc, "merit_cap": cap, "merit_fuel": fuels})
    # We return as list; the main loop nutzt das, um mit resid. Load die marginale Einheit zu finden.
    return marginal_records

def mef_gpkwh_for(fuel: str, eta: float, ef_by_fuel: Dict[str,float]) -> float:
    # EF_th (t/MWh_th) -> EF_el (t/MWh_el) = EF_th / eta
    # g/kWh = t/MWh_el * 1000
    return (ef_by_fuel[fuel] / max(eta, 1e-6)) * 1000.0

# --------------------------- Main ---------------------------

def main(args):
    tz = args.tz

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- Preise
    prices = read_prices(Path(args.fuel_prices))
    prices = ensure_hourly(prices, tz)

    # Emissionsfaktoren + VarOM
    ef_by_fuel = EF_DEFAULT.copy()
    if args.ef_json:
        ef_by_fuel.update(json.loads(Path(args.ef_json).read_text(encoding="utf-8")))
    varom = VAROM_DEFAULT.copy()
    if args.varom_json:
        varom.update(json.loads(Path(args.varom_json).read_text(encoding="utf-8")))

    # ---- Fleet
    fleet = pd.read_csv(args.fleet, sep=";", encoding="utf-8-sig", low_memory=False)
    cap_col = pick_capacity_col(fleet)
    if args.eta_col not in fleet.columns:
        raise KeyError(f"η-Spalte '{args.eta_col}' nicht in Fleet gefunden.")
    units = pd.DataFrame({
        "fuel": fleet["Energieträger"].map(map_fuel),
        "eta":  pd.to_numeric(fleet[args.eta_col], errors="coerce"),
        "mw":   pd.to_numeric(fleet[cap_col], errors="coerce"),
    }).dropna()
    # Filter auf FOSSIL
    units = units[units["fuel"].isin(FOSSIL)].copy()
    # Cleanup / Clipping
    units["eta"] = units["eta"].clip(lower=0.20, upper=0.65)  # thermische Grenzen
    units["mw"]  = units["mw"].clip(lower=0.0)
    # Verfügbarkeit
    units["mw"] *= float(args.therm_avail)

    # ---- DE-Last & DE-Gen (für ND + Pumpspeicher)
    if args.dispatch_de:
        de_gen = read_dispatch_de(Path(args.dispatch_de), tz)
        # in dispatch_2024.csv heißen Pump-Spalten z.B. "Hydro Pumped Storage" (Gen) und "Hydro Pumped Storage.1" (pumping)
        # wir nutzen generisch build_nd_series()
    else:
        de_gen = load_de_gen_neighbor_dir(Path(args.neighbor_gen_dir), tz)

    if args.de_load_file:
        de_load = read_load_file(Path(args.de_load_file), tz)
    else:
        # Standard-Datei im neighbors/out_load/2024
        de_load_path = Path(args.neighbor_load_dir) / "load_DE_LU_2024.csv"
        de_load = read_load_file(de_load_path, tz)

    # Ausrichten auf gemeinsamen Stundenindex (lokal)
    idx = de_load.index
    prices = prices.reindex(idx).interpolate(limit=2)
    de_gen = de_gen.reindex(idx)

    # Nicht-disponible + Pumpspeicher
    de_nd, de_load = build_nd_series(de_gen, tz, args.nd_include_bio, args.ps_mode, de_load)
    residual = (de_load - de_nd).clip(lower=0.0)

    # ---- Merit für DE (unit-wise SRMC)
    merit_records = compute_srmc_unitwise(units, prices, ef_by_fuel, varom)

    # ---- Nachbarpreise + Flüsse
    nprices = read_neighbor_prices(Path(args.neighbor_prices), tz).reindex(idx)
    flows   = read_flows(Path(args.flows), tz).reindex(idx).fillna(0.0)

    # ---- Mustrun (optional, einfacher Sockel auf Residual-Last via gen_quantile Braunkohle)
    if args.mustrun_mode == "gen_quantile":
        # Peak-Maske
        h_start, h_end = [int(x) for x in args.mustrun_peak_hours.split("-")]
        def is_peak(index):
            if h_start <= h_end:
                return (index.hour >= h_start) & (index.hour < h_end)
            else:
                return (index.hour >= h_start) | (index.hour < h_end)

        # Basierend auf realer Braunkohle-Gen (aus de_gen)
        lign_col = None
        for c in de_gen.columns:
            if "lignite" in c.lower() or "brown" in c.lower() or "braunkohle" in c.lower():
                lign_col = c; break
        if lign_col is not None:
            if args.mustrun_monthly:
                mus = []
                for (y,m),sub in de_gen[[lign_col]].groupby([de_gen.index.year, de_gen.index.month]):
                    pk = sub.loc[is_peak(sub.index), lign_col]
                    op = sub.loc[~is_peak(sub.index), lign_col]
                    qpk= pk.quantile(args.mustrun_quantile) if len(pk)>0 else 0.0
                    qop= op.quantile(args.mustrun_quantile) if len(op)>0 else 0.0
                    msk_pk = (de_gen.index.year==y)&(de_gen.index.month==m)&(is_peak(de_gen.index))
                    msk_op = (de_gen.index.year==y)&(de_gen.index.month==m)&(~is_peak(de_gen.index))
                    mus.append(pd.Series(np.where(msk_pk, qpk, np.where(msk_op, qop, 0.0)), index=de_gen.index))
                musrun = pd.concat(mus, axis=1).sum(axis=1)
            else:
                pk = de_gen.loc[is_peak(de_gen.index), lign_col]
                op = de_gen.loc[~is_peak(de_gen.index), lign_col]
                qpk= pk.quantile(args.mustrun_quantile) if len(pk)>0 else 0.0
                qop= op.quantile(args.mustrun_quantile) if len(op)>0 else 0.0
                musrun = pd.Series(np.where(is_peak(de_gen.index), qpk, qop), index=de_gen.index)
            # Mustrun als "belegt" anrechnen → Residual-Last sinkt entsprechend
            residual = (residual - musrun.clip(lower=0.0)).clip(lower=0.0)

    # ---- Hauptloop: pro Stunde Grenz-Einheit DE oder IMPORT
    out_rows = []
    dbg_rows = []

    zones = [c.replace("price_","") for c in nprices.columns if c.startswith("price_")]
    zones = [z for z in zones if z != "DE_LU"]

    for i, ts in enumerate(idx):
        rl = float(residual.iloc[i])
        # 1) Preis-Cluster
        price_de = float(nprices.loc[ts, "price_DE_LU"]) if "price_DE_LU" in nprices.columns else np.nan
        cluster = ["DE_LU"]
        for z in zones:
            col = f"price_{z}"
            if col in nprices.columns:
                if np.isfinite(price_de) and np.isfinite(nprices.loc[ts, col]) and abs(nprices.loc[ts, col]-price_de) <= args.epsilon:
                    cluster.append(z)

        # 2) DE-Grenzeinheit
        mer = merit_records[i]
        de_srmc = np.nan; de_fuel = None; de_eta = np.nan; mef_de = np.nan
        if isinstance(mer, dict) and "merit_srmc" in mer:
            srmc = mer["merit_srmc"]; cap = mer["merit_cap"]; fuels = mer["merit_fuel"]
            if rl <= 0 or len(srmc)==0:
                pass
            else:
                csum = np.cumsum(cap)
                pos = np.searchsorted(csum, rl, side="left")
                if pos >= len(srmc): pos = len(srmc)-1
                de_srmc = float(srmc[pos])
                de_fuel = str(fuels[pos])
                # Für MEF: nimm typischen eta der marginalen Fuel (median der Units dieser Fuel)
                # (alternativ: genaues eta der marginalen Einheit; hier nicht explizit gespeichert)
                # Approximationsweise:
                # -> Wir nehmen den Median-eta der Fuel aus units:
                eta_med = units.loc[units["fuel"]==de_fuel, "eta"].median() if de_fuel else np.nan
                de_eta = float(eta_med) if np.isfinite(eta_med) else 0.40
                mef_de = mef_gpkwh_for(de_fuel, de_eta, ef_by_fuel) if de_fuel else np.nan

        # 3) IMPORT-Kandidat bei Net-Import & Preiskopplung
        net_imp = float(flows.loc[ts, "net_import_total"]) if "net_import_total" in flows.columns else 0.0
        marginal_side = "DE"
        marginal_fuel = de_fuel
        marginal_eta  = de_eta
        marginal_srmc = de_srmc
        mef_gpkwh     = mef_de
        marginal_label= "DE"

        imp_details = []
        if (net_imp > 0.0) and (len(cluster) > 1):
            # MW-Anteile je Grenzland (nur Zonen im Cluster berücksichtigen)
            zone_mw = {}
            for z in zones:
                if z in cluster:
                    col = f"imp_{z.replace('DE_LU_','')}" if z.startswith("DE_LU_") else f"imp_{z.replace('_','_')}"
                    # Mapping der Codes im Flows-File
                    col = f"imp_{z.replace('DE_LU_','').replace('DK1','DK_1').replace('DK2','DK_2').replace('NO2','NO_2').replace('SE4','SE_4')}"
                    if col in flows.columns:
                        zone_mw[z] = float(flows.loc[ts, col])
            imp_sum = sum([v for v in zone_mw.values() if v>0])

            if imp_sum > 0:
                # SRMC je Fuel zur Orientierung (für Import-Fuel-Wahl pro Zone)
                # -> simple: wähle Fuel mit minimaler SRMC in dieser Stunde (ohne Unit-Stack auf Nachbarseite)
                colmap = {"Gas":"gas_eur_mwh_th","Steinkohle":"coal_eur_mwh_th","Braunkohle":"lignite_eur_mwh_th","Oel":"oil_eur_mwh_th"}
                fuel_srmc_now = {f: float(prices.loc[ts, colmap[f]] + prices.loc[ts,"co2_eur_t"]*ef_by_fuel[f]) / 0.45 + varom[f] for f in FOSSIL}  # η~0.45 als pragmatischer Mix

                # Details fürs Labeln
                for z, mw in zone_mw.items():
                    if mw <= 0: 
                        continue
                    # Wähle Fuel mit minimaler SRMC
                    ef = min(fuel_srmc_now, key=fuel_srmc_now.get)
                    srmc_z = fuel_srmc_now[ef]
                    eta_z  = 0.45
                    mef_z  = mef_gpkwh_for(ef, eta_z, ef_by_fuel)
                    imp_details.append( (z, mw, ef, eta_z, srmc_z, mef_z) )

                # Dominante Zone / Fuel
                import_fuel_final = None; import_srmc_final=None; import_mef_final=None; import_label=None
                if args.import_label_mode != "mix" and imp_details:
                    from collections import defaultdict
                    zone_mw_agg = defaultdict(float)
                    fuel_mw_agg = defaultdict(float)
                    fuel_val = {}
                    for z, mw, ef, eta_z, srmc_z, mef_z in imp_details:
                        zone_mw_agg[z] += mw
                        fuel_mw_agg[ef] += mw
                        fuel_val[ef] = (srmc_z, mef_z, z)
                    if args.import_label_mode == "dominant_zone":
                        dom_z = max(zone_mw_agg, key=zone_mw_agg.get)
                        if zone_mw_agg[dom_z]/imp_sum >= args.dominant_share_zone:
                            rec = next(r for r in imp_details if r[0]==dom_z)
                            _, mw, ef, eta_z, srmc_z, mef_z = rec
                            import_fuel_final, import_srmc_final, import_mef_final, import_label = ef, srmc_z, mef_z, f"{dom_z}({ef})"
                    else:
                        dom_f = max(fuel_mw_agg, key=fuel_mw_agg.get)
                        if fuel_mw_agg[dom_f]/imp_sum >= args.dominant_share_fuel:
                            srmc_z, mef_z, z0 = fuel_val[dom_f]
                            import_fuel_final, import_srmc_final, import_mef_final, import_label = dom_f, srmc_z, mef_z, f"{z0}({dom_f})"

                # Entscheidung IMPORT vs DE
                if import_srmc_final is not None:
                    # Wenn Import-SRMC >= DE-SRMC → Import kann marginal sein (gleiches Preisniveau/Cluster)
                    if (not np.isfinite(marginal_srmc)) or (import_srmc_final >= marginal_srmc):
                        marginal_side  = "IMPORT"
                        marginal_fuel  = import_fuel_final or "mix"
                        marginal_eta   = np.nan
                        marginal_srmc  = float(import_srmc_final)
                        mef_gpkwh      = float(import_mef_final)
                        marginal_label = import_label or "IMPORT(mix)"

        out_rows.append({
            "time": ts, "marginal_side": marginal_side, "marginal_fuel": marginal_fuel,
            "mef_g_per_kwh": mef_gpkwh
        })

        dbg_rows.append({
            "time": ts,
            "price_DE": price_de,
            "cluster": ",".join(cluster),
            "residual_MW": rl,
            "DE_srmc": de_srmc,
            "DE_fuel": de_fuel,
            "DE_eta": de_eta,
            "marginal_side": marginal_side,
            "marginal_fuel": marginal_fuel,
            "marginal_label": marginal_label,
            "marginal_srmc": marginal_srmc,
            "mef_g_per_kwh": mef_gpkwh,
            "net_import_total_MW": net_imp
        })

    # ---- Outputs
    tsfmt = "%Y-%m-%d %H:%M:%S%z"
    mef = pd.DataFrame(out_rows).copy()
    mef["time"] = pd.to_datetime(mef["time"]).dt.tz_convert(tz)
    mef = mef.set_index("time").sort_index()
    mef.to_csv(outdir / "mef_track_c_2024.csv", index=True)

    dbg = pd.DataFrame(dbg_rows).copy()
    dbg["time"] = pd.to_datetime(dbg["time"]).dt.tz_convert(tz)
    dbg = dbg.set_index("time").sort_index()
    # hänge die imp_* Spalten aus flows für Diagnose an:
    imp_cols = [c for c in flows.columns if c.startswith("imp_")] + ["net_import_total"]
    dbg = dbg.join(flows[imp_cols], how="left")
    dbg.to_csv(outdir / "_debug_hourly.csv", index=True)

    # Mini-Report
    print(f"[INFO] Stunden: {len(mef)} | Mittel-MEF: {mef['mef_g_per_kwh'].mean():.1f} g/kWh")
    print("Anteile marginal_fuel:\n", mef["marginal_fuel"].value_counts(normalize=True).round(3))
    print("Anteile marginal_side:\n", mef["marginal_side"].value_counts(normalize=True).round(3))
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'} & {outdir/'_debug_hourly.csv'}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
