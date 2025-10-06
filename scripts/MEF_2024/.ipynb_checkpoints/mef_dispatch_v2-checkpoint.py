#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch-Backcast (DE/LU 2024) mit:
- DE: imputierte Wirkungsgrade (eta_col)
- Nachbarn: Effizienz-Spannen oder MC-Sampling (Normalverteilung, min/max-Clipping)
- Import-Fuel-Attribution: dominante Grenze / dominanter Fuel (Schex-Mehrheitslogik)
- Optional: Kapazitätsmaske je Zone×Fuel (2024 installierte Leistung)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

TZ = "Europe/Berlin"

# ----------------------------- CLI -------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track C v2 – MEF Backcast mit Import-Fuel-Attribution & Nachbar-η-Verteilung")

    # DE-Fleet + Preise/Flows
    p.add_argument("--fleet", required=True, help="CSV: DE-Fleet (z. B. Kraftwerke_eff_binned.csv)")
    p.add_argument("--eta_col", default="Imputed_Effizienz_binned", help="Spalte mit imputierter Effizienz")
    p.add_argument("--fuel_prices", required=True, help="CSV: prices_2024.csv (€/MWh_th, EUA €/t)")
    p.add_argument("--flows", required=True, help="CSV: flows_scheduled_DE_LU_2024_net.csv (mit imp_* und evtl. net_import_total)")
    p.add_argument("--start", default=None, help="Start (Europe/Berlin), z. B. 2024-01-01T00:00:00")
    p.add_argument("--end",   default=None, help="Ende exklusiv (Europe/Berlin), z. B. 2025-01-01T00:00:00")
    p.add_argument("--neighbor_fleet", default=None,
               help="CSV mit Nachbar-Fleet (mind. zone,fuel,eta oder heat_rate,capacity_mw). Wird genutzt für zonale η-Distributionen und Kapazitätsmaske.")

    # Nachbarn: Gen/Load/Prices
    p.add_argument("--neighbor_gen_dir",   required=True, help="Dir mit actual_gen_<ZONE>_2024.csv")
    p.add_argument("--neighbor_load_dir",  required=True, help="Dir mit load_<ZONE>_2024.csv")
    p.add_argument("--neighbor_prices",    required=True, help="CSV: price_DE_LU, price_AT, ...")
    p.add_argument("--epsilon", type=float, default=0.01, help="Preis-Kopplungs-Schwelle in €/MWh")
    p.add_argument("--price_anchor", choices=["off","closest","threshold"], default="closest",
                   help="Preis-Nähe beim Seitenentscheid berücksichtigen")
    p.add_argument("--price_tol", type=float, default=30.0,
                   help="Zulässige Abweichung SRMC vs. Preis (€/MWh) bei --price_anchor threshold")

    # Nachbarn: Effizienz-Modelle
    p.add_argument("--nei_eta_mode", choices=["mean","bounds","mc"], default="mean",
                   help="mean=Flottenmittel; bounds=auch min/max debuggen; mc=Normalverteilung mit Clipping")
    p.add_argument("--nei_eta_json", default=None,
                   help="JSON mit η-Parametern je Fuel (global oder je Zone). Siehe default_dists unten.")
    p.add_argument("--nei_mc_draws", type=int, default=50, help="Anzahl Ziehungen je Fuel/Zeitpunkt bei --nei_eta_mode mc")
    p.add_argument("--neighbor_capacity", default=None,
                   help="Optionale CSV: zone,fuel,capacity_mw → maskiert Fuels ohne installierte Leistung")

    # Dispatch-Details
    p.add_argument("--varom_json", default=None, help="JSON: {tech_or_fuel: varOM_eur_per_mwh_el}")
    p.add_argument("--therm_avail", type=float, default=0.95, help="Verfügbarkeit thermischer Einheiten in DE (0..1)")
    p.add_argument("--mustrun_mode", choices=["off","capacity","gen_quantile"], default="gen_quantile")
    p.add_argument("--mustrun_lignite_q", type=float, default=0.20, help="Nur für capacity: Anteil der Lignite-Kapazität")
    p.add_argument("--mustrun_quantile",  type=float, default=0.20, help="Nur für gen_quantile: Quantil der realen Lignite-Gen")
    p.add_argument("--mustrun_peak_hours", default="08-20", help="Peakfenster lokal, z. B. 08-20")
    p.add_argument("--mustrun_monthly", action="store_true", help="Quantile je Monat getrennt")
    p.add_argument("--year", type=int, default=2024)

    # Output
    p.add_argument("--outdir", required=True, help="Output-Ordner")
    return p

# -------------------------- Helper: Zeit & IO --------------------------------

def parse_ts(s: pd.Series) -> pd.DatetimeIndex:
    ser_utc = pd.to_datetime(s, errors="coerce", utc=True)
    return pd.DatetimeIndex(ser_utc).tz_convert(TZ)

def force_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame benötigt DatetimeIndex.")
    return df.resample("1h").mean() if how == "mean" else df.resample("1h").sum()

def read_csv_auto_time(path: str, time_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in time_cols if c in df.columns), df.columns[0])
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_fuel_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    if "net_import_total" not in df.columns:
        imp_cols = [c for c in df.columns if c.startswith("imp_")]
        if imp_cols:
            df["net_import_total"] = df[imp_cols].sum(axis=1)
        else:
            df["net_import_total"] = 0.0
    return df

def load_neighbor_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_neighbor_load(path_dir: str, zone: str) -> pd.Series:
    candidates = list(Path(path_dir).glob(f"load_{zone}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Load-CSV fehlt: load_{zone}_2024*.csv in {path_dir}")
    df = read_csv_auto_time(str(candidates[0]), ["timestamp_cec","timestamp","time","timestamp_brussels","timestamp_utc"])
    load_col = next((c for c in df.columns if "ActualTotalLoad" in c or "load" in c.lower()), df.columns[0])
    return pd.to_numeric(df[load_col], errors="coerce")

NEIGHBOR_TECHS = [
    "Fossil Gas","Fossil Hard coal","Fossil Oil","Fossil Brown coal/Lignite",
    "Nuclear","Biomass",
    "Hydro Run-of-river and poundage","Hydro Water Reservoir","Hydro Pumped Storage",
    "Wind Onshore","Wind Offshore","Solar","Waste",
]

def load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    candidates = list(Path(path_dir).glob(f"actual_gen_{zone}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Gen-CSV fehlt: actual_gen_{zone}_2024*.csv in {path_dir}")
    df_raw = pd.read_csv(candidates[0])
    tcol = next((c for c in ["timestamp_cec","timestamp","time","datetime"] if c in df_raw.columns), df_raw.columns[0])
    df_raw.index = parse_ts(df_raw[tcol]); df_raw = df_raw.drop(columns=[tcol])
    keep = [c for c in df_raw.columns for tech in NEIGHBOR_TECHS if c==tech or c.startswith(tech)]
    df = df_raw[keep].copy()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    # Doubletten gleicher Technamen aggregieren
    col_map, agg = {}, {}
    for c in df.columns:
        key = next((tech for tech in NEIGHBOR_TECHS if c==tech or c.startswith(tech)), c)
        col_map.setdefault(key, []).append(c)
    for key, cols in col_map.items():
        agg[key] = df[cols].sum(axis=1)
    return pd.DataFrame(agg, index=df.index).sort_index()

# -------------------------- Mappings & Defaults ------------------------------

EF_LOOKUP_T_PER_MWH_TH = {
    "Erdgas": 0.201,
    "Steinkohle": 0.335,
    "Braunkohle": 0.383,
    "Heizöl schwer": 0.288,
    "Heizöl leicht / Diesel": 0.266,
}
FOSSIL_TECH_TO_FUEL = {
    "Fossil Gas": ("gas", "Erdgas"),
    "Fossil Hard coal": ("coal", "Steinkohle"),
    "Fossil Brown coal/Lignite": ("lignite", "Braunkohle"),
    "Fossil Oil": ("oil", "Heizöl schwer"),
}
PRICE_COLS = ["gas_eur_mwh_th","coal_eur_mwh_th","lignite_eur_mwh_th","oil_eur_mwh_th","co2_eur_t"]

def _norm(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower().strip()
    t = t.replace("-", " ").replace("/", " ").replace(",", " ")
    return " ".join(t.split())


def map_fuel_to_price_and_ef(raw: str):
    t = _norm(raw)
    if any(k in t for k in ["erdgas","erdölgas","erdolgas","fossilgas"," gas"]): return ("gas","Erdgas")
    if "steinkohle" in t or "hard coal" in t:                                   return ("coal","Steinkohle")
    if "braunkohle" in t or "lignite" in t:                                     return ("lignite","Braunkohle")
    if "heizöl" in t or "heizoel" in t or "diesel" in t or " oil" in t or "öl" in t or "oel" in t:
        return ("oil","Heizöl leicht / Diesel" if ("leicht" in t or "diesel" in t) else "Heizöl schwer")
    return (None,None)

# -------------------------- Fleet & SRMC (DE) --------------------------------

def read_csv_smart(path: str, min_cols: int = 3) -> pd.DataFrame:
    seps = [",",";","\t","|"]
    encs = ["utf-8-sig","cp1252","latin1"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] >= min_cols: return df
            except Exception as e:
                last_err = e; continue
    raise RuntimeError(f"CSV nicht lesbar: {path} – letzter Fehler: {last_err}")
def neighbor_marginal_from_residual(
    load_val: float,
    gen_row: pd.Series,
    fuel_prices_row: pd.Series,
    zone: str,
    dists: Dict[str, dict],
    mode: str = "mean",
    draws: int = 50,
    capacity_mask: Optional[Dict[Tuple[str,str], float]] = None
) -> Tuple[Optional[str], Optional[float], Optional[float], dict]:
    """
    Bestimmt den marginalen Fuel in einer Nachbarzone per Residuallast.
    - load_val: Last der Zone in MW
    - gen_row: Zeitreihe der tatsächlichen Generation (MW pro Fuel)
    - fuel_prices_row: Brennstoffpreise + CO2 für den Zeitschritt
    """
    debug = {}
    # Nicht-disponible abziehen (Wind, PV, Hydro, Biomasse, Nuclear …)
    nondisp = [
      "Nuclear","Solar","Wind Onshore","Wind Offshore",
      "Hydro Run-of-river and poundage","Hydro Pumped Storage",  # <- beides NICHT regelbar
      "Biomass","Waste"
    ]

    nd_val = gen_row.get(nondisp, pd.Series(0.0)).sum()
    residual = max(load_val - nd_val, 0.0)
    # zusätzlich: regelbare Wasserkraft als steuerbare Technologie
    if "Hydro Water Reservoir" in gen_row.index:
        cap = gen_row["Hydro Water Reservoir"]
        if np.isfinite(cap) and cap > 1.0:
            # Kein Brennstoff/CO2 → operative SRMC ~ 0; MEF ~ 0 g/kWh
            ef_name = "Hydro Reservoir"
            srmc    = 0.0
            eta_eff = 1.0
            cand_units.append((ef_name, srmc, eta_eff, cap))

    # Kandidaten = alle *disponiblen* Techs mit Kapazität >0
    cand_units = []
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in gen_row.index: 
            continue
        cap = gen_row[tech]
        if not (np.isfinite(cap) and cap > 1.0):
            continue

        fuel_th = fuel_prices_row.get(f"{pk}_eur_mwh_th", np.nan)
        co2     = fuel_prices_row.get("co2_eur_t", np.nan)
        if not (np.isfinite(fuel_th) and np.isfinite(co2)):
            continue

        d = dists.get(zone, {}).get(ef_name, DEFAULT_NEI_DISTS.get(ef_name))
        m, s, lo, hi = d["mean"], d["std"], d["min"], d["max"]

        if mode == "mean":
            eta_eff = m
        elif mode == "mc":
            eta_eff = float(np.mean(truncated_normal(m, s, lo, hi, size=draws)))
        else:  # bounds
            eta_eff = m

        srmc = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_eff,1e-6)
        cand_units.append((ef_name, srmc, eta_eff, cap))

    if not cand_units or residual <= 0:
        return None, None, None, debug

    # Merit-Order: sortiere nach SRMC
    cand_units = sorted(cand_units, key=lambda x: x[1])
    cumcap, cum = [], 0.0
    for ef_name, srmc, eta_eff, cap in cand_units:
        cum += cap
        cumcap.append((ef_name, srmc, eta_eff, cap, cum))
        if cum >= residual:
            debug["residual"] = residual
            debug["cumcap"]   = cum
            return ef_name, srmc, eta_eff, debug

    # Falls Residual > Summe → letzter Fuel marginal
    ef_name, srmc, eta_eff, _cap, _cum = cumcap[-1]
    return ef_name, srmc, eta_eff, debug

def load_fleet(path: str, eta_col: Optional[str]) -> pd.DataFrame:
    df = read_csv_smart(path, min_cols=5)
    pcol = next((c for c in ["MW Nettonennleistung der Einheit","Leistung_MW","Nettonennleistung der Einheit","Nettonennleistung","p_mw","P_MW"] if c in df.columns), None)
    fcol = next((c for c in ["Hauptbrennstoff der Einheit","Energieträger","Hauptbrennstoff","Brennstoff","fuel","Fuel"] if c in df.columns), None)
    idcol = next((c for c in df.columns if "MaStR" in c or "Mastr" in c or "unit_id" in c.lower()), df.columns[0])
    namecol = next((c for c in df.columns if "Anzeige-Name" in c or "Name der Einheit" in c or "Name" in c), idcol)
    if not pcol or not fcol: raise ValueError("Fleet: Leistungs- oder Brennstoffspalte fehlt.")

    eta_cols = [eta_col] if eta_col else []
    eta_cols += ["Effizienz","Effizienz_imputiert","eta","Eta","wirkungsgrad","Imputed_Effizienz_binned"]
    use_eta = next((c for c in eta_cols if c in df.columns), None)
    if use_eta is None: raise ValueError(f"Keine Effizienzspalte gefunden: {eta_cols}")

    out = pd.DataFrame({
        "unit_id": df[idcol].astype(str),
        "unit_name": df[namecol].astype(str),
        "fuel_raw": df[fcol].astype(str),
        "eta": pd.to_numeric(df[use_eta], errors="coerce"),
        "p_mw": pd.to_numeric(df[pcol], errors="coerce"),
    })

    price_key, ef_key = [], []
    for f in out["fuel_raw"]:
        pk, ek = map_fuel_to_price_and_ef(f)
        price_key.append(pk); ef_key.append(ek)
    out["price_key"] = price_key; out["ef_key"] = ef_key
    out = out[(out["price_key"].notna()) & (out["ef_key"].notna())].copy()

    # Einheiten prüfen und plausibilisieren
    eta_clean = pd.to_numeric(out["eta"], errors="coerce").to_numpy()
    if np.nanmedian(eta_clean) > 1.5:  # Prozent → Anteil
        eta_clean = eta_clean/100.0
    eta_clean = np.clip(eta_clean, 0.20, 0.65)  # Thermik-Range
    out["eta"] = eta_clean
    out["available_mw"] = pd.to_numeric(out["p_mw"], errors="coerce").fillna(0.0).clip(lower=0).astype("float32")
    return out.dropna(subset=["eta"])

def compute_unit_srmc_series(fleet: pd.DataFrame, fuel_prices: pd.DataFrame, varom_map: Dict[str, float]) -> Dict[str, pd.Series]:
    srmc_by_unit = {}
    co2 = fuel_prices["co2_eur_t"]
    for _, r in fleet.iterrows():
        price_col = f"{r['price_key']}_eur_mwh_th"
        fuel_th = fuel_prices[price_col]
        ef_th   = EF_LOOKUP_T_PER_MWH_TH.get(r["ef_key"], 0.30)
        eta     = max(float(r["eta"]), 1e-6)
        varom   = varom_map.get(r["ef_key"], varom_map.get(r["price_key"], 0.0))
        srmc = (fuel_th + co2 * ef_th) / eta + varom
        srmc_by_unit[r["unit_id"]] = srmc.astype("float32")
    return srmc_by_unit

# ------------------- Nachbarn: η-Verteilungen & Import-Fuel -------------------

DEFAULT_NEI_DISTS = {
    # Globale Defaults – können via --nei_eta_json überschrieben werden (optional je Zone)
    # Werte grob aus Literaturspannen; std ~ Mittelwert/12
    "Erdgas":      {"mean": 0.52, "std": 0.043, "min": 0.35, "max": 0.60},  # CCGT/OCGT-Mix
    "Steinkohle":  {"mean": 0.41, "std": 0.030, "min": 0.34, "max": 0.45},  # sub/superkritisch
    "Braunkohle":  {"mean": 0.40, "std": 0.028, "min": 0.33, "max": 0.43},
    "Heizöl schwer":{"mean": 0.36, "std": 0.020, "min": 0.32, "max": 0.40},
}
def _norm_zone(z: str) -> str:
    """Harmonisiere Zonencodes wie in price_/load_/gen-Dateien (AT, BE, CH, CZ, DK_1→DK_1 etc.)."""
    z = str(z or "").strip()
    return z.replace("-", "_").upper()

def _map_neighbor_fuel(s: str) -> Optional[str]:
    """
    Mappt beliebige Fuel-Bezeichnungen auf die in deinem Skript verwendeten EF-Namen:
    {'Erdgas','Steinkohle','Braunkohle','Heizöl schwer','Heizöl leicht / Diesel'}.
    """
    t = _norm(s)
    if any(k in t for k in ["gas","erdgas","ccgt","ocgt","erdölgas","erdolgas","fossil gas"]):
        return "Erdgas"
    if any(k in t for k in ["hard coal","steinkohle","coal","kohlekraft"]):
        return "Steinkohle"
    if any(k in t for k in ["lignite","braunkohle","brown coal"]):
        return "Braunkohle"
    if any(k in t for k in ["diesel","leicht","light oil"]):
        return "Heizöl leicht / Diesel"
    if any(k in t for k in ["oil","heizöl","heizoel","heavy oil","hfo"]):
        return "Heizöl schwer"
    return None

def _eta_from_row(r) -> Optional[float]:
    """
    Liefert eine elektrische Effizienz (Anteil 0..1). Akzeptiert:
    - 'eta' oder 'effizienz' in 0..1 oder in Prozent (0..100)
    - 'heat_rate' (z.B. kJ/kWh_el oder GJ/MWh_el) -> eta = 3.6 MJ/kWh / HR
    Clipt auf realistische Range.
    """
    cand_cols = [c for c in r.index if str(c).lower() in ("eta","effizienz","wirkungsgrad","eta_el")]
    if cand_cols:
        val = pd.to_numeric(r[cand_cols[0]], errors="coerce")
        if pd.isna(val): return None
        if val > 1.5:  # Prozent
            val = val / 100.0
        return float(np.clip(val, 0.20, 0.65))
    # Heat rate?
    for c in r.index:
        lc = str(c).lower()
        if "heat_rate" in lc or "heatrate" in lc or ("hr" == lc):
            hr = pd.to_numeric(r[c], errors="coerce")
            if not pd.isna(hr) and hr > 0:
                # Versuche Einheiten robust: kJ/kWh → hr ~ 10000..15000; GJ/MWh → hr ~ 9..12
                # Wir normieren auf MJ/kWh: 1 kWh_el = 3.6 MJ_el → eta = 3.6 / (HR_MJ_per_kWh)
                # Falls GJ/MWh: 1 GJ/MWh = 1 MJ/kWh
                HR = float(hr)
                if HR > 2000: # kJ/kWh
                    HR = HR / 1000.0
                elif HR < 50: # GJ/MWh (≈ MJ/kWh)
                    HR = HR * 1.0
                eta = 3.6 / HR
                return float(np.clip(eta, 0.20, 0.65))
    return None

def load_neighbor_fleet(path: str) -> tuple[dict, dict]:
    """
    Liest eine Fleet-CSV der Nachbarn und baut:
    - nei_dists_zonal: {ZONE: {Fuel: {'mean':m,'std':s,'min':lo,'max':hi}}}
    - cap_mask: {(ZONE,Fuel): capacity_mw}
    Erwartete Spalten (flexibel): zone|bidding_zone, fuel|brennstoff|energieträger, eta|effizienz|heat_rate, capacity_mw|leistung_mw
    """
    df = read_csv_smart(path, min_cols=3)
    cols = {c.lower(): c for c in df.columns}
    zcol = cols.get("zone") or cols.get("bidding_zone") or cols.get("country") or list(df.columns)[0]
    fcol = cols.get("fuel") or cols.get("brennstoff") or cols.get("energieträger") or cols.get("energietraeger")
    pcol = cols.get("capacity_mw") or cols.get("leistung_mw") or cols.get("mw") or None

    if fcol is None:
        raise ValueError("neighbor_fleet: keine 'fuel'/'Brennstoff'-Spalte gefunden.")
    if zcol is None:
        raise ValueError("neighbor_fleet: keine 'zone'/'bidding_zone'-Spalte gefunden.")

    df["_zone"] = df[zcol].map(_norm_zone)
    df["_fuel"] = df[fcol].map(_map_neighbor_fuel)
    if pcol is not None:
        df["_cap"] = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0)
    else:
        df["_cap"] = 0.0

    # Eta ableiten
    df["_eta"] = df.apply(_eta_from_row, axis=1)

    df = df[ df["_fuel"].notna() & df["_zone"].notna() ].copy()
    # Kapazitätsmaske
    cap_mask = {(z,f): float(sub["_cap"].sum()) for (z,f), sub in df.groupby(["_zone","_fuel"], dropna=True)}

    # Zonen-spezifische η-Parameter
    nei_dists_zonal: dict = {}
    for (z, f), sub in df.groupby(["_zone","_fuel"], dropna=True):
        etas = pd.to_numeric(sub["_eta"], errors="coerce").dropna()
        if len(etas) == 0:
            continue
        m  = float(etas.mean())
        sd = float(np.std(etas)) if len(etas) > 1 else max(0.02, m/12.0)
        lo = float(np.quantile(etas, 0.05)) if len(etas) >= 5 else max(0.20, m - 2*sd)
        hi = float(np.quantile(etas, 0.95)) if len(etas) >= 5 else min(0.65, m + 2*sd)
        nei_dists_zonal.setdefault(z, {})[f] = {"mean": m, "std": sd, "min": lo, "max": hi}

    return nei_dists_zonal, cap_mask

def truncated_normal(mean, std, lo, hi, size):
    rnd = np.random.normal(mean, std, size=size)
    return np.clip(rnd, lo, hi)

def cluster_zones_by_price(nei_prices: pd.DataFrame, eps: float) -> Dict[pd.Timestamp, List[str]]:
    zones = [c.replace("price_", "") for c in nei_prices.columns if c.startswith("price_")]
    clusters = {}
    for t, row in nei_prices.iterrows():
        p_de = row.get("price_DE_LU", np.nan)
        cluster = ["DE_LU"]
        if not pd.isna(p_de):
            for z in zones:
                if z == "DE_LU": continue
                pz = row.get(f"price_{z}", np.nan)
                if not pd.isna(pz) and abs(pz - p_de) <= eps:
                    cluster.append(z)
        clusters[t] = cluster
    return clusters

def neighbor_marginal_from_gen_dist(
    row_gen: pd.Series,
    fuel_prices_row: pd.Series,
    zone: str,
    dists: Dict[str, dict],
    mode: str = "mean",
    draws: int = 50,
    capacity_mask: Optional[Dict[Tuple[str,str], float]] = None
) -> Tuple[Optional[str], Optional[float], Optional[float], dict]:
    """
    Bestimmt pro Zone (in einer Stunde) einen fossilen marginalen Kandidaten inkl. SRMC/η.
    Gibt zusätzlich ein debug dict zurück.
    """
    cand = []
    debug = {}
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in row_gen.index: 
            continue
        mw = row_gen.get(tech, 0.0)
        if not (np.isfinite(mw) and mw > 1.0):
            continue

        # Maskiere Fuels ohne installierte Kapazität (optional)
        if capacity_mask is not None:
            cap = capacity_mask.get((zone, ef_name), None)
            if cap is not None and cap <= 1.0:
                continue

        fuel_th = fuel_prices_row.get(f"{pk}_eur_mwh_th", np.nan)
        co2     = fuel_prices_row.get("co2_eur_t", np.nan)
        if not (np.isfinite(fuel_th) and np.isfinite(co2)):
            continue

        # Hole Zonen- oder Global-Distribution
        d = dists.get(zone, {}).get(ef_name, None) or dists.get(ef_name, None) or DEFAULT_NEI_DISTS[ef_name]
        m, s, lo, hi = d["mean"], d["std"], d["min"], d["max"]

        if mode == "mean":
            eta_eff = m
            srmc    = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_eff,1e-6)
            cand.append((ef_name, srmc, eta_eff, tech))
            debug[ef_name] = {"eta_used": eta_eff, "srmc": srmc, "mode": "mean"}

        elif mode == "bounds":
            eta_low, eta_high = lo, hi
            srmc_low  = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_high,1e-6)  # effizient = niedrige SRMC
            srmc_high = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_low,1e-6)   # ineffizient = hohe SRMC
            # Für Auswahl nehmen wir den Mittelwert
            eta_eff = m
            srmc_mid= (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_eff,1e-6)
            cand.append((ef_name, srmc_mid, eta_eff, tech))
            debug[ef_name] = {"eta_mean": m, "eta_min": lo, "eta_max": hi,
                              "srmc_mid": srmc_mid, "srmc_low": srmc_low, "srmc_high": srmc_high,
                              "mode": "bounds"}

        else:  # mc
            etas = truncated_normal(m, s, lo, hi, size=draws)
            srmc_draws = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / np.maximum(etas,1e-6)
            srmc_avg = float(np.mean(srmc_draws))
            eta_eff  = float(np.mean(etas))
            cand.append((ef_name, srmc_avg, eta_eff, tech))
            debug[ef_name] = {"eta_mean": float(np.mean(etas)), "eta_std": float(np.std(etas)),
                              "srmc_avg": srmc_avg, "draws": draws, "mode": "mc"}

    if not cand:
        return None, None, None, debug

    # marginal = höchster SRMC
    ef_name, srmc, eta_eff, tech = sorted(cand, key=lambda x: x[1])[-1]
    return ef_name, srmc, eta_eff, debug

# ------------------------------ Main -----------------------------------------

def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Preise
    fuel_prices = load_fuel_prices(args.fuel_prices)
    miss = [c for c in PRICE_COLS if c not in fuel_prices.columns]
    if miss: raise ValueError(f"Fehlende Preisspalten in prices: {miss}")

    # 2) DE-Fleet & SRMC
    fleet_all = load_fleet(args.fleet, args.eta_col)
    fleet = fleet_all[fleet_all["ef_key"].isin(EF_LOOKUP_T_PER_MWH_TH.keys())].copy().reset_index(drop=True)

    varom_map = {}
    if args.varom_json and Path(args.varom_json).exists():
        varom_map = json.load(open(args.varom_json, "r", encoding="utf-8"))

    srmc_by_unit = compute_unit_srmc_series(fleet, fuel_prices, varom_map)
    units = list(srmc_by_unit.keys())
    SRMC = pd.concat([srmc_by_unit[u].rename(u) for u in units], axis=1).astype("float32")

    fleet_idxed = fleet.set_index("unit_id")
    common = [u for u in SRMC.columns if u in fleet_idxed.index]
    SRMC = SRMC.loc[:, common]
    fleet_idxed = fleet_idxed.loc[common]
    units    = list(SRMC.columns)
    cap_base = fleet_idxed["available_mw"].astype("float32").to_numpy()
    eta_arr  = fleet_idxed["eta"].astype("float32").to_numpy()
    ef_keys  = fleet_idxed["ef_key"].astype(str).to_numpy()

    # 3) Flows & Nachbarpreise
    flows = load_flows(args.flows)
    nei_prices = load_neighbor_prices(args.neighbor_prices)
    clusters = cluster_zones_by_price(nei_prices, args.epsilon)

    # 4) Nachbar-Gen/Load + DE/LU
    zones = sorted([c.replace("price_", "") for c in nei_prices.columns if c.startswith("price_")])
    load_by_zone, gen_by_zone = {}, {}
    for z in zones:
        try: load_by_zone[z] = load_neighbor_load(args.neighbor_load_dir, z)
        except Exception: pass
        try: gen_by_zone[z] = load_neighbor_gen(args.neighbor_gen_dir, z)
        except Exception: pass

    if "DE_LU" not in load_by_zone: raise RuntimeError("load_DE_LU_2024.csv fehlt.")
    if "DE_LU" not in gen_by_zone:  raise RuntimeError("actual_gen_DE_LU_2024.csv fehlt.")
    de_load = load_by_zone["DE_LU"]; de_gen = gen_by_zone["DE_LU"]

    # Nicht-disponible in DE (PS-Gen NICHT abziehen; Pump-Last steckt in de_load)
    nondisp_cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage","Biomass","Waste"]
    nd_present = [c for c in nondisp_cols if c in de_gen.columns]
    de_nondisp = de_gen[nd_present].sum(axis=1).reindex(de_load.index).fillna(0.0)

    # 5) Zeitfenster
    idx_common = sorted(de_load.index.intersection(fuel_prices.index).intersection(flows.index).intersection(nei_prices.index))
    def _to_berlin(ts_str: Optional[str]):
        if ts_str is None: return None
        ts = pd.Timestamp(ts_str)
        return ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
    start = _to_berlin(args.start) or idx_common[0]
    end   = _to_berlin(args.end)   or (idx_common[-1] + pd.Timedelta(hours=1))
    idx   = [t for t in idx_common if (t >= start and t < end)]
    print(f"[INFO] Stunden im Lauf: {len(idx)} | Fenster: {start} .. {end} (exkl.)")

    # 6) Mustrun (zeitvariabel aus Lignite-Gen)
    h_start, h_end = [int(x) for x in args.mustrun_peak_hours.split("-")]
    def is_peak(ix: pd.DatetimeIndex):
        if h_start <= h_end:
            return (ix.hour >= h_start) & (ix.hour < h_end)
        else:
            return (ix.hour >= h_start) | (ix.hour < h_end)

    lign_profile = pd.Series(0.0, index=pd.DatetimeIndex(idx))
    if args.mustrun_mode == "gen_quantile" and "Fossil Brown coal/Lignite" in gen_by_zone["DE_LU"].columns:
        lign = gen_by_zone["DE_LU"]["Fossil Brown coal/Lignite"].reindex(lign_profile.index)
        out = pd.Series(index=lign.index, dtype="float64")
        if args.mustrun_monthly:
            for mth, sub in lign.groupby(lign.index.month):
                pk = is_peak(sub.index); op = ~pk
                qpk = np.nanquantile(sub[pk], args.mustrun_quantile) if pk.any() else 0.0
                qop = np.nanquantile(sub[op], args.mustrun_quantile) if op.any() else 0.0
                mask_m = (lign.index.month == mth)
                out.loc[mask_m &  is_peak(lign.index)] = qpk
                out.loc[mask_m & ~is_peak(lign.index)] = qop
        else:
            pk = is_peak(lign.index); op = ~pk
            qpk = np.nanquantile(lign[pk], args.mustrun_quantile) if pk.any() else 0.0
            qop = np.nanquantile(lign[op], args.mustrun_quantile) if op.any() else 0.0
            out[ pk] = qpk; out[~pk] = qop
        lign_profile = out.fillna(0.0).clip(lower=0.0)

    # Lignite-Anteile zur proportionalen Verteilung (für gen_quantile)
    lign_mask  = (ef_keys == "Braunkohle")
    lign_total = float(cap_base[lign_mask].sum())
    lign_share = np.zeros_like(cap_base)
    if lign_total > 0:
        lign_share[lign_mask] = cap_base[lign_mask] / lign_total

    # 7) Nachbar-η-Parameter / Kapazitätsmaske
    # Start mit globalen Defaults
    nei_dists = DEFAULT_NEI_DISTS.copy()
    
    # 3.1 Optional: zonal/fuel-spezifische Distanzen & Kapazität aus Nachbar-Fleet
    fleet_dists = {}
    cap_mask = None
    if args.neighbor_fleet and Path(args.neighbor_fleet).exists():
        fleet_dists, cap_mask_from_fleet = load_neighbor_fleet(args.neighbor_fleet)
        # Merge: zonale Distanzen > globale Defaults
        # Struktur: { "NL": {"Erdgas": {...}}, "AT": {"Steinkohle": {...}}, ... }
        for z, fuels in fleet_dists.items():
            nei_dists.setdefault(z, {})
            for f, d in fuels.items():
                nei_dists[z][f] = d
        cap_mask = cap_mask_from_fleet
    
    # 3.2 Optional: JSON-Overrides anwenden (überschreiben Fleet/Defaults)
    if args.nei_eta_json and Path(args.nei_eta_json).exists():
        with open(args.nei_eta_json, "r", encoding="utf-8") as f:
            user_d = json.load(f)
        # erlaubt: global je Fuel ODER je Zone
        # { "Erdgas": {...} } oder { "NL": {"Erdgas": {...}} }
        for k, v in user_d.items():
            if isinstance(v, dict) and all(isinstance(vv, dict) for vv in v.values()):
                # zonal
                nei_dists.setdefault(k, {}).update(v)
            else:
                # global fuel
                nei_dists[k] = v if isinstance(v, dict) else v
    
    # 3.3 Optional: Kapazitätsmaske zusätzlich/alternativ aus Datei
    if args.neighbor_capacity and Path(args.neighbor_capacity).exists():
        dfc = pd.read_csv(args.neighbor_capacity)
        cap_mask = cap_mask or {}
        for _, r in dfc.iterrows():
            zone = str(r["zone"]).strip()
            fuel = str(r["fuel"]).strip()
            cap  = float(r["capacity_mw"])
            cap_mask[(zone, fuel)] = cap

    # 8) Hauptschleife
    results, debug_rows = [], []
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    imp_to_zone = {c: c.replace("imp_", "").replace("_", "") for c in imp_cols}

    for t in idx:
        L  = float(de_load.get(t, np.nan))
        ND = float(de_nondisp.get(t, 0.0))
        if not np.isfinite(L): 
            continue

        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0
        residual = L - ND - net_imp
        if residual < 0: residual = 0.0

        # Kapazität DE je Stunde (ThermAvail + Mustrun)
        cap_t = cap_base * float(args.therm_avail)
        lignite_mustrun_enforced = 0.0
        if args.mustrun_mode == "capacity" and lign_total > 0 and args.mustrun_lignite_q > 0.0:
            cap_t[lign_mask] = np.maximum(cap_t[lign_mask], cap_base[lign_mask] * float(args.mustrun_lignite_q))
            lignite_mustrun_enforced = float(cap_t[lign_mask].sum())
        elif args.mustrun_mode == "gen_quantile" and lign_total > 0:
            need = float(lign_profile.get(t, 0.0))
            if need > 0.0:
                target = lign_share * need
                cap_t = np.maximum(cap_t, target.astype(cap_t.dtype))
                lignite_mustrun_enforced = float(target[lign_mask].sum())

        # Domestic marginal
        if (residual <= 0) or (t not in SRMC.index):
            unit_id = ef_dom = eta_dom = srmc_dom = None
        else:
            srmc_t = SRMC.loc[t].to_numpy()
            order  = np.argsort(srmc_t, kind="mergesort")
            cumcap = np.cumsum(cap_t[order])
            pos = np.searchsorted(cumcap, residual, side="left")
            if pos >= len(order): pos = len(order) - 1
            uidx = order[pos]
            unit_id = units[uidx]; ef_dom = ef_keys[uidx]; eta_dom = float(eta_arr[uidx]); srmc_dom = float(srmc_t[uidx])

        # Cluster & Importe
        cluster = clusters.get(t, ["DE_LU"])
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan

        # Pro Zone: marginaler Kandidat aus Gen + η-Verteilung
        imp_details = []   # [(zone, mw, ef_name, eta_z, srmc_z, mef_z)]
        zone_mw = defaultdict(float)
        for c in imp_cols:
            mw = float(flows.loc[t, c]) if t in flows.index else 0.0
            if mw <= 1e-6: 
                continue
            z = imp_to_zone[c]
            if z not in cluster: 
                continue
            if z not in gen_by_zone or t not in gen_by_zone[z].index:
                continue
            gen_row = gen_by_zone[z].loc[t]
            fp_row  = fuel_prices.loc[t] if t in fuel_prices.index else None
            if fp_row is None: 
                continue
            load_val = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
            ef_name, srmc_z, eta_z, dbg = neighbor_marginal_from_residual(
                load_val, gen_row, fp_row, z, nei_dists, args.nei_eta_mode, int(args.nei_mc_draws), cap_mask
            )

            if ef_name is None: 
                continue
            ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)
            mef_z = (ef_th / max(eta_z,1e-6)) * 1000.0
            imp_details.append((z, mw, ef_name, eta_z, srmc_z, mef_z))
            zone_mw[z] += mw

        import_fuel_final = None
        import_srmc_final = None
        import_mef_final  = None
        import_label      = None

        if imp_details:
            imp_sum = sum(zone_mw.values())
            dom_z = max(zone_mw, key=zone_mw.get)
            dom_share = zone_mw[dom_z] / imp_sum if imp_sum > 0 else 0.0
            dom_rec = next((r for r in imp_details if r[0] == dom_z), None)
            if dom_share >= 0.60 and dom_rec is not None:
                _, mw, ef, eta_z, srmc_z, mef_z = dom_rec
                import_fuel_final = ef
                import_srmc_final = srmc_z
                import_mef_final  = mef_z
                import_label      = f"{dom_z}({ef})"
            else:
                fuel_mw = defaultdict(float); fuel_val = {}
                for z, mw, ef, eta_z, srmc_z, mef_z in imp_details:
                    fuel_mw[ef] += mw
                    fuel_val[ef] = (srmc_z, mef_z, z)
                if fuel_mw:
                    best_fuel = max(fuel_mw, key=fuel_mw.get)
                    if (fuel_mw[best_fuel] / imp_sum) >= 0.40:
                        srmc_z, mef_z, z0 = fuel_val[best_fuel]
                        import_fuel_final = best_fuel
                        import_srmc_final = srmc_z
                        import_mef_final  = mef_z
                        import_label      = f"{z0}({best_fuel})"

        import_marg_srmc = import_marg_mef = None
        if imp_details:
            w = sum(mw for (_z, mw, *_rest) in imp_details)
            if w > 0:
                import_marg_srmc = sum(mw*srmc for (_z,mw,_ef,_eta,srmc,_mef) in imp_details) / w
                import_marg_mef  = sum(mw*mef  for (_z,mw,_ef,_eta,_srmc,mef) in imp_details) / w

        # Finale Zuordnung
        marginal_side = "DE"
        marginal_label = unit_id if unit_id else "none"
        marginal_fuel  = ef_dom
        marginal_eta   = eta_dom if eta_dom is not None else np.nan
        marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan

        if unit_id is not None and ef_dom is not None and eta_dom is not None:
            mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom,1e-6)) * 1000.0
        else:
            mef_dom = np.nan

        if (import_marg_srmc is not None) and (net_imp > 0.0) and (len(cluster) > 1):
            if (not np.isfinite(marginal_srmc)) or (import_marg_srmc >= marginal_srmc):
                marginal_side  = "IMPORT"
                marginal_label = import_label or ",".join(sorted({z for (z, *_rest) in imp_details}))
                marginal_fuel  = import_fuel_final or "mix"
                marginal_eta   = np.nan
                marginal_srmc  = float(import_srmc_final) if import_srmc_final is not None else float(import_marg_srmc)
                mef_gpkwh      = float(import_mef_final)  if import_mef_final  is not None else float(import_marg_mef)
            else:
                mef_gpkwh = mef_dom
        else:
            mef_gpkwh = mef_dom
        # ... wir haben: marginal_srmc (DE), import_marg_srmc (Import), p_de
        choose_side = None
        
        if (net_imp > 0.0) and (len(cluster) > 1) and np.isfinite(p_de):
            cand = []
            if np.isfinite(marginal_srmc):
                cand.append(("DE", abs(marginal_srmc - p_de), marginal_srmc))
            if import_marg_srmc is not None:
                cand.append(("IMPORT", abs(import_marg_srmc - p_de), float(import_marg_srmc)))
        
            if args.price_anchor == "closest" and cand:
                choose_side = min(cand, key=lambda x: x[1])[0]
            elif args.price_anchor == "threshold" and cand:
                valid = [c for c in cand if c[1] <= float(args.price_tol)]
                if valid:
                    choose_side = min(valid, key=lambda x: x[1])[0]
        
        # Fallback: alte Regel (SRMC-Vergleich), wenn Anchoring nicht entschieden hat
        if choose_side is None and (import_marg_srmc is not None) and (net_imp > 0.0) and (len(cluster) > 1):
            choose_side = "IMPORT" if (not np.isfinite(marginal_srmc) or import_marg_srmc >= marginal_srmc) else "DE"
        
        # Anwenden
        if choose_side == "IMPORT":
            marginal_side  = "IMPORT"
            marginal_label = import_label or ",".join(sorted({z for (z, *_rest) in imp_details}))
            marginal_fuel  = import_fuel_final or "mix"
            marginal_eta   = np.nan
            marginal_srmc  = float(import_srmc_final) if import_srmc_final is not None else float(import_marg_srmc)
            mef_gpkwh      = float(import_mef_final)  if import_mef_final  is not None else float(import_marg_mef)
        else:
            marginal_side  = "DE"
            # (DE-Block wie gehabt)

        results.append({
            "timestamp": t,
            "mef_g_per_kwh": mef_gpkwh,
            "marginal_side": marginal_side,
            "marginal_label": marginal_label,
            "marginal_fuel": marginal_fuel,
            "marginal_eta": marginal_eta,
            "marginal_srmc_eur_per_mwh": marginal_srmc,
            "price_DE": p_de,
            "net_import_total_MW": net_imp,
            "cluster_zones": "|".join(cluster),
            "residual_domestic_fossil_MW": residual,
        })
        debug_rows.append({
            "timestamp": t,
            "DE_unit_marginal": unit_id,
            "DE_fuel": ef_dom,
            "DE_eta": eta_dom,
            "DE_srmc": srmc_dom,
            "IMPORT_srmc_w": import_marg_srmc,
            "IMPORT_mef_gpkwh_w": import_marg_mef,
            "IMPORT_label": import_label or "",
            "cluster": "|".join(cluster),
            "net_import_total_MW": net_imp,
            "price_DE": p_de,
            "ND_MW": ND,
            "Load_MW": L,
            "LIGNITE_MUSTRUN_ENFORCED_MW": lignite_mustrun_enforced,
        })

    # 9) Outputs
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    (outdir / "analysis").mkdir(exist_ok=True, parents=True)
    df_res.to_csv(outdir / "mef_track_c_2024.csv", index=True)
    df_dbg.to_csv(outdir / "_debug_hourly.csv", index=True)
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'}")
    print(f"[OK] Debug:       {outdir/'_debug_hourly.csv'}")

if __name__ == "__main__":
    main(build_parser().parse_args())
