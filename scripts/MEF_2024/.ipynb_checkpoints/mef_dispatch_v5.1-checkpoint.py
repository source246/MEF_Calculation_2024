#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch-Backcast (DE/LU 2024) mit:
- DE: imputierte Wirkungsgrade (eta_col) → SRMC pro Einheit
- Nachbarn: η-Spannen/MC je Fuel + Kapazitätsmaske (optional)
- Import-Fall bei Preiskopplung: Export-Stack der gekoppelten Zonen
  (Reservoir-Hydro als einzige regelbare Hydro, RoR/Pumpspeicher nicht-disponibel)
- Marginaler Block des gemeinsamen Stacks bestimmt Fuel & MEF
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np

TZ = "Europe/Berlin"

# ----------------------------- CLI -------------------------------------------
def _stack_has_zero_srmc(blocks):
    return any((f in ("EE","Reservoir Hydro")) and (mw > 1e-6) and (abs(srmc) <= 1e-6)
               for (f, srmc, mw, eta, z) in blocks)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track C v2 – MEF Backcast mit Export-Stack-Logik")
    # DE-Fleet + Preise/Flows
    p.add_argument("--fleet", required=True, help="CSV: DE-Fleet (z. B. Kraftwerke_eff_binned.csv)")
    p.add_argument("--eta_col", default="Imputed_Effizienz_binned", help="Spalte mit imputierter Effizienz")
    p.add_argument("--fuel_prices", required=True, help="CSV: prices_2024.csv (€/MWh_th, EUA €/t)")
    p.add_argument("--flows", required=True, help="CSV: flows_scheduled_DE_LU_2024_net.csv")
    p.add_argument("--start", default=None)
    p.add_argument("--end",   default=None)
    p.add_argument("--neighbor_fleet", default=None,
                   help="CSV: zone,fuel,eta/heat_rate,capacity_mw → zonale η-Dists + Kapazitätsmaske")
    # --- Peaker-Heuristik ---
    p.add_argument("--peak_switch", action="store_true",
                   help="Aktiviere Preis-zu-Peaker-Override (OCGT/Öl) bei hohen Preisen.")
    p.add_argument("--peak_price_thresholds", default="180,260",
                   help="Schwellen in €/MWh: 'p1,p2' -> p1≈OCGT, p2≈Öl/Diesel.")
    p.add_argument("--peak_eta_ocgt", type=float, default=0.36,
                   help="η für OCGT-Peaker (el.).")
    p.add_argument("--peak_eta_oil", type=float, default=0.33,
                   help="η für öl-/dieselbefeuerte Turbinen/Engines als Peaker.")


    # Fossiler Mustrun (neu: q_all mit q=10%)
    p.add_argument("--fossil_mustrun_mode",
                   choices=["off","min_all","min_peak","min_peak_monthly","q_all"],
                   default="q_all",
                   help="q_all = unteres Quantil über alle Stunden je Fuel")
    p.add_argument("--fossil_mustrun_q", type=float, default=0.10,
                   help="Quantil für q_all (z.B. 0.10 = 10 %)")
    p.add_argument("--fossil_mustrun_fuels",
                   default="Erdgas,Steinkohle,Heizöl schwer,Heizöl leicht / Diesel",
                   help="Braunkohle hier i.d.R. NICHT listen – wird separat behandelt.")

    # Kopplung / Preisanker
    p.add_argument("--neighbor_gen_dir",   required=True)
    p.add_argument("--neighbor_load_dir",  required=True)
    p.add_argument("--neighbor_prices",    required=True)
    p.add_argument("--epsilon", type=float, default=5.0,   # EINMAL definiert
                   help="Preis-Kopplungs-Schwelle in €/MWh")
    p.add_argument("--price_anchor", choices=["off","closest","threshold"], default="closest")
    p.add_argument("--price_tol", type=float, default=30.0)

    # Nachbarn: Effizienz-Modelle
    p.add_argument("--nei_eta_mode", choices=["mean","bounds","mc"], default="mean")
    p.add_argument("--nei_eta_json", default=None)
    p.add_argument("--nei_mc_draws", type=int, default=50)
    p.add_argument("--neighbor_capacity", default=None)

    # Optionale Mustrun-Shares (bleiben wie gehabt)
    p.add_argument("--de_mustrun_gas_share", type=float, default=0.0)
    p.add_argument("--de_mustrun_coal_share", type=float, default=0.0)
    p.add_argument("--de_mustrun_oil_share",  type=float, default=0.0)
    p.add_argument("--nei_mustrun_gas_share", type=float, default=0.0)
    p.add_argument("--nei_mustrun_coal_share", type=float, default=0.0)
    p.add_argument("--nei_mustrun_oil_share",  type=float, default=0.0)

    # Dispatch-Details
    p.add_argument("--varom_json", default=None)
    p.add_argument("--therm_avail", type=float, default=0.95)
    p.add_argument("--mustrun_mode", choices=["off","capacity","gen_quantile"], default="gen_quantile")
    p.add_argument("--mustrun_lignite_q", type=float, default=0.20)
    p.add_argument("--mustrun_quantile",  type=float, default=0.20)
    p.add_argument("--mustrun_peak_hours", default="08-20")
    p.add_argument("--mustrun_monthly", action="store_true")
    p.add_argument("--ee_price_threshold", type=float, default=5.0,
                   help="Nur wenn Preis ≤ Schwelle und kein Netto-Import → MEF=0")
    p.add_argument("--year", type=int, default=2024)

    # Output
    p.add_argument("--outdir", required=True)
    return p


# -------------------------- Helper: Zeit & IO --------------------------------
# --- Robust Missing Handling ---
def _robust_series(s: pd.Series, kind: str, max_gap_h: int = 6) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # 1) zeitliche Interpolation (begrenzen, damit nicht über Wochen „phantomisiert“ wird)
    s_interp = s.interpolate(limit=max_gap_h, limit_direction="both")
    # 2) ffill/bfill Restlöcher
    s_fill = s_interp.ffill().bfill()
    # 3) falls noch NaN (komplette Ausfälle), nimm stündlichen Median
    if s_fill.isna().any():
        by_hour = s_fill.groupby(s_fill.index.hour).transform(
            lambda x: x.fillna(np.nanmedian(x))
        )
        s_fill = s_fill.fillna(by_hour)
    # 4) letzte Sicherung
    if s_fill.isna().any():
        s_fill = s_fill.fillna(0.0)
    return s_fill
def _parse_two_floats(csv_str: str, default=(180.0, 260.0)):
    try:
        a, b = [float(x) for x in str(csv_str).split(",")[:2]]
        return a, b
    except Exception:
        return default

def robustize_load_gen(de_load: pd.Series, de_gen: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    # Non-Disp erkennen
    nd_cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore",
               "Hydro Run-of-river and poundage","Hydro Pumped Storage","Biomass","Waste"]
    present = [c for c in nd_cols if c in de_gen.columns]
    # Impute Gen je Spalte
    for c in de_gen.columns:
        de_gen[c] = _robust_series(de_gen[c], f"gen:{c}")
    # Impute Load
    de_load = _robust_series(de_load, "load")
    # Cross-Check: Residual nie negativ durch Artefakte
    nd_sum = de_gen[present].sum(axis=1) if present else pd.Series(0.0, index=de_load.index)
    # Wenn ND_sum unrealistisch > Load (Artefakt), skaliere ND auf Load (sanft, nicht >5%/h Änderung)
    over = (nd_sum > de_load) & de_load.notna()
    if over.any():
        scale = (de_load / nd_sum).clip(upper=1.0).fillna(1.0)
        # sanfte Glättung des Skalierungsfaktors
        scale = scale.ewm(span=3, min_periods=1).mean()
        for c in present:
            de_gen[c] = (de_gen[c] * scale).fillna(de_gen[c])
    return de_load, de_gen
def reservoir_shadow_price_series(nei_prices: pd.DataFrame, zone: str, window_h: int = 24*7) -> pd.Series:
    col = f"price_{zone}"
    if col not in nei_prices.columns:
        # fallback auf DE/LU-Preis
        col = "price_DE_LU"
    p = pd.to_numeric(nei_prices[col], errors="coerce").astype(float)
    # rollierender Median als Opportunitätskosten-Proxi
    sp = p.rolling(window_h, min_periods=12).median().bfill().ffill()
    # untere Kappung, damit bei Datenlöchern nicht 0 herauskommt
    return sp.clip(lower=0.0)

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
FUEL_KEY_BY_EF = {
    "Erdgas": ("gas", "Fossil Gas"),
    "Steinkohle": ("coal", "Fossil Hard coal"),
    "Braunkohle": ("lignite", "Fossil Brown coal/Lignite"),
    "Heizöl schwer": ("oil", "Fossil Oil"),
    "Heizöl leicht / Diesel": ("oil", "Fossil Oil"),
}

def fossil_mustrun_shares_for_DE(args) -> dict:
    # Rückgabe pro EF-Name (wie im EF_LOOKUP): Anteil 0..1
    return {
        "Erdgas": float(getattr(args, "de_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "de_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,  # Braunkohle-Mustrun kommt bereits separat über dein lignite-Profil
        "Heizöl schwer": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
        "Heizöl leicht / Diesel": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
    }

def fossil_mustrun_shares_for_NEI(args) -> dict:
    return {
        "Erdgas": float(getattr(args, "nei_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "nei_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,  # optional: könnte man auch setzen; hier 0, damit nicht doppelt implizit
        "Heizöl schwer": float(getattr(args, "nei_mustrun_oil_share", 0.0) or 0.0),
        "Heizöl leicht / Diesel": float(getattr(args, "nei_mustrun_oil_share", 0.0) or 0.0),
    }

def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    if "net_import_total" not in df.columns:
        imp_cols = [c for c in df.columns if c.startswith("imp_")]
        df["net_import_total"] = df[imp_cols].sum(axis=1) if imp_cols else 0.0
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
    # Doppelte Technamen aggregieren
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
    seps = [",",";","\t","|"]; encs = ["utf-8-sig","cp1252","latin1"]; last_err=None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] >= min_cols: return df
            except Exception as e:
                last_err = e; continue
    raise RuntimeError(f"CSV nicht lesbar: {path} – letzter Fehler: {last_err}")

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

    eta_clean = pd.to_numeric(out["eta"], errors="coerce").to_numpy()
    if np.nanmedian(eta_clean) > 1.5:  # Prozent → Anteil
        eta_clean = eta_clean/100.0
    eta_clean = np.clip(eta_clean, 0.20, 0.65)
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

# ------------------- Nachbarn: η-Verteilungen & Kapazität --------------------

DEFAULT_NEI_DISTS = {
    "Erdgas":      {"mean": 0.52, "std": 0.043, "min": 0.35, "max": 0.60},
    "Steinkohle":  {"mean": 0.41, "std": 0.030, "min": 0.34, "max": 0.45},
    "Braunkohle":  {"mean": 0.40, "std": 0.028, "min": 0.33, "max": 0.43},
    "Heizöl schwer":{"mean": 0.36, "std": 0.020, "min": 0.32, "max": 0.40},
}

def _norm_zone(z: str) -> str:
    return str(z or "").strip().replace("-", "_").upper()

def _map_neighbor_fuel(s: str) -> Optional[str]:
    t = _norm(s)
    if any(k in t for k in ["gas","erdgas","ccgt","ocgt","erdölgas","erdolgas","fossil gas"]): return "Erdgas"
    if any(k in t for k in ["hard coal","steinkohle","coal","kohlekraft"]):                    return "Steinkohle"
    if any(k in t for k in ["lignite","braunkohle","brown coal"]):                             return "Braunkohle"
    if any(k in t for k in ["diesel","leicht","light oil"]):                                   return "Heizöl leicht / Diesel"
    if any(k in t for k in ["oil","heizöl","heizoel","heavy oil","hfo"]):                      return "Heizöl schwer"
    return None

def _eta_from_row(r) -> Optional[float]:
    cand_cols = [c for c in r.index if str(c).lower() in ("eta","effizienz","wirkungsgrad","eta_el")]
    if cand_cols:
        val = pd.to_numeric(r[cand_cols[0]], errors="coerce")
        if pd.isna(val): return None
        if val > 1.5: val = val/100.0
        return float(np.clip(val, 0.20, 0.65))
    for c in r.index:
        lc = str(c).lower()
        if "heat_rate" in lc or "heatrate" in lc or (lc=="hr"):
            hr = pd.to_numeric(r[c], errors="coerce")
            if not pd.isna(hr) and hr > 0:
                HR = float(hr)
                if HR > 2000: HR = HR/1000.0     # kJ/kWh → MJ/kWh
                elif HR < 50: HR = HR*1.0        # GJ/MWh → MJ/kWh
                eta = 3.6 / HR
                return float(np.clip(eta, 0.20, 0.65))
    return None

def load_neighbor_fleet(path: str) -> tuple[dict, dict]:
    df = read_csv_smart(path, min_cols=3)
    cols = {c.lower(): c for c in df.columns}
    zcol = cols.get("zone") or cols.get("bidding_zone") or cols.get("country") or list(df.columns)[0]
    fcol = cols.get("fuel") or cols.get("brennstoff") or cols.get("energieträger") or cols.get("energietraeger")
    pcol = cols.get("capacity_mw") or cols.get("leistung_mw") or cols.get("mw") or None
    if fcol is None: raise ValueError("neighbor_fleet: 'fuel' fehlt.")
    if zcol is None: raise ValueError("neighbor_fleet: 'zone' fehlt.")

    df["_zone"] = df[zcol].map(_norm_zone)
    df["_fuel"] = df[fcol].map(_map_neighbor_fuel)
    df["_cap"]  = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0) if pcol is not None else 0.0
    df["_eta"]  = df.apply(_eta_from_row, axis=1)
    df = df[df["_fuel"].notna() & df["_zone"].notna()].copy()

    cap_mask = {(z,f): float(sub["_cap"].sum()) for (z,f), sub in df.groupby(["_zone","_fuel"], dropna=True)}
    nei_dists_zonal: dict = {}
    for (z, f), sub in df.groupby(["_zone","_fuel"], dropna=True):
        etas = pd.to_numeric(sub["_eta"], errors="coerce").dropna()
        if len(etas)==0: continue
        m  = float(etas.mean())
        sd = float(np.std(etas)) if len(etas)>1 else max(0.02, m/12.0)
        lo = float(np.quantile(etas, 0.05)) if len(etas)>=5 else max(0.20, m-2*sd)
        hi = float(np.quantile(etas, 0.95)) if len(etas)>=5 else min(0.65, m+2*sd)
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
        coupled = []
        if not pd.isna(p_de):
            for z in zones:
                if z == "DE_LU":
                    continue
                pz = row.get(f"price_{z}", np.nan)
                if not pd.isna(pz) and abs(pz - p_de) <= eps:
                    coupled.append(z)
        clusters[t] = coupled
    return clusters


# ------------------- Export-Stack-Logik (Reservoir-Hydro erlaubt) ------------

NONDISP = [
    "Nuclear","Solar","Wind Onshore","Wind Offshore",
    "Hydro Run-of-river and poundage","Hydro Pumped Storage",
    "Biomass","Waste",
]

def exportable_blocks_for_zone(
    t: pd.Timestamp,
    zone: str,
    gen_z_row: pd.Series,
    load_z_t: float,
    fuel_prices_row: pd.Series,
    nei_dists: Dict[str, dict],
    mode: str,
    draws: int,
    cap_mask: Optional[Dict[Tuple[str,str], float]],
    reservoir_sp_map: Dict[Tuple[str, pd.Timestamp], float],
    min_total_zone_t: float,                 # NEW
    min_by_fuel_zone_t: Dict[str, float],    # NEW (Brennstoffnamen wie in EF_LOOKUP)
) -> List[Tuple[str, float, float, float, str]]:

    """
    Liefert exportierbare Blöcke je Zone:
    [(fuel_name, srmc, mw_exportable, eta_used, zone)]
    Regelbar: Reservoir Hydro (SRMC≈0) + fossile Fuels.
    RoR & Pumpspeicher gelten als nicht-disponibel.
    """
    nd = float(pd.to_numeric(gen_z_row.reindex(NONDISP).fillna(0.0)).sum())
    min_total = float(min_total_zone_t)  # wird als zusätzlicher Parameter reingereicht
    
    # Roh-Residual vor Kappung (kann negativ sein): <0 ⇒ EE-Überschuss in der Zone
    residual_raw = float(load_z_t) - nd - min_total
    residual_z = max(residual_raw, 0.0)
    
    # 2) Sammle regelbare Kandidaten (Reservoir + fossile) mit MW
    candidates = []  # (fuel, srmc, mw, eta, zone)
    
    # NEU: EE-Überschuss dieser Zone als exportierbarer Block (SRMC≈0, MEF=0)
    # (Begründung: negativer Residual bedeutet, die Zone könnte ohne Preisänderung exportieren.)
    if residual_raw <= -1e-6:
        ee_surplus_mw = abs(residual_raw)
        candidates.append(("EE", 0.0, ee_surplus_mw, 1.0, zone))


    # Reservoir-Hydro mit Opportunitätskosten (rollierender Medianpreis)
    if "Hydro Water Reservoir" in gen_z_row.index:
        mw_res = float(gen_z_row.get("Hydro Water Reservoir", 0.0))
        if np.isfinite(mw_res) and mw_res > 0:
            # opportunistischer Preis: nähere dich an den Marktpreis der Zone an
            srmc_res = float(reservoir_sp_map.get((zone, t), 0.0))
            candidates.append(("Reservoir Hydro", srmc_res, mw_res, 1.0, zone))
    


    # Fossile
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in gen_z_row.index:
            continue
    
        mw_raw = float(gen_z_row.get(tech, 0.0))
        if not (np.isfinite(mw_raw) and mw_raw > 0):
            continue
    
        # Kapazitätsmaske
        if cap_mask is not None:
            cap = cap_mask.get((zone, ef_name), None)
            if cap is not None and cap <= 1.0:
                continue
    
        # Preise + Effizienzannahmen
        fuel_th = fuel_prices_row.get(f"{pk}_eur_mwh_th", np.nan)
        co2     = fuel_prices_row.get("co2_eur_t", np.nan)
        if not (np.isfinite(fuel_th) and np.isfinite(co2)):
            continue
    
        d = (nei_dists.get(zone, {}).get(ef_name)
             or nei_dists.get(ef_name)
             or DEFAULT_NEI_DISTS[ef_name])
        m, s, lo, hi = d["mean"], d["std"], d["min"], d["max"]
        eta_eff = m if mode != "mc" else float(np.mean(truncated_normal(m, s, lo, hi, size=draws)))
        srmc    = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH[ef_name]) / max(eta_eff, 1e-6)
    
        # Mindesterzeugung je Fuel abziehen → exportierbarer Anteil
        mw_mustrun = float(min_by_fuel_zone_t.get(ef_name, 0.0))
        mw = max(mw_raw - mw_mustrun, 0.0)
        if mw <= 1e-6:
            continue
    
        candidates.append((ef_name, float(srmc), float(mw), float(eta_eff), zone))
    


    if not candidates:
        return []

    # 3) Innerzonal: sortiere nach SRMC↑ und ziehe bis Residual gedeckt ist → „benötigt“
    fossil_sorted = [c for c in candidates if c[0] != "Reservoir Hydro"]
    fossil_sorted = sorted(fossil_sorted, key=lambda x: x[1])
    need = residual_z
    used_by_fuel = defaultdict(float)
    for (fuel, srmc, mw, eta, z) in fossil_sorted:
        if need <= 1e-9:
            break
        take = min(max(need, 0.0), mw)
        used_by_fuel[fuel] += take
        need -= take

    # 4) Exportierbarer Überschuss = MW - „benötigt“ je Fuel; Reservoir komplett exportierbar
    export_blocks = []
    for (fuel, srmc, mw, eta, z) in candidates:
        if fuel == "Reservoir Hydro":
            if mw > 1e-6:
                export_blocks.append((fuel, srmc, mw, eta, z))
        else:
            used = used_by_fuel.get(fuel, 0.0)
            surplus = max(mw - used, 0.0)
            if surplus > 1e-6:
                export_blocks.append((fuel, srmc, surplus, eta, z))
    return export_blocks
# ------------------------------ Validation & Plots ---------------------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PALETTE = {
    "DE": "#1f77b4",
    "IMPORT": "#d62728",
    "EE": "#2ca02c",
    "price": "#444444",
    "warn": "#ff7f0e",
    "ok": "#2ca02c",
    "mix": "#7f7f7f",
}

def _pct(x, y):
    return 0.0 if y == 0 else 100.0 * (x / y)

def validate_run(df_res: pd.DataFrame,
                 df_dbg: pd.DataFrame,
                 flows: pd.DataFrame,
                 prices: pd.DataFrame,
                 epsilon_price: float,
                 price_anchor_mode: str,
                 tol_balance_mw: float = 1.0):
    out = pd.DataFrame(index=df_res.index).copy()

    out["price_DE"] = df_res["price_DE"]
    out["marginal_srmc"] = df_res["marginal_srmc_eur_per_mwh"]
    out["marginal_side"] = df_res["marginal_side"]
    out["marginal_label"] = df_res["marginal_label"]
    out["marginal_fuel"] = df_res["marginal_fuel"]
    out["mef_gpkwh"] = df_res["mef_g_per_kwh"]
    out["net_import_total_MW"] = df_res["net_import_total_MW"]

    # Kopplungs-Deltas
    for c in [c for c in prices.columns if c.startswith("price_") and c != "price_DE_LU"]:
        out[f"abs_{c}_minus_DE"] = (prices[c] - prices["price_DE_LU"]).abs().reindex(out.index)

    # Preisankerprüfung
    out["IMPORT_anchor_ok"] = True
    if price_anchor_mode in ("closest", "threshold"):
        imp_srmc = df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index)
        de_srmc  = df_dbg["DE_srmc"].reindex(out.index)
        p_de     = out["price_DE"]
        if price_anchor_mode == "closest":
            out.loc[out["marginal_side"]=="IMPORT","IMPORT_anchor_ok"] = (
                (imp_srmc - p_de).abs() <= (de_srmc - p_de).abs()
            )


    # EE-Überschuss-Logik
    out["EE_surplus_flag"] = (df_res["residual_domestic_fossil_MW"] <= 1e-6)
    out["EE_surplus_mef_ok"] = ~(out["EE_surplus_flag"]) | (out["mef_gpkwh"] <= 1e-6) | (out["marginal_side"]=="IMPORT")

    # Import-Stack-Logik
    out["IMPORT_has_block"] = ~df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index).isna()
    mask_import = (out["marginal_side"]=="IMPORT")
    out["IMPORT_logic_ok"] = True
    out.loc[mask_import, "IMPORT_logic_ok"] = (
        (out.loc[mask_import, "net_import_total_MW"] > 0.0) & (out.loc[mask_import, "IMPORT_has_block"])
    )
    
    # Preis-Sanity
    chosen_srmc = out["marginal_srmc"]
    abs_cols = [c for c in out.columns if c.startswith("abs_price_")]
    min_abs_diff = pd.concat([out[c] for c in abs_cols], axis=1).min(axis=1, skipna=True) if abs_cols else pd.Series(np.nan, index=out.index)
    out["suspect_price_deviation"] = ((out["price_DE"] - chosen_srmc).abs() > 100.0) & (min_abs_diff <= epsilon_price)

    summary = {
        "N_hours": len(out),
        "share_IMPORT": _pct((out["marginal_side"]=="IMPORT").sum(), len(out)),
        "share_anchor_ok_when_IMPORT": _pct(out.loc[mask_import, "IMPORT_anchor_ok"].sum(), max(mask_import.sum(),1)),
        "share_EE_surplus_mef_ok": _pct(out["EE_surplus_mef_ok"].sum(), len(out)),
        "share_IMPORT_logic_ok": _pct(out["IMPORT_logic_ok"].sum(), len(out)),
        "share_suspect_price_dev": _pct(out["suspect_price_deviation"].sum(), len(out)),
        "corr_price_vs_srmc": float(pd.concat([out["price_DE"], chosen_srmc], axis=1).dropna().corr().iloc[0,1]) if out[["price_DE","marginal_srmc"]].dropna().shape[0] >= 3 else np.nan,
    }
    summ_df = pd.DataFrame(summary, index=["summary"])
    return out, summ_df

def write_validation_report(outdir: Path,
                            df_val: pd.DataFrame,
                            df_sum: pd.DataFrame) -> None:
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)
    df_val.to_csv(outdir / "analysis" / "_validation.csv", index=True)
    df_sum.to_csv(outdir / "analysis" / "_validation_summary.csv", index=True)
    print("[VALIDATION] geschrieben:",
          outdir / "analysis" / "_validation.csv",
          "und",
          outdir / "analysis" / "_validation_summary.csv")

def _ts_ax(ax, tzlabel=""):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if tzlabel:
        ax.set_xlabel(f"Zeit ({tzlabel})")

def make_validation_plots(outdir: Path,
                          df_res: pd.DataFrame,
                          df_dbg: pd.DataFrame,
                          df_val: pd.DataFrame,
                          prices: pd.DataFrame) -> None:
    """Erzeugt PNGs in outdir/analysis/plots"""
    pdir = outdir / "analysis" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) Timeseries: Day-Ahead-Preis vs. gewählte SRMC (+ Seite als Farbe)
    fig, ax = plt.subplots(figsize=(12, 4))
    srmc = df_res["marginal_srmc_eur_per_mwh"]
    price = df_res["price_DE"]
    ax.plot(price.index, price.values, label="Preis (DE/LU)", linewidth=1.3, color=PALETTE["price"])
    ax.plot(srmc.index, srmc.values, label="gewählter SRMC", linewidth=1.3, color=PALETTE["DE"])
    _ts_ax(ax, tzlabel="Europe/Berlin")
    ax.set_ylabel("€/MWh")
    ax.legend()
    fig.tight_layout(); fig.savefig(pdir / "timeseries_price_vs_chosen_srmc.png", dpi=160); plt.close(fig)

    # 2) Scatter: Preis vs. SRMC (farblich DE/IMPORT)
    side = df_res["marginal_side"].reindex(srmc.index)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    m_de = (side=="DE")
    m_im = (side=="IMPORT")
    ax.scatter(srmc[m_de], price[m_de], s=10, alpha=0.6, label="DE", color=PALETTE["DE"])
    ax.scatter(srmc[m_im], price[m_im], s=10, alpha=0.6, label="IMPORT", color=PALETTE["IMPORT"])
    lims = [np.nanmin([srmc.min(), price.min()]), np.nanmax([srmc.max(), price.max()])]
    ax.plot(lims, lims, linestyle="--", linewidth=1, color=PALETTE["price"])
    ax.set_xlabel("gewählter SRMC [€/MWh]"); ax.set_ylabel("Preis [€/MWh]")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(pdir / "scatter_price_vs_srmc.png", dpi=160); plt.close(fig)

    # 3) Histogramm der Preisabweichung (Preis − SRMC)
    err = (price - srmc).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=60)
    ax.set_xlabel("Preis − SRMC [€/MWh]"); ax.set_ylabel("Häufigkeit")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(pdir / "hist_price_minus_srmc.png", dpi=160); plt.close(fig)

    # 4) Tagesprofil: mittlere Abweichung nach Stunde (Heatmap: Monat×Stunde)
    dfh = (price - srmc).to_frame("err")
    dfh["month"] = dfh.index.month
    dfh["hour"] = dfh.index.hour
    piv = dfh.pivot_table(index="month", columns="hour", values="err", aggfunc="median")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_xlabel("Stunde"); ax.set_ylabel("Monat"); ax.set_title("Median(Preis−SRMC)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="€/MWh")
    fig.tight_layout(); fig.savefig(pdir / "heatmap_median_err_month_hour.png", dpi=160); plt.close(fig)

    # 5) MEF-Verteilung nach marginal_fuel (Boxplot)
    dfm = df_res[["marginal_fuel","mef_g_per_kwh"]].dropna()
    if not dfm.empty:
        order = dfm.groupby("marginal_fuel")["mef_g_per_kwh"].median().sort_values().index
        data = [dfm.loc[dfm["marginal_fuel"]==f, "mef_g_per_kwh"].values for f in order]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("MEF [g/kWh]"); ax.set_title("MEF nach marginaler Technologie")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(pdir / "box_mef_by_fuel.png", dpi=160); plt.close(fig)

    # 6) Anteile DE vs IMPORT (rollierender 7-Tage-Mittelwert)
    side_num = (side=="IMPORT").astype(float).rolling(24*7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(side_num.index, 100.0*side_num.values, color=PALETTE["IMPORT"], label="IMPORT-Anteil (roll. 7T)")
    _ts_ax(ax, tzlabel="Europe/Berlin"); ax.set_ylabel("Anteil [%]"); ax.legend()
    fig.tight_layout(); fig.savefig(pdir / "share_import_rolling.png", dpi=160); plt.close(fig)

    # 7) Import-Stack: wie oft welcher Import-Fuel marginal?
    lab = df_dbg["IMPORT_label"].fillna("")
    fuels = lab.str.extract(r"\((.+)\)")[0].fillna("n/a")
    counts = fuels[side=="IMPORT"].value_counts()
    if counts.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Stunden"); ax.set_title("Import-marginale Fuels (Häufigkeit)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(pdir / "bar_import_marginal_fuels.png", dpi=160); plt.close(fig)

    # 8) Flag-Timeline: suspekt/ok
    sus = df_val["suspect_price_deviation"].reindex(df_res.index).fillna(False)
    fig, ax = plt.subplots(figsize=(12, 0.8))
    ax.plot(sus.index, sus.astype(int).values, linewidth=1)
    _ts_ax(ax, tzlabel="Europe/Berlin")
    ax.set_yticks([0,1]); ax.set_yticklabels(["ok","suspekt"])
    ax.set_title("Preis-Abweichung Flag")
    fig.tight_layout(); fig.savefig(pdir / "flag_price_deviation_timeline.png", dpi=160); plt.close(fig)
def compute_fossil_min_profiles(
    gen_df: pd.DataFrame,
    fuels_select: List[str],
    peak_hours: str,
    mode: str,            # "off"|"min_all"|"min_peak"|"min_peak_monthly"|"q_all"
    q: float = 0.10,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:

    if mode == "off" or gen_df is None or gen_df.empty:
        idx = gen_df.index if (gen_df is not None and isinstance(gen_df.index, pd.DatetimeIndex)) else pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx), {f: pd.Series(0.0, index=idx) for f in fuels_select}

    TECH2FUEL = {
        "Fossil Gas": "Erdgas",
        "Fossil Hard coal": "Steinkohle",
        "Fossil Oil": "Heizöl schwer",
        # "Fossil Brown coal/Lignite": "Braunkohle",  # i.d.R. separat
    }
    tech_cols = [c for c in TECH2FUEL.keys() if c in gen_df.columns]

    h_start, h_end = [int(x) for x in peak_hours.split("-")]
    def is_peak(ix: pd.DatetimeIndex):
        return ((ix.hour >= h_start) & (ix.hour < h_end)) if h_start <= h_end else ((ix.hour >= h_start) | (ix.hour < h_end))

    idx = gen_df.index
    min_by_fuel = {f: pd.Series(0.0, index=idx, dtype="float64") for f in fuels_select}

    if mode == "min_all":
        for tech in tech_cols:
            f = TECH2FUEL[tech]
            if f not in fuels_select: continue
            m = float(pd.to_numeric(gen_df[tech], errors="coerce").min(skipna=True))
            min_by_fuel[f] = pd.Series(max(m,0.0), index=idx, dtype="float64")

    elif mode == "min_peak":
        pk_mask = is_peak(idx); op_mask = ~pk_mask
        for tech in tech_cols:
            f = TECH2FUEL[tech]
            if f not in fuels_select: continue
            s = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            out = pd.Series(0.0, index=idx, dtype="float64")
            out[ pk_mask] = float(s[pk_mask].min()) if pk_mask.any() else 0.0
            out[ op_mask] = float(s[op_mask].min()) if op_mask.any() else 0.0
            min_by_fuel[f] = out

    elif mode == "min_peak_monthly":
        months = sorted(idx.unique().month); pk_all = is_peak(idx)
        for tech in tech_cols:
            f = TECH2FUEL[tech]
            if f not in fuels_select: continue
            s = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            out = pd.Series(0.0, index=idx, dtype="float64")
            for m in months:
                msk = (idx.month==m)
                pk_m = msk & pk_all; op_m = msk & (~pk_all)
                out[ pk_m] = float(s[pk_m].min()) if pk_m.any() else 0.0
                out[ op_m] = float(s[op_m].min()) if op_m.any() else 0.0
            min_by_fuel[f] = out

    elif mode == "q_all":
        for tech in tech_cols:
            f = TECH2FUEL[tech]
            if f not in fuels_select: continue
            s = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            qv = float(np.nanquantile(s, q)) if len(s) else 0.0
            min_by_fuel[f] = pd.Series(max(qv,0.0), index=idx, dtype="float64")

    # Nachbereitung
    for f in min_by_fuel:
        min_by_fuel[f] = pd.to_numeric(min_by_fuel[f], errors="coerce").fillna(0.0).clip(lower=0.0)

    total_min = sum(min_by_fuel.values()) if min_by_fuel else pd.Series(0.0, index=idx)
    return total_min, min_by_fuel

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
    SRMC = SRMC.loc[:, common]; fleet_idxed = fleet_idxed.loc[common]
    units    = list(SRMC.columns)
    cap_base = fleet_idxed["available_mw"].astype("float32").to_numpy()
    eta_arr  = fleet_idxed["eta"].astype("float32").to_numpy()
    ef_keys  = fleet_idxed["ef_key"].astype(str).to_numpy()

    # 3) Flows & Nachbarpreise
    flows = load_flows(args.flows)
    nei_prices = load_neighbor_prices(args.neighbor_prices)
    clusters = cluster_zones_by_price(nei_prices, args.epsilon)
    zones = sorted([c.replace("price_", "") for c in nei_prices.columns if c.startswith("price_")])
    # Opportunitätskosten-Map für Reservoir je Zone & Stunde
    reservoir_sp_map = {}
    for z in zones:
        sp = reservoir_shadow_price_series(nei_prices, z)
        # in dict[str, float] pro Zeitpunkt, für schnellen Zugriff
        for tt, val in sp.items():
            reservoir_sp_map[(z, tt)] = float(val) if np.isfinite(val) else 0.0

    # 4) Nachbar-Gen/Load + DE/LU
    load_by_zone, gen_by_zone = {}, {}
    for z in zones:
        try: load_by_zone[z] = load_neighbor_load(args.neighbor_load_dir, z)
        except Exception: pass
        try: gen_by_zone[z] = load_neighbor_gen(args.neighbor_gen_dir, z)
        except Exception: pass

    if "DE_LU" not in load_by_zone: raise RuntimeError("load_DE_LU_2024.csv fehlt.")
    if "DE_LU" not in gen_by_zone:  raise RuntimeError("actual_gen_DE_LU_2024.csv fehlt.")
    de_load = load_by_zone["DE_LU"]; de_gen = gen_by_zone["DE_LU"]
    de_load, de_gen = robustize_load_gen(de_load, de_gen)

    # --- Fossile Mindesterzeugung für alle Zonen vorbereiten ---
    fossil_list = [s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()]
    nei_min_total_by_zone: Dict[str, pd.Series] = {}
    nei_min_by_zone_fuel: Dict[str, Dict[str, pd.Series]] = {}
        
    for z in zones:
        if z not in gen_by_zone:
            continue
        z_gen = gen_by_zone[z]
        total_min, by_fuel = compute_fossil_min_profiles(
            gen_df=z_gen, fuels_select=fossil_list,
            peak_hours=args.mustrun_peak_hours, mode=args.fossil_mustrun_mode,
            q=float(args.fossil_mustrun_q),
        )
        nei_min_total_by_zone[z] = total_min
        nei_min_by_zone_fuel[z]  = by_fuel
  
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

    # 6) Nicht-disponible in DE (PS-Gen NICHT abziehen; Pump-Last steckt in de_load)
    nd_cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage","Biomass","Waste"]
    nd_present = [c for c in nd_cols if c in de_gen.columns]
    de_nondisp = de_gen[nd_present].sum(axis=1).reindex(de_load.index).fillna(0.0)
    # --- Fossile Mindesterzeugung (DE) ---
    fossil_list = [s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()]
    de_min_total, de_min_by_fuel = compute_fossil_min_profiles(
        gen_df=de_gen.reindex(de_load.index), fuels_select=fossil_list,
        peak_hours=args.mustrun_peak_hours, mode=args.fossil_mustrun_mode,
        q=float(args.fossil_mustrun_q),
    )

    # 7) Mustrun (zeitvariabel aus Lignite-Gen)
    h_start, h_end = [int(x) for x in args.mustrun_peak_hours.split("-")]
    def is_peak(ix: pd.DatetimeIndex):
        return ((ix.hour >= h_start) & (ix.hour < h_end)) if h_start <= h_end else ((ix.hour >= h_start) | (ix.hour < h_end))

    lign_profile = pd.Series(0.0, index=pd.DatetimeIndex(idx))
    if args.mustrun_mode == "gen_quantile" and "Fossil Brown coal/Lignite" in de_gen.columns:
        lign = de_gen["Fossil Brown coal/Lignite"].reindex(lign_profile.index)
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

    lign_mask  = (ef_keys == "Braunkohle")
    lign_total = float(cap_base[lign_mask].sum())
    lign_share = np.zeros_like(cap_base)
    if lign_total > 0:
        lign_share[lign_mask] = cap_base[lign_mask] / lign_total

    # 8) Nachbar-η-Parameter / Kapazitätsmaske
    nei_dists = DEFAULT_NEI_DISTS.copy()
    cap_mask = None
    if args.neighbor_fleet and Path(args.neighbor_fleet).exists():
        fleet_dists, cap_mask_from_fleet = load_neighbor_fleet(args.neighbor_fleet)
        for z, fuels in fleet_dists.items():
            nei_dists.setdefault(z, {})
            for f, d in fuels.items():
                nei_dists[z][f] = d
        cap_mask = cap_mask_from_fleet
    if args.nei_eta_json and Path(args.nei_eta_json).exists():
        with open(args.nei_eta_json, "r", encoding="utf-8") as f:
            user_d = json.load(f)
        for k, v in user_d.items():
            if isinstance(v, dict) and all(isinstance(vv, dict) for vv in v.values()):  # zonal
                nei_dists.setdefault(k, {}).update(v)
            else:  # global fuel
                nei_dists[k] = v if isinstance(v, dict) else v
    if args.neighbor_capacity and Path(args.neighbor_capacity).exists():
        dfc = pd.read_csv(args.neighbor_capacity)
        cap_mask = cap_mask or {}
        for _, r in dfc.iterrows():
            zone = str(r["zone"]).strip()
            fuel = str(r["fuel"]).strip()
            cap  = float(r["capacity_mw"])
            cap_mask[(zone, fuel)] = cap

    # 9) Hauptschleife
    results, debug_rows = [], []
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    imp_to_zone = {c: c.replace("imp_", "").replace("_", "") for c in imp_cols}


    for t in idx:
        L  = float(de_load.get(t, np.nan))
        ND = float(de_nondisp.get(t, 0.0))
        if not np.isfinite(L):
            continue
    
        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0
        # 6a) Preis & Kopplungsinfo VOR jedweden Abbrüchen
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan
        fp_row = fuel_prices.loc[t] if t in fuel_prices.index else None
        cluster_all = clusters.get(t, ["DE_LU"])
        coupled_neighbors = [z for z in cluster_all if z != "DE_LU"]
        coupling_active = len(coupled_neighbors) > 0
    
        # 1) Residuallasten (vor / nach Handel)
        residual_domestic_fossil = L - ND - de_min_total.get(t, 0.0)
        residual_after_trade     = residual_domestic_fossil - net_imp
        residual = max(residual_after_trade, 0.0)
        
        # --- Zwei-Stufen-Null-MEF ---
        # FEE-only Summe
        fee_cols = ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage","Hydro Pumped Storage"]
        fee_cols = [c for c in fee_cols if c in de_gen.columns]
        FEE = float(de_gen[fee_cols].reindex([t]).sum(axis=1).iloc[0]) if fee_cols else 0.0
        
        # Non-Disp zusätzlich (Biomasse, Waste, ggf. Nuclear)
        nd_extra_cols = ["Biomass","Waste","Nuclear"]
        nd_extra_cols = [c for c in nd_extra_cols if c in de_gen.columns]
        ND_EXTRA = float(de_gen[nd_extra_cols].reindex([t]).sum(axis=1).iloc[0]) if nd_extra_cols else 0.0
        
        fee_covers      = (L <= FEE + 1e-6)
        nondisp_covers  = (L <= (FEE + ND_EXTRA) + 1e-6)

        # 3) DE-Mustrun (optionale zusätzliche Minderung der Residuallast)
        de_mustrun_shares = fossil_mustrun_shares_for_DE(args)
        mustrun_de_total = 0.0
        for ef_name, share in de_mustrun_shares.items():
            if share <= 0.0:
                continue
            mask = (ef_keys == ef_name)
            if not np.any(mask):
                continue
            cap_ef = float((cap_base[mask] * float(args.therm_avail)).sum())
            mustrun_de_total += share * cap_ef
        residual = max(residual - mustrun_de_total, 0.0)
    
        # 4) DE-Kapazitätsprofil (inkl. Lignite-Mustrun)
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
    
        # 5) DE-marginal (nur wenn nach allen Abzügen noch Bedarf besteht)
        if (residual <= 0) or (t not in SRMC.index):
            unit_id = ef_dom = None
            eta_dom = srmc_dom = None
            mef_dom = np.nan
        else:
            srmc_t = SRMC.loc[t].to_numpy()
            order  = np.argsort(srmc_t, kind="mergesort")
            cumcap = np.cumsum(cap_t[order])
            pos = np.searchsorted(cumcap, residual, side="left")
            if pos >= len(order): pos = len(order) - 1
            uidx = order[pos]
            unit_id = units[uidx]
            ef_dom  = ef_keys[uidx]
            eta_dom = float(eta_arr[uidx])
            srmc_dom = float(srmc_t[uidx])
            mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom,1e-6)) * 1000.0 if ef_dom is not None else np.nan
    
        # 6) Preis-Cluster
        cluster_all = clusters.get(t, ["DE_LU"])
        # Nur Nachbarn (ohne DE_LU) für Kopplungs-Entscheid & Reporting
        coupled_neighbors = [z for z in cluster_all if z != "DE_LU"]
        coupling_active = len(coupled_neighbors) > 0
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan

        # EE-Überschuss nur bei sehr niedrigen Preisen und ohne Netto-Import
        ee_surplus = (
            (residual_domestic_fossil <= 1e-6) and
            np.isfinite(p_de) and (p_de <= float(args.ee_price_threshold)) and
            (net_imp <= 0.0)
        )


        # 7) Import-Stack NUR bauen, wenn KEIN EE-Überschuss ...
        marginal_import_label = None
        import_marg_srmc = None
        import_marg_mef  = None
        
        # cluster enthält jetzt nur AUSLAND; Kopplung liegt vor, wenn cluster != []
        if (not ee_surplus) and (net_imp > 0.0) and coupled_neighbors:
            I = net_imp
            stack_all = []
        
            if imp_cols:
                for c in imp_cols:
                    mw_imp_from_c = 0.0
                    if (t in flows.index) and (c in flows.columns):
                        val = flows.at[t, c]
                        mw_imp_from_c = float(val) if np.isfinite(val) else 0.0
                    if mw_imp_from_c <= 1e-6:
                        continue
        
                    z = imp_to_zone[c]
                    if z not in coupled_neighbors or z not in gen_by_zone or t not in gen_by_zone[z].index:
                        continue
        
                    gen_row = gen_by_zone[z].loc[t]
                    load_z  = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
                    fp_row  = fuel_prices.loc[t] if t in fuel_prices.index else None
                    if (not np.isfinite(load_z)) or (fp_row is None):
                        continue
        
                    min_total_t = float(nei_min_total_by_zone.get(z, pd.Series(0.0)).reindex([t]).fillna(0.0).iloc[0]) \
                                  if z in nei_min_total_by_zone else 0.0
                    min_by_fuel_t = {f: float(s.reindex([t]).fillna(0.0).iloc[0])
                                     for f, s in nei_min_by_zone_fuel.get(z, {}).items()}
        
                    blocks = exportable_blocks_for_zone(
                        t, z, gen_row, load_z, fp_row, nei_dists,
                        args.nei_eta_mode, int(args.nei_mc_draws), cap_mask,
                        reservoir_sp_map=reservoir_sp_map,
                        min_total_zone_t=min_total_t,
                        min_by_fuel_zone_t=min_by_fuel_t,
                    )
                    stack_all.extend(blocks)
            else:
                # Fallback: keine per-Interkonnektor-Flows → nimm alle gekoppelten Zonen
                for z in coupled_neighbors:
                    if z not in gen_by_zone or t not in gen_by_zone[z].index:
                        continue
                    gen_row = gen_by_zone[z].loc[t]
                    load_z  = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
                    fp_row  = fuel_prices.loc[t] if t in fuel_prices.index else None
                    if (not np.isfinite(load_z)) or (fp_row is None):
                        continue
        
                    min_total_t = float(nei_min_total_by_zone.get(z, pd.Series(0.0)).reindex([t]).fillna(0.0).iloc[0]) \
                                  if z in nei_min_total_by_zone else 0.0
                    min_by_fuel_t = {f: float(s.reindex([t]).fillna(0.0).iloc[0])
                                     for f, s in nei_min_by_zone_fuel.get(z, {}).items()}
        
                    blocks = exportable_blocks_for_zone(
                        t, z, gen_row, load_z, fp_row, nei_dists,
                        args.nei_eta_mode, int(args.nei_mc_draws), cap_mask,
                        reservoir_sp_map=reservoir_sp_map,
                        min_total_zone_t=min_total_t,
                        min_by_fuel_zone_t=min_by_fuel_t,
                    )
                    stack_all.extend(blocks)
        
            stack_all.sort(key=lambda x: x[1])
            I_remaining = I
            marg_block = None
            for (fuel, srmc, mw, eta, z) in stack_all:
                if I_remaining <= 1e-6:
                    break
                take = min(mw, I_remaining)
                I_remaining -= take
                marg_block = (fuel, srmc, eta, z)
        
            if marg_block is not None:
                fuel_m, srmc_m, eta_m, z_m = marg_block
                import_marg_srmc = float(srmc_m)
                if fuel_m in ("Reservoir Hydro", "EE"):
                    import_marg_mef  = 0.0
                    lbl = "Reservoir Hydro" if fuel_m == "Reservoir Hydro" else "EE"
                    marginal_import_label = f"{z_m}({lbl})"
                else:
                    ef_th = EF_LOOKUP_T_PER_MWH_TH.get(fuel_m, 0.30)
                    import_marg_mef  = float((ef_th / max(eta_m,1e-6)) * 1000.0)
                    marginal_import_label = f"{z_m}({fuel_m})"

        # --- Negativpreis-Regel auf Importseite (nur Import-Marginal ggf. auf 0 zwingen) ---
        if (net_imp > 0.0) and coupling_active and np.isfinite(p_de) and (p_de < 0.0) and stack_all:
            if _stack_has_zero_srmc(stack_all):
                # 0-SRMC-Blöcke bevorzugen und marginalen Importblock mit SRMC=0 setzen
                I_remaining = net_imp
                for (fuel, srmc, mw, eta, z) in [b for b in stack_all if b[0] in ("EE","Reservoir Hydro") and b[2] > 1e-6]:
                    if I_remaining <= 1e-6: break
                    take = min(mw, I_remaining)
                    I_remaining -= take
                    import_marg_srmc = 0.0
                    import_marg_mef  = 0.0
                    marginal_import_label = f"{z}({fuel})"
    
        # 8) Seitenwahl (IMMER ausführen, egal ob p_de < 0 oder nicht)
        if ee_surplus:
            marginal_side  = "DE"
            marginal_label = "EE_surplus"
            marginal_fuel  = "EE"
            marginal_eta   = np.nan
            marginal_srmc  = 0.0
            mef_gpkwh      = 0.0
        else:
            # Standard: in DE bestimmen
            marginal_side = "DE"
            marginal_label = unit_id if unit_id else "none"
            marginal_fuel  = ef_dom
            marginal_eta   = eta_dom if eta_dom is not None else np.nan
            marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan
            mef_gpkwh      = mef_dom
    
            # Import-Kandidat vs. Preisanker vergleichen
            if (import_marg_srmc is not None) and (net_imp > 0.0) and coupling_active:
                choose_side = None
                if args.price_anchor in ("closest","threshold") and np.isfinite(p_de):
                    cand = []
                    if np.isfinite(marginal_srmc): cand.append(("DE", abs(marginal_srmc - p_de)))
                    cand.append(("IMPORT", abs(import_marg_srmc - p_de)))
                    if args.price_anchor == "closest":
                        choose_side = min(cand, key=lambda x: x[1])[0]
                    else:
                        valid = [c for c in cand if c[1] <= float(args.price_tol)]
                        if valid: choose_side = min(valid, key=lambda x: x[1])[0]
                if choose_side is None:
                    choose_side = "IMPORT" if (not np.isfinite(marginal_srmc) or import_marg_srmc <= marginal_srmc) else "DE"
    
                if choose_side == "IMPORT":
                    marginal_side  = "IMPORT"
                    marginal_label = marginal_import_label or "import_stack"
                    marginal_fuel  = (marginal_import_label.split("(")[-1].rstrip(")")) if marginal_import_label else "mix"
                    marginal_eta   = np.nan
                    marginal_srmc  = float(import_marg_srmc)
                    mef_gpkwh      = float(import_marg_mef)
        # --- Peaker-Override bei hohen Preisen (nach choose_side, vor finalem Override) ---
        if getattr(args, "peak_switch", False):
            thr1, thr2 = _parse_two_floats(getattr(args, "peak_price_thresholds", "180,260"))
            # Bedingungen: Preis bekannt & sehr hoch, Bedarf > 0, kein EE-only-Surplus
            if np.isfinite(p_de) and (residual > 0.0) and (p_de >= thr1) and not ee_surplus:
                co2 = float(fuel_prices.loc[t, "co2_eur_t"]) if t in fuel_prices.index else np.nan

                # Kandidat 1: OCGT (Gas) als Standard-Peaker
                ocgt_eta = float(getattr(args, "peak_eta_ocgt", 0.36))
                gas_th   = float(fuel_prices.loc[t, "gas_eur_mwh_th"]) if t in fuel_prices.index else np.nan
                ocgt_srmc = (gas_th + co2 * EF_LOOKUP_T_PER_MWH_TH["Erdgas"]) / max(ocgt_eta, 1e-6) if np.isfinite(gas_th) else np.nan

                # Kandidat 2: Öl/Diesel-Peaker bei extremen Preisen
                oil_eta  = float(getattr(args, "peak_eta_oil", 0.33))
                oil_th   = float(fuel_prices.loc[t, "oil_eur_mwh_th"]) if t in fuel_prices.index else np.nan
                oil_srmc = (oil_th + co2 * EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"]) / max(oil_eta, 1e-6) if np.isfinite(oil_th) else np.nan

                # Schwellenlogik: ab thr2 -> Öl/Diesel, sonst OCGT
                if (p_de >= thr2) and np.isfinite(oil_srmc):
                    marginal_side  = "DE"
                    marginal_label = "DE_peaker_override_oil"
                    marginal_fuel  = "Heizöl schwer"
                    marginal_eta   = oil_eta
                    marginal_srmc  = float(oil_srmc)
                    mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"] / max(oil_eta,1e-6)) * 1000.0
                elif np.isfinite(ocgt_srmc):
                    marginal_side  = "DE"
                    marginal_label = "DE_peaker_override_ocgt"
                    marginal_fuel  = "Erdgas"
                    marginal_eta   = ocgt_eta
                    marginal_srmc  = float(ocgt_srmc)
                    mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["Erdgas"] / max(ocgt_eta,1e-6)) * 1000.0
                # Falls Preise extrem hoch, aber *Import* schon gewählt und Import-SRMC << p_de,
                # lässt du die Import-Entscheidung stehen (kein else nötig).
    
            # Finaler Überschuss-Override (nur wenn wirklich keine inländische Fossil-Residuallast und kein Import)
            if (residual_domestic_fossil <= 1e-6) and (net_imp <= 0.0):
                marginal_side  = "DE"
                if L <= FEE + 1e-6:
                    marginal_label = "FEE_only_surplus_override"; marginal_fuel = "EE"
                else:
                    marginal_label = "NonDisp_only_surplus_override"; marginal_fuel = "NonDisp"
                marginal_eta = np.nan; marginal_srmc = 0.0; mef_gpkwh = 0.0
    # --- Peaker-Override: aktiv, wenn Flag gesetzt und Preis vorhanden ---
    if getattr(args, "peak_switch", False) and np.isfinite(p_de):
        # Schwellen robust parsen (z.B. "180,260")
        try:
            thr = [float(x) for x in str(getattr(args, "peak_price_thresholds", "180,260")).split(",")]
            thr_gas = thr[0] if len(thr) > 0 else 180.0
            thr_oil = thr[1] if len(thr) > 1 else 260.0
        except Exception:
            thr_gas, thr_oil = 180.0, 260.0
    
        # Nur greifen lassen, wenn wir NICHT schon EE-surplus gewählt haben
        if not ee_surplus:
            if p_de >= thr_oil:
                # Öl/Diesel-Peaker
                marginal_side  = "DE"
                marginal_label = "DE_Peaker_Oil"
                marginal_fuel  = "Heizöl schwer"
                marginal_eta   = float(getattr(args, "peak_eta_oil", 0.33))
                ef_th          = EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"]
                marginal_srmc  = (fuel_prices.loc[t, "oil_eur_mwh_th"]
                                  + fuel_prices.loc[t, "co2_eur_t"] * ef_th) / max(marginal_eta, 1e-6)
                mef_gpkwh      = (ef_th / max(marginal_eta, 1e-6)) * 1000.0
    
            elif p_de >= thr_gas:
                # Gas-OCGT-Peaker
                marginal_side  = "DE"
                marginal_label = "DE_Peaker_OCGT"
                marginal_fuel  = "Erdgas"
                marginal_eta   = float(getattr(args, "peak_eta_ocgt", 0.36))
                ef_th          = EF_LOOKUP_T_PER_MWH_TH["Erdgas"]
                marginal_srmc  = (fuel_prices.loc[t, "gas_eur_mwh_th"]
                                  + fuel_prices.loc[t, "co2_eur_t"] * ef_th) / max(marginal_eta, 1e-6)
                mef_gpkwh      = (ef_th / max(marginal_eta, 1e-6)) * 1000.0
    
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
            "cluster_zones": "|".join(coupled_neighbors),
            "residual_domestic_fossil_MW": residual_domestic_fossil,
            "residual_after_trade_MW": residual,
        })
        debug_rows.append({
            "timestamp": t,
            "DE_unit_marginal": unit_id,
            "DE_fuel": ef_dom,
            "DE_eta": eta_dom,
            "DE_srmc": srmc_dom,
            "IMPORT_stack_srmc_marg": import_marg_srmc,
            "IMPORT_stack_mef_marg": import_marg_mef,
            "IMPORT_label": marginal_import_label or "",
            "cluster": "|".join(coupled_neighbors),
            "net_import_total_MW": net_imp,
            "price_DE": p_de,
            "ND_MW": ND,
            "Load_MW": L,
            "LIGNITE_MUSTRUN_ENFORCED_MW": lignite_mustrun_enforced,
        })

    


    # 10) Outputs
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    (outdir / "analysis").mkdir(exist_ok=True, parents=True)
    df_res.to_csv(outdir / "mef_track_c_2024.csv", index=True)
    df_dbg.to_csv(outdir / "_debug_hourly.csv", index=True)
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'}")
    print(f"[OK] Debug:       {outdir/'_debug_hourly.csv'}")

    # --- Validierung & Plots ---
    try:
        df_val, df_sum = validate_run(df_res, df_dbg, flows, nei_prices,
                                      epsilon_price=float(args.epsilon),
                                      price_anchor_mode=str(args.price_anchor))
        write_validation_report(outdir, df_val, df_sum)
        make_validation_plots(outdir, df_res, df_dbg, df_val, nei_prices)

        print("[VALIDATION] Summary:")
        for k, v in df_sum.iloc[0].items():
            print(f"  {k}: {v}")
    except Exception as e:
        print("[VALIDATION] Warnung – Validierung/Plots übersprungen wegen:", e)


if __name__ == "__main__":
    main(build_parser().parse_args())
