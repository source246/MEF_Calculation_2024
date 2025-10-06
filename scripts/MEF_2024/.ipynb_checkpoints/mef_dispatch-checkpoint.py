#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch-Backcast (Merit-Stack + Flüsse) für DE/LU (2024)
(Version mit zeitvariablem Lignite-Mustrun per Quantilprofil Peak/Off-peak, monatlich)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# ----------------------------- CLI -------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track C – Dispatch-Backcast (Merit-Stack + Flüsse)")
    p.add_argument("--fleet", required=True, help="CSV: fleet_de_units.csv (MaStR)")

    # NEU: sauberer Modus-Schalter
    p.add_argument("--mustrun_mode", choices=["off", "capacity", "gen_quantile"], default="off",
                   help="off=kein Mustrun; capacity=fester Anteil je Lignite-Einheit (--mustrun_lignite_q); gen_quantile=Profil aus realer Lignite-Gen (Quantil)")

    p.add_argument("--mustrun_lignite_q", type=float, default=0.0,
                   help="Nur im Mode 'capacity': fester Kapazitätsanteil je Lignite-Einheit (0..1)")

    # Parameter für gen_quantile
    p.add_argument("--mustrun_quantile", type=float, default=0.20,
                   help="Quantil (z.B. 0.20) für das Lignite-Mustrun-Profil (gen_quantile)")
    p.add_argument("--mustrun_peak_hours", default="08-20",
                   help="Peak-Fenster (lokal), z.B. 08-20 (Start inkl., Ende exkl.)")
    p.add_argument("--mustrun_monthly", action="store_true",
                   help="Quantile getrennt je Monat bilden")

    p.add_argument("--fuel_prices", required=True, help="CSV: prices_2024.csv")
    p.add_argument("--flows", required=True, help="CSV: flows_scheduled_DE_LU_2024_net.csv")
    p.add_argument("--start", default=None, help="Start (Europe/Berlin), z.B. 2024-01-01T00:00:00")
    p.add_argument("--end",   default=None, help="Ende exklusiv (Europe/Berlin), z.B. 2025-01-01T00:00:00")
    p.add_argument("--neighbor_gen_dir", required=True, help="Dir mit actual_gen_<ZONE>_2024.csv")
    p.add_argument("--neighbor_load_dir", required=True, help="Dir mit load_<ZONE>_2024.csv")
    p.add_argument("--neighbor_prices", required=True, help="CSV mit price_DE_LU, price_AT,...")
    p.add_argument("--outdir", required=True, help="Output-Ordner")
    p.add_argument("--epsilon", type=float, default=0.01, help="Preis-Kopplungs-Schwelle in €/MWh")
    p.add_argument("--eta_col", default=None, help="Effizienzspalte in Fleet (oder auto)")
    p.add_argument("--therm_avail", type=float, default=1.0, help="Verfügbarkeit thermischer Einheiten (0..1)")
    p.add_argument("--varom_json", default=None, help="JSON: {tech_or_fuel: varOM_eur_per_mwh_el}")
    p.add_argument("--year", type=int, default=2024)
    return p

# -------------------------- Helper: Time & IO --------------------------------

TZ = "Europe/Berlin"

def parse_ts(s: pd.Series) -> pd.DatetimeIndex:
    ser_utc = pd.to_datetime(s, errors="coerce", utc=True)
    return pd.DatetimeIndex(ser_utc).tz_convert(TZ)

def read_csv_smart(path: str, min_cols: int = 3) -> pd.DataFrame:
    seps = [",", ";", "\t", "|"]
    encs = ["utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] >= min_cols:
                    return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"CSV nicht lesbar: {path} – letzter Fehler: {last_err}")

def force_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame benötigt DatetimeIndex.")
    rule = "1h"
    return df.resample(rule).mean() if how == "mean" else df.resample(rule).sum()

def read_csv_auto_time(path: str, time_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in time_cols if c in df.columns), df.columns[0])
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

# -------------------------- Mappings & Defaults ------------------------------

EF_LOOKUP_T_PER_MWH_TH = {
    "Erdgas": 0.201,
    "Steinkohle": 0.335,
    "Braunkohle": 0.383,
    "Heizöl schwer": 0.288,
    "Heizöl leicht / Diesel": 0.266,
}
ETA_DEFAULT_BY_FUEL = {
    "Erdgas": 0.50, "Steinkohle": 0.42, "Braunkohle": 0.40, "Heizöl schwer": 0.38, "Heizöl leicht / Diesel": 0.38,
}
NEIGHBOR_TECHS = [
    "Fossil Gas","Fossil Hard coal","Fossil Oil","Fossil Brown coal/Lignite",
    "Nuclear","Biomass",
    "Hydro Run-of-river and poundage","Hydro Water Reservoir","Hydro Pumped Storage",
    "Wind Onshore","Wind Offshore","Solar","Waste",
]
FOSSIL_TECH_TO_FUEL = {
    "Fossil Gas": ("gas", "Erdgas"),
    "Fossil Hard coal": ("coal", "Steinkohle"),
    "Fossil Brown coal/Lignite": ("lignite", "Braunkohle"),
    "Fossil Oil": ("oil", "Heizöl schwer"),
}
PRICE_COLS = ["gas_eur_mwh_th","coal_eur_mwh_th","lignite_eur_mwh_th","oil_eur_mwh_th","co2_eur_t"]

def _norm(text: str) -> str:
    t = (text or "").lower().strip()
    t = t.replace("-", " ").replace("/", " ").replace(",", " ")
    return " ".join(t.split())

def map_fuel_to_price_and_ef(raw: str):
    t = _norm(raw)
    if any(k in t for k in ["erdgas","erdölgas","erdolgas","fossilgas","gas "]): return ("gas","Erdgas")
    if any(k in t for k in ["steinkohle","steinkohlen","wirbelschichtkohle"]):   return ("coal","Steinkohle")
    if any(k in t for k in ["braunkohle","rohbraunkohle","rohbraunkohlen"]):     return ("lignite","Braunkohle")
    if "heizöl" in t or "heizoel" in t or "diesel" in t or " öl" in t or "oel" in t:
        return ("oil","Heizöl leicht / Diesel" if "leicht" in t or "diesel" in t else "Heizöl schwer")
    return (None,None)

# -------------------------- Load Inputs --------------------------------------

def load_fuel_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    miss = [c for c in PRICE_COLS if c not in df.columns]
    if miss: raise ValueError(f"Fehlende Preisspalten: {miss}")
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_neighbor_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")

def load_neighbor_load(path_dir: str, zone: str) -> pd.Series:
    candidates = list(Path(path_dir).glob(f"load_{zone}_2024*.csv"))
    if not candidates: raise FileNotFoundError(f"Load-CSV fehlt: load_{zone}_2024*.csv in {path_dir}")
    df = read_csv_auto_time(str(candidates[0]), ["timestamp_cec","timestamp","time","timestamp_brussels","timestamp_utc"])
    load_col = next((c for c in df.columns if "ActualTotalLoad" in c), df.columns[0])
    return pd.to_numeric(df[load_col], errors="coerce")

def load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    candidates = list(Path(path_dir).glob(f"actual_gen_{zone}_2024*.csv"))
    if not candidates: raise FileNotFoundError(f"Gen-CSV fehlt: actual_gen_{zone}_2024*.csv in {path_dir}")
    df_raw = pd.read_csv(candidates[0])
    tcol = next((c for c in ["timestamp_cec","timestamp","time","datetime"] if c in df_raw.columns), df_raw.columns[0])
    df_raw.index = parse_ts(df_raw[tcol]); df_raw = df_raw.drop(columns=[tcol])
    keep = [c for c in df_raw.columns for tech in NEIGHBOR_TECHS if c==tech or c.startswith(tech)]
    df = df_raw[keep].copy()
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    # Duplikate gleicher Tech-Namen aggregieren
    col_map, agg = {}, {}
    for c in df.columns:
        key = next((tech for tech in NEIGHBOR_TECHS if c==tech or c.startswith(tech)), c)
        col_map.setdefault(key, []).append(c)
    for key, cols in col_map.items(): agg[key] = df[cols].sum(axis=1)
    return pd.DataFrame(agg, index=df.index).sort_index()

# -------------------------- Fleet & SRMC -------------------------------------

def load_fleet(path: str, eta_col: Optional[str]) -> pd.DataFrame:
    df = read_csv_smart(path, min_cols=5)
    eta_cols = [eta_col] if eta_col else ["Effizienz","Effizienz_imputiert","eta","Eta","wirkungsgrad"]
    use_eta = next((c for c in eta_cols if c and c in df.columns), None)
    if use_eta is None: raise ValueError(f"Keine Effizienzspalte gefunden: {eta_cols}")

    pcol = next((c for c in ["MW Nettonennleistung der Einheit","Leistung_MW","Nettonennleistung der Einheit","Nettonennleistung"] if c in df.columns), None)
    if pcol is None: raise ValueError("Keine Leistungs-Spalte (MW).")
    fcol = next((c for c in ["Hauptbrennstoff der Einheit","Energieträger","Hauptbrennstoff","Brennstoff"] if c in df.columns), None)
    if fcol is None: raise ValueError("Keine Brennstoff-Spalte.")

    idcol = next((c for c in df.columns if "MaStR" in c), df.columns[0])
    namecol = next((c for c in df.columns if "Anzeige-Name" in c or "Name der Einheit" in c or "Name" in c), idcol)

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

    # Effizienz plausibilisieren
    eta_def = out["ef_key"].map(lambda k: ETA_DEFAULT_BY_FUEL.get(k, np.nan)).to_numpy()
    eta_clean = pd.to_numeric(out["eta"], errors="coerce").to_numpy()
    if np.nanmedian(eta_clean) > 1.5: eta_clean = eta_clean / 100.0
    bad = (~np.isfinite(eta_clean)) | (eta_clean < 0.05) | (eta_clean > 0.80)
    eta_clean[bad] = eta_def[bad]
    out["eta"] = eta_clean

    out["available_mw"] = (
        pd.to_numeric(out["p_mw"], errors="coerce")
          .fillna(0.0).clip(lower=0).astype("float32")
    )

    return out.dropna(subset=["eta"])

def compute_unit_srmc_series(fleet: pd.DataFrame, fuel_prices: pd.DataFrame, varom_map: Dict[str, float]) -> Dict[str, pd.Series]:
    co2 = fuel_prices["co2_eur_t"]
    srmc_by_unit = {}
    for _, r in fleet.iterrows():
        price_col = f"{r['price_key']}_eur_mwh_th"
        fuel_th = fuel_prices[price_col]
        ef_th   = EF_LOOKUP_T_PER_MWH_TH.get(r["ef_key"], 0.30)
        eta     = r["eta"]
        varom   = varom_map.get(r["ef_key"], varom_map.get(r["price_key"], 0.0))
        srmc = (fuel_th + co2 * ef_th) / max(eta, 1e-6) + varom
        srmc_by_unit[r["unit_id"]] = srmc
    return srmc_by_unit

def _parse_peak_window(s: str) -> tuple[int,int]:
    a, b = s.split("-"); return int(a), int(b)

def build_lignite_mustrun_profile(
    de_gen: pd.DataFrame,
    idx: list[pd.Timestamp],
    quantile: float = 0.20,
    peak_window: str = "08-20",
    monthly: bool = True
) -> pd.Series:
    """
    Liefert Serie mustrun_MW[t] aus realer Lignite-Gen (Fossil Brown coal/Lignite).
    - getrennt nach Peak/Off-Peak (lokale Uhrzeit)
    - optional je Monat eigene Quantile
    """
    tech_col = "Fossil Brown coal/Lignite"
    if tech_col not in de_gen.columns:
        return pd.Series(0.0, index=pd.DatetimeIndex(idx))

    # auf den Zielfenster-Index abbilden
    lign = de_gen[tech_col].reindex(pd.DatetimeIndex(idx)).astype("float64")

    # Peak-Fenster vorbereiten
    h_start, h_end = map(int, peak_window.split("-"))  # z.B. "08-20" -> 8..20
    def _is_peak_hours(index: pd.DatetimeIndex) -> np.ndarray:
        # z.B. "08-20" -> Peak ist 08:00–19:59
        if h_start <= h_end:
            return (index.hour >= h_start) & (index.hour < h_end)
        else:
            # falls Fenster über Mitternacht ginge (hier nicht der Fall, aber robust)
            return (index.hour >= h_start) | (index.hour < h_end)

    # Ergebnis-Container
    out = pd.Series(index=lign.index, dtype="float64")

    if monthly:
        # je Monat und Peak/Off-Peak getrennte Quantile
        for month, sub in lign.groupby(lign.index.month):
            if len(sub) == 0:
                continue
            peak_mask    = _is_peak_hours(sub.index)
            offpeak_mask = ~peak_mask

            q_peak    = np.nanquantile(sub[peak_mask],    quantile) if peak_mask.any()    else 0.0
            q_offpeak = np.nanquantile(sub[offpeak_mask], quantile) if offpeak_mask.any() else 0.0

            mask_month = (lign.index.month == month)
            # wichtige: Masken immer auf den jeweiligen Index beziehen
            out.loc[mask_month &  _is_peak_hours(lign.index)] = q_peak
            out.loc[mask_month & ~_is_peak_hours(lign.index)] = q_offpeak
    else:
        peak_mask_all    = _is_peak_hours(lign.index)
        offpeak_mask_all = ~peak_mask_all
        q_peak    = np.nanquantile(lign[peak_mask_all],    quantile) if peak_mask_all.any()    else 0.0
        q_offpeak = np.nanquantile(lign[offpeak_mask_all], quantile) if offpeak_mask_all.any() else 0.0
        out[ peak_mask_all] = q_peak
        out[~peak_mask_all] = q_offpeak

    # Negative/NaN weg
    out = out.fillna(0.0).clip(lower=0.0)
    return out


# -------------------------- Merit & MEF --------------------------------------

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

def neighbor_marginal_from_gen(row_gen: pd.Series, fuel_prices_row: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    cand = []
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in row_gen.index: continue
        mw = row_gen.get(tech, 0.0)
        if pd.isna(mw) or mw <= 1.0: continue
        fuel_th = fuel_prices_row.get(f"{pk}_eur_mwh_th", np.nan)
        co2     = fuel_prices_row.get("co2_eur_t", np.nan)
        if pd.isna(fuel_th) or pd.isna(co2): continue
        ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)
        eta   = ETA_DEFAULT_BY_FUEL.get(ef_name, 0.40)
        srmc  = (fuel_th + co2 * ef_th) / max(eta, 1e-6)
        cand.append((ef_name, srmc, eta))
    if not cand: return None, None, None
    ef_name, srmc, eta = sorted(cand, key=lambda x: x[1])[-1]
    return ef_name, srmc, eta

# ------------------------------ Main -----------------------------------------

def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Preise
    fuel_prices = load_fuel_prices(args.fuel_prices)

    # 2) Fleet & SRMC
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

    # 3) Flows & Preise Nachbarn
    flows = load_flows(args.flows)
    if "net_import_total" not in flows.columns:
        imp_cols = [c for c in flows.columns if c.startswith("imp_")]
        flows["net_import_total"] = flows[imp_cols].sum(axis=1)

    nei_prices = load_neighbor_prices(args.neighbor_prices)
    clusters = cluster_zones_by_price(nei_prices, args.epsilon)

    # 4) Nachbar-Gen/Last + DE/LU
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

    nondisp_cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    nd_present = [c for c in nondisp_cols if c in de_gen.columns]
    de_nondisp = de_gen[nd_present].sum(axis=1).reindex(de_load.index).fillna(0.0)

    # 5) Gemeinsamer Index + Zeitfenster
    idx = sorted(de_load.index.intersection(fuel_prices.index).intersection(flows.index).intersection(nei_prices.index))
    def _to_berlin(ts_str: Optional[str]):
        if ts_str is None: return None
        ts = pd.Timestamp(ts_str)
        return ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
    start = _to_berlin(args.start) or idx[0]
    end   = _to_berlin(args.end)   or (idx[-1] + pd.Timedelta(hours=1))
    idx = [t for t in idx if (t >= start and t < end)]
    print(f"[INFO] Stunden im Lauf: {len(idx)} | Fenster: {start} .. {end} (exkl.)")

    # 6) Mustrun-Vorbereitung
    therm_avail = float(args.therm_avail)
    lign_mask   = (ef_keys == "Braunkohle")
    lign_total  = float(cap_base[lign_mask].sum())
    # Anteil jeder Lignite-Einheit an der Gesamt-Lignite-Kapazität (für Proportionalverteilung)
    lign_share = np.zeros_like(cap_base)
    if lign_total > 0:
        lign_share[lign_mask] = cap_base[lign_mask] / lign_total

    lignite_profile = None          # stündliches MW-Profil (nur für gen_quantile)
    mustrun_q_static = float(args.mustrun_lignite_q) if args.mustrun_mode == "capacity" else 0.0

    if args.mustrun_mode == "gen_quantile":
        lignite_profile = build_lignite_mustrun_profile(
            de_gen=de_gen,
            idx=idx,
            quantile=float(args.mustrun_quantile),
            peak_window=str(args.mustrun_peak_hours),
            monthly=bool(args.mustrun_monthly)
        )

    # 7) Hauptschleife
    results, debug_rows = [], []
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    imp_to_zone = {c: c.replace("imp_", "").replace("_", "") for c in imp_cols}

    for t in idx:
        L  = float(de_load.get(t, np.nan))
        ND = float(de_nondisp.get(t, 0.0))
        if not np.isfinite(L): continue
        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0

        residual = L - ND - net_imp
        if residual < 0: residual = 0.0

        # ---- Kapazität je Stunde aufbauen (Zeitvariable Mustrun) ----
        cap_t = cap_base * therm_avail

        lignite_mustrun_enforced = 0.0
        if args.mustrun_mode == "capacity" and mustrun_q_static > 0.0 and lign_total > 0:
            # statisch pro Einheit
            cap_t[lign_mask] = np.maximum(cap_t[lign_mask], cap_base[lign_mask] * mustrun_q_static)
            lignite_mustrun_enforced = float(cap_t[lign_mask].sum())

        elif args.mustrun_mode == "gen_quantile" and lignite_profile is not None and lign_total > 0:
            need = float(lignite_profile.get(t, 0.0))  # MW
            if need > 0.0:
                # proportional auf lignite-Einheiten verteilen
                target = lign_share * need
                cap_t = np.maximum(cap_t, target.astype(cap_t.dtype))
                lignite_mustrun_enforced = float(target[lign_mask].sum())

        # ---- Domestic marginal bestimmen ----
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

        # ---- Cluster & Importe ----
        cluster = clusters.get(t, ["DE_LU"])
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan

        imp_details, imp_sum = [], 0.0
        if len(cluster) > 1:
            for c in imp_cols:
                mw = float(flows.loc[t, c]) if t in flows.index else 0.0
                if mw <= 1e-6: continue
                z = imp_to_zone[c]
                if z not in cluster: continue
                if z not in gen_by_zone: continue
                if t not in gen_by_zone[z].index: continue
                gen_row = gen_by_zone[z].loc[t]
                fp_row  = fuel_prices.loc[t] if t in fuel_prices.index else None
                if fp_row is None: continue
                ef_name, srmc_z, eta_z = neighbor_marginal_from_gen(gen_row, fp_row)
                if ef_name is None: continue
                ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)
                mef_z = (ef_th / max(eta_z, 1e-6)) * 1000.0
                imp_details.append((z, mw, ef_name, eta_z, srmc_z, mef_z))
                imp_sum += mw

        import_marg_srmc = import_marg_mef = None
        if imp_details and imp_sum > 0:
            import_marg_srmc = sum(mw*srmc for (_z,mw,_ef,_eta,srmc,_mef) in imp_details) / imp_sum
            import_marg_mef  = sum(mw*mef  for (_z,mw,_ef,_eta,_srmc,mef) in imp_details) / imp_sum

        # ---- Finale Zuordnung ----
        marginal_side = "DE"
        marginal_label = unit_id if unit_id else "none"
        marginal_fuel  = ef_dom
        marginal_eta   = eta_dom if eta_dom is not None else np.nan
        marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan

        if unit_id is not None and ef_dom is not None and eta_dom is not None:
            mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom, 1e-6)) * 1000.0
        else:
            mef_dom = np.nan

        if (import_marg_srmc is not None) and (net_imp > 0.0) and (len(cluster) > 1):
            if (not np.isfinite(marginal_srmc)) or (import_marg_srmc >= marginal_srmc):
                marginal_side = "IMPORT"
                marginal_label = ",".join(sorted({z for (z, *_rest) in imp_details}))
                marginal_fuel  = "mix"
                marginal_eta   = np.nan
                marginal_srmc  = float(import_marg_srmc)
                mef_gpkwh      = float(import_marg_mef)
            else:
                mef_gpkwh = mef_dom
        else:
            mef_gpkwh = mef_dom

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
            "cluster": "|".join(cluster),
            "net_import_total_MW": net_imp,
            "price_DE": p_de,
            "ND_MW": ND,
            "Load_MW": L,
            "LIGNITE_MUSTRUN_ENFORCED_MW": lignite_mustrun_enforced,
        })

    # 8) Outputs
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    df_res.to_csv(outdir / "mef_track_c_2024.csv", index=True)
    df_dbg.to_csv(outdir / "_debug_hourly.csv", index=True)
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'}")
    print(f"[OK] Debug:       {outdir/'_debug_hourly.csv'}")

if __name__ == "__main__":
    main(build_parser().parse_args())
