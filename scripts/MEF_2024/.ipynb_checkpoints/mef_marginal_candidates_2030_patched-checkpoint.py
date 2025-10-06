#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MEF / marginale Kandidaten für DE+LU mit Preisanker (ε), ONR-Offers/SRMC und Speicher-Logik.

Überarbeitete 2030-Version nach Vorgaben:
- Import-Kandidaten nur aus Zonen, die (i) ε-gekoppelt sind und (ii) in der Stunde
  einen Nettofluss NACH DE/LU haben.
- Export-Stack der Nachbarn auf Basis "was bleibt übrig": Load − (fluktuierende + Non-Dispatchables).
  (Kein Mustrun-Abzug auf Nachbarseite.)
- Zwei-Stufen-0-MEF (FEE-only, dann Non-Disp-only) nur, wenn kein Import aus gekoppelten Zonen.
- Optionaler Peaker-Override (OCGT/Öl) bei hohen Preisen.
- PSP-Rolle via --psp_role steuerbar (allow|fallback|off) bleibt erhalten.

Outputs:
- mef_results.csv
- candidates_debug.csv (flache Kandidatenliste mit SRMC-Bändern je Stunde)
- candidates_by_hour.jsonl (JSONL je Stunde mit allen Kandidaten)
"""

import argparse
import json
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
import pytz
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CLI
# -----------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="Build marginal candidates & MEF decisions (DE/LU focus) with price-anchor coupling."
    )

    # Kerndateien
    p.add_argument("--eff", required=True, help="TYNDP/Effizienz-JSON (z.B. tyndp_2024_efficiencies_full.json)")
    p.add_argument("--prices", required=True, help="Day-Ahead-Preise (neighbor_prices_YYYY.csv)")
    p.add_argument("--flows", required=True, help="Netto-Importe nach DE/LU (flows_DE_LU_YYYY_net.csv)")
    p.add_argument("--de_gen", required=True, help="Erzeugung DE/LU (actual_gen_DE_LU_YYYY.csv)")
    p.add_argument("--nei_gen_dir", required=True, help="Ordner mit actual_gen_XX_YYYY.csv (Nachbarn)")
    p.add_argument("--nei_load_dir", required=True, help="Ordner mit load_XX_YYYY*.csv (Nachbarn) – für Export-Stack (ND-Residual).")
    p.add_argument("--fuel_ts", required=False, default=None, help="Zeitreihe Fuel/CO2-Preise (prices_YYYY.csv)")
    p.add_argument("--dispatched_dir", required=False, default=None,
                   help="Ordner, der ONR-Block-Parameter/Offers enthält (z.B. other_nonres_params_YYYY.json)")
    p.add_argument("--de_load", type=str, default=None,
                   help="Pfad zur DE/LU-Load-CSV (load_DE_LU_2030.csv). Überschreibt/ergänzt 'Load'.")
    # Steuerung
    p.add_argument("--co2_basis", choices=["el", "th"], default="el",
                   help="EF-Basis: 'el' (t/MWh_el, via η) oder 'th' (t/MWh_th). Default el.")
    p.add_argument("--eps_coupling", type=float, default=0.01,
                   help="Preisanker-Toleranz ε in €/MWh (|p_DE - p_X| <= ε).")
    p.add_argument("--tol_window", type=float, default=10.0,
                   help="Toleranz beim Kandidaten-Pick (Distanz Preis-Band).")
    p.add_argument("--onr_offer_band", type=float, default=10.0, help="±€/MWh Band um ONR-Offers.")
    p.add_argument("--fossil_band", type=float, default=5.0, help="±€/MWh Zusatzband für Fossile.")
    p.add_argument("--pump_band_up", type=float, default=50.0,
                   help="Max. Aufschlag über Pump-Mittel / P05 für Reservoir-Turbinen.")
    p.add_argument("--include_importers_within_tol", action="store_true",
                   help="(veraltet) – wir nutzen harte ε-Kopplung + tatsächlichen Importfluss.")
    p.add_argument("--psp_role", choices=["allow","fallback","off"], default="off",
                   help="Rolle der PSP-Turbinen bei der Kandidatenwahl: 'allow' = normal zulassen, 'fallback' = nur wenn sonst niemand passt, 'off' = nie als Kandidat.")
    p.add_argument("--allow_nd_import_price_setting", action="store_true",
                   help="Nicht-Dispatchables (Nuclear/Biomass/Waste) dürfen im IMPORT-Stack preissetzend sein (bei ND-Überschuss).")
    p.add_argument("--nd_price_setting", action="store_true",
                   help="Aktiviere 'Non-Dispatchables preissetzend' in DE, wenn residual after Non-Disp = 0.")
    # Peaker-Override (optional)
    p.add_argument("--peak_switch", action="store_true", help="Aktiviere Peaker-Override bei hohen Preisen (OCGT/Öl).")
    p.add_argument("--peak_price_thresholds", default="180,260",
                   help="Schwellen in €/MWh: 'p1,p2' -> p1≈OCGT, p2≈Öl/Diesel.")
    p.add_argument("--peak_eta_ocgt", type=float, default=0.36, help="η für OCGT (el.).")
    p.add_argument("--peak_eta_oil", type=float, default=0.33, help="η für öl-/diesel-Peaker (el.).")

    # Zeitfenster & Zeitzone
    p.add_argument("--start", type=str, default=None, help='Start UTC, z.B. "2030-01-01 00:00"')
    p.add_argument("--end", type=str, default=None, help='Ende UTC (exklusiv), z.B. "2031-01-01 00:00"')
    p.add_argument("--out_tz", type=str, default="UTC", help="Ausgabe-TZ (z.B. Europe/Berlin)")

    # Output
    p.add_argument("--outdir", required=True, help="Output-Ordner für CSV/JSONL")

    # Optionale Kompatibilitäts-Args (werden akzeptiert, nicht zwingend genutzt)
    p.add_argument("--nei_eta_mode", default="mean")
    p.add_argument("--nei_mc_draws", type=int, default=0)
    p.add_argument("--price_anchor", default="closest")
    p.add_argument("--price_tol", type=float, default=30.0)
    p.add_argument("--fossil_mustrun_mode", default="q_all")
    p.add_argument("--fossil_mustrun_q", type=float, default=0)
    p.add_argument("--disable_mustrun", action="store_true",
                   help="Ignoriere fossilen Mustrun komplett (kein Floor, MR=0).")
    return p.parse_args()


# -----------------------------
# Utilities
# -----------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower())


def read_de_load_csv(path: str, tz_hint: str = "UTC"):
    import pandas as pd
    df = pd.read_csv(path)
    cols = { _norm(c): c for c in df.columns }

    # Pflicht: timestamp & load_MW
    ts_col = cols.get("timestamp")
    if ts_col is None:
        raise ValueError("In de_load fehlt die Spalte 'timestamp'.")
    load_col = None
    for key in ("load_mw", "load", "demand_mw"):
        if key in cols:
            load_col = cols[key]; break
    if load_col is None:
        raise ValueError("In de_load fehlt eine Spalte 'load_MW' (oder 'load').")

    # optionale Spalten tolerant greifen
    def pick(*cands):
        for k in cands:
            if k in cols: return cols[k]
        return None

    pump_open   = pick("pump_storage_open_loop_pump_mw")
    pump_closed = pick("pump_storage_closed_loop_pump_mw")
    batt_charge = pick("battery_storage_charge_load_mw", "battery_charge_load_mw")
    el_load     = pick("electrolyser_load_mw", "electrolyser_mw")
    dsr_exp     = pick("demand_side_response_explicit_mw", "dsr_explicit_mw")
    dsr_imp     = pick("demand_side_response_implicit_mw", "dsr_implicit_mw")

    # numerisch machen & fehlende = 0
    for c in [load_col, pump_open, pump_closed, batt_charge, el_load, dsr_exp, dsr_imp]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["_pump_load"] = 0.0
    if pump_open is not None:   df["_pump_load"] += df[pump_open]
    if pump_closed is not None: df["_pump_load"] += df[pump_closed]

    df["_dsr_exp"] = df[dsr_exp] if dsr_exp is not None else 0.0
    df["_dsr_imp"] = df[dsr_imp] if dsr_imp is not None else 0.0

    df["Load_effective"] = (
         df[load_col]
       + df["_pump_load"]
       + (df[batt_charge] if batt_charge is not None else 0.0)
       + (df[el_load]     if el_load     is not None else 0.0)
       + df["_dsr_exp"]
       + df["_dsr_imp"]
    )

    # Zeit
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError("Konnte timestamps in de_load nicht parsen.")
    df["timestamp_utc"] = ts

    return df[["timestamp_utc", "Load_effective", "_pump_load"]].rename(
        columns={"_pump_load": "Pump_Load"}
    )


def parse_time_index(df: pd.DataFrame, tcol_like="timestamp") -> pd.DataFrame:
    tcols = [c for c in df.columns if tcol_like in c.lower()]
    if not tcols:
        raise ValueError("Keine Zeitspalte gefunden (erwarte 'timestamp' im Spaltennamen).")
    tcol = tcols[0]
    df = df.copy()
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    return df.set_index(tcol).sort_index()


def load_csv_indexed(path: str) -> pd.DataFrame:
    return parse_time_index(pd.read_csv(path))


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def any_contains(s: str, needles: List[str]) -> bool:
    s_low = s.lower()
    return any(n.lower() in s_low for n in needles)


# ---- Neighbor load + Reservoir-Shadow-Price --------------------------------

def load_nei_load_dir(path_dir: str) -> Dict[str, pd.Series]:
    """Liest je Zone die erste passende load_{ZONE}_YYYY*.csv und liefert {zone: hourly Series}."""
    from pathlib import Path
    out: Dict[str, pd.Series] = {}
    p = Path(path_dir)
    for f in sorted(p.glob("load_*_*.csv")):
        try:
            zone = f.stem.split("_")[1]
        except Exception:
            continue
        df = pd.read_csv(f)
        tcol = next((c for c in df.columns if "time" in c.lower() or "stamp" in c.lower()), df.columns[0])
        ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        df.index = ts
        valcol = next((c for c in df.columns if c != tcol and "load" in c.lower()), None)
        if valcol is None:
            # versuche generischen ersten Nicht-Zeit-Index
            val_candidates = [c for c in df.columns if c != tcol]
            if not val_candidates:
                continue
            valcol = val_candidates[0]
        ser = pd.to_numeric(df[valcol], errors="coerce").fillna(0.0).groupby(pd.Grouper(freq="1H")).mean()
        out[zone] = ser
    return out


def reservoir_shadow_price_series(prices_df: pd.DataFrame, zone: str, window_h: int = 24*7) -> pd.Series:
    """Opportunitätskosten ≈ rollierender Median des Zonenpreises."""
    col = f"price_{zone}"
    base = prices_df[col] if col in prices_df.columns else prices_df["price_DE_LU"]
    p = pd.to_numeric(base, errors="coerce")
    sp = p.rolling(window_h, min_periods=12).median().bfill().ffill()
    return sp.clip(lower=0.0)


# -----------------------------
# Config-Lookups (Labels)
# -----------------------------

ND_LABELS = [
    "Wind", "Solar", "Run-of-river", "ROR", "Hydro - Run-of-river",
    "Nuclear",
    "Biomass", "Waste", "Geothermal", "Marine",
    "Other Renewables"
]
PUMP_MATCH = ["Pumped Storage - Pump", "PSP Pump"]
ONR_MATCH  = ["Other Non-Renewables", "Others non-renewable", "ONR", "Others non renewable"]
TURB_MATCH = ["Pumped Storage - Turbine", "PSP Turbine",
              "Pump Storage - Open Loop (turbine)", "Pump Storage - Closed Loop (turbine)"]
PUMP_MATCH += ["Pump Storage - Open Loop (pump)", "Pump Storage - Closed Loop (pump)"]
ND_LABELS += ["Others renewable", "Others renewables"]

# Für Zero-MEF-Stufe 1/2:
FEE_COLS = [
    "Solar","Wind","Wind Onshore","Wind Offshore","Run-of-river","ROR","Hydro - Run-of-river",
]
NONDISP_EXTRA = ["Biomass","Waste","Nuclear","Geothermal","Marine","Other Renewables","Others renewable","Others renewables"]
# für ND-Preissetzung fokussieren wir auf klare Kandidaten:
ND_PRICE_TECHS = ["Nuclear","Biomass","Waste"]
DELU_PRICE_COL = "price_DE_LU"
PRICE_PREFIX = "price_"
FLOW_PREFIX = "imp_"

# -----------------------------
# Params aus Effizienz-JSON
# -----------------------------

@dataclass
class TechParam:
    fuel: str
    type_name: str
    eta_min: float
    eta_max: float
    eta_std: float
    co2_t_per_mwh_th: float
    vom_eur_per_mwh_el: float


def build_param_lut(tyndp_rows) -> Dict[Tuple[str, str], TechParam]:
    lut = {}
    for r in tyndp_rows:
        key = (r["fuel"], r["type"])
        lut[key] = TechParam(
            fuel=r["fuel"],
            type_name=r["type"],
            eta_min=r["eta_min_ncv"],
            eta_max=r["eta_max_ncv"],
            eta_std=r["eta_std_ncv"],
            co2_t_per_mwh_th=r["co2_ef_t_per_mwh_th"],
            vom_eur_per_mwh_el=r["vom_eur_per_mwh_el"]
        )
    return lut


# -----------------------------
# Fuel/CO2 Preise
# -----------------------------

FUEL_TS_COLS = {
    "gas": "gas_eur_mwh_th",
    "coal": "coal_eur_mwh_th",
    "lignite": "lignite_eur_mwh_th",
    "oil": "oil_eur_mwh_th",
    "co2": "co2_eur_t",
}
FUEL_MAP = {
    "Natural Gas": "gas",
    "Hard coal": "coal",
    "Lignite": "lignite",
    "Light oil": "oil",
    "Heavy oil": "oil",
    "Oil shale": "coal",
}
FUEL_CONST_EUR_PER_GJ = {
    "Nuclear": 1.7,
    "Lignite": 1.8,     # G2 default
    "Hard coal": 1.8,
    "Natural Gas": 6.3,
    "Light oil": 11.7,
    "Heavy oil": 9.6,
    "Oil shale": 1.9,
}
CO2_DEFAULT = 113.4  # €/t


def fuel_price_eur_mwh_th(fuel: str, ts_row: Optional[pd.Series]) -> float:
    if ts_row is not None:
        key = FUEL_MAP.get(fuel)
        if key and FUEL_TS_COLS[key] in ts_row.index:
            return float(ts_row[FUEL_TS_COLS[key]])
    gj = FUEL_CONST_EUR_PER_GJ.get(fuel)
    if gj is None:
        gj = 6.3 if "gas" in fuel.lower() else 1.8
    return gj * 3.6


def co2_price(ts_row: Optional[pd.Series]) -> float:
    if ts_row is not None and FUEL_TS_COLS["co2"] in ts_row.index:
        return float(ts_row[FUEL_TS_COLS["co2"]])
    return CO2_DEFAULT


# -----------------------------
# SRMC & MEF
# -----------------------------

def srmc_range_el(tp: TechParam, fuel: str, ts_row: Optional[pd.Series],
                  fossil_pad: float) -> Tuple[float, float, float]:
    """ SRMC [€/MWh_el] (min/mean/max) inkl. η-Streuung + VOM, plus Fossil-Band. """
    fuel_th = fuel_price_eur_mwh_th(fuel, ts_row)
    co2p = co2_price(ts_row)

    def srmc_eta(eta):
        fuel_term = fuel_th / max(eta, 1e-6)
        co2_term = (co2p * tp.co2_t_per_mwh_th) / max(eta, 1e-6)
        return fuel_term + co2_term + tp.vom_eur_per_mwh_el

    lo = srmc_eta(tp.eta_max)
    md = srmc_eta(tp.eta_std)
    hi = srmc_eta(tp.eta_min)

    if any(k in fuel.lower() for k in ["coal", "lignite", "gas", "oil", "shale"]):
        lo = max(0.0, lo - fossil_pad)
        hi = hi + fossil_pad
    return lo, md, hi


def mef_from_tp(tp: TechParam, basis="el") -> float:
    """ Rückgabe in g/kWh_el. """
    if basis == "th":
        ef_t_per_mwh_el = tp.co2_t_per_mwh_th / max(tp.eta_std, 1e-6)
    else:
        ef_t_per_mwh_el = tp.co2_t_per_mwh_th / max(tp.eta_std, 1e-6)
    return ef_t_per_mwh_el * 1000.0


# -----------------------------
# ND / Mustrun / Speicher
# -----------------------------

def cols_matching(df: pd.DataFrame, needles: List[str]) -> List[str]:
    return [c for c in df.columns if any_contains(c, needles)]


def fossil_mustrun_series(df: pd.DataFrame, fossil_cols: List[str], q=0.02) -> pd.Series:
    """ Heuristik: Tages-Pq je Spalte, dann Summe. """
    parts = []
    for c in fossil_cols:
        s = df[c].fillna(0.0)
        daily = s.resample("D").quantile(q)
        parts.append(daily.reindex(s.index, method="ffill"))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return sum(parts)


def hourly_p05_by_month(prices: pd.Series) -> pd.Series:
    df = prices.to_frame("p")
    df["m"] = df.index.month
    df["h"] = df.index.hour
    p05 = df.groupby(["m", "h"])["p"].quantile(0.05)
    out = []
    for ts in df.index:
        out.append(p05.loc[(ts.month, ts.hour)])
    return pd.Series(out, index=df.index)


def turbine_min_price(prices_de, gen_de, p05_hbm, band_up):
    pump_cols = cols_matching(gen_de, PUMP_MATCH)
    turb_cols = cols_matching(gen_de, TURB_MATCH)
    if not turb_cols:
        return pd.Series(np.nan, index=gen_de.index)

    pump_tot = gen_de[pump_cols].sum(axis=1) if pump_cols else pd.Series(0.0, index=gen_de.index)
    turb_tot = gen_de[turb_cols].sum(axis=1)

    out = pd.Series(np.nan, index=gen_de.index)
    acc_prices = []
    last_pump_mean = np.nan

    for ts in gen_de.index:
        p = float(prices_de.loc[ts])
        pump = float(pump_tot.loc[ts])
        turb = float(turb_tot.loc[ts])

        if pump > 1e-3:
            acc_prices.append(p)
            last_pump_mean = np.nan  # in Pump-Phase neu sammeln

        if turb > 1e-3:
            if not np.isnan(last_pump_mean) or acc_prices:
                base_min = (np.mean(acc_prices) if acc_prices else last_pump_mean)
                last_pump_mean = base_min
                acc_prices = []  # Pump-Block abgeschlossen
            else:
                base_min = 0.0
            p05 = float(p05_hbm.loc[ts]) if p05_hbm is not None else 0.0
            out.loc[ts] = max(base_min, p05)
    return out


# -----------------------------
# ONR Offers
# -----------------------------

def load_onr_offers(dispatched_dir: Optional[str]) -> List[dict]:
    if not dispatched_dir or not os.path.isdir(dispatched_dir):
        return []
    # Suche typische Dateien
    files = [f for f in os.listdir(dispatched_dir) if f.lower().endswith(".json")]
    out = []
    for f in files:
        try:
            j = load_json(os.path.join(dispatched_dir, f))
            if isinstance(j, list):
                out.extend(j)
            elif isinstance(j, dict):
                out.append(j)
        except Exception:
            pass
    return out


def onr_offer_range(offers: List[dict], country: str, eps: float) -> List[Tuple[str, float, float]]:
    out = []
    for blk in offers:
        if blk.get("country", "").lower() != country.lower():
            continue
        if "offer_eur_mwh" in blk:
            p = float(blk["offer_eur_mwh"])
            out.append((blk.get("block_id", "blk"), p - eps, p + eps))
    return out


# -----------------------------
# Hauptlogik
# -----------------------------

def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Laden
    prices = load_csv_indexed(args.prices)
    flows = load_csv_indexed(args.flows)
    gen_de = load_csv_indexed(args.de_gen)
    fuel_ts = load_csv_indexed(args.fuel_ts) if args.fuel_ts and os.path.exists(args.fuel_ts) else None
    tyndp = load_json(args.eff)
    PARAM = build_param_lut(tyndp)
    onr_offers_all = load_onr_offers(args.dispatched_dir)

    if args.de_load is not None:
        # Load-Datei einlesen (enthält timestamp_utc, Load_effective, Pump_Load)
        de_load_df = read_de_load_csv(args.de_load).set_index("timestamp_utc")
        # Index-Join auf Zeitachse (beide UTC)
        gen_de = gen_de.join(de_load_df, how="left")
        # 'Load' befüllen/ersetzen, falls nötig
        if "Load" not in gen_de.columns:
            gen_de["Load"] = gen_de["Load_effective"]
        else:
            gen_de["Load"] = gen_de["Load"].fillna(gen_de["Load_effective"])
        # Pump-Last bereitstellen (falls in Gen-Zeitreihe nicht vorhanden)
        if "Pump_Load" not in gen_de.columns:
            gen_de["Pump_Load"] = de_load_df["Pump_Load"].reindex(gen_de.index)
        gen_de.drop(columns=["Load_effective"], errors="ignore", inplace=True)

    # Safety: Wenn immer noch keine Load-Spalte vorhanden → harter Abbruch
    if "Load" not in gen_de.columns:
        raise ValueError("Es konnte keine Last ermittelt werden (weder in de_gen noch via --de_load).")

    # Filter Zeitfenster (UTC)
    if args.start:
        prices = prices.loc[pd.Timestamp(args.start, tz="UTC"):]
        flows = flows.loc[pd.Timestamp(args.start, tz="UTC"):]
        gen_de = gen_de.loc[pd.Timestamp(args.start, tz="UTC"):]
        if fuel_ts is not None: fuel_ts = fuel_ts.loc[pd.Timestamp(args.start, tz="UTC"):]
    if args.end:
        prices = prices.loc[:pd.Timestamp(args.end, tz="UTC")]
        flows = flows.loc[:pd.Timestamp(args.end, tz="UTC")]
        gen_de = gen_de.loc[:pd.Timestamp(args.end, tz="UTC")]
        if fuel_ts is not None: fuel_ts = fuel_ts.loc[:pd.Timestamp(args.end, tz="UTC")]

    # Spalten
    assert DELU_PRICE_COL in prices.columns, f"{DELU_PRICE_COL} fehlt in {args.prices}"
    load_col = "Load" if "Load" in gen_de.columns else ("load" if "load" in gen_de.columns else None)
    if load_col is None:
        raise ValueError("In de_gen wird eine Spalte 'Load' (oder 'load') erwartet.")

    # ND / ONR / Speicher / Fossil-Spalten
    ND_cols = cols_matching(gen_de, ND_LABELS)
    ONR_cols = cols_matching(gen_de, ONR_MATCH)
    PUMP_cols = cols_matching(gen_de, PUMP_MATCH)
    TURB_cols = cols_matching(gen_de, TURB_MATCH)
    EXCLUDE = {load_col, "Pump_Load", "timestamp_local"}
    EXCLUDE |= {c for c in gen_de.columns if c.lower().endswith("_load") or "load" in c.lower()}
    fossil_cols = [c for c in gen_de.columns
                   if c not in ND_cols + ONR_cols + PUMP_cols + TURB_cols and c not in EXCLUDE]

    ND_DE = gen_de[ND_cols].sum(axis=1) if ND_cols else pd.Series(0.0, index=gen_de.index)
    MR_DE = fossil_mustrun_series(gen_de, fossil_cols, q=args.fossil_mustrun_q)
    if args.disable_mustrun or args.fossil_mustrun_q < 0:
        MR_DE = pd.Series(0.0, index=gen_de.index)
    P_DE = prices[DELU_PRICE_COL]
    P05_HBM = hourly_p05_by_month(P_DE)
    TURB_MIN = turbine_min_price(P_DE, gen_de, P05_HBM, args.pump_band_up)

    # Hilfsfunktionen intern
    def coupled_zones(ts):
        p_de = float(prices.loc[ts, DELU_PRICE_COL])
        zs = []
        for c in prices.columns:
            if not c.startswith(PRICE_PREFIX) or c == DELU_PRICE_COL:
                continue
            if abs(float(prices.loc[ts, c]) - p_de) <= args.eps_coupling:
                zs.append(c.replace(PRICE_PREFIX, ""))
        return zs

    # Vor den Hilfsfunktionen (einmalig nach dem Laden von prices/flows):
    PRICE_ZONES = [c.replace(PRICE_PREFIX, "") for c in prices.columns if c.startswith(PRICE_PREFIX)]
    FLOW_ZONES  = [c.replace(FLOW_PREFIX,  "") for c in flows.columns  if c.startswith(FLOW_PREFIX)]

    # einfache Alias-Heuristik: exakte Treffer, ohne "00", Prefix-Vergleich
    ZONE_ALIAS: Dict[str, str] = {}
    for z in PRICE_ZONES:
        cand = [f for f in FLOW_ZONES
                if f == z or f.replace("00","") == z.replace("00","") or f.startswith(z[:2]) or z.startswith(f[:2])]
        if cand:
            # nimm den spezifischsten Kandidaten
            ZONE_ALIAS[z] = max(cand, key=len)

    def import_from(zone, ts):
        # bevorzugt Alias, sonst fallback: fuzzy alle passenden imp_-Spalten summieren
        z = ZONE_ALIAS.get(zone, zone)
        exact = f"{FLOW_PREFIX}{z}"
        if exact in flows.columns:
            return float(flows.loc[ts, exact])
        # Fallback: alle imp_-Spalten, deren Suffix "ähnlich" ist, summieren
        cols = [c for c in flows.columns if c.startswith(FLOW_PREFIX)
                and (c.endswith(zone) or c.endswith(z) or c.endswith(zone.replace("00","")))]
        return float(flows.loc[ts, cols].sum()) if cols else 0.0

    def tp_lookup(fuel, tname="old 1"):
        return PARAM.get((fuel, tname)) or PARAM.get((fuel, "-")) or PARAM.get((fuel, "new")) or \
               TechParam(fuel=fuel, type_name=tname, eta_min=0.4, eta_max=0.5, eta_std=0.45,
                         co2_t_per_mwh_th=0.2, vom_eur_per_mwh_el=1.5)

    def guess_fuel(colname: str) -> str:
        n = colname.lower()
        if "lignite" in n: return "Lignite"
        if "coal" in n:   return "Hard coal"
        if "gas" in n:    return "Natural Gas"
        if "light oil" in n: return "Light oil"
        if "heavy oil" in n: return "Heavy oil"
        if "shale" in n:  return "Oil shale"
        return "Natural Gas"

    # ---- oberhalb von dom_candidates / nei_candidates definieren ----
    TYPE_RULES = [
        (r"ccgt.*present.*1", ("Natural Gas","CCGT present 1")),
        (r"ccgt.*present.*2", ("Natural Gas","CCGT present 2")),
        (r"ccgt.*new",        ("Natural Gas","CCGT new")),
        (r"ccgt.*old.*1",     ("Natural Gas","CCGT old 1")),
        (r"ccgt.*old.*2",     ("Natural Gas","CCGT old 2")),
        (r"ocgt.*old",        ("Natural Gas","OCGT old")),
        (r"ocgt.*new",        ("Natural Gas","OCGT new")),
        (r"hard.*coal.*new",  ("Hard coal","new")),
        (r"hard.*coal.*old.*1",("Hard coal","old 1")),
        (r"hard.*coal.*old.*2",("Hard coal","old 2")),
        (r"lignite.*new",     ("Lignite","new")),
        (r"lignite.*old.*1",  ("Lignite","old 1")),
        (r"lignite.*old.*2",  ("Lignite","old 2")),
        (r"light.*oil",       ("Light oil","-")),
        (r"heavy.*oil",       ("Heavy oil","-")),
        (r"oil.*shale.*new",  ("Oil shale","new")),
        (r"oil.*shale.*old",  ("Oil shale","old")),
    ]

    def infer_fuel_type(colname: str):
        n = colname.lower()
        for pat,(fu,ty) in TYPE_RULES:
            if re.search(pat, n): return fu, ty
        return guess_fuel(colname), "-"

    def dom_candidates(ts) -> List[Dict]:
        out: List[Dict] = []
        ts_row_prices = fuel_ts.loc[ts] if (fuel_ts is not None and ts in fuel_ts.index) else None

        # Fossile (alle aktiven MW als Kandidaten)
        for c in fossil_cols:
            gen = float(gen_de.loc[ts, c])
            if gen <= 1e-6:
                continue
            fuel, tname = infer_fuel_type(c)
            tp = tp_lookup(fuel, tname)
            lo, md, hi = srmc_range_el(tp, fuel, ts_row_prices, args.fossil_band)
            out.append(dict(land="DE_LU", unit=c, fuel=fuel, type=tp.type_name,
                            srmc_lo=lo, srmc_md=md, srmc_hi=hi, gen_MW=gen, label="DE_fossil"))

        # ONR
        if ONR_cols:
            offers = onr_offer_range(onr_offers_all, "DE_LU", args.onr_offer_band)
            for c in ONR_cols:
                gen = float(gen_de.loc[ts, c])
                if gen <= 1e-6:
                    continue
                if offers:
                    for blk_id, plo, phi in offers:
                        out.append(dict(land="DE_LU", unit=f"ONR_{blk_id}", fuel="ONR", type="-",
                                        srmc_lo=plo, srmc_md=(plo+phi)/2, srmc_hi=phi, gen_MW=gen, label="DE_ONR_offer"))
                else:
                    # Fallback: SRMC wie Gas (oder per Blockdaten erweitern)
                    fuel = "Natural Gas"
                    tp = tp_lookup(fuel)
                    lo, md, hi = srmc_range_el(tp, fuel, ts_row_prices, args.fossil_band)
                    out.append(dict(land="DE_LU", unit=c, fuel="ONR", type="-",
                                    srmc_lo=lo, srmc_md=md, srmc_hi=hi, gen_MW=gen, label="DE_ONR_srmc"))

        # PSP-Turbinen (optional)
        turb = float(gen_de.loc[ts, TURB_cols].sum()) if TURB_cols else 0.0
        if turb > 1e-6 and args.psp_role != "off":
            pmin = float(TURB_MIN.loc[ts]) if not math.isnan(TURB_MIN.loc[ts]) else 0.0
            out.append(dict(
                land="DE_LU", unit="PSP_Turbine", fuel="Pumped Storage", type="turbine",
                srmc_lo=pmin, srmc_md=pmin + args.pump_band_up/2.0,
                srmc_hi=pmin + args.pump_band_up, gen_MW=turb, label="DE_psp_turbine"))
        return out



    # Cache Nachbar-Gen & -Load & Reservoir-Shadow-Prices
    _nei_cache: Dict[str, Optional[pd.DataFrame]] = {}
    _nei_load: Dict[str, pd.Series] = load_nei_load_dir(args.nei_load_dir) if args.nei_load_dir else {}
    _reservoir_sp: Dict[Tuple[str, pd.Timestamp], float] = {}

    def load_nei(zone: str) -> Optional[pd.DataFrame]:
        if zone in _nei_cache:
            return _nei_cache[zone]
        path = os.path.join(args.nei_gen_dir, f"actual_gen_{zone}_2030.csv")
        if not os.path.exists(path):
            path = os.path.join(args.nei_gen_dir, f"actual_gen_{zone}.csv")
        if not os.path.exists(path):
            _nei_cache[zone] = None
        else:
            _nei_cache[zone] = load_csv_indexed(path)
        return _nei_cache[zone]

    # Shadow-Prices vorbereiten
    for c in prices.columns:
        if not c.startswith(PRICE_PREFIX):
            continue
        z = c.replace(PRICE_PREFIX, "")
        sp = reservoir_shadow_price_series(prices, z)
        for tt, val in sp.items():
            _reservoir_sp[(z, tt)] = float(val) if np.isfinite(val) else 0.0

    def coupled_importing_zones(ts) -> List[str]:
        """Nur Zonen, die ε-gekoppelt sind UND in der Stunde import nach DE/LU liefern."""
        zs: List[str] = []
        for z in coupled_zones(ts):
            if import_from(z, ts) > 1e-6:
                zs.append(z)
        return zs

    def nei_candidates(ts, zones: List[str]) -> List[Dict]:
        """
        Export-Stack je Zone (nur ND abziehen). Liefert exportierbare Blöcke als Kandidaten:
        - Reservoir (SRMC≈Shadow-Price) mit vollem MW,
        - Fossile: MW_export = max(gen_fossil − benötigter Anteil zur Deckung (Load−ND), 0).
        """
        out: List[Dict] = []
        ts_row_prices = fuel_ts.loc[ts] if (fuel_ts is not None and ts in fuel_ts.index) else None
        for z in zones:
            if import_from(z, ts) <= 1e-6:
                continue
            g = load_nei(z)
            if g is None or ts not in g.index:
                continue

            # Load & ND (fluktuierende + Non-Disp) der Zone
            load_ser = _nei_load.get(z)
            if load_ser is None or ts not in load_ser.index:
                continue
            load_z = float(load_ser.loc[ts])
            row = g.loc[ts]

            fee_sum = float(pd.to_numeric(row.reindex(FEE_COLS).fillna(0.0)).sum())
            nd_extra = float(pd.to_numeric(row.reindex(NONDISP_EXTRA).fillna(0.0)).sum())
            ND_sum = fee_sum + nd_extra
            residual_need = max(load_z - ND_sum, 0.0)  # nur ND abziehen (kein Mustrun)

            # Reservoir-Block (Opportunitätskosten via Shadow-Price)
            # akzeptiere typische Spaltennamen
            res_keys = [k for k in row.index if "Hydro Water Reservoir" in k or "Reservoir" in k]
            res_mw = float(pd.to_numeric(row[res_keys], errors="coerce").sum()) if res_keys else 0.0
            if res_mw > 1e-6 and args.psp_role != "off":
                sp = _reservoir_sp.get((z, ts), float(prices.loc[ts, f"{PRICE_PREFIX}{z}"]))
                out.append(dict(land=z, unit="Reservoir", fuel="Hydro Reservoir", type="turbine",
                                srmc_lo=sp, srmc_md=sp, srmc_hi=sp, gen_MW=res_mw, label="NEI_reservoir_shadow"))

            # Fossile Exportkapazitäten
            NDc = cols_matching(g, ND_LABELS)
            TURBc = cols_matching(g, TURB_MATCH)
            ONRc = cols_matching(g, ONR_MATCH)
            fossilc = [c for c in g.columns if c not in NDc + TURBc + ONRc and c.lower() != "load"]

            # innerzonal sortiert nach SRMC, um „benötigt“ abzuziehen
            pool: List[Tuple[float,str,str,TechParam,float]] = []  # (md, col, fuel, tp, avail)
            for c in fossilc:
                avail = float(row.get(c, 0.0))
                if avail <= 1e-6:
                    continue
                fuel, tname = infer_fuel_type(c)
                tp = tp_lookup(fuel, tname)
                lo, md, hi = srmc_range_el(tp, fuel, ts_row_prices, args.fossil_band)
                pool.append((md, c, fuel, tp, avail))
            pool.sort(key=lambda x: x[0])

            need = residual_need
            used_by: Dict[str, float] = {}
            for md, c, fuel, tp, avail in pool:
                if need <= 1e-6:
                    break
                take = min(avail, max(need, 0.0))
                used_by[c] = take
                need -= take

            # exportierbarer Überschuss = MW - „benötigt“
            for md, c, fuel, tp, avail in pool:
                surplus = max(avail - float(used_by.get(c, 0.0)), 0.0)
                if surplus <= 1e-6:
                    continue
                lo, md2, hi = srmc_range_el(tp, fuel, ts_row_prices, args.fossil_band)
                out.append(dict(land=z, unit=c, fuel=fuel, type=tp.type_name,
                                srmc_lo=lo, srmc_md=md2, srmc_hi=hi, gen_MW=surplus, label="NEI_exportable_fossil"))

            # PSP-Turbinen als Kandidat (Preis ~ Zonenpreis), wenn erlaubt
            if args.psp_role != "off":
                for c in TURBc:
                    val = float(row.get(c, 0.0))
                    if val <= 1e-6:
                        continue
                    p_x = float(prices.loc[ts, f"{PRICE_PREFIX}{z}"]) if f"{PRICE_PREFIX}{z}" in prices.columns else float(prices.loc[ts, DELU_PRICE_COL])
                    out.append(dict(
                        land=z, unit="PSP_Turbine", fuel="Pumped Storage", type="turbine",
                        srmc_lo=p_x, srmc_md=p_x + args.pump_band_up/2.0,
                        srmc_hi=p_x + args.pump_band_up, gen_MW=val, label="NEI_psp_turbine"))
        return out

    # Output-Container
    results: List[Dict] = []
    cand_records: List[pd.DataFrame] = []  # flache Liste für CSV
    jsonl_path = os.path.join(args.outdir, "candidates_by_hour.jsonl")
    # JSONL bei neuem Lauf leeren (optional, aber nützlich)
    open(jsonl_path, "w", encoding="utf-8").close()
    csv_res_path = os.path.join(args.outdir, "mef_results.csv")
    csv_cand_path = os.path.join(args.outdir, "candidates_debug.csv")

    # Loop
    for ts in gen_de.index:
        if ts not in prices.index or ts not in flows.index:
            continue

        L = float(gen_de.loc[ts, load_col])
        p_de = float(prices.loc[ts, DELU_PRICE_COL])
        ND = float(ND_DE.loc[ts]) if ts in ND_DE.index else 0.0
        residual_no_mr = L - ND

        # Zwei-Stufen-0-MEF (nur ohne Import aus gekoppelten Zonen)
        fee = 0.0
        for c in FEE_COLS:
            if c in gen_de.columns:
                fee += float(gen_de.loc[ts, c])
        nd_extra = 0.0
        for c in NONDISP_EXTRA:
            if c in gen_de.columns:
                nd_extra += float(gen_de.loc[ts, c])
        net_imp_total = float(flows.loc[ts, [c for c in flows.columns if c.startswith("imp_")]].sum())
        coupled_imp_active = len(coupled_importing_zones(ts)) > 0

        if L <= fee + 1e-6 and not (net_imp_total > 0.0 and coupled_imp_active):
            results.append(dict(timestamp=str(ts), mef_g_per_kwh=0.0, marginal_side="DE",
                                marginal_label="FEE_only_surplus", marginal_fuel="EE",
                                marginal_eta=np.nan, marginal_srmc_eur_per_mwh=0.0,
                                price_DE=p_de, cluster_zones="|".join(coupled_zones(ts)),
                                residual_domestic_fossil_MW=0.0, residual_after_trade_MW=0.0))
            continue
        if L <= (fee + nd_extra) + 1e-6 and not (net_imp_total > 0.0 and coupled_imp_active):
            results.append(dict(timestamp=str(ts), mef_g_per_kwh=0.0, marginal_side="DE",
                                marginal_label="NonDisp_only_surplus", marginal_fuel="NonDisp",
                                marginal_eta=np.nan, marginal_srmc_eur_per_mwh=0.0,
                                price_DE=p_de, cluster_zones="|".join(coupled_zones(ts)),
                                residual_domestic_fossil_MW=0.0, residual_after_trade_MW=0.0))
            continue

        # B) fossiler Mustrun reicht (DE) – beibehalten
        MR = float(MR_DE.loc[ts]) if ts in MR_DE.index else 0.0
        if (not args.disable_mustrun) and (residual_no_mr <= MR + 1e-6):
            # Mustrun-Floor ~ Mix pro Fuel (Anteile aus Fossilspalten in Stunde ts)
            mix: Dict[str,float] = {}
            eta_map: Dict[str,float] = {}
            ef_map: Dict[str,float] = {}
            for c in fossil_cols:
                v = float(gen_de.loc[ts, c])
                if v <= 1e-6: continue
                fuel = guess_fuel(c)
                mix[fuel] = mix.get(fuel, 0.0) + v
                tp = tp_lookup(fuel)
                eta_map[fuel] = tp.eta_std
                ef_map[fuel] = tp.co2_t_per_mwh_th
            # g/kWh
            num = 0.0
            den = 0.0
            for f, mw in mix.items():
                eta = max(eta_map.get(f, 0.45), 1e-6)
                ef_el = ef_map.get(f, 0.2) / eta  # t/MWh_el
                num += ef_el * mw
                den += mw
            mef = (num/den)*1000.0 if den > 0 else 0.0
            results.append(dict(timestamp=str(ts), mef_g_per_kwh=mef, marginal_side="DE",
                                marginal_label="MEF_mustrun_floor", marginal_fuel="MustrunMix",
                                marginal_eta=np.nan, marginal_srmc_eur_per_mwh=np.nan,
                                price_DE=p_de, cluster_zones="|".join(coupled_zones(ts)),
                                residual_domestic_fossil_MW=residual_no_mr - MR, residual_after_trade_MW=0.0))
            continue

        # C) reguläre Merit-Order: DE vs. gekoppelte Import-Zonen (nur ε-gekoppelt + importierend)
        resid_dom_fossil = residual_no_mr - MR
        cz = coupled_importing_zones(ts)
        imp_total = sum(import_from(z, ts) for z in cz)
        resid_after_trade = resid_dom_fossil - imp_total

        nei_cands = nei_candidates(ts, cz)
        dom_cands = dom_candidates(ts)
        cluster = "|".join(cz)

        if resid_after_trade > 1e-6 or not nei_cands:
            cands = dom_cands
            side = "DE"
        else:
            cands = nei_cands if nei_cands else dom_cands
            side = "NEI"

        if not cands:
            results.append(dict(timestamp=str(ts), mef_g_per_kwh=0.0, marginal_side=side,
                                marginal_label="fallback_no_candidates", marginal_fuel="",
                                marginal_eta=np.nan, marginal_srmc_eur_per_mwh=np.nan,
                                price_DE=p_de, cluster_zones=cluster,
                                residual_domestic_fossil_MW=resid_dom_fossil,
                                residual_after_trade_MW=resid_after_trade))
            continue

        # PSP-Fallback-Regel (falls gewünscht): wenn psp_role=="fallback", dann PSP-Kandidaten nur nehmen,
        # wenn keine Nicht-PSP-Bänder den Preis treffen.
        dfc = pd.DataFrame(cands)
        if args.psp_role == "fallback" and ("Pumped Storage" in dfc["fuel"].values):
            non_psp = dfc[dfc["fuel"] != "Pumped Storage"].copy()
            non_psp["price_hit"] = np.where(
                (non_psp["srmc_lo"] - 1e-9 <= p_de) & (p_de <= non_psp["srmc_hi"] + 1e-9),
                0.0,
                np.minimum(abs(non_psp["srmc_lo"] - p_de), abs(non_psp["srmc_hi"] - p_de))
            )
            if not non_psp.empty and (non_psp["price_hit"] == 0.0).any():
                dfc = non_psp

        # Pick: Band trifft Preis, sonst kleinste Distanz
        dfc["price_hit"] = np.where(
            (dfc["srmc_lo"] - 1e-9 <= p_de) & (p_de <= dfc["srmc_hi"] + 1e-9),
            0.0,
            np.minimum(abs(dfc["srmc_lo"] - p_de), abs(dfc["srmc_hi"] - p_de))
        )
        dfc = dfc.sort_values(["price_hit", "srmc_md"]).reset_index(drop=True)
        best = dfc.iloc[0].to_dict()

        # Peaker-Override (optional) – nur wenn Bedarf > 0 und kein Zero-MEF-Fall
        if args.peak_switch and np.isfinite(p_de) and resid_after_trade > 1e-6:
            try:
                thr_parts = [float(x) for x in str(getattr(args, "peak_price_thresholds", "180,260")).split(",")]
                thr_gas = thr_parts[0] if len(thr_parts) > 0 else 180.0
                thr_oil = thr_parts[1] if len(thr_parts) > 1 else 260.0
            except Exception:
                thr_gas, thr_oil = 180.0, 260.0
            if p_de >= thr_gas:
                ts_row_prices = fuel_ts.loc[ts] if (fuel_ts is not None and ts in fuel_ts.index) else None
                co2p = co2_price(ts_row_prices)
                gas_th = fuel_price_eur_mwh_th("Natural Gas", ts_row_prices)
                oil_th = fuel_price_eur_mwh_th("Heavy oil",  ts_row_prices)
                # konservative EF (t/MWh_th) – falls Param fehlt
                ef_g = PARAM.get(("Natural Gas","-"), TechParam("Natural Gas","-",0.3,0.6,0.36,0.201,0.0)).co2_t_per_mwh_th
                ef_o = PARAM.get(("Heavy oil","-"),  TechParam("Heavy oil","-",0.3,0.6,0.33,0.288,0.0)).co2_t_per_mwh_th
                ocgt_eta = float(getattr(args, "peak_eta_ocgt", 0.36))
                oil_eta  = float(getattr(args, "peak_eta_oil", 0.33))
                ocgt_srmc = (gas_th + co2p*ef_g)/max(ocgt_eta,1e-6)
                oil_srmc  = (oil_th + co2p*ef_o)/max(oil_eta,1e-6)
                if p_de >= thr_oil:
                    best = dict(land="DE_LU", unit="Peaker_Oil", fuel="Heavy oil", type="-",
                                srmc_lo=oil_srmc, srmc_md=oil_srmc, srmc_hi=oil_srmc, gen_MW=np.nan, label="DE_peaker_oil")
                    side = "DE"
                else:
                    best = dict(land="DE_LU", unit="Peaker_OCGT", fuel="Natural Gas", type="-",
                                srmc_lo=ocgt_srmc, srmc_md=ocgt_srmc, srmc_hi=ocgt_srmc, gen_MW=np.nan, label="DE_peaker_ocgt")
                    side = "DE"

        # MEF
        if best["fuel"] in ("Hydro Reservoir", "Pumped Storage"):
            mef_gpkwh = 0.0
            tp_used = None
        elif best["fuel"] == "ONR":
            tp_used = tp_lookup("Natural Gas")
            mef_gpkwh = mef_from_tp(tp_used, basis=args.co2_basis)
        else:
            tp_used = tp_lookup(str(best["fuel"]), best.get("type", "old 1"))
            mef_gpkwh = mef_from_tp(tp_used, basis=args.co2_basis)

        results.append(dict(timestamp=str(ts), mef_g_per_kwh=mef_gpkwh, marginal_side=side,
                            marginal_label=best["label"], marginal_fuel=best["fuel"],
                            marginal_eta=(tp_used.eta_std if tp_used is not None else np.nan),
                            marginal_srmc_eur_per_mwh=float(best["srmc_md"]),
                            price_DE=p_de, cluster_zones=cluster,
                            residual_domestic_fossil_MW=resid_dom_fossil,
                            residual_after_trade_MW=resid_after_trade))

        # Debug-Export
        dfc["timestamp"] = str(ts)
        dfc["cluster_used"] = cluster
        cand_records.append(dfc)

        # JSONL (Kandidatenliste der Stunde)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": str(ts),
                "candidates": dfc.drop(columns=["price_hit"]).to_dict(orient="records")
            }) + "\n")

    # Exporte
    res_df = pd.DataFrame(results)
    # Zeitzone ausgeben
    if args.out_tz and len(res_df) > 0:
        tz = pytz.timezone(args.out_tz)
        res_df["timestamp_local"] = pd.to_datetime(res_df["timestamp"], utc=True).dt.tz_convert(tz)
        res_df = res_df[["timestamp", "timestamp_local"] + [c for c in res_df.columns if c not in ("timestamp", "timestamp_local")]]

    res_df.to_csv(csv_res_path, index=False)

    if cand_records:
        dbg_df = pd.concat(cand_records, ignore_index=True)
        dbg_df.to_csv(csv_cand_path, index=False)

    print(f"Wrote: {csv_res_path}")
    print(f"Wrote: {csv_cand_path}")
    print(f"Wrote: {jsonl_path}")


if __name__ == "__main__":
    main()
