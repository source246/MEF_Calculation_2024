#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track C – Dispatch-Backcast (Merit-Stack + Flüsse) für DE/LU (2024)

Was das Skript tut (kurz):
- Liest Brennstoff- & EUA-Preise (th-Basis), Flotte (MaStR, mit Effizienz),
  Grenzflüsse, Nachbar-Preise, Nachbar-Gen und Nachbar-Last (stündlich, TZ Europe/Berlin).
- Baut stündlich einen domestic Fossil-Merit-Stack (unit-basiert, SRMC €/MWh_el).
- Prüft Preis-Kopplung (ε-Regel). Wenn gekoppelt und Importe > 0:
  ermittelt import-seitig die marginale fossile Technologie pro Zone aus der realen
  Erzeugung und vergleicht SRMC (Cluster-Clearing).
- Liefert stündlichen MEF (gCO2/kWh) und Debug-Infos.

Autor: Yannick Schönmeier
"""

from __future__ import annotations
import argparse, os, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import pandas as pd
import numpy as np

# ----------------------------- CLI -------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track C – Dispatch-Backcast (Merit-Stack + Flüsse)")
    p.add_argument("--fleet", required=True, help="CSV: fleet_de_units.csv (MaStR)")
    p.add_argument("--mustrun_mode", choices=["capacity","gen_quantile","off"], default="off",
                   help="off=kein Mustrun, capacity=alte Logik --mustrun_lignite_q, gen_quantile=profilbasiert aus realer Lignite-Gen")
    p.add_argument("--mustrun_quantile", type=float, default=0.20,
                   help="Quantil für gen_quantile, z.B. 0.20 = 20. Perzentil.")
    p.add_argument("--mustrun_peak_hours", default="08-20",
                   help="Peak-Fenster (lokale Uhrzeit, inkl. Start exkl. Ende), Rest = Off-Peak.")
    p.add_argument("--mustrun_monthly", action="store_true",
                   help="Wenn gesetzt: Quantile separat je Monat bilden (empfohlen).")

    p.add_argument("--fuel_prices", required=True, help="CSV: prices_2024.csv (time, gas_eur_mwh_th, coal_eur_mwh_th, lignite_eur_mwh_th, oil_eur_mwh_th, co2_eur_t)")
    p.add_argument("--flows", required=True, help="CSV: flows_scheduled_DE_LU_2024_net.csv (time, imp_AT,..., net_import_total)")
    # build_parser()
    p.add_argument("--start", default=None, help="Start (Europe/Berlin), z.B. 2024-01-01T00:00:00")
    p.add_argument("--end",   default=None, help="Ende exklusiv (Europe/Berlin), z.B. 2024-01-08T00:00:00")

    p.add_argument("--neighbor_gen_dir", required=True, help="Dir mit actual_gen_<ZONE>_2024.csv")
    p.add_argument("--neighbor_load_dir", required=True, help="Dir mit load_<ZONE>_2024.csv")
    p.add_argument("--neighbor_prices", required=True, help="CSV mit price_DE_LU, price_AT,...")
    p.add_argument("--outdir", required=True, help="Output-Ordner")
    p.add_argument("--epsilon", type=float, default=0.01, help="Preis-Kopplungs-Schwelle in €/MWh (default 0.01)")
    p.add_argument("--eta_col", default=None, help="Spaltenname Effizienz in Fleet; falls None: ['Effizienz','Effizienz_imputiert']")
    p.add_argument("--therm_avail", type=float, default=1.0, help="Pauschale Verfügbarkeit thermischer Einheiten (0..1)")
    p.add_argument("--mustrun_lignite_q", type=float, default=0.0, help="Optionaler Braunkohle-Mustrun als Anteil der Kapazität (0..1)")
    p.add_argument("--varom_json", default=None, help="JSON: {tech_or_fuel: varOM_eur_per_mwh_el}")
    p.add_argument("--year", type=int, default=2024)
    return p

# -------------------------- Helper: Time & IO --------------------------------

TZ = "Europe/Berlin"

def parse_ts(s: pd.Series) -> pd.DatetimeIndex:
    # robust: alles direkt als UTC parsen (funktioniert auch bei '...+00:00' und reinen Strings)
    ser_utc = pd.to_datetime(s, errors="coerce", utc=True)
    # immer Europe/Berlin als Index zurückgeben
    return pd.DatetimeIndex(ser_utc).tz_convert(TZ)

def domestic_marginal_fast(t, residual):
    srmc_t = SRMC.loc[t].to_numpy()
    order = np.argsort(srmc_t)
    cumcap = np.cumsum(cap[order])
    pos = np.searchsorted(cumcap, residual, side="left")
    if pos >= len(order): pos = len(order)-1
    uidx = order[pos]
    return units[uidx], ef_keys[uidx], float(eta_arr[uidx]), float(srmc_t[uidx])


    
def read_csv_smart(path: str, min_cols: int = 3) -> pd.DataFrame:
    """
    Robuster CSV-Reader für 'wilde' Dateien (Trennzeichen/Encoding unklar).
    Probiert mehrere Kombinationen, bis Spaltenanzahl plausibel ist.
    """
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



def write_domestic_emissions_mix(outdir: Path, de_gen: pd.DataFrame):
    """
    Berechnet stündliche inländische Emissionen (tCO2/h) und Mix-EF (g/kWh)
    aus realer DE/LU-Erzeugung:
      nur fossile Spalten werden belegt:
        Fossil Gas, Fossil Hard coal, Fossil Brown coal/Lignite, Fossil Oil
    Annahmen:
      - EF_LOOKUP_T_PER_MWH_TH (tCO2/MWh_th) aus Skript
      - ETA_DEFAULT_BY_FUEL (eta) aus Skript (konservativ)
    """
    # Map Neighbor-Tech -> (EF fuel name for BAFA, default eta)
    tech_map = {
        "Fossil Gas": ("Erdgas", ETA_DEFAULT_BY_FUEL.get("Erdgas", 0.50)),
        "Fossil Hard coal": ("Steinkohle", ETA_DEFAULT_BY_FUEL.get("Steinkohle", 0.42)),
        "Fossil Brown coal/Lignite": ("Braunkohle", ETA_DEFAULT_BY_FUEL.get("Braunkohle", 0.40)),
        "Fossil Oil": ("Heizöl schwer", ETA_DEFAULT_BY_FUEL.get("Heizöl schwer", 0.38))}
    # Gen-Spalten filtern
    cols = [c for c in tech_map if c in de_gen.columns]
    if not cols:
        return  # nichts zu tun

    gen_fossil_mwh = de_gen[cols].copy()  # stündliche MWh (ENTSO-E liefert MW -> nach Resample mean; hier 1h -> MWh)
    # Emissionen pro fossil tech (tCO2/h) = Gen_el_MWh * (EF_th / eta)
    emis_parts = []
    for tech in cols:
        ef_name, eta = tech_map[tech]
        ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)  # tCO2/MWh_th
        emis_t = gen_fossil_mwh[tech] * (ef_th / max(eta, 1e-6))
        emis_parts.append(emis_t.rename(tech))

    emis_total_t_per_h = pd.concat(emis_parts, axis=1).sum(axis=1) if emis_parts else pd.Series(0.0, index=de_gen.index)
    # Gesamt-Erzeugung (alle techs, falls vorhanden)
    gen_total_mwh = de_gen.sum(axis=1)
    # Mix-EF (g/kWh) = (tCO2/MWh) * 1000
    ef_mix_g_per_kwh = (emis_total_t_per_h / gen_total_mwh).replace([np.inf, -np.inf], np.nan) * 1000.0

    idx_local_naive = emis_total_t_per_h.index.tz_convert("Europe/Berlin").tz_localize(None)
    
    out = pd.DataFrame({
        "Datetime": idx_local_naive.strftime("%Y-%m-%dT%H:%M:%S"),
        "Absolute Emissionen": emis_total_t_per_h.round(3),
        "CO₂-Emissionsfaktor des Strommix": ef_mix_g_per_kwh.round(3),
    })
    out.to_csv(outdir / "emissions_domestic_2024.csv", index=False)

def force_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Resample auf Stunde. `how`=mean/sum je nach Größe (Last/Gen meist 'mean')."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame benötigt DatetimeIndex.")
    rule = "1h"
    if how == "mean":
        return df.resample(rule).mean()
    elif how == "sum":
        return df.resample(rule).sum()
    else:
        raise ValueError("how must be 'mean' or 'sum'")

def read_csv_auto_time(path: str, time_cols: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Zeitspalte suchen
    tcol = None
    for c in time_cols:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        # Notfall: erste Spalte als Zeit
        tcol = df.columns[0]
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    # numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Stunde erzwingen, falls feinere Auflösung (Load oft 15-min)
    df = force_hourly(df, how="mean")
    return df

# -------------------------- Mappings & Defaults ------------------------------

# Map MaStR-Hauptbrennstoff → (fuel_price_key, bafa_fuel_name_for_EF)
MASTR_FUEL_MAP = {
    "Erdgas": ("gas", "Erdgas"),
    "Steinkohlen": ("coal", "Steinkohle"),
    "Steinkohle": ("coal", "Steinkohle"),
    "Rohbraunkohlen": ("lignite", "Braunkohle"),
    "Braunkohle": ("lignite", "Braunkohle"),
    "Heizöl": ("oil", "Heizöl schwer"),
    "Heizöl leicht": ("oil", "Heizöl leicht / Diesel"),
    "Diesel": ("oil", "Heizöl leicht / Diesel"),
    "Öl": ("oil", "Heizöl schwer"),
    "Fossilgas": ("gas", "Erdgas"),
    # Fallbacks:
    "Gas": ("gas", "Erdgas"),
    "Coal": ("coal", "Steinkohle"),
    "Lignite": ("lignite", "Braunkohle"),
    "Oil": ("oil", "Heizöl schwer"),
}

# BAFA-ähnliche EF (th-Basis) – tCO2/MWh_th (nur die benötigten fossilen)
EF_LOOKUP_T_PER_MWH_TH = {
    "Erdgas": 0.201,
    "Steinkohle": 0.335,
    "Braunkohle": 0.383,
    "Heizöl schwer": 0.288,
    "Heizöl leicht / Diesel": 0.266,
}

# Default-η nach fossiler Tech (wenn in Fleet fehlt)
ETA_DEFAULT_BY_FUEL = {
    "Erdgas": 0.50,        # Mix aus CCGT/OCGT (konservativ)
    "Steinkohle": 0.42,
    "Braunkohle": 0.40,
    "Heizöl schwer": 0.38,
    "Heizöl leicht / Diesel": 0.38,
}

# Neighbor gen tech headers, wie sie auf ENTSO-E CSVs typischerweise heißen:
NEIGHBOR_TECHS = [
    "Fossil Gas",
    "Fossil Hard coal",
    "Fossil Oil",
    "Fossil Brown coal/Lignite",
    "Nuclear",
    "Biomass",
    "Hydro Run-of-river and poundage",
    "Hydro Water Reservoir",
    "Hydro Pumped Storage",
    "Wind Onshore",
    "Wind Offshore",
    "Solar",
    "Waste",
]

FOSSIL_TECH_TO_FUEL = {
    "Fossil Gas": ("gas", "Erdgas"),
    "Fossil Hard coal": ("coal", "Steinkohle"),
    "Fossil Brown coal/Lignite": ("lignite", "Braunkohle"),
    "Fossil Oil": ("oil", "Heizöl schwer"),
}

PRICE_COLS = ["gas_eur_mwh_th", "coal_eur_mwh_th", "lignite_eur_mwh_th", "oil_eur_mwh_th", "co2_eur_t"]

# -------------------------- Load Inputs --------------------------------------

def load_fuel_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol_guess = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    tcol = tcol_guess[0] if tcol_guess else df.columns[0]
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    # Sicherstellen, dass alle Preis-Spalten da sind:
    miss = [c for c in PRICE_COLS if c not in df.columns]
    if miss:
        raise ValueError(f"Fehlende Preisspalten: {miss}")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    return df

def load_flows(path: str) -> pd.DataFrame:
    # time, imp_AT,..., net_import_total
    df = pd.read_csv(path)
    tcol_guess = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    tcol = tcol_guess[0] if tcol_guess else df.columns[0]
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    return df

def load_neighbor_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol_guess = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()]
    tcol = tcol_guess[0] if tcol_guess else df.columns[0]
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    return df

def load_neighbor_load(path_dir: str, zone: str) -> pd.Series:
    path = Path(path_dir) / f"load_{zone}_2024.csv"
    if not path.exists():
        # manche heißen load_<ZONE>_2024.error.csv (wir ignorieren .error)
        candidates = list(Path(path_dir).glob(f"load_{zone}_2024*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Load-CSV für Zone {zone} nicht gefunden in {path_dir}")
        path = candidates[0]
    df = read_csv_auto_time(str(path), ["timestamp_cec", "timestamp", "time", "timestamp_brussels", "timestamp_utc"])
    # Spalte mit 'ActualTotalLoad_MW' finden:
    load_col = None
    for c in df.columns:
        if "ActualTotalLoad" in c:
            load_col = c; break
    if load_col is None:
        # Fallback: erste Spalte
        load_col = df.columns[0]
    s = pd.to_numeric(df[load_col], errors="coerce")
    return s

def load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    path = Path(path_dir) / f"actual_gen_{zone}_2024.csv"
    if not path.exists():
        candidates = list(Path(path_dir).glob(f"actual_gen_{zone}_2024*.csv"))
        if not candidates:
            raise FileNotFoundError(f"Gen-CSV für Zone {zone} nicht gefunden: {path}")
        path = candidates[0]
    df_raw = pd.read_csv(path)
    # Zeitspalte finden
    tcol = None
    for c in ["timestamp_cec", "timestamp", "time", "datetime"]:
        if c in df_raw.columns:
            tcol = c; break
    if tcol is None:
        tcol = df_raw.columns[0]
    df_raw.index = parse_ts(df_raw[tcol])
    df_raw = df_raw.drop(columns=[tcol])

    # Nur valide Tech-Spalten nehmen (manche Dateien haben hinten 'Actual Aggregated...' etc.)
    keep_cols = []
    for col in df_raw.columns:
        base = col.strip()
        # harte Filter: exakte Namen oder begin mit diesen Namen
        for tech in NEIGHBOR_TECHS:
            if base == tech or base.startswith(tech):
                keep_cols.append(col); break

    df = df_raw[keep_cols].copy()
    # Numerisch & Stunde
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    # Duplikate gleicher Tech-Namen aggregieren (falls es z. B. zwei 'Fossil Hard coal*' gab):
    # Map auf "Klartext"-Tech:
    col_map = {}
    for c in df.columns:
        key = None
        for tech in NEIGHBOR_TECHS:
            if c == tech or c.startswith(tech):
                key = tech; break
        if key is None: key = c
        col_map.setdefault(key, []).append(c)
    agg = {}
    for key, cols in col_map.items():
        agg[key] = df[cols].sum(axis=1)
    out = pd.DataFrame(agg, index=df.index).sort_index()
    return out

# -------------------------- Fleet & SRMC -------------------------------------
# --- robuste Normalisierung + Fossil-Mapping (ersetzt MASTR_FUEL_MAP) ---

def _fix_broken_umlauts(s: str) -> str:
    # häufige Windows/CSV-Encoding-Artefakte beheben
    repl = {
        "Ã¶": "ö", "ÃÖ": "Ö", "Ã¤": "ä", "Ã„": "Ä",
        "Ã¼": "ü", "Ãœ": "Ü", "ÃŸ": "ß",
        "Ã¶l": "öl", "erdÃ¶l": "erdöl", "erdÃ¶lgas": "erdölgas",
        "Ã¶": "ö"
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s

def _norm(text: str) -> str:
    t = (text or "").strip()
    t = _fix_broken_umlauts(t)
    t = t.lower()
    # Vereinheitlichen
    t = t.replace("-", " ").replace("/", " ").replace(",", " ")
    t = " ".join(t.split())
    return t

# Welche Einträge zählen wir zum fossil dispatch?
# (Alles andere ignorieren wir später in load_fleet)
def map_fuel_to_price_and_ef(raw: str):
    t = _norm(raw)

    # Gas
    if any(k in t for k in ["erdgas", "erdölgas", "erdolgas", "fossilgas", "gas "]):
        return ("gas", "Erdgas")

    # Steinkohle (inkl. Varianten)
    if any(k in t for k in ["steinkohle", "steinkohlen", "wirbelschichtkohle", "staub  und trockenkohle"]):
        return ("coal", "Steinkohle")

    # Braunkohle
    if any(k in t for k in ["braunkohle", "rohbraunkohle", "braunkohlenbrikett", "rohbraunkohlen"]):
        return ("lignite", "Braunkohle")

    # Öl (schwer/leicht/Diesel)
    if "heizöl" in t or "heizoel" in t or "öl " in t or " oel" in t or "diesel" in t:
        if "leicht" in t or "diesel" in t:
            return ("oil", "Heizöl leicht / Diesel")
        else:
            return ("oil", "Heizöl schwer")

    # Sonst: nicht-fossil bzw. unbekannt -> None (wird gefiltert)
    return (None, None)

def load_fleet(path: str, eta_col: Optional[str]) -> pd.DataFrame:
    df = read_csv_smart(path, min_cols=5)  # <— robust gegen \t/; und Windows-Encoding

    # Effizienz-Spalte(n)
    eta_cols = [eta_col] if eta_col else ["Effizienz", "Effizienz_imputiert", "eta", "Eta", "wirkungsgrad"]
    use_eta = next((c for c in eta_cols if c and c in df.columns), None)
    if use_eta is None:
        raise ValueError(f"Keine Effizienzspalte gefunden. Kandidaten: {eta_cols}")

    # Leistung (MW) – mehrere mögliche Namen
    p_cols = [
        "MW Nettonennleistung der Einheit",
        "Leistung_MW",
        "Nettonennleistung der Einheit",
        "Nettonennleistung"
    ]
    pcol = next((c for c in p_cols if c in df.columns), None)
    if pcol is None:
        raise ValueError("Keine Leistungs-Spalte gefunden (MW).")

    # Brennstoff
    fuel_cols = ["Hauptbrennstoff der Einheit", "Energieträger", "Hauptbrennstoff", "Brennstoff"]
    fcol = next((c for c in fuel_cols if c in df.columns), None)
    if fcol is None:
        raise ValueError("Keine Brennstoff-Spalte gefunden.")

    # IDs/Namen für Debug
    id_cols = [c for c in df.columns if "MaStR" in c]
    idcol = id_cols[0] if id_cols else df.columns[0]
    name_cols = [c for c in df.columns if "Anzeige-Name" in c or "Name der Einheit" in c or "Name" in c]
    namecol = name_cols[0] if name_cols else idcol

    out = pd.DataFrame({
        "unit_id": df[idcol].astype(str),
        "unit_name": df[namecol].astype(str),
        "fuel_raw": df[fcol].astype(str),
        "eta": pd.to_numeric(df[use_eta], errors="coerce"),
        "p_mw": pd.to_numeric(df[pcol], errors="coerce"),
    })

    # --- NEU: Fossil-Mapping per Normalizer ---
    price_key, ef_key, eta_default = [], [], []
    for f in out["fuel_raw"]:
        pk, ek = map_fuel_to_price_and_ef(f)
        price_key.append(pk)
        ef_key.append(ek)
        eta_default.append(ETA_DEFAULT_BY_FUEL.get(ek, np.nan) if ek else np.nan)

    out["price_key"] = price_key
    out["ef_key"] = ef_key

    # Nur echte Fossil-Einheiten behalten (pk/ef vorhanden)
    out = out[(out["price_key"].notna()) & (out["ef_key"].notna())].copy()

    # Default-η passend zu den JETZT gefilterten Zeilen neu ableiten
    eta_def_arr = out["ef_key"].map(lambda k: ETA_DEFAULT_BY_FUEL.get(k, np.nan)).to_numpy()

    # Effizienz säubern und ggf. von Prozent auf Anteil bringen
    eta_clean = pd.to_numeric(out["eta"], errors="coerce").to_numpy()
    # falls jemand 55 statt 0.55 eingetragen hat:
    if np.nanmedian(eta_clean) > 1.5:
        eta_clean = eta_clean / 100.0

    mask_bad = (~np.isfinite(eta_clean)) | (eta_clean < 0.05) | (eta_clean > 0.80)
    eta_clean[mask_bad] = eta_def_arr[mask_bad]
    out["eta"] = eta_clean

    out["available_mw"] = (
        pd.to_numeric(out["p_mw"], errors="coerce")
          .fillna(0.0)
          .clip(lower=0)
          .astype("float32")
    )
    out = out[(out["price_key"].notna()) & (out["ef_key"].notna())].copy()
    out = out[out["available_mw"].notna()]         # sollte nach fillna eh passen
    out = out[np.isfinite(out["eta"])]              # Effizienz muss endlich sein

    return out.dropna(subset=["eta"])




def compute_unit_srmc_series(fleet: pd.DataFrame, fuel_prices: pd.DataFrame, varom_map: Dict[str, float]) -> Dict[str, pd.Series]:
    """
    Liefert dict unit_id -> SRMC_series (€/MWh_el).
    """
    # Hilf-Serien
    co2 = fuel_prices["co2_eur_t"]
    srmc_by_unit = {}
    # Cache pro (price_key, ef_key, eta, varom) Kombi? → hier direkt pro Einheit.
    for i, r in fleet.iterrows():
        # Preisreihe
        pk = r["price_key"]
        price_col = f"{pk}_eur_mwh_th"
        if price_col not in fuel_prices.columns:
            raise ValueError(f"Preis-Spalte fehlt: {price_col}")
        fuel_th = fuel_prices[price_col]
        ef_key = r["ef_key"]
        ef = EF_LOOKUP_T_PER_MWH_TH.get(ef_key, np.nan)
        if np.isnan(ef):
            # konservativ: 0.30 t/MWh_th
            ef = 0.30
        eta = r["eta"]
        varom = 0.0
        # varOM map: fuelname oder price_key
        if ef_key in varom_map:
            varom = varom_map[ef_key]
        elif pk in varom_map:
            varom = varom_map[pk]

        srmc = (fuel_th + co2 * ef) / max(eta, 1e-6) + varom
        srmc_by_unit[r["unit_id"]] = srmc
    return srmc_by_unit

def capacities_with_modifiers(cap_base, ef_keys, therm_avail, mustrun_q):
    cap = cap_base * float(therm_avail)
    if mustrun_q > 0:
        is_lignite = (ef_keys == "Braunkohle")
        cap = np.maximum(cap, cap_base * float(mustrun_q) * is_lignite.astype("float32"))
    return cap

    
def domestic_marginal_fast(t: pd.Timestamp, residual: float):
    if residual <= 0 or t not in SRMC.index:
        return None, None, None, 0.0
    srmc_t = SRMC.loc[t].to_numpy()  # float32
    # sortiere Einheiten nach SRMC
    order = np.argsort(srmc_t)
    cumcap = np.cumsum(cap[order])
    pos = np.searchsorted(cumcap, residual, side="left")
    if pos >= len(order):
        pos = len(order) - 1
    uidx = order[pos]
    return units[uidx], ef_keys[uidx], float(eta_arr[uidx]), float(srmc_t[uidx])
# -------------------------- Merit & MEF --------------------------------------

def cluster_zones_by_price(nei_prices: pd.DataFrame, eps: float) -> Dict[pd.Timestamp, List[str]]:
    """
    Liefert pro Stunde die Zonen, die mit DE/LU gekoppelt sind (inkl. 'DE_LU').
    """
    zones = [c.replace("price_", "") for c in nei_prices.columns if c.startswith("price_")]
    assert "DE_LU" in zones, "Spalte price_DE_LU fehlt in neighbor_prices"
    clusters = {}
    for t, row in nei_prices.iterrows():
        p_de = row[f"price_DE_LU"]
        if pd.isna(p_de):
            clusters[t] = ["DE_LU"]; continue
        cluster = ["DE_LU"]
        for z in zones:
            if z == "DE_LU": continue
            pz = row.get(f"price_{z}", np.nan)
            if pd.isna(pz): 
                continue
            if abs(pz - p_de) <= eps:
                cluster.append(z)
        clusters[t] = cluster
    return clusters

def neighbor_marginal_from_gen(row_gen: pd.Series, fuel_prices_row: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Bestimmt marginale fossil-Tech in einer Zone aus realer Gen (höchstes SRMC unter den >0 MW Techs).
    Rückgabe: (fuel_name_for_EF, srmc_eur_per_mwh_el, eta_used)
    """
    # Sammle fossil MW und SRMCs
    cand = []
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in row_gen.index:
            continue
        mw = row_gen.get(tech, 0.0)
        if pd.isna(mw) or mw <= 1.0:
            continue
        price_col = f"{pk}_eur_mwh_th"
        if price_col not in fuel_prices_row.index:
            continue
        fuel_th = fuel_prices_row[price_col]
        co2 = fuel_prices_row["co2_eur_t"]
        ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)
        eta = ETA_DEFAULT_BY_FUEL.get(ef_name, 0.40)
        srmc = (fuel_th + co2 * ef_th) / max(eta, 1e-6)
        cand.append((ef_name, srmc, eta))
    if not cand:
        return None, None, None
    # Marginal = teuerste eingesetzte
    ef_name, srmc, eta = sorted(cand, key=lambda x: x[1])[-1]
    return ef_name, srmc, eta

def build_domestic_merit_and_marginal(
    t: pd.Timestamp,
    residual_mw: float,
    fleet: pd.DataFrame,
    srmc_by_unit: Dict[str, pd.Series],
    therm_avail: float,
    mustrun_lignite_q: float,
) -> Tuple[Optional[str], Optional[str], Optional[float], float]:
    """
    Füllt domestic-fossil Stack bis residual_mw.
    Rückgabe: (unit_id, fuel_name (EF-Key), eta, marginal_srmc)
    """
    if residual_mw <= 0:
        return None, None, None, 0.0

    # konsistente Datenstruktur
    rows = []  # dicts mit einheitlichen Keys
    for _, r in fleet.iterrows():
        unit_id = r["unit_id"]
        avail = float((r["available_mw"] or 0.0) * float(therm_avail))
        if avail <= 0:
            continue

        # optionaler Braunkohle-"Mustrun": hier nur als Verfügbarkeitsuntergrenze interpretieren
        if "Braunkohle" in r["ef_key"] and mustrun_lignite_q > 0:
            avail = max(avail, (r["available_mw"] or 0.0) * float(mustrun_lignite_q))

        s = srmc_by_unit.get(unit_id)
        if s is None or t not in s.index:
            continue
        srmc_t = float(s.loc[t])
        if not np.isfinite(srmc_t):
            continue

        rows.append({
            "unit_id": unit_id,
            "srmc": srmc_t,
            "avail": avail,
            "ef_key": r["ef_key"],
            "eta": float(r["eta"]),
        })

    if not rows:
        return None, None, None, 0.0

    rows.sort(key=lambda x: x["srmc"])
    cum = 0.0
    marginal = None

    for r in rows:
        if cum + r["avail"] >= residual_mw - 1e-6:
            marginal = r
            break
        cum += r["avail"]

    if marginal is None:
        # Stack reicht nicht: setze teuerste Einheit als marginal (aber gib 4 Werte zurück)
        marginal = rows[-1]
        cum = sum(rr["avail"] for rr in rows)

    return marginal["unit_id"], marginal["ef_key"], marginal["eta"], marginal["srmc"]
def _parse_peak_window(s: str) -> tuple[int,int]:
    # "08-20" -> (8, 20)
    a, b = s.split("-")
    return int(a), int(b)

def build_lignite_mustrun_profile(
    de_gen: pd.DataFrame,
    idx: list[pd.Timestamp],
    quantile: float = 0.20,
    peak_window: str = "08-20",
    monthly: bool = True
) -> pd.Series:
    """
    Liefert Serie mustrun_MW[t] aus realer Lignite-Gen (Fossil Brown coal/Lignite).
    - getrennt nach Peak/Off-Peak
    - optional je Monat eigene Quantile
    """
    if "Fossil Brown coal/Lignite" not in de_gen.columns:
        return pd.Series(0.0, index=pd.DatetimeIndex(idx))

    lign = de_gen["Fossil Brown coal/Lignite"].copy()
    lign = lign.reindex(pd.DatetimeIndex(idx)).astype("float64")

    h_start, h_end = _parse_peak_window(peak_window)  # z.B. 8..20
    is_peak = lign.index.hour.isin(range(h_start, h_end))

    out = pd.Series(index=lign.index, dtype="float64")

    if monthly:
        # je Monat und Peak/Off-Peak getrennte Quantile
        for month, sub in lign.groupby(lign.index.month):
            peak_vals    = sub[is_peak[sub.index]]
            offpeak_vals = sub[~is_peak[sub.index]]

            q_peak    = np.nanquantile(peak_vals, quantile)    if len(peak_vals)    else 0.0
            q_offpeak = np.nanquantile(offpeak_vals, quantile) if len(offpeak_vals) else 0.0

            mask = (lign.index.month == month)
            out.loc[mask &  is_peak] = q_peak
            out.loc[mask & ~is_peak] = q_offpeak
    else:
        q_peak    = np.nanquantile(lign[is_peak],    quantile) if is_peak.any() else 0.0
        q_offpeak = np.nanquantile(lign[~is_peak],   quantile)
        out[ is_peak] = q_peak
        out[~is_peak] = q_offpeak

    # Negative/NaN weg
    out = out.fillna(0.0).clip(lower=0.0)
    return out


# ------------------------------ Main -----------------------------------------

def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) Preise
    fuel_prices = load_fuel_prices(args.fuel_prices)
    
    # 2) Fleet
    fleet_all = load_fleet(args.fleet, args.eta_col)
    fleet = fleet_all[fleet_all["ef_key"].isin(EF_LOOKUP_T_PER_MWH_TH.keys())].copy().reset_index(drop=True)
    
    # varOM
    varom_map = {}
    if args.varom_json and Path(args.varom_json).exists():
        varom_map = json.load(open(args.varom_json, "r", encoding="utf-8"))
    
    # 3) SRMC je Einheit (Series)  <<-- MUSS VOR dem Vektorisierungsblock stehen
    srmc_by_unit = compute_unit_srmc_series(fleet, fuel_prices, varom_map)
    
    # --- Vektorisierter Domestic-Merit (nach compute_unit_srmc_series) ---
    units = list(srmc_by_unit.keys())
    SRMC = pd.concat([srmc_by_unit[u].rename(u) for u in units], axis=1).astype("float32")
    
    # Fleet exakt an SRMC-Spalten AUSRICHTEN (wichtig!)
    fleet_idxed = fleet.set_index("unit_id")
    # 1) SRMC auf Schnittmenge reduzieren, Spalten sortenidentisch zum Fleet-Index ziehen
    common = [u for u in SRMC.columns if u in fleet_idxed.index]
    SRMC = SRMC.loc[:, common]
    fleet_idxed = fleet_idxed.loc[common]
    
    # 2) UNBEDINGT: gleiche Reihenfolge sicherstellen
    assert list(SRMC.columns) == list(fleet_idxed.index), "SRMC/Fleet nicht ausgerichtet!"
    
    _bad = SRMC.isna().any(axis=0)
    if _bad.any():
        print("[WARN] drop units with NaN SRMC:", [u for u,b in zip(SRMC.columns, _bad) if b])
        SRMC = SRMC.loc[:, ~_bad]
        fleet_idxed = fleet_idxed.loc[SRMC.columns]
    
    # danach erst:
    units    = list(SRMC.columns)
    cap_base = fleet_idxed["available_mw"].astype("float32").to_numpy()
    eta_arr  = fleet_idxed["eta"].astype("float32").to_numpy()
    ef_keys  = fleet_idxed["ef_key"].astype(str).to_numpy()
    
    # und zur Sicherheit:
    if not np.isfinite(cap_base).all():
        bad = np.where(~np.isfinite(cap_base))[0][:10]
        print("[WARN] non-finite caps at:", [units[i] for i in bad])
        cap_base = np.nan_to_num(cap_base, nan=0.0, posinf=0.0, neginf=0.0)


    test_t = pd.Timestamp("2024-01-15 19:00", tz="Europe/Berlin")
    if test_t in SRMC.index:
        s = SRMC.loc[test_t].to_numpy()
        order = np.argsort(s)  # billig -> teuer
        # zeig die 3 billigsten und 3 teuersten Units samt Fuel:
        for tag, idxs in [("CHEAP", order[:3]), ("EXPENSIVE", order[-3:])]:
            print(f"[{tag}]")
            for i in idxs:
                print(units[i], ef_keys[i], float(eta_arr[i]), float(s[i]))
    print("Sum cap fossil (MW):", cap_base.sum())
    print("Braunkohle MW:", cap_base[(ef_keys=="Braunkohle")].sum())
    print("Steinkohle MW:", cap_base[(ef_keys=="Steinkohle")].sum())
    print("Erdgas MW:", cap_base[(ef_keys=="Erdgas")].sum())
          
    cap = cap_base * float(args.therm_avail)
    if args.mustrun_lignite_q > 0:
        is_lignite = (ef_keys == "Braunkohle")
        cap = np.maximum(cap, cap_base * float(args.mustrun_lignite_q) * is_lignite.astype("float32"))
    
    def domestic_marginal_fast(t: pd.Timestamp, residual: float):
        if (residual <= 0) or (t not in SRMC.index):
            return None, None, None, 0.0
        srmc_t = SRMC.loc[t].to_numpy()
        order = np.argsort(srmc_t, kind="mergesort")
        cumcap = np.cumsum(cap[order])
        pos = np.searchsorted(cumcap, residual, side="left")
        if pos >= len(order): pos = len(order) - 1
        uidx = order[pos]
        return units[uidx], ef_keys[uidx], float(eta_arr[uidx]), float(srmc_t[uidx])
    
    # 4) Flows
    flows = load_flows(args.flows)

    if "net_import_total" not in flows.columns:
        # Erzeuge net_import_total
        imp_cols = [c for c in flows.columns if c.startswith("imp_")]
        flows["net_import_total"] = flows[imp_cols].sum(axis=1)

    # 5) Neighbor prices & Cluster
    nei_prices = load_neighbor_prices(args.neighbor_prices)
    clusters = cluster_zones_by_price(nei_prices, args.epsilon)

    # 6) Nachbar-Gen/Last + DE/LU Load/Gen
    # Zonen aus neighbor_prices ableiten
    zones = sorted([c.replace("price_", "") for c in nei_prices.columns if c.startswith("price_")])
    # Load je Zone
    load_by_zone = {}
    for z in zones:
        try:
            load_by_zone[z] = load_neighbor_load(args.neighbor_load_dir, z)
        except Exception:
            # optional: still weiter
            pass
    # Gen je Zone
    gen_by_zone = {}
    for z in zones:
        try:
            gen_by_zone[z] = load_neighbor_gen(args.neighbor_gen_dir, z)
        except Exception:
            pass

    # 7) DE/LU Last & Non-Dispatchables
    if "DE_LU" not in load_by_zone:
        raise RuntimeError("DE/LU Last konnte nicht geladen werden (load_DE_LU_2024.csv erwartet).")
    de_load = load_by_zone["DE_LU"]
    if "DE_LU" not in gen_by_zone:
        raise RuntimeError("DE/LU Gen konnte nicht geladen werden (actual_gen_DE_LU_2024.csv erwartet).")
    de_gen = gen_by_zone["DE_LU"]

    # Non-dispatchables (konfigurierbar – hier Kernenergie, Solar, Wind, RoR)
    nondisp_cols = [
        "Nuclear",
        "Solar",
        "Wind Onshore",
        "Wind Offshore",
        "Hydro Run-of-river and poundage",
    ]
    nd_present = [c for c in nondisp_cols if c in de_gen.columns]
    de_nondisp = de_gen[nd_present].sum(axis=1).reindex(de_load.index).fillna(0.0)
    start = pd.Timestamp("2024-01-01 00:00:00")
    end   = pd.Timestamp("2024-01-02 00:00:00")
    
    # ---- Zeitindex schneiden ----
    # Basierend auf den 4 Kernquellen
    # ---- Zeitindex schneiden ----
    idx = sorted(
        de_load.index
        .intersection(fuel_prices.index)
        .intersection(flows.index)
        .intersection(nei_prices.index)
    )
    
    if not idx:
        raise RuntimeError(
            "Kein gemeinsamer Zeitindex! Prüfe Resampling/Zeitspalten. "
            f"len(de_load)={len(de_load)}, len(fuel_prices)={len(fuel_prices)}, "
            f"len(flows)={len(flows)}, len(nei_prices)={len(nei_prices)}"
        )
    
    def _to_berlin(ts_str: str):
        ts = pd.Timestamp(ts_str)
        return ts.tz_localize("Europe/Berlin") if ts.tz is None else ts.tz_convert("Europe/Berlin")
    
    if args.start:
        start = _to_berlin(args.start)
    else:
        start = idx[0]
    if args.end:
        end = _to_berlin(args.end)
    else:
        end = idx[-1] + pd.Timedelta(hours=1)  # exklusives Ende
    
    idx = [t for t in idx if (t >= start and t < end)]
    
    if not idx:
        raise RuntimeError(f"Nach Filter kein Zeitstempel in [{start} .. {end}) vorhanden.")
    
    print(f"[INFO] Stunden im Lauf: {len(idx)} | Fenster: {start} .. {end} (exkl.)")


    # 8) Hauptschleife: pro Stunde marginal bestimmen
    results = []
    debug_rows = []

    # Welche Import-Spalten sind vorhanden?
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    # Map imp_ col to zone name
    # Typical: imp_AT -> AT, imp_DK_1 -> DK1, imp_DK_2 -> DK2, imp_NO_2 -> NO2, imp_SE_4 -> SE4
    def impcol_to_zone(c: str) -> str:
        z = c.replace("imp_", "")
        return z.replace("_", "")
    imp_to_zone = {c: impcol_to_zone(c) for c in imp_cols}

    for t in idx:
        L = float(de_load.get(t, np.nan))
        if not np.isfinite(L):
            continue
        ND = float(de_nondisp.get(t, 0.0))
        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0

        # Residual, der domestic-fossil zu decken hat:
        residual = L - ND - net_imp
        if residual < 0: residual = 0.0

        # Domestic-marginal:
        unit_id, ef_dom, eta_dom, srmc_dom = domestic_marginal_fast(t, residual)

        # Cluster-Check
        cluster = clusters.get(t, ["DE_LU"])
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan

        # Import-marginal (nur, wenn gekoppelt & es gibt Importe > 0)
        imp_details = []
        imp_sum = 0.0
        import_marg_srmc = None
        import_marg_mef_g_per_kwh = None
        if len(cluster) > 1:
            # Sammle importierende Zonen im Cluster
            for c in imp_cols:
                mw = float(flows.loc[t, c]) if t in flows.index else 0.0
                if mw <= 1e-6:
                    continue
                z = imp_to_zone[c]
                # Nur Zonen im Preis-Cluster
                if z not in cluster:
                    continue
                if z not in gen_by_zone:
                    continue
                gen_row = gen_by_zone[z].reindex([t]).iloc[0] if t in gen_by_zone[z].index else None
                if gen_row is None:
                    continue
                fuel_prices_row = fuel_prices.reindex([t]).iloc[0] if t in fuel_prices.index else None
                if fuel_prices_row is None:
                    continue
                ef_name, srmc_z, eta_z = neighbor_marginal_from_gen(gen_row, fuel_prices_row)
                if ef_name is None or srmc_z is None or eta_z is None:
                    continue
                # MEF_z (g/kWh)
                ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.30)
                mef_z_g_per_kwh = (ef_th / max(eta_z, 1e-6)) * 1000.0
                imp_details.append((z, mw, ef_name, eta_z, srmc_z, mef_z_g_per_kwh))
                imp_sum += mw

            if imp_details and imp_sum > 0:
                # Fluss-gewichtet
                w_srmc = sum(mw * srmc for (_z, mw, _efn, _eta, srmc, _mef) in imp_details) / imp_sum
                w_mef = sum(mw * mef for (_z, mw, _efn, _eta, _srmc, mef) in imp_details) / imp_sum
                import_marg_srmc = w_srmc
                import_marg_mef_g_per_kwh = w_mef

        # Entscheidung: marginale Seite
        marginal_side = "DE"
        marginal_label = unit_id if unit_id else "none"
        marginal_fuel = ef_dom
        marginal_eta = eta_dom if eta_dom is not None else np.nan
        marginal_srmc = srmc_dom if srmc_dom is not None else np.nan
        mef_g_per_kwh = np.nan

        if unit_id is not None and ef_dom is not None and eta_dom is not None:
            ef_th = EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30)
            mef_dom = (ef_th / max(eta_dom, 1e-6)) * 1000.0
        else:
            mef_dom = np.nan

        if (import_marg_srmc is not None) and (net_imp > 0.0) and (len(cluster) > 1):
            # cleare Preis durch max(SRMC_dom, SRMC_imp)
            if (not np.isfinite(marginal_srmc)) or (import_marg_srmc >= marginal_srmc):
                marginal_side = "IMPORT"
                marginal_label = ",".join(sorted({z for (z, *_rest) in imp_details}))
                marginal_fuel = "mix"
                marginal_eta = np.nan
                marginal_srmc = import_marg_srmc
                mef_g_per_kwh = import_marg_mef_g_per_kwh
            else:
                mef_g_per_kwh = mef_dom
        else:
            mef_g_per_kwh = mef_dom

        results.append({
            "timestamp": t,
            "mef_g_per_kwh": mef_g_per_kwh,
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
            "IMPORT_mef_gpkwh_w": import_marg_mef_g_per_kwh,
            "cluster": "|".join(cluster),
            "net_import_total_MW": net_imp,
            "price_DE": p_de,
            "ND_MW": ND,
            "Load_MW": L,
        })

    # 9) Outputs
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    df_res.to_csv(outdir / "mef_track_c_2024.csv", index=True)
    df_dbg.to_csv(outdir / "_debug_hourly.csv", index=True)
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'}")
    print(f"[OK] Debug:       {outdir/'_debug_hourly.csv'}")
    # optionaler Mix-Output
    try:
        write_domestic_emissions_mix(outdir, de_gen)
        print(f"[OK] Domestic emissions/mix: {outdir/'emissions_domestic_2024.csv'}")
    except Exception as e:
        print(f"[WARN] domestic emissions not written: {e}")

if __name__ == "__main__":
    main(build_parser().parse_args())
