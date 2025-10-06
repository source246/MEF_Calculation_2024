#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mef_dispatch_mustrun.py – Track C Merit-Order Backcast mit Lignite-Mustrun
- Parst ENTSO-E actual_gen_DE_LU_2024.csv (Multi-Header: Technik | Messart | timestamp_cec)
- Bildet Wind/Solar/RoR aus 'Actual Aggregated'
- Load-Proxy = Summe aller 'Actual Aggregated' + Pumpspeicher-'Actual Consumption'
- Lignite-Mustrun = Quantil je Monat & Peak/Off-Peak aus 'Fossil Brown coal/Lignite | Actual Aggregated'
- Residual-Last -= Mustrun; Lignite-Kapazität -= Mustrun
- Merit-Order per SRMC; MEF aus marginalem Fuel (Fuel-EF & η)

Outputs:
  <outdir>/mef_track_c_2024.csv
  <outdir>/_debug_hourly.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse, re
import numpy as np
import pandas as pd

# ---------- CLI ----------
p = argparse.ArgumentParser()
p.add_argument("--start", required=True, type=str)
p.add_argument("--end",   required=True, type=str)

p.add_argument("--fleet", required=True, type=str, help=".../fleet_de_units.csv")
p.add_argument("--fuel_prices", required=True, type=str, help=".../prices_2024.csv")

# Du kannst hier entweder ein 'dispatch_2024.csv' (einfache Zeitachsen-Datei) ODER
# direkt die ENTSO-E 'actual_gen_DE_LU_2024.csv' angeben – das Skript erkennt beides.
p.add_argument("--de_timeseries", required=True, type=str,
               help="Pfad zu dispatch_2024.csv ODER actual_gen_DE_LU_2024.csv")

# Optional: reale Generation für Mustrun – wenn du ohnehin actual_gen als de_timeseries nimmst,
# wird das automatisch wiederverwendet.
p.add_argument("--neighbor_gen_dir", required=False, default=None, type=str)

p.add_argument("--outdir", required=True, type=str)
p.add_argument("--eta_col", default="Effizienz_imputiert", type=str)
p.add_argument("--mustrun_lignite_q", default=0.0, type=float, help="Quantil (0..1) für Lignite-Mustrun")
p.add_argument("--peak_window", default="08-20", type=str, help="z.B. 08-20 oder 20-08")
p.add_argument("--monthly_mustrun", action="store_true", default=True)
p.add_argument("--epsilon", default=0.01, type=float)
args = p.parse_args()

OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)


# ---------- Helfer ----------
def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Findet eine Spalte über exakte/partielle Matches (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    # exakt
    for k in candidates:
        if k.lower() in lower: return lower[k.lower()]
    # fuzzy
    for c in df.columns:
        lc = c.lower()
        for k in candidates:
            if k.lower() in lc: return c
    return None

def _norm_fuel(s: str) -> str:
    t = re.sub(r"[-_/]", " ", str(s or "").strip().lower())
    t = re.sub(r"\s+", " ", t)
    if re.search(r"braun", t):                  return "Braunkohle"
    if re.search(r"stein", t):                  return "Steinkohle"
    if re.search(r"gas|erdgas|fossilgas", t):   return "Erdgas"
    if re.search(r"heizöl schwer|heavy|hfo", t):return "Heizöl schwer"
    if re.search(r"heizöl leicht|diesel|lfo",t):return "Heizöl leicht / Diesel"
    return "mix"

def _peak_mask(index: pd.DatetimeIndex, window: str = "08-20") -> pd.Series:
    a, b = (int(x) for x in window.split("-"))
    h = index.hour
    if a < b:  # 08-20
        return (h >= a) & (h < b)
    # über Mitternacht: 20-08
    return (h >= a) | (h < b)

def _squash_cols(mi_df: pd.DataFrame) -> pd.DataFrame:
    """Kombiniert Multi-Header (level0|level1|level2) in 'A | B' Namen."""
    def join(name_tuple):
        parts = [str(x).strip() for x in name_tuple if str(x).strip() and str(x).lower() != "nan"]
        return " | ".join(parts)
    mi_df = mi_df.copy()
    mi_df.columns = [join(t) for t in mi_df.columns.to_list()]
    return mi_df


# ---------- Konstanten ----------
EF_KG_PER_GJ = {  # IPCC-typisch
    "Erdgas": 56.1,
    "Steinkohle": 94.6,
    "Braunkohle": 101.2,
    "Heizöl schwer": 77.4,
    "Heizöl leicht / Diesel": 74.1,
}
ETA_DEFAULT = {  # fallback η_el
    "Erdgas": 0.52,
    "Steinkohle": 0.40,
    "Braunkohle": 0.33,
    "Heizöl schwer": 0.38,
    "Heizöl leicht / Diesel": 0.38,
}
FUEL_ORDER = ["Braunkohle", "Steinkohle", "Erdgas", "Heizöl schwer", "Heizöl leicht / Diesel", "mix"]


# ---------- Fleet laden ----------
print("[1/7] Lese Fleet…")
fleet = pd.read_csv(args.fleet, sep=None, engine="python")

cap_col = _find_col(fleet, [
    "mw nettonennleistung der einheit","leistung_mw","nettonennleistung der einheit",
    "nettonennleistung","capacity","mw"
]) or fleet.columns[0]
fuel_col = _find_col(fleet, [
    "hauptbrennstoff der einheit","energieträger","hauptbrennstoff","brennstoff","fuel"
]) or fleet.columns[1]

fleet["capacity_MW"] = pd.to_numeric(fleet[cap_col], errors="coerce")
fleet["fuel_group"] = fleet[fuel_col].map(_norm_fuel)

eta_col = args.eta_col if args.eta_col in fleet.columns else None
fleet["eta"] = pd.to_numeric(fleet[eta_col], errors="coerce") if eta_col else np.nan

# Fallback-η je Fuel
for f in FUEL_ORDER:
    sel = fleet["fuel_group"] == f
    m = np.nanmedian(fleet.loc[sel, "eta"])
    if np.isnan(m): m = ETA_DEFAULT.get(f, 0.38)
    fleet.loc[sel, "eta"] = fleet.loc[sel, "eta"].fillna(m)

# gewichtetes η je Fuel
eta_by_fuel = (fleet.assign(w=fleet["capacity_MW"])
               .groupby("fuel_group", group_keys=False)
               .apply(lambda g: np.average(g["eta"], weights=np.clip(g["w"].values, 1e-6, None))))

cap_by_fuel = fleet.groupby("fuel_group")["capacity_MW"].sum()


# ---------- Brennstoffpreise ----------
print("[2/7] Lese Brennstoffpreise…")
fuel_prices = pd.read_csv(args.fuel_prices)
tcol_fp = _find_col(fuel_prices, ["timestamp","time","datetime","datum"])
if tcol_fp is None:
    raise ValueError("Keine Zeitspalte in fuel_prices gefunden.")
fuel_prices[tcol_fp] = pd.to_datetime(fuel_prices[tcol_fp], utc=True).dt.tz_convert("Europe/Berlin")
fuel_prices = fuel_prices.set_index(tcol_fp).sort_index()

co2_col = _find_col(fuel_prices, ["co2","eua","co2_price"])
co2_price = fuel_prices[co2_col] if co2_col else pd.Series(0.0, index=fuel_prices.index)

def _find_price_col(df: pd.DataFrame, fuel: str) -> str | None:
    keys = {
        "Erdgas": ["gas","erdgas"],
        "Steinkohle": ["hardcoal","steinkohle","coal"],
        "Braunkohle": ["lignite","braunkohle","brown"],
        "Heizöl schwer": ["heizöl schwer","hfo","heavy"],
        "Heizöl leicht / Diesel": ["heizöl leicht","diesel","lfo","light"],
    }
    for k in keys.get(fuel, []):
        c = _find_col(df, [k])
        if c: return c
    return None

price_cols = {f: _find_price_col(fuel_prices, f) for f in FUEL_ORDER}


# ---------- DE Timeseries laden ----------
print("[3/7] Lese DE-Zeitreihen…")
def load_de_timeseries(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Liefert:
      de_ts -> DataFrame mit ['Load_MW','Wind_MW','Solar_MW','RoR_MW'] (index = Zeit)
      de_gen -> (falls aus actual_gen) Original-Gen-DF mit flachen Spaltennamen, sonst leer
    """
    # Versuch: Multi-Header (ENTSO-E actual_gen)
    try:
        tmp = pd.read_csv(path, header=[0,1,2])
        # Heuristik: enthält zweite Headerzeile 'Actual Aggregated'?
        if any("Actual Aggregated" in str(x) for x in tmp.columns.get_level_values(1)):
            df = _squash_cols(tmp)
            tcol = _find_col(df, ["timestamp_cec","timestamp","time","datetime","datum","MTU (CET/CEST)","MTU"])
            if tcol is None:
                raise ValueError("Keine Zeitspalte in ENTSO-E-Datei gefunden.")
            df[tcol] = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
            df = df.set_index(tcol).sort_index()

            # Komponenten finden
            def pick(col_like: str, meas: str = "Actual Aggregated"):
                # wählt Spalte, die beide Teilstrings enthält
                for c in df.columns:
                    lc = c.lower()
                    if col_like.lower() in lc and meas.lower() in lc:
                        return c
                return None

            wind_on  = pick("Wind Onshore", "Actual Aggregated")
            wind_off = pick("Wind Offshore", "Actual Aggregated")
            solar    = pick("Solar", "Actual Aggregated")
            ror      = pick("Hydro Run-of-river and poundage", "Actual Aggregated")
            ps_cons  = pick("Hydro Pumped Storage", "Actual Consumption")

            # Load-Proxy = Summe aller 'Actual Aggregated' + Pumpspeicher-Verbrauch
            agg_cols = [c for c in df.columns if "actual aggregated" in c.lower()]
            load_proxy = df[agg_cols].sum(axis=1)
            if ps_cons and ps_cons in df.columns:
                load_proxy = load_proxy.add(pd.to_numeric(df[ps_cons], errors="coerce").fillna(0.0), fill_value=0.0)

            de_ts = pd.DataFrame(index=df.index)
            de_ts["Load_MW"]  = pd.to_numeric(load_proxy, errors="coerce")
            de_ts["Wind_MW"]  = pd.to_numeric(df[[c for c in [wind_on, wind_off] if c]].sum(axis=1), errors="coerce") if (wind_on or wind_off) else 0.0
            de_ts["Solar_MW"] = pd.to_numeric(df[solar], errors="coerce") if solar else 0.0
            de_ts["RoR_MW"]   = pd.to_numeric(df[ror], errors="coerce") if ror else 0.0

            return de_ts, df  # df = de_gen (flattened)
    except Exception:
        pass

    # Fallback: einfache Zeitreihen-Datei (dispatch_2024.csv)
    simple = pd.read_csv(path)
    tcol = _find_col(simple, ["timestamp","time","datetime","datum","MTU (CET/CEST)","MTU"])
    if tcol is None:
        raise ValueError("Keine Zeitspalte in de_timeseries gefunden.")
    simple[tcol] = pd.to_datetime(simple[tcol], utc=True).dt.tz_convert("Europe/Berlin")
    simple = simple.set_index(tcol).sort_index()

    load_col  = _find_col(simple, ["gesamtlast","load","verbrauch","da gesamtlast","total load"])
    wind_cols = [c for c in simple.columns if re.search(r"wind", c, re.I)]
    solar_col = _find_col(simple, ["solar","pv"])
    ror_cols  = [c for c in simple.columns if re.search(r"(ror|laufwasser|run[\s-]?of[\s-]?river|hydro)", c, re.I)]

    de_ts = pd.DataFrame(index=simple.index)
    if load_col is None:
        raise ValueError("Keine Lastspalte in de_timeseries gefunden.")
    de_ts["Load_MW"]  = pd.to_numeric(simple[load_col], errors="coerce")
    de_ts["Wind_MW"]  = pd.to_numeric(simple[wind_cols].sum(axis=1), errors="coerce") if wind_cols else 0.0
    de_ts["Solar_MW"] = pd.to_numeric(simple[solar_col], errors="coerce") if solar_col else 0.0
    de_ts["RoR_MW"]   = pd.to_numeric(simple[ror_cols].sum(axis=1), errors="coerce") if ror_cols else 0.0
    return de_ts, pd.DataFrame(index=de_ts.index)

de_ts, de_gen_flat = load_de_timeseries(args.de_timeseries)

# Arbeitsindex
idx = pd.date_range(args.start, args.end, tz="Europe/Berlin", freq="H", inclusive="left")
de_ts = de_ts.reindex(idx).interpolate(limit_direction="both")


# ---------- DE_LU-Gen (für Mustrun) ----------
print("[4/7] Lese reale Lignite-Gen für Mustrun…")
if de_gen_flat.empty and args.neighbor_gen_dir:
    # optionaler separater Pfad (falls de_timeseries kein actual_gen war)
    alt = Path(args.neighbor_gen_dir) / "actual_gen_DE_LU_2024.csv"
    if alt.exists():
        df_mi = pd.read_csv(alt, header=[0,1,2])
        de_gen_flat = _squash_cols(df_mi)
        tcol = _find_col(de_gen_flat, ["timestamp_cec","timestamp","time","datetime","datum","MTU (CET/CEST)","MTU"])
        de_gen_flat[tcol] = pd.to_datetime(de_gen_flat[tcol], utc=True).dt.tz_convert("Europe/Berlin")
        de_gen_flat = de_gen_flat.set_index(tcol).sort_index()
        de_gen_flat = de_gen_flat.reindex(idx).interpolate(limit_direction="both")

# ---------- Lignite-Mustrun ----------
def build_lignite_mustrun_profile(de_gen_flat: pd.DataFrame,
                                  idx: pd.DatetimeIndex,
                                  quantile: float = 0.20,
                                  peak_window: str = "08-20",
                                  monthly: bool = True) -> pd.Series:
    """
    Nutzt Spalte wie 'Fossil Brown coal/Lignite | Actual Aggregated' (falls vorhanden).
    """
    if de_gen_flat.empty:
        return pd.Series(0.0, index=idx)

    # Spalte finden
    lign_col = None
    for c in de_gen_flat.columns:
        cl = c.lower()
        if "fossil brown coal/lignite" in cl and "actual aggregated" in cl:
            lign_col = c; break
    if lign_col is None:
        # Fallback: jede Spalte, die 'lignite' enthält
        lign_col = _find_col(de_gen_flat, ["lignite","braunkohle"])
        if lign_col is None:
            return pd.Series(0.0, index=idx)

    lign = pd.to_numeric(de_gen_flat[lign_col], errors="coerce").reindex(idx)
    is_peak = _peak_mask(idx, peak_window)
    out = pd.Series(index=idx, dtype="float64")

    if monthly:
        for m in np.unique(idx.month):
            m_mask = (idx.month == m)
            p_mask =  m_mask &  is_peak
            o_mask =  m_mask & ~is_peak
            q_p = np.nanquantile(lign[p_mask], quantile) if p_mask.any() else 0.0
            q_o = np.nanquantile(lign[o_mask], quantile) if o_mask.any() else 0.0
            out.loc[p_mask] = q_p
            out.loc[o_mask] = q_o
    else:
        q_p = np.nanquantile(lign[ is_peak], quantile) if is_peak.any()     else 0.0
        q_o = np.nanquantile(lign[~is_peak], quantile) if (~is_peak).any() else 0.0
        out[ is_peak] = q_p; out[~is_peak] = q_o

    return out.fillna(0.0).clip(lower=0.0)

print("[5/7] Baue Mustrun-Profil…")
m_q = float(args.mustrun_lignite_q or 0.0)
mustrun = build_lignite_mustrun_profile(de_gen_flat, idx, m_q, args.peak_window, args.monthly_mustrun)

# Residual: Load - (RES + RoR)
residual = (de_ts["Load_MW"] - de_ts[["Wind_MW","Solar_MW","RoR_MW"]].sum(axis=1)).clip(lower=0.0)
residual_after_mr = (residual - mustrun).clip(lower=0.0)


# ---------- SRMC & Merit-Order ----------
print("[6/7] SRMC & Merit-Order…")
def srmc_series(fuel: str) -> pd.Series:
    price_col = price_cols.get(fuel)
    if price_col and price_col in fuel_prices.columns:
        p = pd.to_numeric(fuel_prices[price_col], errors="coerce").reindex(idx).interpolate(limit_direction="both")
    else:
        p = pd.Series(1e6, index=idx, dtype="float64")  # kein Preis -> prohibitiv
    eta = float(eta_by_fuel.get(fuel, ETA_DEFAULT.get(fuel, 0.38)))
    ef_t_per_mwh_el = (EF_KG_PER_GJ.get(fuel, 90.0)/1000.0 * 3.6) / max(eta, 1e-6)
    co2 = pd.to_numeric(co2_price, errors="coerce").reindex(idx).fillna(method="ffill").fillna(0.0)
    return (p / max(eta, 1e-6)) + co2 * ef_t_per_mwh_el

srmc = pd.DataFrame({f: srmc_series(f) for f in FUEL_ORDER if f in cap_by_fuel.index}, index=idx)

# verfügbare Kapazität je Fuel (statisch)
cap = cap_by_fuel.reindex(list(srmc.columns)).fillna(0.0).to_dict()
cap_time = pd.DataFrame({f: float(cap[f]) for f in srmc.columns}, index=idx)

# Lignite: Kapazität um Mustrun reduzieren
if "Braunkohle" in cap_time.columns:
    cap_time["Braunkohle"] = (cap_time["Braunkohle"] - mustrun).clip(lower=0.0)

marginal_fuel = []
marginal_cost = []
gen_split = {f: np.zeros(len(idx)) for f in srmc.columns}

for i, t in enumerate(idx):
    need = residual_after_mr.iat[i]
    if need <= args.epsilon:
        # formal: nimm billigsten Fuel als "marginal" (alles durch RES+MR gedeckt)
        fstar = srmc.columns[srmc.loc[t].values.argmin()]
        marginal_fuel.append(fstar); marginal_cost.append(float(srmc.at[t,fstar]))
        continue

    order = srmc.loc[t].sort_values().index.tolist()
    used_fuel = None
    for f in order:
        avail = float(cap_time.at[t, f])
        take = min(avail, max(need, 0.0))
        if take > 0:
            gen_split[f][i] += take
            need -= take
            used_fuel = f
        if need <= args.epsilon:
            break
    if used_fuel is None:
        used_fuel = order[0]
    marginal_fuel.append(used_fuel)
    marginal_cost.append(float(srmc.at[t, used_fuel]))

def ef_g_per_kwh(fuel: str) -> float:
    eta = float(eta_by_fuel.get(fuel, ETA_DEFAULT.get(fuel, 0.38)))
    t_per_mwh = (EF_KG_PER_GJ.get(fuel, 90.0)/1000.0 * 3.6) / max(eta, 1e-6)
    return 1000.0 * t_per_mwh

ef_map = {f: ef_g_per_kwh(f) for f in srmc.columns}
mef_gpkwh = [ef_map.get(f, np.nan) for f in marginal_fuel]


# ---------- Outputs ----------
print("[7/7] Schreibe Outputs…]")
out = pd.DataFrame({
    "timestamp": idx,
    "zone": "DE_LU",
    "selection": "dispatch_mustrun" if m_q > 0 else "dispatch_base",
    "marginal_tech": marginal_fuel,
    "marginal_fuel": marginal_fuel,
    "marginal_origin": "DE",
    "MEF_tCO2_MWh": np.array(mef_gpkwh) / 1000.0,
    "mef_g_per_kwh": mef_gpkwh,
    "SRMC_domestic": marginal_cost,
    "mustrun_lignite_MW": mustrun.values,
    "residual_MW_before_mr": residual.values,
    "residual_MW_after_mr": residual_after_mr.values,
})
out_path = OUT / "mef_track_c_2024.csv"
out.to_csv(out_path, index=False)
print(f"[OK] {out_path}")

debug = pd.DataFrame({"timestamp": idx})
for k in ["Load_MW","Wind_MW","Solar_MW","RoR_MW"]:
    debug[k] = de_ts[k].values
debug["Residual_before_mr"] = residual.values
debug["Lignite_mustrun_MW"] = mustrun.values
debug["Residual_after_mr"]  = residual_after_mr.values
for f in srmc.columns:
    debug[f"SRMC_{f}"] = srmc[f].values
    debug[f"Cap_{f}_MW"] = cap_time[f].values if f in cap_time.columns else 0.0
    debug[f"Gen_{f}_MW"] = gen_split[f]
debug["marginal_fuel"] = marginal_fuel

dbg_path = OUT / "_debug_hourly.csv"
debug.to_csv(dbg_path, index=False)
print(f"[OK] {dbg_path}\nFertig.")
