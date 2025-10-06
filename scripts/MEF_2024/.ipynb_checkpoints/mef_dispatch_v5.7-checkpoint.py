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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
TZ = "Europe/Berlin"

# ----------------------------- CLI -------------------------------------------
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
    # Peaker-Heuristik
    p.add_argument("--peak_switch", action="store_true",
                   help="Aktiviere Preis-zu-Peaker-Override (OCGT/Öl) bei hohen Preisen.")
    p.add_argument("--peak_price_thresholds", default="180,260",
                   help="Schwellen in €/MWh: 'p1,p2' -> p1≈OCGT, p2≈Öl/Diesel.")
    p.add_argument("--peak_eta_ocgt", type=float, default=0.36,
                   help="η für OCGT-Peaker (el.).")
    p.add_argument("--peak_eta_oil", type=float, default=0.33,
                   help="η für öl-/dieselbefeuerte Turbinen/Engines als Peaker.")
    # Nuklear-SRMC (€/MWh_el) für Export-Stack (MEF bleibt 0)
    p.add_argument("--nuclear_srmc_eur_mwh", type=float, default=None,
                   help="Fixer SRMC Kernenergie (€/MWh_el). Wenn gesetzt, überschreibt die Berechnung.")
    p.add_argument("--nuclear_fuel_eur_mwh_th", type=float, default=5.5,
                   help="Brennstoffkosten Kernenergie (€/MWh_th).")
    p.add_argument("--nuclear_eta", type=float, default=0.33,
                   help="Elektrischer Wirkungsgrad Kernenergie (Anteil).")
    p.add_argument("--nuclear_varom_eur_mwh", type=float, default=1.2,
                   help="Sonstige variable Kosten Kernenergie (€/MWh_el).")
    p.add_argument("--biomass_srmc_eur_mwh", type=float, default=35.0,
                   help="Grenzkosten Biomasse (€/MWh_el), nur für Preis-Logik.")
    p.add_argument("--waste_srmc_eur_mwh", type=float, default=1.0,
                   help="Grenzkosten Waste (€/MWh_el) – default: 1 €/MWh (Gebot).")
    p.add_argument("--biomass_mef_gpkwh", type=float, default=0.0,
                   help="MEF für Biomasse (g/kWh) – default 0.")
    p.add_argument("--waste_mef_gpkwh", type=float, default=0.0,
                   help="MEF für Waste (g/kWh) – default 0.")
    # NEU: kostenbasierte MU-Logik
    p.add_argument("--mu_cost_mode",
                   choices=["off","q_vs_cost"],
                   default="off",
                   help="Kostenbasierte MU-Ermittlung: q_vs_cost = MU = q-Quantil der Gen in Stunden mit Preis ≤ α·SRMC.")
    p.add_argument("--mu_cost_alpha", type=float, default=0.75,
                   help="α in Preis ≤ α·SRMC (z. B. 0.75).")
    p.add_argument("--mu_cost_q", type=float, default=0.50,
                   help="q-Quantil innerhalb der 'unter Kosten'-Stunden (z. B. 0.5 = Median).")
    p.add_argument("--mu_cost_monthly", action="store_true",
                   help="Kostenbasiertes MU getrennt pro Monat berechnen.")
    p.add_argument("--mu_cost_use_peak", action="store_true",
                   help="Kostenbasiertes MU zusätzlich Peak/Offpeak je Monat trennen (Fenster via --mustrun_peak_hours).")
    
    # Optional: DE anstatt fixer Shares aus Kosten ableiten
    p.add_argument("--de_fossil_mustrun_from_cost", action="store_true",
                   help="DE-fossiler MU aus Kostenlogik ableiten (ersetzt de_mustrun_*_share).")

    p.add_argument("--coupled_import_anyflow", action="store_true", default=True,
                   help="Importseite aktivieren, sobald irgendeine gekoppelte Leitung >0 MW nach DE liefert (Nettoimport nicht erforderlich).")
    # --- MU-Bid Logik (optional rollierend) ---
    p.add_argument("--mu_bid_mode", choices=["default","rolling"], default="default",
                   help="Gebote für MU (Waste/Nuclear/Biomass/Fossil) als 'default' oder rollierender Schnitt (Preis < α·SRMC).")
    p.add_argument("--mu_bid_window_h", type=int, default=168,
                   help="Fenster für rollierende MU-Bids (Stunden), z. B. 168=7 Tage.")

    # Negativbepreisung Mustrun (fossil + optional Nuklear)
    p.add_argument("--mustrun_neg_pricing_enable", action="store_true",
                   help="Wenn Preis < pct*SRMC: markiere Mustrun-Mengen als negativ bepreist.")
    p.add_argument("--mustrun_neg_price_threshold_pct", type=float, default=0.75,
                   help="Schwelle α für Negativbepreisung (Preis < α*SRMC).")
    p.add_argument("--mustrun_neg_price_value", type=float, default=-10.0,
                   help="Preis für negativ bepreiste Mustrun-Mengen [€/MWh].")
    # Export-Stack: gewünschte Bids
    p.add_argument("--waste_bid_eur_mwh_export", type=float, default=1.0,
                   help="Gebot Waste im Export-Stack (€/MWh).")
    p.add_argument("--biomass_bid_eur_mwh_export", type=float, default=35.0,
                   help="Gebot Biomasse im Export-Stack (€/MWh).")
    p.add_argument("--mustrun_bid_eur_mwh", type=float, default=1.0,
                   help="Gebot fossiler Mustrun im Export-Stack, wenn Negativ-Bepreisung aus (€/MWh).")
    p.add_argument("--mustrun_neg_share", type=float, default=1.0,
                   help="Anteil (0..1) der Mustrun-Mengen, die bei Unterschreiten der Schwelle negativ bepreist werden sollen.")

    # Nuklear-Mustrun
    p.add_argument("--de_nuclear_mustrun_share", type=float, default=0.0,
                    help="Anteil der verfügbaren DE-Nuklearkapazität als Mustrun (0..1).")
    p.add_argument("--nei_nuclear_mustrun_share", type=float, default=0.0,
                    help="Anteil der verfügbaren NEI-Nuklearkapazität als Mustrun (0..1).")
    # Biomasse-Mustrun (optional)
    p.add_argument("--de_biomass_mustrun_share", type=float, default=0.0,
                   help="Anteil Biomasse als Mustrun (DE).")
    p.add_argument("--nei_biomass_mustrun_share", type=float, default=0.0,
                   help="Anteil Biomasse als Mustrun (Nachbarn).")
    # Fossiler Mustrun
    p.add_argument("--fossil_mustrun_mode",
                   choices=["off","min_all","min_peak","min_peak_monthly","q_all"],
                   default="q_all",
                   help="q_all = unteres Quantil über alle Stunden je Fuel")
    p.add_argument("--fossil_mustrun_q", type=float, default=0.10,
                   help="Quantil für q_all (z.B. 0.10 = 10 %)")
    p.add_argument("--fossil_mustrun_fuels",
                   default="Erdgas,Steinkohle,Heizöl schwer,Heizöl leicht / Diesel",
                   help="Braunkohle i.d.R. NICHT listen – wird separat behandelt.")
        # --- Korrelation / Diagnostics ---
    p.add_argument("--corr_drop_neg_prices", action="store_true", default=True,
                   help="Negative Preise bei der Korrelation ignorieren.")
    p.add_argument("--corr_cap_mode",
                   choices=["none","absolute","peaker_min","peaker_max"],
                   default="peaker_min",
                   help=("Preis-Cap für Korrelation: "
                         "none=kein Cap, absolute=fester Grenzwert, "
                         "peaker_min=min(OCGT,Oil)-SRMC als Cap, "
                         "peaker_max=max(OCGT,Oil)-SRMC als Cap."))
    p.add_argument("--corr_cap_value", type=float, default=500.0,
                   help="Absolute Obergrenze (nur bei --corr_cap_mode=absolute).")
    p.add_argument("--corr_offenders_topn", type=int, default=500,
                   help="Top-N Ausreißer in analysis/_corr_offenders.csv.")
    p.add_argument("--corr_cap_tol", type=float, default=1.0,
                   help="Toleranz in €/MWh beim Cap-Vergleich.")
    

    # Kopplung / Preisanker
    p.add_argument("--neighbor_gen_dir",   required=True)
    p.add_argument("--neighbor_load_dir",  required=True)
    p.add_argument("--neighbor_prices",    required=True)
    p.add_argument("--epsilon", type=float, default=5.0, help="Preis-Kopplungs-Schwelle in €/MWh")
    p.add_argument("--price_anchor", choices=["off","closest","threshold"], default="closest")
    p.add_argument("--price_tol", type=float, default=30.0)

    # Nachbarn: Effizienz-Modelle
    p.add_argument("--nei_eta_mode", choices=["mean","bounds","mc"], default="mean")
    p.add_argument("--nei_eta_json", default=None)
    p.add_argument("--nei_mc_draws", type=int, default=50)
    p.add_argument("--neighbor_capacity", default=None)

    # Optionale Mustrun-Shares
    p.add_argument("--de_mustrun_gas_share", type=float, default=0.0)
    p.add_argument("--de_mustrun_coal_share", type=float, default=0.0)
    p.add_argument("--de_mustrun_oil_share",  type=float, default=0.0)
    p.add_argument("--nei_mustrun_gas_share", type=float, default=0.0)
    p.add_argument("--nei_mustrun_coal_share", type=float, default=0.0)
    p.add_argument("--nei_mustrun_oil_share",  type=float, default=0.0)
    # Bid PSP
    p.add_argument("--psp_srmc_floor_eur_mwh", type=float, default=60.0,
               help="Oberes Band für PSP-SRMC (preisfolgend).")
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
def _robust_series(s: pd.Series, kind: str, max_gap_h: int = 6) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s_interp = s.interpolate(limit=max_gap_h, limit_direction="both")
    s_fill = s_interp.ffill().bfill()
    if s_fill.isna().any():
        by_hour = s_fill.groupby(s_fill.index.hour).transform(lambda x: x.fillna(np.nanmedian(x)))
        s_fill = s_fill.fillna(by_hour)
    if s_fill.isna().any():
        s_fill = s_fill.fillna(0.0)
    return s_fill
def _compute_peaker_srmc_series(fuel_prices: pd.DataFrame,
                                eta_ocgt: float,
                                eta_oil: float) -> tuple[pd.Series, pd.Series]:
    """SRMC-Reihen für OCGT (Gas) und Öl-Peaker, €/MWh_el."""
    ef_gas_th = EF_LOOKUP_T_PER_MWH_TH["Erdgas"]
    ef_oil_th = EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"]
    gas = pd.to_numeric(fuel_prices["gas_eur_mwh_th"], errors="coerce")
    oil = pd.to_numeric(fuel_prices["oil_eur_mwh_th"], errors="coerce")
    co2 = pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce")
    srmc_ocgt = (gas + co2 * ef_gas_th) / max(eta_ocgt, 1e-6)
    srmc_oil  = (oil + co2 * ef_oil_th) / max(eta_oil,  1e-6)
    return srmc_ocgt.astype(float), srmc_oil.astype(float)

def _eta_repr_for_fuel_DE(fleet_df: pd.DataFrame, ef_name: str, default=0.42) -> float:
    """Robuster η-Vertreter je Fuel aus DE-Fleet (Median)."""
    sub = fleet_df[fleet_df["ef_key"]==ef_name]["eta"].dropna().astype(float)
    if sub.empty: return float(default)
    m = float(np.nanmedian(sub))
    return float(np.clip(m, 0.20, 0.65))

def _tech_srmc_series_from_eta(fuel_prices: pd.DataFrame, ef_name: str, eta_eff: float) -> pd.Series:
    key = FUEL_KEY_BY_EF.get(ef_name, (None,None))[0]
    if key is None: return pd.Series(np.nan, index=fuel_prices.index)
    ef_th = EF_LOOKUP_T_PER_MWH_TH[ef_name]
    th = pd.to_numeric(fuel_prices[f"{key}_eur_mwh_th"], errors="coerce")
    co2 = pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce")
    srmc = (th + co2*ef_th) / max(float(eta_eff), 1e-6)
    return srmc.astype(float)

def _split_peak_mask(ix: pd.DatetimeIndex, peak_hours: str) -> tuple[pd.Series,pd.Series]:
    h_start, h_end = [int(x) for x in peak_hours.split("-")]
    peak = ((ix.hour >= h_start) & (ix.hour < h_end)) if h_start <= h_end else ((ix.hour >= h_start) | (ix.hour < h_end))
    return pd.Series(peak, index=ix), ~pd.Series(peak, index=ix)


def _filtered_corr_and_offenders(outdir: Path,
                                 df_res: pd.DataFrame,
                                 df_dbg: pd.DataFrame,
                                 df_val: Optional[pd.DataFrame],
                                 fuel_prices: pd.DataFrame,
                                 args) -> float:
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)

    price = df_res["price_DE"]                     # €/MWh
    srmc  = df_res["marginal_srmc_eur_per_mwh"]    # €/MWh

    mask = price.notna() & srmc.notna()

    # 1) Negativpreise immer raus (dein Wunsch)
    if getattr(args, "corr_drop_neg_prices", True):
        mask &= (price >= 0.0)

    # 2) Cap-Regel: „alles über Peaker-SRMC nicht in Korrelation“
    mode = getattr(args, "corr_cap_mode", "peaker_min")
    tol  = float(getattr(args, "corr_cap_tol", 1.0))

    if mode in ("peaker_min", "peaker_max"):
        eta_ocgt = float(getattr(args, "peak_eta_ocgt", 0.36))
        eta_oil  = float(getattr(args, "peak_eta_oil", 0.33))
        s_ocgt, s_oil = _compute_peaker_srmc_series(fuel_prices, eta_ocgt, eta_oil)
        if mode == "peaker_min":
            cap_series = pd.concat([s_ocgt, s_oil], axis=1).min(axis=1)  # strenger
        else:
            cap_series = pd.concat([s_ocgt, s_oil], axis=1).max(axis=1)  # großzügiger
        cap_series = cap_series.reindex(price.index)
        mask &= (price <= cap_series + tol)
    elif mode == "absolute":
        cap = float(getattr(args, "corr_cap_value", 500.0))
        cap_series = pd.Series(cap, index=price.index)
        mask &= (price <= cap + tol)
    else:
        cap_series = pd.Series(np.inf, index=price.index)
    # 3) PSP-preissetzende Stunden ausschließen (arbitragegetrieben)
    flex_mask = (
        (df_res["marginal_fuel"] == "Hydro Pumped Storage") |
        (df_res.get("flag_psp_price_setting", pd.Series(False, index=df_res.index))) |
        (
            (df_res["marginal_side"] == "IMPORT") &
            df_dbg["IMPORT_label"].fillna("").str.contains(r"\(Hydro Pumped Storage\)", case=False)
        )
    )
    
    mask &= ~flex_mask

    pr = price[mask]
    sr = srmc[mask]
    corr = float(pd.concat([pr, sr], axis=1).dropna().corr().iloc[0,1]) if pr.size >= 3 else np.nan
    
    # ---- Offender-Datei: „wer versaut die Korrelation?“ ----
    base = pd.DataFrame({
        "price_DE": pr,
        "chosen_SRMC": sr,
        "abs_error": (pr - sr).abs(),
    })
    # Kontexte / Regeln als Spalten
    base["marginal_side"]  = df_res["marginal_side"].reindex(base.index)
    base["marginal_fuel"]  = df_res["marginal_fuel"].reindex(base.index)
    base["marginal_label"] = df_res["marginal_label"].reindex(base.index)
    base["net_import_MW"]  = df_res["net_import_total_MW"].reindex(base.index)
    base["cluster_zones"]  = df_res["cluster_zones"].reindex(base.index)
    base["residual_domestic_fossil_MW"] = df_res["residual_domestic_fossil_MW"].reindex(base.index)
    base["residual_after_trade_MW"]     = df_res["residual_after_trade_MW"].reindex(base.index)

    # Regel-Flags aus Debug + Validation ableiten
    base["rule_price_neg_or_zero"] = (price.reindex(base.index) <= 0.0)
    base["rule_peaker_cap_exceeded"] = (price.reindex(base.index) > cap_series.reindex(base.index) + tol)
    if df_val is not None:
        for col in ["IMPORT_anchor_ok","EE_surplus_flag","suspect_price_deviation","IMPORT_logic_ok"]:
            if col in df_val.columns:
                base[col] = df_val[col].reindex(base.index)

    # noch mehr: hat dein Skript EE/NonDisp/Mustrun/Peaker-Override gesetzt?
    lbl = base["marginal_label"].fillna("")
    base["rule_ee_surplus"]        = lbl.str.contains("EE_surplus|FEE_only", case=False, regex=True)
    base["rule_nondisp_price_set"] = lbl.str.contains("NonDisp_only", case=False, regex=True)
    base["rule_peaker_override"]   = lbl.str.contains("peaker_override", case=False, regex=True)
    base["cap_mode"]   = mode
    base["cap_value"]  = cap_series.reindex(base.index)

    offenders = base.sort_values("abs_error", ascending=False).head(int(getattr(args,"corr_offenders_topn",500))).copy()
    offenders.index.name = "timestamp"
    offenders.to_csv(outdir / "analysis" / "_corr_offenders.csv")

    with open(outdir / "analysis" / "_corr_offenders_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Filtered corr (Pearson): {corr:.4f}\n")
        f.write(f"N points after filter: {pr.size}\n")
        f.write(f"Cap mode: {mode}\n")
        f.write(f"Drop negative prices: {getattr(args, 'corr_drop_neg_prices', True)}\n")
    return corr


def _parse_two_floats(csv_str: str, default=(180.0, 260.0)):
    try:
        a, b = [float(x) for x in str(csv_str).split(",")[:2]]
        return a, b
    except Exception:
        return default

def robustize_load_gen(de_load: pd.Series, de_gen: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    nd_cols = ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage",]
    present = [c for c in nd_cols if c in de_gen.columns]
    for c in de_gen.columns:
        de_gen[c] = _robust_series(de_gen[c], f"gen:{c}")
    de_load = _robust_series(de_load, "load")
    nd_sum = de_gen[present].sum(axis=1) if present else pd.Series(0.0, index=de_load.index)
    over = (nd_sum > de_load) & de_load.notna()
    if over.any():
        scale = (de_load / nd_sum).clip(upper=1.0).fillna(1.0).ewm(span=3, min_periods=1).mean()
        for c in present:
            de_gen[c] = (de_gen[c] * scale).fillna(de_gen[c])
    return de_load, de_gen

def reservoir_shadow_price_series(nei_prices: pd.DataFrame, zone: str, window_h: int = 24*7) -> pd.Series:
    col = f"price_{zone}" if f"price_{zone}" in nei_prices.columns else "price_DE_LU"
    p = pd.to_numeric(nei_prices[col], errors="coerce").astype(float)
    sp = p.rolling(window_h, min_periods=12).median().bfill().ffill()
    return sp.clip(lower=0.0)
def _rolling_mu_bid_series(price: pd.Series, thresh: pd.Series | float, window_h: int, alpha: float, default_bid: float) -> pd.Series:
    """
    Rollierendes MU-Gebot: Mittel der Preise in Stunden mit Preis <= α * Schwelle (SRMC oder Floor).
    Fällt auf default_bid zurück, wenn zu wenig Punkte im Fenster sind.
    """
    if not isinstance(thresh, pd.Series):
        thresh = pd.Series(thresh, index=price.index)
    mask = price <= (alpha * thresh.reindex(price.index))
    roll = price.where(mask).rolling(window_h, min_periods=max(12, window_h//7)).mean()
    return roll.fillna(default_bid)
def compute_fossil_min_profiles_cost_based(
    gen_df: pd.DataFrame,
    price_series: pd.Series,
    fuel_prices: pd.DataFrame,
    fuels_select: List[str],
    peak_hours: str,
    q: float,
    alpha: float,
    monthly: bool,
    use_peak_split: bool,
    eta_source: str,
    nei_dists: Optional[dict] = None,
    fleet_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    MU je Fuel = q-Quantil der tatsächlichen Gen in Stunden mit Preis ≤ α·SRMC_fuel.
    eta_source: 'DE' (Fleet) | 'NEI' (nei_dists mean) | 'default'
    """
    if gen_df is None or gen_df.empty:
        idx = pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx), {f: pd.Series(0.0, index=idx) for f in fuels_select}

    ix = gen_df.index
    price = pd.to_numeric(price_series.reindex(ix), errors="coerce")
    out_by_fuel = {f: pd.Series(0.0, index=ix, dtype="float64") for f in fuels_select}

    # η je Fuel
    eta_map = {}
    for f in fuels_select:
        if eta_source == "DE" and fleet_df is not None:
            eta_map[f] = _eta_repr_for_fuel_DE(fleet_df, f)
        elif eta_source == "NEI" and nei_dists is not None:
            d = nei_dists.get(f) or DEFAULT_NEI_DISTS.get(f)
            eta_map[f] = float(d["mean"]) if d else 0.42
        else:
            d = DEFAULT_NEI_DISTS.get(f, {"mean":0.42})
            eta_map[f] = float(d["mean"])

    # SRMC je Fuel (Serie)
    srmc_fuel = {f: _tech_srmc_series_from_eta(fuel_prices, f, eta_map[f]).reindex(ix) for f in fuels_select}

    # Segmentierung
    if not monthly:
        segments = [("all", pd.Series(True, index=ix))]
    else:
        segments = []
        for m in sorted(ix.month.unique()):
            msk_m = (ix.month == m)
            if use_peak_split:
                pk, op = _split_peak_mask(ix, peak_hours)
                segments.append((f"m{m:02d}_pk", (msk_m & pk)))
                segments.append((f"m{m:02d}_op", (msk_m & op)))
            else:
                segments.append((f"m{m:02d}", msk_m))

    for f in fuels_select:
        tech_col = {
            "Erdgas": "Fossil Gas",
            "Steinkohle": "Fossil Hard coal",
            "Heizöl schwer": "Fossil Oil",
            "Heizöl leicht / Diesel": "Fossil Oil",
            "Braunkohle": "Fossil Brown coal/Lignite",
        }[f]
        if tech_col not in gen_df.columns:
            continue
        g = pd.to_numeric(gen_df[tech_col].reindex(ix), errors="coerce").fillna(0.0)
        s = srmc_fuel[f]; thr = alpha * s

        mu_series = pd.Series(0.0, index=ix, dtype="float64")
        for _, segmask in segments:
            mask = segmask & (price <= thr)
            if mask.any():
                qv = float(np.nanquantile(g[mask], q)) if g[mask].size else 0.0
                mu_series.loc[segmask] = max(qv, 0.0)
            else:
                mu_series.loc[segmask] = 0.0
        out_by_fuel[f] = mu_series.clip(lower=0.0)

    total_min = sum(out_by_fuel.values()) if out_by_fuel else pd.Series(0.0, index=ix)
    return total_min, out_by_fuel

def precompute_mu_bid_fn(nei_prices: pd.DataFrame, fuel_prices: pd.DataFrame, args, zones: list[str]):
    """
    Erzeuge eine Lookup-Funktion mu_bid(zone, tech, t, default) für MU-Bids.
    tech in {"waste","nuclear","biomass","fossil"}.
    """
    mode = str(getattr(args, "mu_bid_mode", "default"))
    if mode != "rolling":
        # trivialer Fallback: immer default
        return lambda zone, tech, t, default: float(default)

    alpha   = float(getattr(args, "mustrun_neg_price_threshold_pct", 0.75))
    window  = int(getattr(args, "mu_bid_window_h", 168))
    # Referenz-SRMCs/Floors (zonenunabhängig; Preise zonenspezifisch)
    ef_g = EF_LOOKUP_T_PER_MWH_TH["Erdgas"]; ef_c = EF_LOOKUP_T_PER_MWH_TH["Steinkohle"]
    eta_ref = 0.40  # „Durchschnitts“-Effizienz als Floor für MU-Fossil
    gas_ref  = (pd.to_numeric(fuel_prices["gas_eur_mwh_th"], errors="coerce") + pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce") * ef_g) / max(eta_ref,1e-6)
    coal_ref = (pd.to_numeric(fuel_prices["coal_eur_mwh_th"], errors="coerce") + pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce") * ef_c) / max(eta_ref,1e-6)
    fossil_floor = pd.concat([gas_ref, coal_ref], axis=1).min(axis=1)

    # Konstanten
    nuc_srmc = ( float(getattr(args,"nuclear_srmc_eur_mwh", None))
                 if getattr(args,"nuclear_srmc_eur_mwh", None) is not None
                 else ( float(getattr(args,"nuclear_fuel_eur_mwh_th",5.5)) / max(float(getattr(args,"nuclear_eta",0.33)),1e-6)
                        + float(getattr(args,"nuclear_varom_eur_mwh",1.2)) ) )
    bio_srmc = float(getattr(args, "biomass_srmc_eur_mwh", 35.0))

    # Default-Bids
    def_bid_waste   = float(getattr(args, "waste_bid_eur_mwh_export", 1.0))
    def_bid_biomass = float(getattr(args, "biomass_bid_eur_mwh_export",
                          getattr(args,"biomass_srmc_eur_mwh",35.0)))
    def_bid_mu      = float(getattr(args, "mustrun_bid_eur_mwh", 1.0))

    # Vorberechnen je Zone
    store: dict[tuple[str,str], pd.Series] = {}
    for z in zones:
        pcol = f"price_{z}" if f"price_{z}" in nei_prices.columns else "price_DE_LU"
        pz = pd.to_numeric(nei_prices[pcol], errors="coerce").astype(float)

        store[(z,"waste")]   = _rolling_mu_bid_series(pz, 1.0,      window, alpha, def_bid_waste)
        store[(z,"nuclear")] = _rolling_mu_bid_series(pz, nuc_srmc, window, alpha, def_bid_mu)
        store[(z,"biomass")] = _rolling_mu_bid_series(pz, bio_srmc, window, alpha, def_bid_biomass)
        store[(z,"fossil")]  = _rolling_mu_bid_series(pz, fossil_floor.reindex(pz.index), window, alpha, def_bid_mu)

    def _mu_bid(zone: str, tech: str, t: pd.Timestamp, default: float) -> float:
        ser = store.get((zone, tech))
        if ser is None: return float(default)
        try:
            val = float(ser.loc[t])
            if np.isfinite(val): return val
        except Exception:
            pass
        return float(default)
    return _mu_bid

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
    df.index = parse_ts(df[tcol]); df = df.drop(columns=[tcol])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
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
    return {
        "Erdgas": float(getattr(args, "de_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "de_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,
        "Heizöl schwer": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
        "Heizöl leicht / Diesel": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
    }

def fossil_mustrun_shares_for_NEI(args) -> dict:
    return {
        "Erdgas": float(getattr(args, "nei_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "nei_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,
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
    from pathlib import Path
    candidates = list(Path(path_dir).glob(f"load_{zone}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Load-CSV fehlt: load_{zone}_2024*.csv in {path_dir}")
    df = read_csv_auto_time(str(candidates[0]), ["timestamp_cec","timestamp","time","timestamp_brussels","timestamp_utc"])
    load_col = next((c for c in df.columns if "ActualTotalLoad" in c or "load" in c.lower()), df.columns[0])
    return pd.to_numeric(df[load_col], errors="coerce")

NEIGHBOR_TECHS = [
    "Fossil Gas","Fossil Hard coal","Fossil Oil","Fossil Brown coal/Lignite","Fossil Coal-derived gas",
    "Nuclear","Biomass",
    "Hydro Run-of-river and poundage","Hydro Water Reservoir","Hydro Pumped Storage",
    "Wind Onshore","Wind Offshore","Solar","Waste",
    "Geothermal","Other renewable","Other",
]


def load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    """
    Liest eine 'actual_gen_<ZONE>_2024*.csv' Datei robust ein und gibt ein
    stündlich gemitteltes Wide-DF mit ENTSO-E Tech-Spalten zurück.
    Unterstützt sowohl 'wide' (Spalten je Technologie) als auch 'long' Format
    (Spalten: <timestamp, tech, value>).
    """
    from pathlib import Path

    # 1) Datei finden (Unterstrich/Bindestrich tolerant)
    z = zone.strip()
    pats = [f"actual_gen_{z}_2024*.csv"]
    alt = z.replace("-", "_") if "-" in z else z.replace("_", "-")
    if alt != z:
        pats.append(f"actual_gen_{alt}_2024*.csv")

    candidates = []
    for pat in pats:
        candidates += list(Path(path_dir).glob(pat))
    if not candidates:
        raise FileNotFoundError(f"Gen-CSV fehlt: actual_gen_{zone}_2024*.csv in {path_dir}")

    # 2) Einlesen (Separator/Encoding automatisch), Zeitspalte → Index
    df_raw = read_csv_smart(str(candidates[0]), min_cols=2)
    tcol = next((c for c in ["timestamp_cec","timestamp","time","datetime","MTU"] if c in df_raw.columns), df_raw.columns[0])
    df_raw.index = parse_ts(df_raw[tcol])
    df_raw = df_raw.drop(columns=[tcol])

    # 3) Falls schon 'wide': Spalten übernehmen, sonst pivotieren
    def _is_tech_col(c: str) -> bool:
        lc = str(c).lower()
        return any(k in lc for k in ["fossil", "nuclear", "wind", "solar", "hydro", "biomass", "waste", "geothermal", "other"])

    wide_cols = [c for c in df_raw.columns if _is_tech_col(c)]
    if len(wide_cols) >= 2:
        df_wide = df_raw.copy()
    else:
        # Long-Format erkennen
        cols_lc = {c.lower(): c for c in df_raw.columns}
        tech_col = next((df_raw.columns[i] for i,c in enumerate(df_raw.columns)
                         if any(k in c.lower() for k in ["productiontype","type","technology","tech","fuel"])), None)
        val_col  = next((df_raw.columns[i] for i,c in enumerate(df_raw.columns)
                         if any(k in c.lower() for k in ["actual","generation","gen","mw","value"])), None)
        if tech_col is None or val_col is None:
            raise ValueError(f"Unbekanntes Gen-Format in {candidates[0].name}: brauche eine Tech- und eine Wertspalte.")
        df_wide = (
            df_raw
            .assign(**{val_col: pd.to_numeric(df_raw[val_col], errors="coerce")})
            .pivot_table(index=df_raw.index, columns=tech_col, values=val_col, aggfunc="sum")
        )

    # 4) Synonyme auf ENTSO-E-Labels mappen
    ALIAS = {
        "Hydro PumpedStorage": "Hydro Pumped Storage",
        "Pumped Storage": "Hydro Pumped Storage",
        "Run-of-river": "Hydro Run-of-river and poundage",
        "Hydro Run-of-river": "Hydro Run-of-river and poundage",
        "Hard coal": "Fossil Hard coal",
        "Brown coal": "Fossil Brown coal/Lignite",
        "Lignite": "Fossil Brown coal/Lignite",
        "Oil": "Fossil Oil",
        "Biomasse": "Biomass",
        "Abfall": "Waste",
        "Wasserkraft Pumpspeicher": "Hydro Pumped Storage",
        "Wasserkraft Laufwasser": "Hydro Run-of-river and poundage",
    }
    ren = {}
    for c in list(df_wide.columns):
        base = str(c).strip()
        ren[c] = ALIAS.get(base, base)
    df_wide = df_wide.rename(columns=ren)

    # 5) Nur relevante Techs behalten, numerisch machen
    keep = [c for c in NEIGHBOR_TECHS if c in df_wide.columns]
    if not keep:
        # Wenn gar nichts matcht, lasse alles drin – aber numerisch
        keep = list(df_wide.columns)
    for c in keep:
        df_wide[c] = pd.to_numeric(df_wide[c], errors="coerce")
    df_wide = df_wide[keep]

    # 6) Auf stündlich mitteln und zurück
    return force_hourly(df_wide, "mean")



# -------------------------- Mappings & Defaults ------------------------------
EF_LOOKUP_T_PER_MWH_TH = {"Erdgas":0.201,"Steinkohle":0.335,"Braunkohle":0.383,"Heizöl schwer":0.288,"Heizöl leicht / Diesel":0.266}
FOSSIL_TECH_TO_FUEL = {
    "Fossil Gas": ("gas", "Erdgas"),
    "Fossil Hard coal": ("coal", "Steinkohle"),
    "Fossil Brown coal/Lignite": ("lignite", "Braunkohle"),
    "Fossil Coal-derived gas": ("gas", "Erdgas"),
    "Fossil Oil": ("oil", "Heizöl schwer"),
}
PRICE_COLS = ["gas_eur_mwh_th","coal_eur_mwh_th","lignite_eur_mwh_th","oil_eur_mwh_th","co2_eur_t"]

def _norm(text: str) -> str:
    if pd.isna(text): return ""
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

def _norm_zone(z: str) -> str: return str(z or "").strip().replace("-", "_").upper()

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
                if z == "DE_LU": continue
                pz = row.get(f"price_{z}", np.nan)
                if not pd.isna(pz) and abs(pz - p_de) <= eps:
                    coupled.append(z)
        clusters[t] = coupled
    return clusters

# ------------------- Export-Stack-Logik (Reservoir-Hydro erlaubt) ------------
NONDISP = ["Nuclear","Solar","Wind Onshore","Wind Offshore",
           "Hydro Run-of-river and poundage",
           "Biomass","Waste"]

def _stack_has_zero_srmc(blocks):
    return any((f in ("EE","Reservoir Hydro")) and (mw > 1e-6) and (abs(srmc) <= 1e-6)
               for (f, srmc, mw, eta, z) in blocks)

def exportable_blocks_for_zone(
    t: pd.Timestamp,
    zone: str,
    gen_z_row: pd.Series,
    load_z_t: float,
    fuel_prices_row: pd.Series,
    nei_dists: Dict[str, dict],
    mode: str,
    draws: int,
    args,
    cap_mask: Optional[Dict[Tuple[str,str], float]],
    reservoir_sp_map: Dict[Tuple[str, pd.Timestamp], float],
    min_total_zone_t: float,
    min_by_fuel_zone_t: Dict[str, float],
    price_zone: Optional[float],
    mu_bid_getter = None,
) -> List[Tuple[int, str, float, float, float, str]]:
    """
    Rückgabe sortiert nach Gruppen (kleiner = früher):
      0=EE (0 €/MWh),
      1=MU Waste (rollierendes Bid / fix 1 €),
      2=MU Nuclear (rollierendes Bid / fix),
      3=MU Biomass (rollierendes Bid / fix),
      4=MU Fossil (rollierendes Bid / fix),
      5=PSP (0..Floor),
      6=Reservoir (Shadow-Price),
      7=Nuclear FLEX (SRMC),
      8=Fossil FLEX (SRMC je Fuel)
    MW = exportierbarer Überschuss *nachdem* die Stufen der RL-Leiter lokal bedient sind.
    """
    mu_bid = mu_bid_getter or (lambda z, tech, tt, default: float(default))

    # Read techs
    fee = float(pd.to_numeric(gen_z_row.reindex(
        ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    ).fillna(0.0)).sum())
    waste = float(gen_z_row.get("Waste", 0.0))
    bio   = float(gen_z_row.get("Biomass", 0.0))
    nuc   = float(gen_z_row.get("Nuclear", 0.0))

    # MU-Shares (Biomasse/Nuclear parametrisierbar; Waste = vollständig MU)
    bio_mu_share = float(getattr(args, "nei_biomass_mustrun_share", 0.0))
    if zone == "DE_LU":
        bio_mu_share = float(getattr(args, "de_biomass_mustrun_share", 0.0))
    bio_mu   = max(min(bio, bio * bio_mu_share), 0.0)
    bio_flex = max(bio - bio_mu, 0.0)

    nuc_mu_share = float(getattr(args, "nei_nuclear_mustrun_share", 0.0))
    if zone == "DE_LU":
        nuc_mu_share = float(getattr(args, "de_nuclear_mustrun_share", 0.0))
    nuc_mu   = max(min(nuc, nuc * nuc_mu_share), 0.0)
    nuc_flex = max(nuc - nuc_mu, 0.0)

    # Nuklear-SRMC
    if getattr(args, "nuclear_srmc_eur_mwh", None) is not None:
        nuc_srmc = float(args.nuclear_srmc_eur_mwh)
    else:
        f = float(getattr(args, "nuclear_fuel_eur_mwh_th", 5.5))
        eta = max(float(getattr(args, "nuclear_eta", 0.33)), 1e-6)
        varo = float(getattr(args, "nuclear_varom_eur_mwh", 1.2))
        nuc_srmc = f/eta + varo

    # RL-Leiter gemäß gewünschter Reihenfolge
    RL1 = max(load_z_t - fee, 0.0)                # nach FEE
    take_waste_mu = min(waste, RL1);              RL2 = max(RL1 - take_waste_mu, 0.0)
    take_nuc_mu   = min(nuc_mu, RL2);             RL3 = max(RL2 - take_nuc_mu, 0.0)
    take_bio_mu   = min(bio_mu, RL3);             RL4 = max(RL3 - take_bio_mu, 0.0)
    mu_foss_total = float(min_total_zone_t)       # nur Fossil-MU (aus compute_fossil_min_profiles)
    take_mu_foss  = min(mu_foss_total, RL4);      RL5 = max(RL4 - take_mu_foss, 0.0)

    # PSP, Reservoir
    mw_psp = float(gen_z_row.get("Hydro Pumped Storage", 0.0)) if "Hydro Pumped Storage" in gen_z_row.index else 0.0
    mw_res = float(gen_z_row.get("Hydro Water Reservoir", 0.0)) if "Hydro Water Reservoir" in gen_z_row.index else 0.0
    take_psp = min(mw_psp, RL5);                  RL6 = max(RL5 - take_psp, 0.0)
    take_res = min(mw_res, RL6);                  RL7 = max(RL6 - take_res, 0.0)  # verbleibende RL wird durch FLEX gedeckt

    # Surplus je Stufe → exportierbare Blöcke (in Gruppen-Reihenfolge)
    out: List[Tuple[int,str,float,float,float,str]] = []

    # G0 EE-Surplus
    if fee - (load_z_t - RL1) > 1e-6:
        out.append((0, "EE", 0.0, float(fee - (load_z_t - RL1)), 1.0, zone))

    # G1 MU Waste (Gebot: rollierend oder fix 1 €)
    waste_surplus = max(waste - take_waste_mu, 0.0)
    if waste_surplus > 1e-6:
        bid_w = mu_bid(zone, "waste", t, float(getattr(args, "waste_bid_eur_mwh_export", 1.0)))
        out.append((1, "Waste", float(bid_w), float(waste_surplus), 1.0, zone))

    # G2 MU Nuclear (Gebot: rollierend oder mustrun_bid)
    nuc_mu_surplus = max(nuc_mu - take_nuc_mu, 0.0)
    if nuc_mu_surplus > 1e-6:
        bid_n = mu_bid(zone, "nuclear", t, float(getattr(args, "mustrun_bid_eur_mwh", 1.0)))
        out.append((2, "Nuclear", float(bid_n), float(nuc_mu_surplus), 1.0, zone))

    # G3 MU Biomass (Gebot: rollierend oder biomass_bid)
    bio_mu_surplus = max(bio_mu - take_bio_mu, 0.0)
    if bio_mu_surplus > 1e-6:
        bid_b = mu_bid(zone, "biomass", t, float(getattr(args, "biomass_bid_eur_mwh_export",
                                             getattr(args,"biomass_srmc_eur_mwh",35.0))))
        out.append((3, "Biomass", float(bid_b), float(bio_mu_surplus), 1.0, zone))

    # G4 MU Fossil (Gebot: rollierend oder mustrun_bid)
    mu_foss_surplus = max(mu_foss_total - take_mu_foss, 0.0)
    if mu_foss_surplus > 1e-6:
        bid_f = mu_bid(zone, "fossil", t, float(getattr(args, "mustrun_bid_eur_mwh", 1.0)))
        out.append((4, "Mustrun (fossil)", float(bid_f), float(mu_foss_surplus), 1.0, zone))

    # G5 PSP (0 bis Floor)
    if mw_psp - take_psp > 1e-6:
        out.append((5, "Hydro Pumped Storage", 0.0, float(mw_psp - take_psp), 1.0, zone))

    # G6 Reservoir (Shadow-Price)
    if mw_res - take_res > 1e-6:
        srmc_res = float(reservoir_sp_map.get((zone, t), 0.0))
        srmc_res = float(np.clip(srmc_res, 0.0, float(getattr(args,"psp_srmc_floor_eur_mwh",60.0))))
        out.append((6, "Reservoir Hydro", srmc_res, float(mw_res - take_res), 1.0, zone))

    # G7 Nuclear FLEX (positives SRMC)
    if nuc_flex > 1e-6:
        out.append((7, "Nuclear", float(nuc_srmc), float(nuc_flex), 1.0, zone))

    # G8 Fossil FLEX (Gas/Coal/Oil mit η-Modell)
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
        mw_mustrun = float(min_by_fuel_zone_t.get(ef_name, 0.0))
        mw = max(mw_raw - mw_mustrun, 0.0)
        if mw <= 1e-6:
            continue
        out.append((8, ef_name, float(srmc), float(mw), float(eta_eff), zone))

    # Sortierung: erst Gruppe, dann SRMC
    out.sort(key=lambda x: (x[0], x[2]))
    return out

def residual_need_after_local_steps(
    t: pd.Timestamp,
    zone: str,
    gen_z_row: pd.Series,
    load_z_t: float,
    min_total_zone_t: float,
    args=None,   # <— neu
) -> float:
    """
    Liefert die 'Restlast nach lokalen Stufen' (analog RL7) der Zone:
    Last – FEE – MU(Waste) – MU(Nuclear) – MU(Biomass) – MU(Fossil) – PSP – Reservoir, min 0.
    Nutzt exakt die gleiche Reihenfolge wie exportable_blocks_for_zone().
    """
    fee = float(pd.to_numeric(gen_z_row.reindex(
        ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    ).fillna(0.0)).sum())
    waste = float(gen_z_row.get("Waste", 0.0))
    bio   = float(gen_z_row.get("Biomass", 0.0))
    nuc   = float(gen_z_row.get("Nuclear", 0.0))

    # Shares robust aus args (oder 0.0 falls None)
    bio_mu_share = float(getattr(args, "nei_biomass_mustrun_share", 0.0)) if args is not None else 0.0
    if zone == "DE_LU":
        bio_mu_share = float(getattr(args, "de_biomass_mustrun_share", bio_mu_share)) if args is not None else bio_mu_share
    bio_mu = max(min(bio, bio * bio_mu_share), 0.0)

    nuc_mu_share = float(getattr(args, "nei_nuclear_mustrun_share", 0.0)) if args is not None else 0.0
    if zone == "DE_LU":
        nuc_mu_share = float(getattr(args, "de_nuclear_mustrun_share", nuc_mu_share)) if args is not None else nuc_mu_share
    nuc_mu = max(min(nuc, nuc * nuc_mu_share), 0.0)

    mu_foss_total = float(min_total_zone_t)

    RL1 = max(load_z_t - fee, 0.0)
    RL2 = max(RL1 - min(waste, RL1), 0.0)
    RL3 = max(RL2 - min(nuc_mu, RL2), 0.0)
    RL4 = max(RL3 - min(bio_mu, RL3), 0.0)
    RL5 = max(RL4 - min(mu_foss_total, RL4), 0.0)

    mw_psp = float(gen_z_row.get("Hydro Pumped Storage", 0.0)) if "Hydro Pumped Storage" in gen_z_row.index else 0.0
    RL6 = max(RL5 - min(mw_psp, RL5), 0.0)

    mw_res = float(gen_z_row.get("Hydro Water Reservoir", 0.0)) if "Hydro Water Reservoir" in gen_z_row.index else 0.0
    RL7 = max(RL6 - min(mw_res, RL6), 0.0)

    return RL7


# ------------------------------ Validation & Plots ---------------------------
PALETTE = {
    "DE": "#1f77b4", "IMPORT": "#d62728", "EE": "#2ca02c",
    "price": "#444444", "warn": "#ff7f0e", "ok": "#2ca02c", "mix": "#7f7f7f",
}
def _pct(x, y): return 0.0 if y == 0 else 100.0 * (x / y)

def validate_run(df_res: pd.DataFrame, df_dbg: pd.DataFrame, flows: pd.DataFrame,
                 prices: pd.DataFrame, epsilon_price: float, price_anchor_mode: str,
                 tol_balance_mw: float = 1.0):
    out = pd.DataFrame(index=df_res.index).copy()
    out["price_DE"] = df_res["price_DE"]
    out["marginal_srmc"] = df_res["marginal_srmc_eur_per_mwh"]
    out["marginal_side"] = df_res["marginal_side"]
    out["marginal_label"] = df_res["marginal_label"]
    out["marginal_fuel"] = df_res["marginal_fuel"]
    out["mef_gpkwh"] = df_res["mef_g_per_kwh"]
    out["net_import_total_MW"] = df_res["net_import_total_MW"]

    for c in [c for c in prices.columns if c.startswith("price_") and c != "price_DE_LU"]:
        out[f"abs_{c}_minus_DE"] = (prices[c] - prices["price_DE_LU"]).abs().reindex(out.index)

    out["IMPORT_anchor_ok"] = True
    if price_anchor_mode in ("closest", "threshold"):
        imp_srmc = df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index)
        de_srmc  = df_dbg["DE_srmc"].reindex(out.index)
        p_de     = out["price_DE"]
        if price_anchor_mode == "closest":
            out.loc[out["marginal_side"]=="IMPORT","IMPORT_anchor_ok"] = (
                (imp_srmc - p_de).abs() <= (de_srmc - p_de).abs()
            )

    out["EE_surplus_flag"] = (df_res["residual_domestic_fossil_MW"] <= 1e-6)
    out["EE_surplus_mef_ok"] = ~(out["EE_surplus_flag"]) | (out["mef_gpkwh"] <= 1e-6) | (out["marginal_side"]=="IMPORT")

    out["IMPORT_has_block"] = ~df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index).isna()
    mask_import = (out["marginal_side"]=="IMPORT")
    out["IMPORT_logic_ok"] = True
    out.loc[mask_import, "IMPORT_logic_ok"] = (
        (out.loc[mask_import, "net_import_total_MW"] > 0.0) & (out.loc[mask_import, "IMPORT_has_block"])
    )

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

def write_validation_report(outdir: Path, df_val: pd.DataFrame, df_sum: pd.DataFrame) -> None:
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)
    df_val.to_csv(outdir / "analysis" / "_validation.csv", index=True)
    df_sum.to_csv(outdir / "analysis" / "_validation_summary.csv", index=True)
    print("[VALIDATION] geschrieben:", outdir / "analysis" / "_validation.csv", "und", outdir / "analysis" / "_validation_summary.csv")
def write_negative_price_gen_summary(outdir: Path, nei_prices: pd.DataFrame, gen_by_zone: Dict[str,pd.DataFrame]):
    rows = []
    for z, g in gen_by_zone.items():
        pcol = f"price_{z}" if f"price_{z}" in nei_prices.columns else None
        if pcol is None:
            continue
        pp = nei_prices[pcol].reindex(g.index)
        mask = pd.to_numeric(pp, errors="coerce") < 0.0
        if not mask.any():
            continue
        sub = g[mask.fillna(False)]
        for col in sub.columns:
            val_mwh = float(pd.to_numeric(sub[col], errors="coerce").fillna(0.0).sum())
            rows.append({"zone": z, "tech": col, "gen_at_negative_price_MWh": val_mwh})
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "analysis" / "_gen_when_price_negative_by_zone_tech.csv", index=False)
def _ts_ax(ax, tzlabel=""):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    if tzlabel: ax.set_xlabel(f"Zeit ({tzlabel})")

# ---------- GLOBAL COLOR SYSTEM (BBH-ish) ------------------------------------
from matplotlib import colors as mcolors
def set_bbh_style():
    plt.rcParams.update({
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "axes.grid": True,
        "grid.alpha": 0.30,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "legend.frameon": True,
    })
set_bbh_style()

GLOBAL = {
    # Grundschema – hier nur Hex anpassen und alles andere folgt automatisch
    "primary":   "#00374B",  # Hauptlinie / Fokus (z.B. Last, DE-Serie)
    "secondary": "#1B6E8C",  # Vergleich / RL1 / Import-Serie
    "tertiary":  "#2F4B7C",  # weitere Vergleichslinie (RL2)
    "accent":    "#9B0028",  # Signal / Hervorhebung (RL3, Events)
    "warning":   "#D1495B",  # Warn-/Öl-Peaker o.ä.
    "success":   "#0F8C4A",  # EE (Wind+PV), „positiv / grün“
    "neutral":   "#444444",  # neutrale/sekundäre Linien (RL4, Achsen)
    "muted":     "#8D99A6",  # gedämpfte Infos (Preis, Nebenlinien)
    "background":"#F4EFE8",  # Legenden-Background (BBH-Beige)
    "outline":   "#8D7B5A",  # Legenden-Rahmen
}

def _to_rgb(hex_):
    return mcolors.to_rgb(hex_)

def _blend(c1, c2, t=0.3):
    r1,g1,b1=_to_rgb(c1); r2,g2,b2=_to_rgb(c2)
    return (r1+(r2-r1)*t, g1+(g2-g1)*t, b1+(b2-b1)*t)

def lighten(hex_, t=0.25):   # zu Weiß
    return _blend(hex_, "#FFFFFF", t)
def darken(hex_, t=0.2):     # zu Schwarz
    return _blend(hex_, "#000000", t)

# Abgeleitete Palette für Code-Komponenten
PALETTE = {
    "DE":      GLOBAL["primary"],
    "IMPORT":  GLOBAL["secondary"],
    "price":   GLOBAL["muted"],
    "bg":      GLOBAL["background"],
    "bg_edge": GLOBAL["outline"],
}

# Tech-Farben (gestapelte Balken, marginal mix) – nach Semantik abgeleitet
TECH_COLORS = {
    "Braunkohle":           darken(GLOBAL["neutral"], 0.15),
    "Steinkohle":           "#A08268",                         # „erdig“
    "Erdgas":               GLOBAL["secondary"],
    "EE":                   GLOBAL["success"],
    "Solar":                lighten(GLOBAL["success"], 0.35),
    "Wind Onshore":         GLOBAL["success"],
    "Wind Offshore":        darken(GLOBAL["success"], 0.15),
    "Reservoir Hydro":      darken(GLOBAL["secondary"], 0.10),
    "Hydro Pumped Storage": lighten(GLOBAL["secondary"], 0.45),
    "MustrunMix":           "#B08B6E",
    "Heizöl schwer":        GLOBAL["warning"],
    "Nuclear":              "#6B3FA0",
    "Waste":                "#8A8D35",
    "Biomasse":             "#6E8B3D",
    "Biomass":              "#6E8B3D",  # engl. Key-Fallback
}
SYNONYMS = {
    "Wind": "Wind Onshore",
    "PV": "Solar",
    "Renewables": "EE",
    "RES": "EE",
    "Pumped Storage": "Hydro Pumped Storage",
    "Hydro PumpedStorage": "Hydro Pumped Storage",
    "Reservoir": "Reservoir Hydro",
    "Hard coal": "Steinkohle",
    "Lignite": "Braunkohle",
    "Oil": "Heizöl schwer",
    "Biomass": "Biomasse",
}

def color_for(label: str) -> str:
    """Mappt beliebige Tech-Labels robust auf das globale Farbschema."""
    if not isinstance(label, str):
        return GLOBAL["muted"]
    lab = SYNONYMS.get(label.strip(), label.strip())
    return TECH_COLORS.get(lab, GLOBAL["muted"])
    
# BBH-Style-Legende
def _legend_bbh(ax, *args, **kwargs):
    leg = ax.legend(*args, **kwargs)
    if leg:
        fr = leg.get_frame()
        fr.set_facecolor(PALETTE["bg"])
        fr.set_edgecolor(PALETTE["bg_edge"])
        fr.set_alpha(0.95); fr.set_linewidth(0.8)
    return leg
# ------------------------------------------------------------------------------

def _safe_sum(df, cols):
    # Robust gegen Series/DF + fehlende Spalten
    if isinstance(df, pd.Series):
        return pd.Series(0.0, index=df.index)

    if not isinstance(df, pd.DataFrame):
        return pd.Series(0.0, index=pd.DatetimeIndex([]))

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)

    # WICHTIG: apply(pd.to_numeric, …) auf die Spalten statt to_numeric auf dem DF
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .sum(axis=1)
    )


def _plot_residuallast_decomposition_per_month(
    outdir: Path,
    month: int,
    de_load: pd.Series,
    de_gen: pd.DataFrame,
    de_min_total: pd.Series,
    flows: pd.DataFrame,
    prices: pd.DataFrame,
):
    pdir = outdir / "analysis" / "plots_rl"
    pdir.mkdir(parents=True, exist_ok=True)

    mask = (de_load.index.month == month)
    if not mask.any(): return
    idx = de_load.index[mask]

    L   = de_load.reindex(idx).astype(float)
    FEE = _safe_sum(de_gen.reindex(idx), ["Solar","Wind Onshore","Wind Offshore"])
    WB  = _safe_sum(de_gen.reindex(idx), ["Waste","Biomasse","Biomass"])
    MU  = de_min_total.reindex(idx).fillna(0.0).astype(float)
    NI  = flows.reindex(idx)["net_import_total"].fillna(0.0).astype(float) if "net_import_total" in flows.columns else 0.0
    P   = prices.reindex(idx)["price_DE_LU"].astype(float) if "price_DE_LU" in prices.columns else None
    PSP = pd.to_numeric(de_gen.reindex(idx).get("Hydro Pumped Storage", pd.Series(0.0, index=idx)), errors="coerce").fillna(0.0).clip(lower=0.0)

    RL1 = (L - FEE).rename("RL1")
    RL2 = (RL1 - WB).rename("RL2")
    RL3 = (RL2 - MU).rename("RL3")
    RL3_PSP = (RL3 - PSP).rename("RL3 − PSP")
    RL4 = (RL3 - NI).rename("RL4 (NI last)")
    RL4_psp_first = (RL3_PSP - NI).rename("RL4 (PSP→NI)")

    fig, ax = plt.subplots(figsize=(18, 6))
    # Hauptkurven (BBH-Schema)
    ax.plot(L.index,   L.values,   label="Last",                    lw=1.8, color=GLOBAL["primary"])
    ax.plot(RL1.index, RL1.values, label="RL1 = Last − FEE",        lw=1.6, color=GLOBAL["secondary"])
    ax.plot(RL2.index, RL2.values, label="RL2 − (Waste+Biomasse)",  lw=1.6, color=GLOBAL["tertiary"])
    ax.plot(RL3.index, RL3.values, label="RL3 − foss. Mindestlast", lw=1.7, color=GLOBAL["accent"])
    ax.plot(RL3_PSP.index, RL3_PSP.values, label="RL3 − PSP",       lw=1.6, color=darken(GLOBAL["secondary"],0.15))
    ax.plot(RL4.index, RL4.values, label="RL4 − nach Handel (NI)",  lw=1.6, color=PALETTE["bg_edge"], alpha=0.9)

    # Kontext-Linien (transparent)
    ax.plot(FEE.index, FEE.values, label="FEE (Solar+Wind)",        lw=1.0, alpha=0.35, color=color_for("EE"))
    ax.plot(WB.index,  WB.values,  label="Waste+Biomasse",          lw=1.0, alpha=0.35, color=color_for("Steinkohle"))  # erdig
    ax.plot(MU.index,  MU.values,  label="Fossile Mindestlast",     lw=1.0, alpha=0.35, color=lighten(GLOBAL["accent"],0.55))
    ax.plot(PSP.index, PSP.values, label="PSP Gen",                 lw=1.0, alpha=0.35, color=lighten(GLOBAL["secondary"],0.55))

    _ts_ax(ax, tzlabel="Europe/Berlin")
    ax.set_ylabel("Leistung [MW]")

    if P is not None:
        ax2 = ax.twinx()
        ax2.plot(P.index, P.values, ls="--", lw=1.2, color=PALETTE["price"], label="Preis (DE/LU)")
        ax2.set_ylabel("Preis [€/MWh]")
        h, l = [], []
        for a in (ax, ax2):
            hi, li = a.get_legend_handles_labels(); h += hi; l += li
        _legend_bbh(ax, handles=h, labels=l, loc="upper left")
    else:
        _legend_bbh(ax, loc="upper left")

    ax.set_title(f"Residuallast-Zerlegung (RL1…RL4) – Monat {month:02d} (mit PSP-Stufe)")
    fig.tight_layout()
    fig.savefig(pdir / f"rl_decomposition_month_{month:02d}.png", dpi=160)
    plt.close(fig)

    # Quantifizierung als CSV
    out_csv = outdir / "analysis" / f"residuallast_decomp_month_{month:02d}.csv"
    pd.DataFrame({
        "L": L, "FEE": FEE, "WB": WB, "MU": MU, "PSP_gen": PSP, "NI": NI,
        "RL1": RL1, "RL2": RL2, "RL3": RL3, "RL3_minus_PSP": RL3_PSP,
        "RL4_original(NI_last)": RL4, "RL4_psp_first": RL4_psp_first
    }).to_csv(out_csv, index=True)


def plot_rl_ladder_week(outdir: Path, week_start: pd.Timestamp,
                        de_load: pd.Series, de_gen: pd.DataFrame,
                        de_min_total: pd.Series, flows: pd.DataFrame):
    pdir = outdir / "analysis" / "plots_rl"; pdir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range(week_start, week_start + pd.Timedelta(days=7), freq="h", inclusive="left", tz=TZ)
    L   = de_load.reindex(idx).astype(float)
    FEE = _safe_sum(de_gen.reindex(idx), ["Solar","Wind Onshore","Wind Offshore"])
    WB  = _safe_sum(de_gen.reindex(idx), ["Waste","Biomasse","Biomass"])
    MU  = de_min_total.reindex(idx).fillna(0.0).astype(float)
    NI  = flows.reindex(idx)["net_import_total"].fillna(0.0).astype(float) if "net_import_total" in flows.columns else 0.0

    RL1 = (L - FEE).clip(lower=0); RL2 = (RL1 - WB).clip(lower=0); RL3 = (RL2 - MU).clip(lower=0); RL4 = (RL3 - NI)
    # Stacked Area: was reduziert die Last?
    fig, ax = plt.subplots(figsize=(18,5))
    ax.fill_between(idx, 0, L, facecolor=lighten(GLOBAL["primary"],0.75), label="Last")
    ax.fill_between(idx, L-FEE, L, facecolor=lighten(color_for("EE"),0.35), label="FEE")
    ax.fill_between(idx, L-(FEE+WB), L-FEE, facecolor=lighten(TECH_COLORS["Waste"],0.35), label="Waste+Biomasse")
    ax.fill_between(idx, L-(FEE+WB+MU), L-(FEE+WB), facecolor=lighten(GLOBAL["accent"],0.55), label="foss. Mindestlast")
    ax.plot(idx, RL3, lw=1.5, color=GLOBAL["accent"], label="RL3")
    ax.plot(idx, RL4, lw=1.5, color=PALETTE["bg_edge"], label="RL4 (nach Handel)")
    _ts_ax(ax, "Europe/Berlin"); ax.set_ylabel("MW")
    _legend_bbh(ax, loc="upper left"); ax.set_title(f"RL-Leiter (Woche ab {week_start:%Y-%m-%d})")
    fig.tight_layout(); fig.savefig(pdir / f"rl_ladder_{week_start:%Y%m%d}.png", dpi=160); plt.close(fig)

def plot_rl_duration_curves(outdir: Path, df: pd.DataFrame):
    # df erwartet Spalten RL1..RL4 (du hast diese je Monat als CSV)
    pdir = outdir / "analysis" / "plots_rl"; pdir.mkdir(parents=True, exist_ok=True)
    for m, sub in df.groupby(df.index.month):
        cols = [c for c in ["RL1","RL2","RL3","RL4_original(NI_last)"] if c in sub.columns]
        if not cols: continue
        fig, ax = plt.subplots(figsize=(8,4))
        for c in cols:
            s = sub[c].dropna().sort_values(ascending=False).values
            ax.plot(np.arange(len(s)), s, lw=1.5, label=c,
                    color={"RL1":GLOBAL["secondary"],"RL2":GLOBAL["tertiary"],
                           "RL3":GLOBAL["accent"],"RL4_original(NI_last)":PALETTE["bg_edge"]}.get(c, GLOBAL["neutral"]))
        ax.set_xlabel("Stunden (absteigend)"); ax.set_ylabel("MW"); ax.set_title(f"RL-Dauerlinien – Monat {m:02d}")
        ax.grid(True, alpha=0.3); _legend_bbh(ax, loc="best")
        fig.tight_layout(); fig.savefig(pdir / f"rl_duration_month_{m:02d}.png", dpi=160); plt.close(fig)

def plot_monthly_contributions(outdir: Path):
    pdir = outdir / "analysis" / "plots_rl"; pdir.mkdir(parents=True, exist_ok=True)
    # nutze die pro-Monat CSVs aus _plot_residuallast_decomposition_per_month
    paths = sorted((outdir/"analysis").glob("residuallast_decomp_month_*.csv"))
    if not paths: return
    rows=[]
    for p in paths:
        m = int(p.stem[-2:])
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        # mittlere Beiträge (MW): wieviel senken FEE/WB/MU/NI im Monatsmittel die Last?
        rows.append({
            "month": m,
            "FEE": d["FEE"].mean(),
            "Waste+Biomasse": d[["WB"]].sum(axis=1).mean(),
            "foss. Mindestlast": d["MU"].mean(),
            "Nettoimporte": d["NI"].mean(),
        })
    dd = pd.DataFrame(rows).set_index("month").sort_index()
    fig, ax = plt.subplots(figsize=(10,4))
    bottom = np.zeros(len(dd))
    for k, col in [("FEE", color_for("EE")),
                   ("Waste+Biomasse", lighten(TECH_COLORS["Waste"],0.35)),
                   ("foss. Mindestlast", lighten(GLOBAL["accent"],0.55)),
                   ("Nettoimporte", PALETTE["bg_edge"])]:
        ax.bar(dd.index, dd[k], bottom=bottom, label=k, edgecolor="none", color=col)
        bottom += dd[k].values
    ax.set_xlabel("Monat"); ax.set_ylabel("mittlere Reduktion [MW]")
    ax.set_title("Durchschnittliche Beiträge zur RL-Reduktion je Monat")
    ax.grid(axis="y", alpha=0.3); _legend_bbh(ax, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(pdir / "rl_contributions_by_month.png", dpi=160); plt.close(fig)

def plot_rl_vs_price(outdir: Path, df_res: pd.DataFrame):
    pdir = outdir / "analysis" / "plots_rl"; pdir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["RL_after_FEE_MW","RL_after_NDext_MW","residual_domestic_fossil_MW","residual_after_trade_MW"] if c in df_res.columns]
    if not cols: return
    price = df_res["price_DE"]
    fig, ax = plt.subplots(figsize=(6,5))
    for c, col in zip(cols, [GLOBAL["secondary"], GLOBAL["tertiary"], GLOBAL["accent"], PALETTE["bg_edge"]]):
        ax.scatter(df_res[c], price, s=8, alpha=0.25, label=c, color=col)
    ax.set_xlabel("MW"); ax.set_ylabel("Preis [€/MWh]"); ax.set_title("Preis vs. RL-Stufe")
    ax.grid(True, alpha=0.3); _legend_bbh(ax, loc="best")
    fig.tight_layout(); fig.savefig(pdir / "scatter_price_vs_rl_stages.png", dpi=160); plt.close(fig)

def _plot_mef_hourly_mean_per_month(outdir: Path, df_res: pd.DataFrame):
    """Für jeden Monat: Ø MEF je Stunde (24 Werte)."""
    pdir = outdir / "analysis" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    s = df_res["mef_g_per_kwh"].dropna()
    if s.empty:
        return
    df = s.to_frame("mef").copy()
    df["month"] = df.index.month
    df["hour"]  = df.index.hour
    piv = df.pivot_table(index="month", columns="hour", values="mef", aggfunc="mean")
    for m in piv.index:
        vals = piv.loc[m].astype(float).values
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.plot(range(24), vals, marker="o", linewidth=1.6)
        ax.set_xlabel("Stunde"); ax.set_ylabel("Ø MEF [g/kWh]")
        ax.set_xticks(range(0,24,2))
        ax.grid(True, alpha=0.3)
        _legend_bbh(ax)  # nur Rahmen, falls später Legenden ergänzt werden
        ax.set_title(f"Ø MEF je Stunde – Monat {int(m):02d}")
        fig.tight_layout()
        fig.savefig(pdir / f"mef_hourly_mean_month_{int(m):02d}.png", dpi=160)
        plt.close(fig)

# REPLACE the existing make_validation_plots with this enhanced version:
def make_validation_plots(outdir: Path,
                          df_res: pd.DataFrame,
                          df_dbg: pd.DataFrame,
                          df_val: pd.DataFrame,
                          prices: pd.DataFrame,
                          de_gen: pd.DataFrame,
                          de_min_total: pd.Series,
                          de_load: pd.Series,
                          flows: pd.DataFrame) -> None:
    """Erzeugt Validierungsplots + RL-Zerlegung + MEF-Stundenmittel je Monat."""
    pdir = outdir / "analysis" / "plots"
    pdir.mkdir(parents=True, exist_ok=True)

    # 1) Timeseries: Preis vs. gewählter SRMC
    fig, ax = plt.subplots(figsize=(12, 4))
    srmc  = df_res["marginal_srmc_eur_per_mwh"]
    price = df_res["price_DE"]
    ax.plot(price.index, price.values, label="Preis (DE/LU)", linewidth=1.3, color=PALETTE["price"])
    ax.plot(srmc.index,  srmc.values,  label="gewählter SRMC", linewidth=1.3, color=PALETTE["DE"])
    _ts_ax(ax, tzlabel="Europe/Berlin")
    ax.set_ylabel("€/MWh")
    ax.grid(True, alpha=0.3)
    _legend_bbh(ax, loc="best")
    fig.tight_layout(); fig.savefig(pdir / "timeseries_price_vs_chosen_srmc.png", dpi=160); plt.close(fig)

    # 2) Scatter: Preis vs. SRMC (DE/IMPORT)
    side = df_res["marginal_side"].reindex(srmc.index)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    m_de = (side=="DE"); m_im = (side=="IMPORT")
    ax.scatter(srmc[m_de], price[m_de], s=10, alpha=0.6, label="DE",      color=PALETTE["DE"])
    ax.scatter(srmc[m_im], price[m_im], s=10, alpha=0.6, label="IMPORT",  color=PALETTE["IMPORT"])
    lims = [np.nanmin([srmc.min(), price.min()]), np.nanmax([srmc.max(), price.max()])]
    ax.plot(lims, lims, linestyle="--", linewidth=1, color=PALETTE["price"])
    ax.set_xlabel("gewählter SRMC [€/MWh]"); ax.set_ylabel("Preis [€/MWh]")
    ax.grid(True, alpha=0.3); _legend_bbh(ax, loc="best")
    fig.tight_layout(); fig.savefig(pdir / "scatter_price_vs_srmc.png", dpi=160); plt.close(fig)

    # 3) Histogramm Preis−SRMC
    err = (price - srmc).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err, bins=60)
    ax.set_xlabel("Preis − SRMC [€/MWh]"); ax.set_ylabel("Häufigkeit")
    ax.grid(True, alpha=0.3); _legend_bbh(ax)
    fig.tight_layout(); fig.savefig(pdir / "hist_price_minus_srmc.png", dpi=160); plt.close(fig)

    # 4) Heatmap Median(Preis−SRMC) Monat×Stunde
    dfh = (price - srmc).to_frame("err")
    dfh["month"] = dfh.index.month; dfh["hour"] = dfh.index.hour
    piv = dfh.pivot_table(index="month", columns="hour", values="err", aggfunc="median")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(piv.values, aspect="auto", origin="lower")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_xlabel("Stunde"); ax.set_ylabel("Monat"); ax.set_title("Median(Preis−SRMC)")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, label="€/MWh")
    fig.tight_layout(); fig.savefig(pdir / "heatmap_median_err_month_hour.png", dpi=160); plt.close(fig)

    # 5) MEF nach marginaler Technologie (Boxplot)
    dfm = df_res[["marginal_fuel","mef_g_per_kwh"]].dropna()
    if not dfm.empty:
        order = dfm.groupby("marginal_fuel")["mef_g_per_kwh"].median().sort_values().index
        data  = [dfm.loc[dfm["marginal_fuel"]==f, "mef_g_per_kwh"].values for f in order]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(data, labels=order, showfliers=False)
        ax.set_ylabel("MEF [g/kWh]"); ax.set_title("MEF nach marginaler Technologie")
        ax.grid(True, axis="y", alpha=0.3); _legend_bbh(ax)
        fig.tight_layout(); fig.savefig(pdir / "box_mef_by_fuel.png", dpi=160); plt.close(fig)

    # 6) Import-Anteil (roll. 7 Tage)
    side_num = (side=="IMPORT").astype(float).rolling(24*7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(side_num.index, 100.0*side_num.values, color=PALETTE["IMPORT"], label="IMPORT-Anteil (roll. 7T)")
    _ts_ax(ax, tzlabel="Europe/Berlin"); ax.set_ylabel("Anteil [%]"); ax.grid(True, alpha=0.3)
    _legend_bbh(ax, loc="best")
    fig.tight_layout(); fig.savefig(pdir / "share_import_rolling.png", dpi=160); plt.close(fig)

    # 7) Import-Stack Häufigkeit
    lab = df_dbg["IMPORT_label"].fillna("")
    fuels = lab.str.extract(r"\((.+)\)")[0].fillna("n/a")
    counts = fuels[side=="IMPORT"].value_counts()
    if counts.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Stunden"); ax.set_title("Import-marginale Fuels (Häufigkeit)")
        ax.grid(True, axis="y", alpha=0.3); _legend_bbh(ax)
        fig.tight_layout(); fig.savefig(pdir / "bar_import_marginal_fuels.png", dpi=160); plt.close(fig)

    # 8) Flag Timeline
    sus = df_val["suspect_price_deviation"].reindex(df_res.index).fillna(False)
    fig, ax = plt.subplots(figsize=(12, 0.8))
    ax.plot(sus.index, sus.astype(int).values, linewidth=1)
    _ts_ax(ax, tzlabel="Europe/Berlin")
    ax.set_yticks([0,1]); ax.set_yticklabels(["ok","suspekt"])
    ax.set_title("Preis-Abweichung Flag"); fig.tight_layout()
    fig.savefig(pdir / "flag_price_deviation_timeline.png", dpi=160); plt.close(fig)

    # 9) NEU: Residuallast-Zerlegung RL1..RL4 (für alle vorhandenen Monate)
    months = sorted(de_load.index.unique().month)
    for m in months:
        _plot_residuallast_decomposition_per_month(outdir, m, de_load, de_gen, de_min_total, flows, prices)

    # 10) NEU: Ø MEF je Stunde pro Monat
    _plot_mef_hourly_mean_per_month(outdir, df_res)
# --- END PATCH ----------------------------------------------------------------


# ----------------- Fossile Mindestprofile (DE & Nachbarn) --------------------
def compute_fossil_min_profiles(
    gen_df: pd.DataFrame, fuels_select: List[str], peak_hours: str,
    mode: str, q: float = 0.10,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    if mode == "off" or gen_df is None or gen_df.empty:
        idx = gen_df.index if (gen_df is not None and isinstance(gen_df.index, pd.DatetimeIndex)) else pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx), {f: pd.Series(0.0, index=idx) for f in fuels_select}

    TECH2FUEL = {"Fossil Gas":"Erdgas","Fossil Hard coal":"Steinkohle","Fossil Oil":"Heizöl schwer"}
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
    zones = sorted([
        c.split("price_",1)[1].strip().replace("-", "_")
        for c in nei_prices.columns if str(c).startswith("price_")
    ])

    # 4) Opportunitätskosten-Map für Reservoir je Zone & Stunde
    reservoir_sp_map = {}
    for z in zones:
        sp = reservoir_shadow_price_series(nei_prices, z)
        for tt, val in sp.items():
            v = float(val) if pd.notna(val) else 0.0
            reservoir_sp_map[(z, tt)] = v

    # 5) Nachbar-Gen/Load + DE/LU
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

    # 6) Fossile Mindestprofile vorbereiten (alle Zonen)
    fossil_list = [s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()]
    nei_min_total_by_zone = {}
    nei_min_by_zone_fuel = {}
    for z in zones:
        if z not in gen_by_zone: continue
        z_gen = gen_by_zone[z]
        if args.mu_cost_mode == "q_vs_cost":
            # Preis-Spalte der Zone (fallback DE/LU)
            pcol = f"price_{z}" if f"price_{z}" in nei_prices.columns else "price_DE_LU"
            total_min, by_fuel = compute_fossil_min_profiles_cost_based(
                gen_df=z_gen,
                price_series=nei_prices[pcol],
                fuel_prices=fuel_prices,
                fuels_select=[s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()],
                peak_hours=args.mustrun_peak_hours,
                q=float(args.mu_cost_q),
                alpha=float(args.mu_cost_alpha),
                monthly=bool(getattr(args, "mu_cost_monthly", False)),
                use_peak_split=bool(getattr(args, "mu_cost_use_peak", False)),
                eta_source="NEI",
                nei_dists=DEFAULT_NEI_DISTS,  # oder zonal dists, wenn vorhanden
                fleet_df=None,
            )
        else:
            total_min, by_fuel = compute_fossil_min_profiles(
                gen_df=z_gen,
                fuels_select=[s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()],
                peak_hours=args.mustrun_peak_hours,
                mode=args.fossil_mustrun_mode,
                q=float(args.fossil_mustrun_q),
            )
        nei_min_total_by_zone[z] = total_min
        nei_min_by_zone_fuel[z]  = by_fuel


    # 7) Zeitfenster
    idx_common = sorted(de_load.index.intersection(fuel_prices.index).intersection(flows.index).intersection(nei_prices.index))
    def _to_berlin(ts_str: Optional[str]):
        if ts_str is None: return None
        ts = pd.Timestamp(ts_str)
        return ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
    start = _to_berlin(args.start) or idx_common[0]
    end   = _to_berlin(args.end)   or (idx_common[-1] + pd.Timedelta(hours=1))
    idx   = [t for t in idx_common if (t >= start and t < end)]
    print(f"[INFO] Stunden im Lauf: {len(idx)} | Fenster: {start} .. {end} (exkl.)")

    # 8) Nicht-disponible in DE (nur für Checks; PSP-Gen NICHT abziehen)
    nd_cols = ["Nuclear","Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage","Biomass","Waste"]
    nd_present = [c for c in nd_cols if c in de_gen.columns]
    de_nondisp = de_gen[nd_present].sum(axis=1).reindex(de_load.index).fillna(0.0)

    # 9) Fossile Mindestprofile (DE)
    if args.mu_cost_mode == "q_vs_cost" and bool(getattr(args, "de_fossil_mustrun_from_cost", False)):
        de_min_total, de_min_by_fuel = compute_fossil_min_profiles_cost_based(
            gen_df=de_gen.reindex(de_load.index),
            price_series=nei_prices["price_DE_LU"],
            fuel_prices=fuel_prices,
            fuels_select=[s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()],
            peak_hours=args.mustrun_peak_hours,
            q=float(args.mu_cost_q),
            alpha=float(args.mu_cost_alpha),
            monthly=bool(getattr(args, "mu_cost_monthly", False)),
            use_peak_split=bool(getattr(args, "mu_cost_use_peak", False)),
            eta_source="DE",
            nei_dists=None,

            fleet_df=fleet_all,   # für η-Median je Fuel
        )
    else:
        de_min_total, de_min_by_fuel = compute_fossil_min_profiles(
            gen_df=de_gen.reindex(de_load.index),
            fuels_select=[s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()],
            peak_hours=args.mustrun_peak_hours,
            mode=args.fossil_mustrun_mode,
            q=float(args.fossil_mustrun_q),
        )

    # 10) Lignite-MU Profil (falls aktiviert)
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

    # Lignite Hilfsgrößen
    lign_mask = (ef_keys == "Braunkohle")
    units_np = np.array(units)
    lign_unit_ids = units_np[lign_mask]
    if lign_unit_ids.size > 0:
        lign_total = float(fleet_idxed.loc[lign_unit_ids, "available_mw"].sum())
    else:
        lign_total = 0.0
    lign_share = np.zeros_like(cap_base)
    if lign_total > 0 and lign_mask.any():
        idx_lign = np.where(lign_mask)[0]
        lign_share[idx_lign] = cap_base[idx_lign] / lign_total

    # 11) Nachbar-η-Parameter / Kapazitätsmaske
    nei_dists = DEFAULT_NEI_DISTS.copy()
    cap_mask = None
    if args.neighbor_fleet and Path(args.neighbor_fleet).exists():
        fleet_dists, cap_mask_from_fleet = load_neighbor_fleet(args.neighbor_fleet)
        for z, fuels in fleet_dists.items():
            nei_dists.setdefault(z, {})
            for f, d in fuels.items(): nei_dists[z][f] = d
        cap_mask = cap_mask_from_fleet
    if args.nei_eta_json and Path(args.nei_eta_json).exists():
        with open(args.nei_eta_json, "r", encoding="utf-8") as f:
            user_d = json.load(f)
        for k, v in user_d.items():
            if isinstance(v, dict) and all(isinstance(vv, dict) for vv in v.values()):
                nei_dists.setdefault(k, {}).update(v)
            else:
                nei_dists[k] = v if isinstance(v, dict) else v
    if args.neighbor_capacity and Path(args.neighbor_capacity).exists():
        dfc = pd.read_csv(args.neighbor_capacity)
        cap_mask = cap_mask or {}
        for _, r in dfc.iterrows():
            cap_mask[(str(r["zone"]).strip(), str(r["fuel"]).strip())] = float(r["capacity_mw"])

    # 12) MU-Bid Funktion vorbereiten (rolling/default)
    mu_bid_fn_global = precompute_mu_bid_fn(nei_prices, fuel_prices, args, zones)

    # 13) Hauptschleife
    results, debug_rows = [], []
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    imp_to_zone = {c: c.replace("imp_", "").replace("_", "") for c in imp_cols}

    for t in idx:
        L  = float(de_load.get(t, np.nan))
        ND = float(de_nondisp.get(t, 0.0))
        if not np.isfinite(L): continue

        # Flüsse
        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0
        pos_imp_total = 0.0
        pos_imp_by_zone = {}
        if imp_cols:
            for c in imp_cols:
                z = imp_to_zone[c]
                if z not in clusters.get(t, []):  # nur gekoppelte Zonen
                    continue
                v = float(flows.at[t, c]) if (t in flows.index and c in flows.columns and np.isfinite(flows.at[t, c])) else 0.0
                if v > 0:
                    pos_imp_total += v
                    pos_imp_by_zone[z] = pos_imp_by_zone.get(z, 0.0) + v

        import_relevant = (pos_imp_total > 1e-6) if bool(getattr(args, "coupled_import_anyflow", True)) else (net_imp > 0.0)
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan
        fp_row = fuel_prices.loc[t] if t in fuel_prices.index else None

        # --------- RL-Leiter (DE) nach gewünschter Reihenfolge ----------
        fee_cols = ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
        fee_cols = [c for c in fee_cols if c in de_gen.columns]
        FEE = float(de_gen[fee_cols].reindex([t]).sum(axis=1).iloc[0]) if fee_cols else 0.0

        waste_t = float(de_gen.get("Waste",   pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Waste"   in de_gen.columns else 0.0
        nuc_t   = float(de_gen.get("Nuclear", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Nuclear" in de_gen.columns else 0.0
        bio_raw = float(de_gen.get("Biomass", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Biomass" in de_gen.columns else 0.0

        # MU-Shares (DE)
        bio_mu_de_share = float(getattr(args, "de_biomass_mustrun_share", 0.0))
        bio_mu_de   = max(min(bio_raw, bio_raw * bio_mu_de_share), 0.0)
        bio_flex_de = max(bio_raw - bio_mu_de, 0.0)
        nuc_mu_de_share = float(getattr(args, "de_nuclear_mustrun_share", 0.0))
        nuc_mu_de = max(min(nuc_t, nuc_t * nuc_mu_de_share), 0.0)
        nuc_flex_de = max(nuc_t - nuc_mu_de, 0.0)

        # Stufen
        RL1 = max(L - FEE, 0.0)                         # nach FEE
        take_waste_mu = min(waste_t, RL1);              RL2 = max(RL1 - take_waste_mu, 0.0)
        take_nuc_mu   = min(nuc_mu_de, RL2);            RL3 = max(RL2 - take_nuc_mu, 0.0)
        take_bio_mu   = min(bio_mu_de, RL3);            RL4 = max(RL3 - take_bio_mu, 0.0)

        # Fossile MU (parametrisierte Shares)
        de_mustrun_shares = fossil_mustrun_shares_for_DE(args)
        mustrun_de_total = 0.0
        for ef_name, share in de_mustrun_shares.items():
            if share <= 0.0: continue
            mask_f = (ef_keys == ef_name)
            if not np.any(mask_f): continue
            cap_ef = float((cap_base[mask_f] * float(args.therm_avail)).sum())
            mustrun_de_total += share * cap_ef
        take_mu_foss = min(mustrun_de_total, RL4);      RL5 = max(RL4 - take_mu_foss, 0.0)

        # Flex (DE) – PSP / Reservoir
        psp_avail = float(de_gen.get("Hydro Pumped Storage", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Hydro Pumped Storage" in de_gen.columns else 0.0
        take_psp  = min(psp_avail, RL5);                RL6 = max(RL5 - take_psp, 0.0)

        de_reservoir_avail = float(de_gen.get("Hydro Water Reservoir", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Hydro Water Reservoir" in de_gen.columns else 0.0
        take_res  = min(de_reservoir_avail, RL6);       RL7 = max(RL6 - take_res, 0.0)
        # Signifikanzregel für Import-Relevanz (überschreibt anyflow)
        import_relevant_signif = (net_imp > 0.0) and (pos_imp_total >= max(0.05 * RL7, 200.0))
        import_relevant = import_relevant_signif
        # Default: Markierungen
        marginal_side  = None
        marginal_label = None
        marginal_fuel  = None
        marginal_eta   = np.nan
        marginal_srmc  = np.nan
        mef_gpkwh      = np.nan
        DEBUG_rule_psp_price = False

        # Preis-Cluster
        cluster_all = clusters.get(t, ["DE_LU"])
        coupled_neighbors = [z for z in cluster_all if z != "DE_LU"]
        coupling_active = len(coupled_neighbors) > 0

        # PSP/Reservoir ggf. direkt preissetzend
        if (RL5 > 0.0) and (psp_avail >= RL5 - 1e-6) and np.isfinite(p_de):
            marginal_side  = "DE"
            marginal_label = "DE_flex_psp_price_setting"
            marginal_fuel  = "Hydro Pumped Storage"
            marginal_eta   = np.nan
            marginal_srmc  = float(max(0.0, min(p_de, getattr(args, "psp_srmc_floor_eur_mwh", 60.0))))
            mef_gpkwh      = 0.0
            DEBUG_rule_psp_price = True

        if (RL6 > 0.0) and (de_reservoir_avail >= RL6 - 1e-6) and np.isfinite(p_de):
            marginal_side  = "DE"
            marginal_label = "DE_flex_reservoir_price_setting"
            marginal_fuel  = "Hydro Water Reservoir"
            reservoir_sp_de = reservoir_sp_map.get(("DE_LU", t), float(np.clip(p_de, 0.0, 60.0)))
            marginal_srmc   = float(np.clip(reservoir_sp_de, 0.0, getattr(args, "psp_srmc_floor_eur_mwh", 60.0)))
            marginal_eta    = np.nan
            mef_gpkwh       = 0.0

        # Residuen (für Reporting)
        residual_domestic_fossil = RL5
        residual_after_trade     = RL7 - max(net_imp, 0.0)
        residual = max(residual_after_trade, 0.0)

        # ---------- DE-Marginal (Fallback/kein Import oder Rest-Fossil) ----------
        # Kapazitätsprofil inkl. Lignite-Erzwingung
        cap_t = cap_base * float(args.therm_avail)
        lignite_mustrun_enforced = 0.0
        if args.mustrun_mode == "capacity" and (ef_keys == "Braunkohle").any() and args.mustrun_lignite_q > 0.0:
            lmask = (ef_keys == "Braunkohle")
            cap_t[lmask] = np.maximum(cap_t[lmask], cap_base[lmask] * float(args.mustrun_lignite_q))
            lignite_mustrun_enforced = float(cap_t[lmask].sum())
        elif args.mustrun_mode == "gen_quantile" and (ef_keys == "Braunkohle").any():
            lmask = (ef_keys == "Braunkohle")
            need = float(lign_profile.get(t, 0.0))
            if need > 0.0:
                total = cap_base[lmask].sum()
                target = np.zeros_like(cap_t)
                if total > 0:
                    target[lmask] = cap_base[lmask] / total * need
                cap_t = np.maximum(cap_t, target.astype(cap_t.dtype))
                lignite_mustrun_enforced = float(target[lmask].sum())

        # DE-Einheit bestimmen (falls später DE gewählt wird)
        unit_id = None; ef_dom = None; eta_dom = None; srmc_dom = None; mef_dom = np.nan
        if (residual > 0) and (t in SRMC.index):
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

        # --------- Kopplung: gepoolter Export-Stack aller gekoppelten Zonen (IMPORT-Richtung) ------
        marginal_import_label = None
        import_marg_srmc = None
        import_marg_mef  = None

        I_needed = RL7
        if (I_needed > 1e-6) and coupling_active and import_relevant and (fp_row is not None):
            stack_all = []
            for z in coupled_neighbors:
                if z not in gen_by_zone or t not in gen_by_zone[z].index:
                    continue
                gen_row = gen_by_zone[z].loc[t]
                load_z  = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
                if not np.isfinite(load_z): 
                    continue
                min_total_t = float(nei_min_total_by_zone.get(z, pd.Series(0.0)).reindex([t]).fillna(0.0).iloc[0]) if z in nei_min_total_by_zone else 0.0
                min_by_fuel_t = {f: float(s.reindex([t]).fillna(0.0).iloc[0]) for f, s in nei_min_by_zone_fuel.get(z, {}).items()}
                pz = float(nei_prices.loc[t, f"price_{z}"]) if (f"price_{z}" in nei_prices.columns and t in nei_prices.index) else np.nan

                blocks = exportable_blocks_for_zone(
                    t=t, zone=z, gen_z_row=gen_row, load_z_t=load_z,
                    fuel_prices_row=fp_row, nei_dists=nei_dists,
                    mode=args.nei_eta_mode, draws=int(args.nei_mc_draws),
                    args=args, cap_mask=cap_mask, reservoir_sp_map=reservoir_sp_map,
                    min_total_zone_t=min_total_t, min_by_fuel_zone_t=min_by_fuel_t,
                    price_zone=pz, mu_bid_getter=mu_bid_fn_global,
                )
                stack_all.extend(blocks)

            stack_all.sort(key=lambda x: (x[0], x[2]))
            I_remaining = I_needed
            marg_block = None
            for (_, fuel, srmc, mw, eta, z) in stack_all:
                if I_remaining <= 1e-6: 
                    break
                take = min(mw, I_remaining)
                I_remaining -= take
                marg_block = (fuel, srmc, eta, z)

            if (marg_block is not None) and (I_remaining <= 1e-6):
                fuel_m, srmc_m, eta_m, z_m = marg_block
                import_marg_srmc = float(srmc_m)
                if fuel_m in ("Reservoir Hydro", "Hydro Pumped Storage", "EE", "Nuclear"):
                    import_marg_mef = 0.0
                else:
                    ef_th = EF_LOOKUP_T_PER_MWH_TH.get(fuel_m, 0.30)
                    import_marg_mef  = float((ef_th / max(eta_m,1e-6)) * 1000.0)
                marginal_import_label = f"{z_m}({fuel_m})"
        # --------- Neu: Export-Richtung symmetrisch behandeln ---------
        # Bedingung: gekoppelt, relevanter Export (negativer Nettofluss) und DE hat Exportüberschuss
        if (net_imp < -1e-6) and coupling_active and (fp_row is not None):
            # 1) Zielbedarf der GEGENSEITE: wahlweise (a) Betrag des gemessenen Exports
            #    oder (b) Summe der RL7 der importierenden Nachbarn. Wir nutzen (b) als Default.
            demand_importing = 0.0
            for z in coupled_neighbors:
                if z not in gen_by_zone or t not in gen_by_zone[z].index: 
                    continue
                gen_row_z = gen_by_zone[z].loc[t]
                load_z    = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
                if not np.isfinite(load_z): 
                    continue
                min_total_t = float(nei_min_total_by_zone.get(z, pd.Series(0.0)).reindex([t]).fillna(0.0).iloc[0]) if z in nei_min_total_by_zone else 0.0
                demand_importing += residual_need_after_local_steps(
                    t, z, gen_row_z, load_z, min_total_t, args
                )

            E_needed = max(demand_importing, 0.0)
            # Fallback: nutze gemessenen Exportbetrag, falls keine RL7 bestimmbar
            if E_needed <= 1e-6:
                E_needed = abs(net_imp)

            # 2) Angebots-Stack der EXPORTSEITE = DE_LU (optional: + weitere Exportzonen, hier streng DE)
            stack_de = []
            gen_row_de = de_gen.loc[t]
            load_de   = float(de_load.reindex([t]).fillna(np.nan).iloc[0]) if 'de_load' in locals() else float(L)
            min_total_de = float(mustrun_de_total) if 'mustrun_de_total' in locals() else 0.0
            pz_de = float(nei_prices.loc[t, "price_DE_LU"]) if ("price_DE_LU" in nei_prices.columns and t in nei_prices.index) else np.nan
            blocks_de = exportable_blocks_for_zone(
                t=t, zone="DE_LU", gen_z_row=gen_row_de, load_z_t=load_de,
                fuel_prices_row=fp_row, nei_dists=nei_dists,
                mode=args.nei_eta_mode, draws=int(args.nei_mc_draws),
                args=args, cap_mask=cap_mask, reservoir_sp_map=reservoir_sp_map,
                min_total_zone_t=min_total_de, min_by_fuel_zone_t={}, price_zone=pz_de,
                mu_bid_getter=mu_bid_fn_global,
            )
            stack_de.extend(blocks_de)
            stack_de.sort(key=lambda x: (x[0], x[2]))  # gleiche Gruppenreihenfolge wie Import

            E_remaining = E_needed
            marg_block_exp = None
            for (_, fuel, srmc, mw, eta, z) in stack_de:
                if E_remaining <= 1e-6:
                    break
                take = min(mw, E_remaining)
                E_remaining -= take
                marg_block_exp = (fuel, srmc, eta, z)
            
            # --- EXPORT-KANDIDAT nur vormerken (nicht sofort setzen) ---
            export_marg_srmc   = None
            export_marg_mef    = None
            export_marg_label  = None
            export_marg_fuel   = None
            export_marg_eta    = np.nan
            
            if (marg_block_exp is not None) and (E_remaining <= 1e-6):
                fuel_m, srmc_m, eta_m, z_m = marg_block_exp
                export_marg_srmc  = float(srmc_m)
                export_marg_fuel  = fuel_m
                export_marg_eta   = eta_m
                if fuel_m in ("Reservoir Hydro","Hydro Pumped Storage","EE","Nuclear"):
                    export_marg_mef = 0.0
                else:
                    ef_th = EF_LOOKUP_T_PER_MWH_TH.get(fuel_m, 0.30)
                    export_marg_mef = float((ef_th / max(eta_m,1e-6)) * 1000.0)
                export_marg_label = f"DE_LU({fuel_m})"
            


        # ------------- Nicht gekoppelt / Restfall in DE -------------
        if (I_needed > 1e-6) and ( (not coupling_active) or (import_marg_srmc is None) ):
            # Rest Waste → 1 €
            rest_after_waste = RL1 - take_waste_mu
            if (rest_after_waste <= 1e-6) and (waste_t > 0.0) and (RL1 > 0.0):
                marginal_side  = "DE"; marginal_label = "RestWaste_only"
                marginal_fuel  = "Waste"; marginal_eta = np.nan
                marginal_srmc  = float(getattr(args,"waste_srmc_eur_mwh",1.0))
                mef_gpkwh      = float(getattr(args,"waste_mef_gpkwh",0.0))
            else:
                # Rest Biomasse FLEX → SRMC konservativ
                rest_after_bio = RL2 - take_bio_mu  # nach Waste+Nuclear
                if (rest_after_bio <= 1e-6) and (bio_flex_de > 0.0) and (RL2 > 0.0):
                    marginal_side  = "DE"; marginal_label = "RestBiomass_only"
                    marginal_fuel  = "Biomass"; marginal_eta = np.nan
                    marginal_srmc  = float(getattr(args,"biomass_srmc_eur_mwh",35.0))
                    mef_gpkwh      = float(getattr(args,"biomass_mef_gpkwh",0.0))
                else:
                    # sonst entscheidet DE-Flex/Fossil (siehe Einheitensuche oben)
                    pass

        # --------------- Seitenwahl / Preisanker ----------------
        if (import_marg_srmc is not None) and coupling_active:
            # Wenn Netto-Export: Export-Kandidat als DE-Alternative benutzen
            de_candidate_srmc = marginal_srmc
            de_candidate_label = marginal_label
            de_candidate_fuel  = marginal_fuel
            de_candidate_eta   = marginal_eta
            de_candidate_mef   = mef_gpkwh
            
            if (net_imp < -1e-6) and (export_marg_srmc is not None):
                de_candidate_srmc  = export_marg_srmc
                de_candidate_label = "DE_export_pool_price_setting"
                de_candidate_fuel  = export_marg_fuel
                de_candidate_eta   = export_marg_eta
                de_candidate_mef   = export_marg_mef
            # DE-Kandidat auf Basis der Einheitensuche setzen, falls noch nicht gesetzt
            if marginal_label is None and unit_id is not None:
                marginal_side  = "DE"
                marginal_label = unit_id
                marginal_fuel  = ef_dom
                marginal_eta   = eta_dom if eta_dom is not None else np.nan
                marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan
                mef_gpkwh      = mef_dom
            choose_side = None
            if args.price_anchor in ("closest","threshold") and np.isfinite(p_de):
                cand = []
                if np.isfinite(de_candidate_srmc): 
                    cand.append(("DE", abs(de_candidate_srmc - p_de)))
                cand.append(("IMPORT", abs(import_marg_srmc - p_de)))
                
                if args.price_anchor == "closest":
                    choose_side = min(cand, key=lambda x: x[1])[0]
                elif args.price_anchor == "threshold":
                    valid = [c for c in cand if c[1] <= float(args.price_tol)]
                    choose_side = min(valid, key=lambda x: x[1])[0] if valid else ("IMPORT" if (not np.isfinite(de_candidate_srmc) or import_marg_srmc <= de_candidate_srmc) else "DE")
                else:
                    choose_side = "IMPORT" if (not np.isfinite(de_candidate_srmc) or import_marg_srmc <= de_candidate_srmc) else "DE"
                
                # WICHTIG: Export-Preissetzung nur bei Preisgleichheit zulassen
                if choose_side == "DE":
                    # Nur wenn Export-Kandidat existiert und preisgleich (oder besser) ist
                    if (net_imp < -1e-6) and (export_marg_srmc is not None):
                        if abs(p_de - export_marg_srmc) <= float(args.epsilon) and \
                           (abs(import_marg_srmc - p_de) >= abs(export_marg_srmc - p_de) - 1e-9):
                            marginal_side  = "DE"
                            marginal_label = de_candidate_label
                            marginal_fuel  = de_candidate_fuel
                            marginal_eta   = de_candidate_eta
                            marginal_srmc  = float(de_candidate_srmc)
                            mef_gpkwh      = float(de_candidate_mef)
                        else:
                            # Export erfüllt die Bedingung nicht → setze IMPORT
                            marginal_side  = "IMPORT"
                            marginal_label = marginal_import_label or "import_stack"
                            marginal_fuel  = (marginal_import_label.split("(")[-1].rstrip(")")) if marginal_import_label else "mix"
                            marginal_eta   = np.nan
                            marginal_srmc  = float(import_marg_srmc)
                            mef_gpkwh      = float(import_marg_mef)
                    else:
                        # Normaler DE-Kandidat (z. B. inländische Einheit)
                        marginal_side  = "DE"
                        marginal_label = de_candidate_label
                        marginal_fuel  = de_candidate_fuel
                        marginal_eta   = de_candidate_eta
                        marginal_srmc  = float(de_candidate_srmc)
                        mef_gpkwh      = float(de_candidate_mef)
                else:
                    marginal_side  = "IMPORT"
                    marginal_label = marginal_import_label or "import_stack"
                    marginal_fuel  = (marginal_import_label.split("(")[-1].rstrip(")")) if marginal_import_label else "mix"
                    marginal_eta   = np.nan
                    marginal_srmc  = float(import_marg_srmc)
                    mef_gpkwh      = float(import_marg_mef)
    
        

        # --------- Peaker-Override (nur wenn DE plausibel marginal ist) ----------
        if getattr(args, "peak_switch", False):
            thr1, thr2 = _parse_two_floats(getattr(args, "peak_price_thresholds", "180,260"))
            # Voraussetzungen: Preis hoch, Restlast > 0, DE ist gewählt, SRMC preisnah.
            if np.isfinite(p_de) and (RL7 > 0.0) and (marginal_side == "DE") and np.isfinite(marginal_srmc):
                if abs(marginal_srmc - p_de) <= float(args.epsilon) and (p_de >= thr1):
                    co2 = float(fuel_prices.loc[t, "co2_eur_t"])
                    gas_th = float(fuel_prices.loc[t, "gas_eur_mwh_th"])
                    oil_th = float(fuel_prices.loc[t, "oil_eur_mwh_th"])
                    ocgt_eta = float(getattr(args, "peak_eta_ocgt", 0.36))
                    oil_eta  = float(getattr(args, "peak_eta_oil", 0.33))
                    ocgt_srmc = (gas_th + co2 * EF_LOOKUP_T_PER_MWH_TH["Erdgas"]) / max(ocgt_eta, 1e-6)
                    oil_srmc  = (oil_th + co2 * EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"]) / max(oil_eta,  1e-6)
                    # Nur ersetzen, wenn die Peaker-SRMC selbst nah am Preis liegen
                    if (p_de >= thr2) and np.isfinite(oil_srmc) and (abs(oil_srmc - p_de) <= float(args.epsilon)):
                        marginal_label = "DE_peaker_override_oil"
                        marginal_fuel  = "Heizöl schwer"
                        marginal_eta   = oil_eta
                        marginal_srmc  = float(oil_srmc)
                        mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"] / max(oil_eta,1e-6)) * 1000.0
                    elif np.isfinite(ocgt_srmc) and (abs(ocgt_srmc - p_de) <= float(args.epsilon)):
                        marginal_label = "DE_peaker_override_ocgt"
                        marginal_fuel  = "Erdgas"
                        marginal_eta   = ocgt_eta
                        marginal_srmc  = float(ocgt_srmc)
                        mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["Erdgas"] / max(ocgt_eta,1e-6))  


        # Mustrun-only Floor (falls alles andere weg und Preis existiert)
        mustrun_only = (RL5 <= 1e-6) and (net_imp <= 0.0) and (L > FEE + 1e-6)
        if mustrun_only and np.isfinite(p_de):
            marginal_side  = "DE"
            marginal_label = "mustrun_floor_price_setting"
            marginal_fuel  = "MustrunMix"
            marginal_eta   = np.nan
            marginal_srmc  = float(np.clip(p_de, 0.0, 60.0))
            mef_gpkwh      = 0.0

        # Falls immer noch nichts gesetzt und DE-Einheit vorhanden → DE wählen
        if (marginal_label is None) and (unit_id is not None):
            marginal_side  = "DE"
            marginal_label = unit_id
            marginal_fuel  = ef_dom
            marginal_eta   = eta_dom if eta_dom is not None else np.nan
            marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan
            mef_gpkwh      = mef_dom

        # ---- Append (einzige Stelle) ----
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
            "flag_psp_price_setting": bool(DEBUG_rule_psp_price),
            "FEE_MW": FEE,
            "ND_EXTRA_MW": (nuc_t + bio_raw + waste_t),
            "RL_after_FEE_MW": RL1,
            "RL_after_MU_WASTE_MW": RL2,
            "RL_after_MU_NUC_MW": RL3,
            "RL_after_MU_BIO_MW": RL4,
            "RL_after_FOSSIL_MU_MW": RL5,
            "RL_after_PSP_MW": RL6,
            "RL_after_RES_MW": RL7,
            "BIO_MW": bio_raw,
            "WASTE_MW": waste_t,
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
            "DEBUG_rule_psp_price_setting": DEBUG_rule_psp_price,
        })

    # 14) Outputs schreiben
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    (outdir / "analysis").mkdir(exist_ok=True, parents=True)
    df_res.to_csv(outdir / "mef_track_c_2024.csv", index=True)
    df_dbg.to_csv(outdir / "_debug_hourly.csv", index=True)
    print(f"[OK] geschrieben: {outdir/'mef_track_c_2024.csv'}")
    print(f"[OK] Debug:       {outdir/'_debug_hourly.csv'}")

    # 15) Validation erstellen
    df_val, df_sum = validate_run(
        df_res=df_res,
        df_dbg=df_dbg,
        flows=flows,
        prices=nei_prices,
        epsilon_price=float(args.epsilon),
        price_anchor_mode=str(args.price_anchor),
    )
    write_validation_report(outdir, df_val, df_sum)

    # 16) Zusätzliche Residuallast-Plots (wie zuvor)
    plots_dir = outdir / "analysis" / "plots_residuallast"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df_plot = pd.DataFrame(index=df_res.index)
    df_plot["price"] = df_res["price_DE"]
    df_plot["residual_domestic_fossil"] = df_res["residual_domestic_fossil_MW"]
    df_plot["residual_after_trade"] = df_res["residual_after_trade_MW"]

    if "FEE_MW" in df_res.columns and "ND_EXTRA_MW" in df_res.columns:
        df_plot["FEE"] = df_res["FEE_MW"]
        df_plot["ND_EXTRA"] = df_res["ND_EXTRA_MW"]
    else:
        print("[WARNUNG] FEE/ND_EXTRA nicht gespeichert → diese Kurven fehlen im Plot.")

    for month, df_m in df_plot.groupby(df_plot.index.month):
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df_m.index, df_m["residual_domestic_fossil"], label="Residuallast nach Mustrun", color="#00374B", linewidth=1.2)
        ax1.plot(df_m.index, df_m["residual_after_trade"], label="Residuallast nach Handel", color="#9B0028", linewidth=1.2)
        if "FEE" in df_m.columns:
            ax1.plot(df_m.index, df_m["FEE"], label="FEE", color="#518E9F", linewidth=1.0, alpha=0.7)
        if "ND_EXTRA" in df_m.columns:
            ax1.plot(df_m.index, df_m["ND_EXTRA"], label="Non-Disp", color="#A08268", linewidth=1.0, alpha=0.7)
        ax1.set_ylabel("Leistung [MW]")
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df_m.index, df_m["price"], label="Preis (DE)", color="#E6F0F7", linewidth=1.0, linestyle="--")
        ax2.set_ylabel("Preis [€/MWh]")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(plots_dir / f"residuallast_month_{month:02d}.png", dpi=160)
        plt.close(fig)

    print(f"[OK] Residuallast-Plots in {plots_dir} geschrieben.")

    # 17) Marginal-Share-Plots & Validierungsplots
    shares_dir = outdir / "analysis" / "plots_marginal_shares"
    shares_dir.mkdir(parents=True, exist_ok=True)

    df_ms = df_res.copy()
    df_ms["month"] = df_ms.index.month
    df_ms["hour"]  = df_ms.index.hour

    FUEL_COLOR = {
        "EE":                  "#007D55",
        "Reservoir Hydro":     "#00374B",
        "Hydro Pumped Storage":"#E6F0F7",
        "Erdgas":              "#518E9F",
        "Steinkohle":          "#A08269",
        "Braunkohle":          "#5F5E5E",
        "Heizöl schwer":       "#DC0C23",
        "Heizöl leicht / Diesel":"#F5644B",
        "Biomass":             "#D2B900",
        "Waste":               "#F3F0E7",
        "Nuclear":             "#9B0028",
        "MustrunMix":          "#B08B6E",
        "NonDisp":             "#518E9F",
        "mix":                 "#E6F0F7",
        "Other":               "#DDDDDD",
    }
    def color_for_share(key): return FUEL_COLOR.get(key, "#BBBBBB")

    OTHER_NAME = "Other"; TOP_K = 8
    for m, sub in df_ms.groupby("month"):
        tab = (
            sub.pivot_table(index="hour", columns="marginal_fuel",
                            values="marginal_srmc_eur_per_mwh",
                            aggfunc="count", fill_value=0)
            .sort_index()
            .reindex(range(24), fill_value=0)
        )
        total = tab.sum(axis=1).replace(0, np.nan)
        col_order = tab.sum(axis=0).sort_values(ascending=False).index.tolist()
        keep_cols = col_order[:TOP_K]
        if len(col_order) > TOP_K:
            tab[OTHER_NAME] = tab[[c for c in col_order[TOP_K:]]].sum(axis=1)
            cols_final = keep_cols + [OTHER_NAME]
        else:
            cols_final = keep_cols
        tab = tab[cols_final]
        share = 100.0 * tab.div(total, axis=0).fillna(0.0)

        fig, ax = plt.subplots(figsize=(12, 4.8))
        x = np.arange(24); bottom = np.zeros_like(x, dtype=float)
        for c in cols_final:
            y = share[c].values.astype(float)
            ax.bar(x, y, bottom=bottom, label=c, width=0.9, edgecolor="none", color=color_for_share(c))
            bottom += y
        ax.set_title(f"Anteil marginaler Technologien je Stunde – Monat {m:02d}")
        ax.set_xlabel("Stunde"); ax.set_ylabel("Anteil [%]"); ax.set_xticks(range(24)); ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        leg = ax.legend(ncol=2, fontsize=8, loc="upper left")
        fig.tight_layout(); fig.savefig(shares_dir / f"marginal_shares_hourly_month_{m:02d}.png", dpi=160); plt.close(fig)

    print(f"[OK] Marginal-Share-Plots in {shares_dir} geschrieben.")

    # Validierungsplots (erweiterte)
   # make_validation_plots(
   #     outdir, df_res, df_dbg, df_val, nei_prices,
   #     de_gen=de_gen, de_min_total=de_min_total, de_load=de_load, flows=flows
   # )

    # 18) Gefilterte Korrelation + Offender
    try:
        corr_filt = _filtered_corr_and_offenders(
            outdir=outdir, df_res=df_res, df_dbg=df_dbg, df_val=df_val,
            fuel_prices=fuel_prices, args=args
        )
        print(f"[VALIDATION] Filtered corr (Preis vs. SRMC) = {corr_filt:.4f}")
        print(f"[VALIDATION] Details: analysis/_corr_offenders.csv und analysis/_corr_offenders_summary.txt")
    except Exception as e:
        print("[VALIDATION] Hinweis – gefilterte Korrelation/Offender konnte nicht berechnet werden:", e)



if __name__ == "__main__":
    main(build_parser().parse_args())
