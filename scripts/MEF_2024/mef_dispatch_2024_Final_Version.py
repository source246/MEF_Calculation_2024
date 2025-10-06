#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track C â€“ Dispatch-Backcast (DE/LU 2024) mit:
- DE: imputierte Wirkungsgrade (eta_col) â†’ SRMC pro Einheit
- Nachbarn: Î·-Spannen/MC je Fuel + KapazitÃ¤tsmaske (optional)
- Import-Fall bei Preiskopplung: Export-Stack der gekoppelten Zonen
  (Reservoir-Hydro als einzige regelbare Hydro, RoR/Pumpspeicher nicht-disponibel)
- Marginaler Block des gemeinsamen Stacks bestimmt Fuel & MEF
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

from modules.io_utils import (
    load_flows,
    load_fleet,
    load_fuel_prices,
    load_neighbor_fleet,
    load_neighbor_gen,
    load_neighbor_load,
    load_neighbor_prices,
    map_unit_to_plant_type,
    prepare_flows_for_mode,
    validate_hourly_index,
)
from modules.mustrun import (
    DEFAULT_NEI_DISTS,
    EF_LOOKUP_T_PER_MWH_TH,
    compute_fossil_min_profiles,
    compute_fossil_min_profiles_cost_based,
    fossil_mustrun_shares_for_DE,
    fossil_mustrun_shares_for_NEI,
    price_based_lignite_mustrun_profile,
    price_based_oil_mustrun_profile,
    truncated_normal,
)
from modules.plots import (
    create_load_coverage_chart,
    generate_enhanced_plots,
    make_validation_plots,
    _plot_mef_hourly_mean_per_month,
)
from modules.validation import (
    _filtered_corr_and_offenders,
    validate_run,
    write_validation_report,
    write_negative_price_gen_summary,
    enhanced_data_time_validation,
    enhanced_price_srmc_validation,
    enhanced_transformation_validation,
    enhanced_dr_optimization_validation,
    run_full_enhanced_validation,
)

from collections import defaultdict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
TZ = "Europe/Berlin"

# ================================= UNIFIED RL LADDER ===================================

def compute_residual_load_ladder(gen_z_row, load_z_t, min_by_fuel_zone_t, min_total_zone_t, args=None, zone: Optional[str] = None):
    """
    Einheitliche RL-Leiter Berechnung: RL0 (Last) → RL8 (nach allen Abzügen)
    Verhindert mehrfache inkonsistente RL-Berechnungen im Code.
    
    WICHTIG: Diese Funktion wird NUR EINMAL pro Zeitstunde aufgerufen!
    Keine mehrfachen inkonsistenten RL-Berechnungen mehr.
    
    Args:
        gen_z_row: Generation data for zone
        load_z_t: Load for timestep
        min_by_fuel_zone_t: Minimum generation by fuel
        min_total_zone_t: Total minimum generation
        args: Command line arguments for parameters
    
    Returns:
        dict: {
            'RL0': load_z_t,           # Ausgangslast
            'RL1': RL1,                # nach FEE
            'RL2': RL2,                # nach Müll MU
            'RL3': RL3,                # nach Nuclear MU
            'RL4': RL4,                # nach Bio MU
            'RL5': RL5,                # nach Oil MU
            'RL6': RL6,                # nach Fossil MU
            'RL7': RL7,                # nach PSP
            'RL8': RL8,                # nach Reservoir
            'takes': {                 # Abzüge pro Stufe für Debugging
                'fee': take_fee,
                'waste_mu': take_waste_mu,
                'nuc_mu': take_nuc_mu,
                'bio_mu': take_bio_mu,
                'oil_mu': take_oil_mu,
                'mu_foss': take_mu_foss,
                'psp': take_psp,
                'res': take_res
            }
        }
    """
    # EE-Abzug (FEE)
    fee_cols = ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    fee_cols = [c for c in fee_cols if c in gen_z_row.index]
    FEE = float(gen_z_row[fee_cols].sum()) if fee_cols else 0.0

    RL0 = load_z_t

    # Option: domestic EE after must-run (konsequent für DE_LU)
    ee_after_mustrun = False
    try:
        ee_after_mustrun = bool(getattr(args, 'domestic_ee_after_mustrun', False)) and (str(zone or '').upper() in ("DE_LU", "DE"))
    except Exception:
        ee_after_mustrun = False

    if ee_after_mustrun:
        # Apply Must-Run first from RL0, then subtract EE from the remaining
        waste = float(min_by_fuel_zone_t.get("Müll (nicht biogen)", 0.0)) if min_by_fuel_zone_t else 0.0
        take_waste_mu = min(waste, RL0)
        RL2 = max(RL0 - take_waste_mu, 0.0)

        nuc_mu = float(min_by_fuel_zone_t.get("Kernenergie", 0.0)) if min_by_fuel_zone_t else 0.0
        take_nuc_mu = min(nuc_mu, RL2)
        RL3 = max(RL2 - take_nuc_mu, 0.0)

        bio_mu = float(min_by_fuel_zone_t.get("Biomasse", 0.0)) if min_by_fuel_zone_t else 0.0
        take_bio_mu = min(bio_mu, RL3)
        RL4 = max(RL3 - take_bio_mu, 0.0)

        oil_mu_total = 0.0
        if min_by_fuel_zone_t:
            oil_mu_total = float(get_fuel_value_robust(min_by_fuel_zone_t, "Heizöl schwer", 0.0) + 
                                get_fuel_value_robust(min_by_fuel_zone_t, "Heizöl leicht / Diesel", 0.0) + 
                                get_fuel_value_robust(min_by_fuel_zone_t, "Fossil Oil", 0.0))
        take_oil_mu = min(oil_mu_total, RL4)
        RL5 = max(RL4 - take_oil_mu, 0.0)

        mu_foss_total = max(float(min_total_zone_t) - oil_mu_total, 0.0)
        take_mu_foss = min(mu_foss_total, RL5)
        RL6 = max(RL5 - take_mu_foss, 0.0)

        # Now subtract EE from the remaining RL6
        take_fee = min(FEE, RL6)
        # MU-first: after removing MU, EE is taken from the remaining RL6.
        # RL1 should reflect the residual after MU and EE, i.e. RL6 - take_fee.
        RL1 = max(RL6 - take_fee, 0.0)
        # NOTE: RL7/RL8 (after PSP/Reservoir) are computed later in the function
        # once flexible units (PSP/Reservoir) are applied. Do not set them here to
        # avoid double-assignment and confusion.
        # Note: 'takes' will report fee taken from RL6 (take_fee)
    else:
        # Default: EE before MU (existing behaviour)
        take_fee = min(FEE, RL0)
        RL1 = max(RL0 - take_fee, 0.0)

        # Must-Run Stufen
        waste = float(min_by_fuel_zone_t.get("Müll (nicht biogen)", 0.0)) if min_by_fuel_zone_t else 0.0
        take_waste_mu = min(waste, RL1)
        RL2 = max(RL1 - take_waste_mu, 0.0)

        nuc_mu = float(min_by_fuel_zone_t.get("Kernenergie", 0.0)) if min_by_fuel_zone_t else 0.0
        take_nuc_mu = min(nuc_mu, RL2)
        RL3 = max(RL2 - take_nuc_mu, 0.0)

        bio_mu = float(min_by_fuel_zone_t.get("Biomasse", 0.0)) if min_by_fuel_zone_t else 0.0
        take_bio_mu = min(bio_mu, RL3)
        RL4 = max(RL3 - take_bio_mu, 0.0)

        oil_mu_total = 0.0
        if min_by_fuel_zone_t:
            oil_mu_total = float(get_fuel_value_robust(min_by_fuel_zone_t, "Heizöl schwer", 0.0) + 
                                get_fuel_value_robust(min_by_fuel_zone_t, "Heizöl leicht / Diesel", 0.0) + 
                                get_fuel_value_robust(min_by_fuel_zone_t, "Fossil Oil", 0.0))
        take_oil_mu = min(oil_mu_total, RL4)
        RL5 = max(RL4 - take_oil_mu, 0.0)

        mu_foss_total = max(float(min_total_zone_t) - oil_mu_total, 0.0)
        take_mu_foss = min(mu_foss_total, RL5)
        RL6 = max(RL5 - take_mu_foss, 0.0)
    
    # Flexible Erzeugung
    mw_psp = gen_z_row.get("Hydro Pumped Storage", 0.0) if "Hydro Pumped Storage" in gen_z_row.index else 0.0
    # coerce NaN -> 0.0 to avoid NaN propagation
    if pd.isna(mw_psp):
        mw_psp = 0.0
    mw_psp = float(mw_psp)
    mw_psp = max(mw_psp, float(getattr(args, "psp_min_avail_mw", 0.0))) if args is not None else mw_psp
    take_psp = min(mw_psp, RL6)
    RL7 = max(RL6 - take_psp, 0.0)
    
    mw_res = gen_z_row.get("Hydro Water Reservoir", 0.0) if "Hydro Water Reservoir" in gen_z_row.index else 0.0
    if pd.isna(mw_res):
        mw_res = 0.0
    mw_res = float(mw_res)
    mw_res = max(mw_res, float(getattr(args, "reservoir_min_avail_mw", 0.0))) if args is not None else mw_res
    take_res = min(mw_res, RL7)
    RL8 = max(RL7 - take_res, 0.0)
    
    # Safety assertion: final residual must not be significantly negative
    # Provide a clearer error when RL8 is NaN so we can debug upstream inputs
    if np.isnan(RL8):
        # include compact context to help track down source of NaNs
        raise AssertionError(
            f"RL8 is NaN for zone={zone!r}, RL0={RL0}, FEE={FEE}, "
            f"min_total_zone_t={min_total_zone_t}, min_by_fuel_zone_t={min_by_fuel_zone_t}"
        )
    try:
        assert RL8 >= -1e-6, f"Negative residual ({RL8})"
    except AssertionError:
        # Raise with context to aid debugging
        raise

    return {
        'RL0': RL0, 'RL1': RL1, 'RL2': RL2, 'RL3': RL3, 'RL4': RL4, 
        'RL5': RL5, 'RL6': RL6, 'RL7': RL7, 'RL8': RL8,
        'domestic_ee_after_mustrun_applied': ee_after_mustrun,
        'takes': {
            'fee': take_fee, 'waste_mu': take_waste_mu, 'nuc_mu': take_nuc_mu,
            'bio_mu': take_bio_mu, 'oil_mu': take_oil_mu, 'mu_foss': take_mu_foss,
            'psp': take_psp, 'res': take_res
        },
        'totals': {
            'oil_mu_total': oil_mu_total,
            'mu_foss_total': mu_foss_total
        }
    }

def determine_import_export_treatment(price_de, price_neighbor, coupling_active, args):
    """
    Bestimmt wie Import/Export behandelt werden soll basierend auf Preiskopplung.
    
    Für preis-gekoppelte Importe (p≈DE):
    - Als Angebotsblöcke in den Preis-Stack aufnehmen (bis zum gemessenen Flow/RAM)
    - NICHT zusätzlich von der Nachfrage abziehen
    
    Für nicht-gekoppelte Importe (p≠DE):
    - Als Nachfrage-Offset behandeln wenn Nachbarzone günstiger (von der Restlast abziehen)
    - NICHT in den Preis-Stack als Angebot aufnehmen
    
    Returns:
        dict: {
            'treatment': 'coupled_supply' | 'uncoupled_demand_offset' | 'no_import',
            'reason': str  # Begründung für Debugging
        }
    """
    eps = float(getattr(args, 'epsilon', 0.01)) if args else 0.01
    
    if coupling_active:
        if abs(price_neighbor - price_de) <= eps:
            return {
                'treatment': 'coupled_supply',
                'reason': f'Price-coupled import: P_DE={price_de:.2f} ≈ P_NEI={price_neighbor:.2f} (Δ={abs(price_neighbor-price_de):.3f} ≤ ε={eps})'
            }
    
    # Nicht gekoppelt - prüfe ob Import sinnvoll
    if price_neighbor < price_de:
        return {
            'treatment': 'uncoupled_demand_offset', 
            'reason': f'Non-coupled cheap import: P_NEI={price_neighbor:.2f} < P_DE={price_de:.2f}'
        }
    
    return {
        'treatment': 'no_import',
        'reason': f'No beneficial import: P_NEI={price_neighbor:.2f} ≥ P_DE={price_de:.2f}, coupled={coupling_active}'
    }

# =====================================================================================

# BBH Farbpalette


PRICE_COLS = [
    "gas_eur_mwh_th",
    "coal_eur_mwh_th",
    "lignite_eur_mwh_th",
    "oil_eur_mwh_th",
    "co2_eur_t",
]

FOSSIL_TECH_TO_FUEL: Dict[str, Tuple[str, str]] = {
    "Fossil Gas": ("gas", "Erdgas"),
    "Fossil Hard coal": ("coal", "Steinkohle"),
    "Fossil Brown coal/Lignite": ("lignite", "Braunkohle"),
    "Fossil Oil": ("oil", "Heizöl schwer"),
    "Nuclear": ("nuclear", "Kernenergie"),
}

# Fuel key normalization for encoding issues
FUEL_ALIASES = {
    # Umlaut encoding variants
    "HeizÃ¶l schwer": "Heizöl schwer",
    "HeizÃ¶l leicht / Diesel": "Heizöl leicht / Diesel",
}

def normalize_fuel_key(fuel_key: str) -> str:
    """Normalize fuel keys to handle encoding issues."""
    return FUEL_ALIASES.get(fuel_key, fuel_key)

def get_fuel_value_robust(fuel_dict: dict, key: str, default=0.0):
    """Get fuel value with alias fallback for encoding issues."""
    if key in fuel_dict:
        return fuel_dict[key]
    normalized = normalize_fuel_key(key)
    if normalized in fuel_dict:
        return fuel_dict[normalized]
    # Try reverse lookup
    for alias, canonical in FUEL_ALIASES.items():
        if canonical == key and alias in fuel_dict:
            return fuel_dict[alias]
    return default


def _zone_alias_candidates(zone: str) -> set[str]:
    base = str(zone or '').strip().upper()
    if not base:
        return set()
    candidates = {base, base.replace('-', '_'), base.replace('_', '-'), base.replace('-', '').replace('_', '')}
    compact = base.replace('-', '').replace('_', '')
    candidates.add(re.sub(r'([A-Z]+)(\d+)$', r'\1_\2', compact))
    candidates.add(re.sub(r'([A-Z]+)(\d+)$', r'\1-\2', compact))
    candidates.add(re.sub(r'([A-Z]+)(\d+)$', r'\1\2', compact))
    return {c for c in candidates if c}

def _build_zone_alias_map(zones: Iterable[str]) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for z in zones:
        for cand in _zone_alias_candidates(z):
            alias_map.setdefault(cand, z)
    return alias_map

def _resolve_zone_name(zone: str, alias_map: Dict[str, str]) -> str:
    for cand in _zone_alias_candidates(zone):
        if cand in alias_map:
            return alias_map[cand]
    return str(zone or '').strip()

def _merge_ambiv_masks(raw_masks: Dict[str, pd.Series], alias_map: Dict[str, str]) -> Dict[str, pd.Series]:
    merged: Dict[str, pd.Series] = {}
    for raw, series in raw_masks.items():
        zone = _resolve_zone_name(raw, alias_map)
        series_bool = series.astype(bool)
        if zone in merged:
            combined_index = merged[zone].index.union(series_bool.index)
            merged[zone] = merged[zone].reindex(combined_index, fill_value=False) | series_bool.reindex(combined_index, fill_value=False)
        else:
            merged[zone] = series_bool
    return merged


def safe_float(val):
    """Convert German decimal strings (comma) to float."""
    if isinstance(val, str):
        val = val.strip().replace(' ', '')
        if not val:
            return float('nan')
        return float(val.replace(',', '.'))
    return float(val)


# ----------------------------- PSP WATER VALUE -------------------------------
def psp_water_value(price_series, t, window_h=48, rte=0.78, floor=60.0):
    """
    PSP Water Value Calculation based on rolling pump price baseline.
    
    Args:
        price_series: Price time series (pandas Series)
        t: Current timestamp
        window_h: Rolling window in hours for baseline calculation
        rte: Round-trip efficiency (0.7-0.85)
        floor: Minimum water value (EUR/MWh)
    
    Returns:
        Water value (EUR/MWh) representing opportunity cost of stored energy
    """
    # Baseline = 30% quantile of last 'window_h' prices (robust against outliers)
    hist = price_series.loc[:t].tail(window_h)
    if hist.empty:
        return floor
    
    base = hist.quantile(0.30)
    # Opportunity cost: "cheap pumped" â†’ discharge cost ~ base / rte
    w = max(floor, base / max(rte, 1e-6))
    return float(w)

def psp_is_discharging(t, de_gen, threshold_mw=10.0):
    """
    Check if PSP is discharging at timestamp t.
    
    Args:
        t: Current timestamp
        de_gen: DE generation data DataFrame
        threshold_mw: Minimum generation to consider as discharging
    
    Returns:
        bool: True if PSP is discharging
    """
    if "Hydro Pumped Storage" not in de_gen.columns:
        return False
    
    psp_gen = float(de_gen.get("Hydro Pumped Storage", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0])
    return psp_gen > threshold_mw

# ----------------------------- Preflight NaN checks --------------------------------
def preflight_check_and_fill_nans(inputs: dict, auto_fill: bool = True):
    """
    Inspect key input DataFrames / dicts for NaNs and optionally fill harmless defaults.

    inputs: mapping with keys like 'neighbor_gen', 'neighbor_load', 'fleet', 'fuel_prices'
    auto_fill: if True, fill NaNs for known harmless columns (hydro PSP/reservoir, FEE cols)

    Returns a dict with counts and modifications applied.
    """
    report = {'checked': {}, 'filled': {}}

    # neighbor_gen: expected as dict zone->DataFrame or a DataFrame
    nei_gen = inputs.get('neighbor_gen')
    if isinstance(nei_gen, dict):
        for zone, df in nei_gen.items():
            if not hasattr(df, 'isna'):
                continue
            na_counts = df.isna().sum().to_dict()
            report['checked'][f'neighbor_gen:{zone}'] = na_counts
            if auto_fill:
                for col in ('Hydro Pumped Storage', 'Hydro Water Reservoir'):
                    if col in df.columns and df[col].isna().any():
                        # record number before fill
                        cnt = int(df[col].isna().sum())
                        df[col].fillna(0.0, inplace=True)
                        report['filled'].setdefault(f'neighbor_gen:{zone}', {})[col] = cnt
                inputs['neighbor_gen'][zone] = df
    elif hasattr(nei_gen, 'isna'):
        na_counts = nei_gen.isna().sum().to_dict()
        report['checked']['neighbor_gen'] = na_counts
        if auto_fill:
            for col in ('Hydro Pumped Storage', 'Hydro Water Reservoir'):
                if col in nei_gen.columns and nei_gen[col].isna().any():
                    cnt = int(nei_gen[col].isna().sum())
                    nei_gen[col].fillna(0.0, inplace=True)
                    report['filled'].setdefault('neighbor_gen', {})[col] = cnt
            inputs['neighbor_gen'] = nei_gen

    # neighbor_load
    nei_load = inputs.get('neighbor_load')
    if isinstance(nei_load, dict):
        for zone, df in nei_load.items():
            if not hasattr(df, 'isna'):
                continue
            na_counts = df.isna().sum().to_dict()
            report['checked'][f'neighbor_load:{zone}'] = na_counts
    elif hasattr(nei_load, 'isna'):
        report['checked']['neighbor_load'] = nei_load.isna().sum().to_dict()

    # fuel_prices
    fp = inputs.get('fuel_prices')
    if isinstance(fp, pd.DataFrame):
        report['checked']['fuel_prices'] = fp.isna().sum().to_dict()

    # fleet
    fleet = inputs.get('fleet')
    if hasattr(fleet, 'isna'):
        try:
            report['checked']['fleet'] = fleet.isna().sum().to_dict()
        except Exception:
            pass

    return report

# ----------------------------- CLI -------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track C v2 â€“ MEF Backcast mit Export-Stack-Logik")
    # DE-Fleet + Preise/Flows
    p.add_argument("--fleet", required=True, help="CSV: DE-Fleet (z. B. Kraftwerke_eff_binned.csv)")
    p.add_argument("--eta_col", default="Imputed_Effizienz_binned", help="Spalte mit imputierter Effizienz")
    p.add_argument("--fuel_prices", required=True, help="CSV: prices_2024.csv (EUR/MWh_th, EUA EUR/t)")
    p.add_argument("--flows", required=True, help="CSV: flows_scheduled_DE_LU_2024_net.csv")
    p.add_argument("--flows_physical", default=None,
                   help="CSV: flows_actual_DE_LU_2024_net.csv (physische Fl?sse, optional)")
    p.add_argument("--flow_mode", choices=["scheduled", "physical", "hybrid"],
                   default="scheduled",
                   help="Welche Fl?sse f?r das Import-Gate genutzt werden (Standard: scheduled).")
    p.add_argument("--flow_hybrid_ambiv_threshold", type=float, default=0.40,
                   help="Schwelle (Anteil an physischer Menge) f?r Ambivalenz-Tagging in Hybrid-Flows.")
    p.add_argument("--start", default=None)
    p.add_argument("--end",   default=None)
    p.add_argument("--neighbor_fleet", default=None,
                   help="CSV: zone,fuel,eta/heat_rate,capacity_mw â†’ zonale Î·-Dists + KapazitÃ¤tsmaske")
    # Peaker-Heuristik
    p.add_argument("--peak_switch", action="store_true",
                   help="Aktiviere Preis-zu-Peaker-Override (OCGT/Ã–l) bei hohen Preisen.")
    p.add_argument("--peak_price_thresholds", default="300,500",
                   help="BASELINE FIX: Schwellen in EUR/MWh: 'p1,p2' -> p1â‰ˆOCGT, p2â‰ˆÃ–l/Diesel. Angehoben von 180,260 um Ã–l-Artefakte zu vermeiden.")
    p.add_argument("--peak_eta_ocgt", type=float, default=0.36,
                   help="Î· fÃ¼r OCGT-Peaker (el.).")
    p.add_argument("--peak_eta_oil", type=float, default=0.33,
                   help="Î· fÃ¼r Ã¶l-/dieselbefeuerte Turbinen/Engines als Peaker.")
    p.add_argument("--apply_non_availability", action="store_true",
                   help="AUDIT RISK: Wendet vordefinierte Nicht-VerfÃ¼gbarkeitsraten an. Erfordert --non_availability_json mit Quellenangabe!")
    p.add_argument("--non_availability_json", type=str, default=None,
                   help="Optional: Pfad zu JSON mit Nicht-VerfÃ¼gbarkeitsraten pro Kraftwerkstyp (Werte in Dezimal-Bruchteil, z.B. 0.13 fÃ¼r 13%%).")
    # Nuklear-SRMC (EUR/MWh_el) fÃ¼r Export-Stack (MEF bleibt 0)
    p.add_argument("--nuclear_srmc_eur_mwh", type=float, default=None,
                   help="Fixer SRMC Kernenergie (EUR/MWh_el). Wenn gesetzt, Ã¼berschreibt die Berechnung.")
    p.add_argument("--nuclear_fuel_eur_mwh_th", type=float, default=5.5,
                   help="Brennstoffkosten Kernenergie (EUR/MWh_th).")
    p.add_argument("--nuclear_eta", type=float, default=0.33,
                   help="Elektrischer Wirkungsgrad Kernenergie (Anteil).")
    p.add_argument("--nuclear_varom_eur_mwh", type=float, default=1.2,
                   help="Sonstige variable Kosten Kernenergie (EUR/MWh_el).")
    p.add_argument("--biomass_srmc_eur_mwh", type=float, default=35.0,
                   help="Grenzkosten Biomasse (EUR/MWh_el), nur fÃ¼r Preis-Logik.")
    p.add_argument("--waste_srmc_eur_mwh", type=float, default=1.0,
                   help="Grenzkosten Waste (EUR/MWh_el) â€“ default: 1 EUR/MWh (Gebot).")
    p.add_argument("--biomass_mef_gpkwh", type=float, default=0.0,
                   help="MEF fÃ¼r Biomasse (g/kWh) â€“ default 0.")
    p.add_argument("--waste_mef_gpkwh", type=float, default=0.0,
                   help="MEF fÃ¼r Waste (g/kWh) â€“ default 0.")
    # NEU: kostenbasierte MU-Logik
    p.add_argument("--mu_cost_mode",
                   choices=["off","q_vs_cost"],
                   default="q_vs_cost",
                   help="BASELINE FIX: Kostenbasierte MU-Ermittlung als Standard: q_vs_cost = MU = q-Quantil der Gen in Stunden mit Preis â‰¤ Î±Â·SRMC.")
    p.add_argument("--mu_cost_alpha", type=float, default=0.75,
                   help="Î± in Preis â‰¤ Î±Â·SRMC (z. B. 0.75).")
    p.add_argument("--mu_cost_q", type=float, default=0.50,
                   help="q-Quantil innerhalb der 'unter Kosten'-Stunden (z. B. 0.5 = Median).")
    p.add_argument("--mu_cost_monthly", action="store_true",
                   help="Kostenbasiertes MU getrennt pro Monat berechnen.")
    p.add_argument("--mu_cost_use_peak", action="store_true",
                   help="Kostenbasiertes MU zusÃ¤tzlich Peak/Offpeak je Monat trennen (Fenster via --mustrun_peak_hours).")
    
    # Optional: DE anstatt fixer Shares aus Kosten ableiten
    p.add_argument("--de_fossil_mustrun_from_cost", action="store_true",
                   help="DE-fossiler MU aus Kostenlogik ableiten (ersetzt de_mustrun_*_share).")

    p.add_argument("--coupled_import_anyflow", action="store_true", default=False,
                   help="CRITICAL FIX: Importseite nur bei echtem Netto-Import aktivieren (default=False). True kann Marginal-Seite kÃ¼nstlich zu IMPORT schieben.")
    # --- MU-Bid Logik (optional rollierend) ---
    p.add_argument("--mu_bid_mode", choices=["default","rolling"], default="default",
                   help="Gebote fÃ¼r MU (Waste/Nuclear/Biomass/Fossil) als 'default' oder rollierender Schnitt (Preis < Î±Â·SRMC).")
    p.add_argument("--mu_bid_window_h", type=int, default=168,
                   help="Fenster fÃ¼r rollierende MU-Bids (Stunden), z. B. 168=7 Tage.")

    # Negativbepreisung Mustrun (fossil + optional Nuklear)
    p.add_argument("--mustrun_neg_pricing_enable", action="store_true", default=False,
                   help="BASELINE FIX: Standardmäßig aus. Wenn Preis < pct*SRMC: markiere Mustrun-Mengen als negativ bepreist.")
    p.add_argument("--mustrun_neg_price_threshold_pct", type=float, default=0.75,
                   help="Schwelle Î± fÃ¼r Negativbepreisung (Preis < Î±*SRMC).")
    p.add_argument("--mustrun_neg_price_value", type=float, default=-10.0,
                   help="Preis fÃ¼r negativ bepreiste Mustrun-Mengen [EUR/MWh].")
    # Export-Stack: gewÃ¼nschte Bids
    p.add_argument("--waste_bid_eur_mwh_export", type=float, default=1.0,
                   help="Gebot Waste im Export-Stack (EUR/MWh).")
    p.add_argument("--biomass_bid_eur_mwh_export", type=float, default=35.0,
                   help="Gebot Biomasse im Export-Stack (EUR/MWh).")
    p.add_argument("--mustrun_bid_eur_mwh", type=float, default=1.0,
                   help="Gebot fossiler Mustrun im Export-Stack, wenn Negativ-Bepreisung aus (EUR/MWh).")
    p.add_argument("--mustrun_neg_share", type=float, default=0.0,
                   help="BASELINE FIX: Anteil (0..1) der Mustrun-Mengen auf 0.0 gesetzt, die bei Unterschreiten der Schwelle negativ bepreist werden sollen.")

    # Nuklear-Mustrun
    p.add_argument("--de_nuclear_mustrun_share", type=float, default=0.0,
                    help="Anteil der verfÃ¼gbaren DE-NuklearkapazitÃ¤t als Mustrun (0..1).")
    p.add_argument("--nei_nuclear_mustrun_share", type=float, default=0.0,
                    help="Anteil der verfÃ¼gbaren NEI-NuklearkapazitÃ¤t als Mustrun (0..1).")
    # Biomasse-Mustrun (optional)
    p.add_argument("--de_biomass_mustrun_share", type=float, default=0.0,
                   help="Anteil Biomasse als Mustrun (DE).")
    p.add_argument("--nei_biomass_mustrun_share", type=float, default=0.0,
                   help="Anteil Biomasse als Mustrun (Nachbarn).")
    # Fossiler Mustrun
    p.add_argument("--fossil_mustrun_mode",
                   choices=["off","min_all","min_peak","min_peak_monthly","q_all"],
                   default="q_all",
                   help="q_all = unteres Quantil Ã¼ber alle Stunden je Fuel")
    p.add_argument("--fossil_mustrun_q", type=float, default=0.10,
                   help="Quantil fÃ¼r q_all (z.B. 0.10 = 10 %%)")
    p.add_argument("--fossil_mustrun_fuels",
                   default="Erdgas,Steinkohle,HeizÃ¶l schwer,HeizÃ¶l leicht / Diesel",
                   help="Braunkohle i.d.R. NICHT listen â€“ wird separat behandelt.")
        # --- Korrelation / Diagnostics ---
    p.add_argument("--corr_drop_neg_prices", action="store_true", default=True,
                   help="Negative Preise bei der Korrelation ignorieren.")
    p.add_argument("--corr_neg_price_cut", type=float, default=-50.0,
                   help="Absolute Schwelle (EUR) unterhalb der negative Preise für Korrelation ausgeschlossen werden.")
    p.add_argument("--corr_cap_mode",
                   choices=["none","absolute","peaker_min","peaker_max"],
                   default="absolute",
                   help=("BASELINE FIX: Preis-Cap fÃ¼r Korrelation auf 'absolute' (500EUR) um echte Knappheitsstunden nicht wegzuschneiden. "
                         "none=kein Cap, absolute=fester Grenzwert, "
                         "peaker_min=min(OCGT,Oil)-SRMC als Cap, "
                         "peaker_max=max(OCGT,Oil)-SRMC als Cap."))
    p.add_argument("--corr_cap_value", type=float, default=500.0,
                   help="Absolute Obergrenze (nur bei --corr_cap_mode=absolute).")
    p.add_argument("--corr_offenders_topn", type=int, default=500,
                   help="Top-N AusreiÃŸer in analysis/_corr_offenders.csv.")
    p.add_argument("--corr_cap_tol", type=float, default=3.0,
                   help="OPTIMIZED: Toleranz 3.0EUR/MWh fÃ¼r bessere r-Korrelation ohne Peaker-Bias (vorher 1.0).")
    

    # Kopplung / Preisanker
    p.add_argument("--neighbor_gen_dir",   required=True)
    p.add_argument("--neighbor_load_dir",  required=True)
    p.add_argument("--neighbor_prices",    required=True)
    
    # JAO FlowBased Boundaries
    p.add_argument("--fb_np_csv", default=None, 
                   help="CSV mit minNP/maxNP/NetPosition/fb_boundary (UTC) aus JAO API")
    
    p.add_argument("--epsilon", type=float, default=0.01, help="Preis-Kopplungs-Schwelle (EUR/MWh). Lower => tighter price-equality; default set to 0.01 for threshold mode.")
    p.add_argument("--price_anchor", choices=["off","closest","threshold"], default="threshold")
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
               help="Oberes Band fÃ¼r PSP-SRMC (preisfolgend).")
    p.add_argument("--psp_rt_eff", type=float, default=0.78, 
               help="Round-trip efficiency (typ. 0.7â€“0.85).")
    p.add_argument("--psp_pump_window_h", type=int, default=48, 
               help="Fenster fÃ¼r Pump-Preis-Baseline.")
    p.add_argument("--psp_accept_band", type=float, default=5.0, 
               help="Â±EUR/MWh Band fÃ¼r PSP-Preissetzung.")
    p.add_argument("--psp_price_cap", type=float, default=180.0, 
               help="CRITICAL FIX: Max Preis fÃ¼r PSP-Preissetzung. ErhÃ¶ht von 80EUR auf 180EUR fÃ¼r realistische Knappheitspreise.")
    p.add_argument("--psp_min_avail_mw", type=float, default=0.0,
                   help="Minimum available PSP MW considered for RL reductions (forces a floor on PSP available capacity).")
    p.add_argument("--reservoir_min_avail_mw", type=float, default=0.0,
                   help="Minimum available Reservoir MW considered for RL reductions (forces a floor on reservoir available capacity).")
    p.add_argument("--reservoir_max_clip", type=float, default=300.0,
                   help="Maximum price clip used when deriving a reservoir base water value from price (EUR/MWh). Set high to avoid artificial clipping at 60 EUR.")
    p.add_argument("--reservoir_max_srmc", type=float, default=300.0,
                   help="Maximum SRMC allowed for Reservoir Hydro when used in merit/export logic (EUR/MWh).")
    # JAO FlowBased Boundary Detection
    p.add_argument("--jao_fb_enable", action="store_true", 
               help="Enable JAO FlowBased boundary detection for Net Position limits.")
    p.add_argument("--jao_fb_tolerance", type=float, default=100.0,
               help="Tolerance (MW) for Net Position boundary detection.")
    # Dispatch-Details
    p.add_argument("--varom_json", default=None)
    p.add_argument("--therm_avail", type=float, default=0.95)
    p.add_argument("--mustrun_mode", choices=["off","capacity","gen_quantile"], default="gen_quantile")
    p.add_argument("--mustrun_lignite_q", type=float, default=0.20)
    p.add_argument("--mustrun_quantile",  type=float, default=0.10)
    p.add_argument("--mustrun_peak_hours", default="08-20")
    p.add_argument("--mustrun_monthly", action="store_true")
    p.add_argument("--lignite_price_floor", type=float, default=30.0,
                   help="Preis-Schwelle fÃ¼r dynamischen Braunkohle-MU (EUR/MWh)")
    p.add_argument("--lignite_price_window", type=int, default=6,
                   help="FenstergrÃ¶ÃŸe (h) fÃ¼r max-GlÃ¤ttung des Braunkohle-MU aus Low-Price-Stunden")
    p.add_argument("--lignite_price_min_hours", type=int, default=3,
                   help="Mindestdauer (h) einer Low-Price-Phase, bevor sie MU auslÃ¶st")

    p.add_argument("--oil_price_floor", type=float, default=20.0,
                   help="Preis-Schwelle fÃ¼r dynamischen Ã–l-MU (EUR/MWh)")
    p.add_argument("--oil_price_window", type=int, default=6,
                   help="FenstergrÃ¶ÃŸe (h) fÃ¼r max-GlÃ¤ttung des Ã–l-MU aus Low-Price-Stunden")
    p.add_argument("--oil_price_min_hours", type=int, default=3,
                   help="Mindestdauer (h) einer Low-Price-Phase, bevor sie Ã–l-MU auslÃ¶st")
    p.add_argument("--enable_oil_mustrun", action="store_true",
                   help="Enable dynamic oil must-run heuristic (default: disabled)."
                   )

    p.add_argument("--ee_price_threshold", type=float, default=10.0,
                   help="MEF=0 bei Preis <= 10 EUR/MWh (starke EE-Überschussstunden). Set default raised to 10 to classify low-price hours as RES-driven for imports/exports.")
    p.add_argument("--import_zero_price_threshold", type=float, default=10.0,
                   help="Preisgrenze (EUR/MWh) unter der Importe als EE-getrieben (MEF=0) klassifiziert werden. Default set to 10 EUR to apply EE-override also for imports.")
    p.add_argument("--ee_gate_tol", type=float, default=1e-02,
                   help="Tolerance (MW) for the EE gate on the residual after FEE. If RL_after_FEE <= ee_gate_tol and price <= ee_price_threshold, classify as EE-driven. Default 1e-2 MW.")
    p.add_argument("--domestic_ee_after_mustrun", action="store_true",
                   help="Wenn gesetzt, werden in der inländischen Zone (DE/DE_LU) EE erst NACH den Must-Run-Abzügen abgezogen (Konsequente MU-first-Logik).")
    p.add_argument("--ee_surplus_order",
                   choices=["ee_before_mu", "mu_before_ee"],
                   default="ee_before_mu",
                   help="Reihenfolge der EE-Überschussbildung im Export-Stack: "
                        "ee_before_mu (Policy-Fidelity) oder mu_before_ee (Must-Run zuerst).")
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
    """SRMC-Reihen fÃ¼r OCGT (Gas) und Ã–l-Peaker, EUR/MWh_el."""
    ef_gas_th = EF_LOOKUP_T_PER_MWH_TH["Erdgas"]
    ef_oil_th = EF_LOOKUP_T_PER_MWH_TH["HeizÃ¶l schwer"]
    gas = pd.to_numeric(fuel_prices["gas_eur_mwh_th"], errors="coerce")
    oil = pd.to_numeric(fuel_prices["oil_eur_mwh_th"], errors="coerce")
    co2 = pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce")
    srmc_ocgt = (gas + co2 * ef_gas_th) / max(eta_ocgt, 1e-6)
    srmc_oil  = (oil + co2 * ef_oil_th) / max(eta_oil,  1e-6)
    return srmc_ocgt.astype(float), srmc_oil.astype(float)


def _parse_two_floats(csv_str: str, default=(180.0, 260.0)):
    try:
        a, b = [float(x) for x in str(csv_str).split(",")[:2]]
        return a, b
    except Exception:
        return default




def robustize_load_gen(de_load: pd.Series, de_gen: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    CRITICAL FIX: Entfernt EE-Skalierung bei Exportlagen.
    
    Vorher: EE wurde gekappt wenn ND>Load, was echte ÃœberschÃ¼sse wegfilterete
    und MU in Negativstunden kÃ¼nstlich runterdrckte.
    
    Jetzt: Nur Robustness-Checks, keine Kappung von EE-Ãœberfahrten.
    """
    # Nur Robustness fÃ¼r NaN/Inf, keine EE-Kappung mehr
    for c in de_gen.columns:
        de_gen[c] = _robust_series(de_gen[c], f"gen:{c}")
    de_load = _robust_series(de_load, "load")
    
    # REMOVED: EE-Skalierung bei Exportlagen 
    # Das war: over = (nd_sum > de_load) & scale EE runter
    # FÃ¼hrte zu kÃ¼nstlich niedrigen MU-Niveaus in Negativstunden
    
    return de_load, de_gen


def energy_budget_reservoir_prices(nei_prices: pd.DataFrame, zone: str, 
                                 reservoir_budget_mwh_per_month: float = None,
                                 epsilon_coupling: float = 0.05,
                                 delta_band: float = 5.0,
                                 smoothing_days: int = 7,
                                 min_water_value: float = 30.0,
                                 max_water_value: float = 180.0) -> pd.Series:
    """
    Enhanced Energie-Budget + Price-Duration-Cutoff Ansatz fÃ¼r Hydro Water Reservoir.
    
    Algorithmus:
    1. Sortiere Stunden nach gekoppeltem Preis (price_DE Â± Îµ) absteigend
    2. Integriere Energie bis Budget Em erreicht â†’ Grenzstunde tÌ‚
    3. Wasserwert wm = Preis(tÌ‚)
    4. GlÃ¤tte wm als rollenden Median und bilde Kandidatenband [wm-Î”, wm+Î”]
    5. Dispatch-Flag: Reservoir ist Kandidat wenn Preis im Band liegt
    
    Args:
        nei_prices: DataFrame mit Preisen
        zone: Zone (z.B. 'AT', 'CH', etc.)
        reservoir_budget_mwh_per_month: Monatliches Energie-Budget in MWh (jetzt 5k-120k statt 1k)
        epsilon_coupling: Preiskopplung-Toleranz in EUR/MWh
        delta_band: Wasserwert-Band Â±Î” in EUR/MWh
        smoothing_days: GlÃ¤ttungsfenster in Tagen
        min_water_value: Minimum Wasserwert (Floor)
        max_water_value: Maximum Wasserwert (Ceil)
    
    Returns:
        pd.Series: Wasserwerte fÃ¼r jede Stunde
    """
    
    # Spalten identifizieren
    col = f"price_{zone}" if f"price_{zone}" in nei_prices.columns else "price_DE_LU"
    base_col = "price_DE_LU" if "price_DE_LU" in nei_prices.columns else col
    
    if col not in nei_prices.columns or base_col not in nei_prices.columns:
        print(f"[WARNING] Preisspalten fÃ¼r {zone} nicht gefunden, fallback auf einfachen Ansatz")
        return reservoir_shadow_price_series(nei_prices, zone)
    
    price_zone = pd.to_numeric(nei_prices[col], errors="coerce").fillna(0)
    price_base = pd.to_numeric(nei_prices[base_col], errors="coerce").fillna(0)
    
    # Gekoppelter Preis: price_DE Â± Îµ
    coupled_price = price_base.copy()
    price_diff = abs(price_zone - price_base)
    coupling_mask = price_diff <= epsilon_coupling
    coupled_price.loc[coupling_mask] = price_zone.loc[coupling_mask]
    
    # Budget-SchÃ¤tzung falls nicht gegeben
    if reservoir_budget_mwh_per_month is None:
        # Konservative SchÃ¤tzung: 15% CF fÃ¼r Reservoir-Hydro
        # Annahme: ~2000 MW installierte Reservoir-Leistung pro Zone
        installed_mw = 2000  # Plausible Annahme
        cf = 0.15  # 15% Capacity Factor
        hours_per_month = 24 * 30.4  # Durchschnittlicher Monat
        reservoir_budget_mwh_per_month = installed_mw * cf * hours_per_month
        print(f"[INFO] {zone}: GeschÃ¤tztes Reservoir-Budget {reservoir_budget_mwh_per_month:.0f} MWh/Monat")
    
    # Wasserwerte je Monat berechnen
    water_values = pd.Series(index=coupled_price.index, dtype=float)
    
    # Gruppierung nach Monat
    coupled_price_with_month = coupled_price.to_frame('price')
    coupled_price_with_month['month'] = coupled_price_with_month.index.month
    coupled_price_with_month['year'] = coupled_price_with_month.index.year
    
    # Zone-specific installed reservoir power proxies (MW)
    # These are realistic estimates for reservoir dispatch capacity per zone
    reservoir_power_proxies = {
        'AT': 6000,     # Austria: ~6 GW reservoir capacity
        'CH': 15000,    # Switzerland: ~15 GW reservoir capacity  
        'NO2': 20000,   # Norway: ~20 GW reservoir capacity
        'SE4': 16000,   # Sweden: ~16 GW reservoir capacity
        'FR': 5000,     # France: ~5 GW reservoir capacity
        'IT': 4000,     # Italy: ~4 GW reservoir capacity
        'ES': 3200,     # Spain: ~3.2 GW reservoir capacity
        'PL': 1500,     # Poland: ~1.5 GW reservoir capacity
        'DE_LU': 1200,  # Germany+Luxembourg: ~1.2 GW reservoir capacity
        'BE': 450,      # Belgium: ~0.45 GW reservoir capacity (Coo)
        'NL': 200,      # Netherlands: minimal reservoir
        'CZ': 900,      # Czech Republic: ~0.9 GW reservoir capacity
        'DK1': 100,     # Denmark West: minimal
        'DK2': 100,     # Denmark East: minimal
        'default': 2000 # Fallback for unknown zones
    }
    
    # Get installed reservoir power for this zone
    reservoir_power_mw = reservoir_power_proxies.get(zone, reservoir_power_proxies['default'])
    
    # Convert monthly budget from MWh to equivalent hours of operation
    # budget_hours = Em_MWh / P_installed_MW (realistic reservoir dispatch)
    budget_hours = max(1.0, reservoir_budget_mwh_per_month / max(reservoir_power_mw, 1.0))
    
    # QA Warning for unrealistic budgets
    hours_in_month = 24 * 30.4  # Average month
    if budget_hours > hours_in_month:
        print(f"[WARNING] Zone {zone}: Budget {reservoir_budget_mwh_per_month:.0f} MWh "
              f"requires {budget_hours:.1f}h but month only has {hours_in_month:.1f}h. "
              f"Using monthly cap.")
        budget_hours = hours_in_month * 0.95  # 95% of month max
    
    for (year, month), group in coupled_price_with_month.groupby(['year', 'month']):
        if len(group) < 24:  # Zu wenig Daten
            continue
            
        # Stunden nach Preis absteigend sortieren
        sorted_hours = group.sort_values('price', ascending=False)
        
        # FIXED: Budget-Hours Integration statt 1 MWh/Stunde
        # Energieintegration bis Budget-Stunden erreicht
        cumulative_hours = 0
        cutoff_price = max_water_value
        
        for idx, (timestamp, row) in enumerate(sorted_hours.iterrows()):
            cumulative_hours += 1  # 1 hour increment
            if cumulative_hours >= budget_hours:
                cutoff_price = row['price']
                break
        
        # Floor/Ceil mit monatlichen Perzentilen
        month_prices = group['price']
        p30_month = month_prices.quantile(0.3)
        p95_month = month_prices.quantile(0.95)
        
        floor_wm = max(min_water_value, p30_month)
        ceil_wm = min(max_water_value, p95_month)
        
        # Wasserwert begrenzen
        monthly_water_value = np.clip(cutoff_price, floor_wm, ceil_wm)
        
        # Allen Stunden des Monats zuweisen
        water_values.loc[group.index] = monthly_water_value
        
        print(f"[HYDRO] {zone} {year}-{month:02d}: Budget={reservoir_budget_mwh_per_month:.0f}MWh, "
              f"Cutoff={cutoff_price:.1f}EUR/MWh, Wasserwert={monthly_water_value:.1f}EUR/MWh "
              f"(Floor={floor_wm:.1f}, Ceil={ceil_wm:.1f})")
    
    # Memory-optimized smoothing using rolling median with chunking for large datasets
    def rolling_median_chunked(series, window, min_periods=12, center=True, chunk_size=2000):
        """Apply rolling median in chunks to avoid memory issues"""
        if len(series) <= chunk_size:
            # Small enough - process directly
            return series.rolling(window, min_periods=min_periods, center=center).median()
        
        print(f"[INFO] Applying rolling median in chunks for series length {len(series)}")
        
        # For large datasets, process in overlapping chunks
        half_window = window // 2
        results = []
        
        for start_idx in range(0, len(series), chunk_size):
            end_idx = min(start_idx + chunk_size, len(series))
            
            # Extend chunk boundaries to handle edge effects
            chunk_start = max(0, start_idx - half_window)
            chunk_end = min(len(series), end_idx + half_window)
            
            chunk_series = series.iloc[chunk_start:chunk_end]
            chunk_result = chunk_series.rolling(window, min_periods=min_periods, center=center).median()
            
            # Extract the relevant part (without padding)
            if start_idx == 0:
                relevant_result = chunk_result.iloc[:end_idx-start_idx]
            else:
                padding = start_idx - chunk_start
                relevant_result = chunk_result.iloc[padding:padding+(end_idx-start_idx)]
            
            results.append(relevant_result)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        return pd.concat(results)
    
    # GlÃ¤ttung als rollender Median (memory-optimized)
    smoothing_hours = smoothing_days * 24
    water_values_smoothed = rolling_median_chunked(
        water_values, smoothing_hours, min_periods=12, center=True
    ).fillna(water_values)
    
    # Wasserwert-Band fÃ¼r Dispatch [wm-Î”, wm+Î”]
    # Hier geben wir den geglÃ¤tteten Wasserwert zurÃ¼ck
    # Das Dispatch-Band wird spÃ¤ter in der calling function verwendet
    water_values_smoothed = water_values_smoothed.fillna(min_water_value)
    
    return water_values_smoothed.clip(lower=min_water_value, upper=max_water_value)


def is_reservoir_dispatch_candidate(current_price: float, water_value: float, delta_band: float = 5.0) -> bool:
    """
    PrÃ¼ft ob Reservoir-Hydro zu aktueller Stunde dispatcht werden sollte.
    
    Args:
        current_price: Aktueller Strompreis in EUR/MWh
        water_value: Wasserwert in EUR/MWh
        delta_band: Band Â±Î” um Wasserwert in EUR/MWh
    
    Returns:
        bool: True wenn Preis im Band [wm-Î”, wm+Î”] liegt
    """
    lower_bound = water_value - delta_band
    upper_bound = water_value + delta_band
    return lower_bound <= current_price <= upper_bound


def reservoir_shadow_price_series(nei_prices: pd.DataFrame, zone: str, window_h: int = 24*7, short_h: int = 24) -> pd.Series:
    col = f"price_{zone}" if f"price_{zone}" in nei_prices.columns else "price_DE_LU"
    base_col = "price_DE_LU" if "price_DE_LU" in nei_prices.columns else col
    price_zone = pd.to_numeric(nei_prices[col], errors="coerce").astype(float)
    price_base = pd.to_numeric(nei_prices[base_col], errors="coerce").astype(float)
    
    # Memory-optimized rolling median using the chunked function from above
    def rolling_median_chunked_simple(series, window, min_periods=12, chunk_size=2000):
        """Simplified chunked rolling median for memory efficiency"""
        if len(series) <= chunk_size:
            return series.rolling(window, min_periods=min_periods).median()
        
        results = []
        for start_idx in range(0, len(series), chunk_size):
            end_idx = min(start_idx + chunk_size, len(series))
            chunk = series.iloc[start_idx:end_idx]
            chunk_result = chunk.rolling(window, min_periods=min_periods).median()
            results.append(chunk_result)
            
            import gc
            gc.collect()
        
        return pd.concat(results)
    
    med_zone = rolling_median_chunked_simple(price_zone, window_h, min_periods=12)
    med_base = rolling_median_chunked_simple(price_base, window_h, min_periods=12)
    
    blended = 0.6 * med_zone + 0.4 * med_base
    fallback = price_zone.fillna(price_base)
    sp = blended.fillna(fallback)
    if short_h > 1:
        sp = sp.rolling(short_h, min_periods=1).mean()
    return sp.clip(lower=0.0)
def _rolling_mu_bid_series(price: pd.Series, thresh: pd.Series | float, window_h: int, alpha: float, default_bid: float) -> pd.Series:
    """
    Rollierendes MU-Gebot: Mittel der Preise in Stunden mit Preis <= Î± * Schwelle (SRMC oder Floor).
    FÃ¤llt auf default_bid zurÃ¼ck, wenn zu wenig Punkte im Fenster sind.
    """
    if not isinstance(thresh, pd.Series):
        thresh = pd.Series(thresh, index=price.index)
    mask = price <= (alpha * thresh.reindex(price.index))
    roll = price.where(mask).rolling(window_h, min_periods=max(12, window_h//7)).mean()
    return roll.fillna(default_bid)


def precompute_mu_bid_fn(nei_prices: pd.DataFrame, fuel_prices: pd.DataFrame, args, zones: list[str]):
    """
    Erzeuge eine Lookup-Funktion mu_bid(zone, tech, t, default) fÃ¼r MU-Bids.
    tech in {"waste","nuclear","biomass","fossil"}.
    """
    mode = str(getattr(args, "mu_bid_mode", "default"))
    if mode != "rolling":
        # trivialer Fallback: immer default
        return lambda zone, tech, t, default: float(default)

    alpha   = float(getattr(args, "mustrun_neg_price_threshold_pct", 0.75))
    window  = int(getattr(args, "mu_bid_window_h", 168))
    # Referenz-SRMCs/Floors (zonenunabhÃ¤ngig; Preise zonenspezifisch)
    ef_g = EF_LOOKUP_T_PER_MWH_TH["Erdgas"]; ef_c = EF_LOOKUP_T_PER_MWH_TH["Steinkohle"]; ef_oil = EF_LOOKUP_T_PER_MWH_TH["Heizöl schwer"]
    eta_ref = 0.40  # â€žDurchschnittsâ€œ-Effizienz als Floor fÃ¼r MU-Fossil
    eta_oil = 0.33  # Öl-Peaker-Effizienz
    gas_ref  = (pd.to_numeric(fuel_prices["gas_eur_mwh_th"], errors="coerce") + pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce") * ef_g) / max(eta_ref,1e-6)
    coal_ref = (pd.to_numeric(fuel_prices["coal_eur_mwh_th"], errors="coerce") + pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce") * ef_c) / max(eta_ref,1e-6)
    oil_ref  = (pd.to_numeric(fuel_prices["oil_eur_mwh_th"], errors="coerce") + pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce") * ef_oil) / max(eta_oil,1e-6)
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
        store[(z,"oil")]     = _rolling_mu_bid_series(pz, oil_ref.reindex(pz.index), window, alpha, float(getattr(args, "mustrun_bid_eur_mwh", 35.0)))

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

# ------------------- Nachbarn: Î·-Verteilungen & KapazitÃ¤t --------------------
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
    RÃ¼ckgabe sortiert nach Gruppen (kleiner = frÃ¼her):
      0=EE (0 EUR/MWh),
      1=MU Waste (rollierendes Bid / fix 1 EUR),
      2=MU Nuclear (rollierendes Bid / fix),
      3=MU Biomass (rollierendes Bid / fix),
      4=MU Oil (rollierendes Bid / fix),
      5=MU Fossil (rollierendes Bid / fix),
      6=PSP (0..Floor),
      7=Reservoir (Shadow-Price),
      8=Nuclear FLEX (SRMC),
      9=Fossil FLEX (SRMC je Fuel)
    MW = exportierbarer Ãœberschuss *nachdem* die Stufen der RL-Leiter lokal bedient sind.
    """
    mu_bid = mu_bid_getter or (lambda z, tech, tt, default: float(default))

    # Read techs
    fee = float(pd.to_numeric(gen_z_row.reindex(
        ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
    ).fillna(0.0)).sum())
    waste = float(gen_z_row.get("Waste", 0.0))
    bio   = float(gen_z_row.get("Biomass", 0.0))
    nuc   = float(gen_z_row.get("Nuclear", 0.0))

    # MU-Shares (Biomasse/Nuclear parametrisierbar; Waste = vollstÃ¤ndig MU)
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

    # RL-Leiter - EINHEITLICHE BERECHNUNG
    rl_ladder = compute_residual_load_ladder(gen_z_row, load_z_t, min_by_fuel_zone_t, min_total_zone_t, args, zone)
    RL1, RL2, RL3, RL4, RL5, RL6, RL7, RL8 = rl_ladder['RL1'], rl_ladder['RL2'], rl_ladder['RL3'], rl_ladder['RL4'], rl_ladder['RL5'], rl_ladder['RL6'], rl_ladder['RL7'], rl_ladder['RL8']
    takes = rl_ladder['takes']
    take_waste_mu, take_nuc_mu, take_bio_mu, take_oil_mu, take_mu_foss, take_psp, take_res = takes['waste_mu'], takes['nuc_mu'], takes['bio_mu'], takes['oil_mu'], takes['mu_foss'], takes['psp'], takes['res']
    
    # Extract totals for surplus calculations
    oil_mu_total = rl_ladder['totals']['oil_mu_total']
    mu_foss_total = rl_ladder['totals']['mu_foss_total']

    # PSP/Reservoir Kapazitäten für Export-Blöcke
    mw_psp = float(gen_z_row.get("Hydro Pumped Storage", 0.0)) if "Hydro Pumped Storage" in gen_z_row.index else 0.0
    mw_psp = max(mw_psp, float(getattr(args, "psp_min_avail_mw", 0.0)))
    mw_res = float(gen_z_row.get("Hydro Water Reservoir", 0.0)) if "Hydro Water Reservoir" in gen_z_row.index else 0.0
    mw_res = max(mw_res, float(getattr(args, "reservoir_min_avail_mw", 0.0)))


    # Surplus je Stufe â†’ exportierbare BlÃ¶cke (in Gruppen-Reihenfolge)
    out: List[Tuple[int,str,float,float,float,str]] = []

    # G0 EE-Surplus
    # Use the total EE available in the row (computed above as `fee`) rather than
    # the ladder's `takes['fee']` which records the locally consumed portion.
    fee_total = float(fee)
    order = getattr(args, "ee_surplus_order", "ee_before_mu")

    if order == "mu_before_ee":
        # MU-first: local MU is removed from load first, EE local consumption = min(fee_total, load_after_mu)
        mu_taken_local = float(
            take_waste_mu + take_nuc_mu + take_bio_mu + take_oil_mu + take_mu_foss
        )
        load_after_mu = max(float(load_z_t) - mu_taken_local, 0.0)
        fee_surplus = max(0.0, fee_total - load_after_mu)
    else:
        # EE-first (default): local EE consumption = load - RL1
        ee_used_local = max(0.0, float(load_z_t) - float(RL1))
        fee_surplus = max(0.0, fee_total - ee_used_local)
        
    if fee_surplus > 1e-6:
        out.append((0, "EE", 0.0, float(fee_surplus), 1.0, zone))

    # G1 MU Waste (Gebot: rollierend oder fix 1 EUR)
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

    mu_oil_surplus = max(oil_mu_total - take_oil_mu, 0.0)
    # Do not offer oil must-run on export stacks: skip adding the MU-Oil block entirely
    # (prevents fossil must-run oil surplus from being selected as import marginal)
    DK_OIL_BLOCK_ZONES = {"DK1", "DK2"}
    OIL_LABELS = {"Heizöl schwer", "HeizÃ¶l schwer"}
    if mu_oil_surplus > 1e-6:
        # intentionally skip adding oil must-run export block
        pass

    # G5 MU Fossil (Gebot: rollierend oder mustrun_bid)
    mu_foss_surplus = max(mu_foss_total - take_mu_foss, 0.0)
    if mu_foss_surplus > 1e-6:
        bid_f = mu_bid(zone, "fossil", t, float(getattr(args, "mustrun_bid_eur_mwh", 1.0)))
        out.append((5, "Mustrun (fossil)", float(bid_f), float(mu_foss_surplus), 1.0, zone))

    # G6 PSP (0 bis Floor)
    if mw_psp - take_psp > 1e-6:
        out.append((6, "Hydro Pumped Storage", 0.0, float(mw_psp - take_psp), 1.0, zone))

    # G7 Reservoir (Shadow-Price)
    if mw_res - take_res > 1e-6:
        srmc_res = float(reservoir_sp_map.get((zone, t), 0.0))
        srmc_res = float(np.clip(srmc_res, 0.0, float(getattr(args,"psp_srmc_floor_eur_mwh",60.0))))
        out.append((7, "Reservoir Hydro", srmc_res, float(mw_res - take_res), 1.0, zone))

    # G8 Nuclear FLEX (positives SRMC)
    if nuc_flex > 1e-6:
        out.append((8, "Nuclear", float(nuc_srmc), float(nuc_flex), 1.0, zone))

    # G9 Fossil FLEX (Gas/Coal/Oil mit Î·-Modell)
    for tech, (pk, ef_name) in FOSSIL_TECH_TO_FUEL.items():
        if tech not in gen_z_row.index: 
            continue
        mw_raw = float(gen_z_row.get(tech, 0.0))
        if not (np.isfinite(mw_raw) and mw_raw > 0): 
            continue
        # PATCH 3: Cap-Mask anwenden statt überspringen
        if cap_mask is not None:
            cap = cap_mask.get((zone, ef_name), None)
            if cap is not None:
                mw_raw = min(mw_raw, max(0.0, float(cap)))
        
        fuel_th = fuel_prices_row.get(f"{pk}_eur_mwh_th", np.nan)
        co2     = fuel_prices_row.get("co2_eur_t", np.nan)
        if not (np.isfinite(fuel_th) and np.isfinite(co2)):
            continue
        d = (nei_dists.get(zone, {}).get(ef_name)
             or nei_dists.get(ef_name)
             or DEFAULT_NEI_DISTS.get(ef_name))
        if not d:
            continue
        m, s, lo, hi = d["mean"], d["std"], d["min"], d["max"]
        eta_eff = m if mode != "mc" else float(np.mean(truncated_normal(m, s, lo, hi, size=draws)))
        srmc    = (fuel_th + co2 * EF_LOOKUP_T_PER_MWH_TH.get(ef_name, 0.0)) / max(eta_eff, 1e-6)
        mw_mustrun = float(min_by_fuel_zone_t.get(ef_name, 0.0))
        mw = max(mw_raw - mw_mustrun, 0.0)
        if mw <= 1e-6: 
            continue
        # Skip fossil oil export blocks for DK zones (prevent oil being treated as exportable marginal)
        if ef_name in OIL_LABELS and zone in DK_OIL_BLOCK_ZONES:
            # don't append oil fossil flex blocks for DK1/DK2
            continue
        out.append((9, ef_name, float(srmc), float(mw), float(eta_eff), zone))

    # Sortierung: erst Gruppe, dann SRMC
    out.sort(key=lambda x: (x[0], x[2]))
    return out

def residual_need_after_local_steps(
    t: pd.Timestamp,
    zone: str,
    gen_z_row: pd.Series,
    load_z_t: float,
    min_total_zone_t: float,
    min_by_fuel_zone_t: Optional[Dict[str, float]] = None,
    args=None,   # <â€” neu
) -> float:
    """
    Liefert die 'Restlast nach lokalen Stufen' (analog RL8) der Zone:
    Last â€“ FEE â€“ MU(Waste) â€“ MU(Nuclear) â€“ MU(Biomass) â€“ MU(Oil) â€“ MU(Fossil) â€“ PSP â€“ Reservoir, min 0.
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

    min_by_fuel_zone_t = min_by_fuel_zone_t or {}
    
    # EINHEITLICHE RL-BERECHNUNG statt Duplikation
    rl_ladder = compute_residual_load_ladder(gen_z_row, load_z_t, min_by_fuel_zone_t, min_total_zone_t, args, zone)
    RL8 = rl_ladder['RL8']

    return RL8


# ------------------------------ Validation & Plots ---------------------------
PALETTE = {
    "DE": "#1f77b4", "IMPORT": "#d62728", "EE": "#2ca02c",
    "price": "#444444", "warn": "#ff7f0e", "ok": "#2ca02c", "mix": "#7f7f7f",
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
    "Oil": "HeizÃ¶l schwer",
    "Biomass": "Biomasse",
}


# BBH-Style-Legende

# ------------------------------------------------------------------------------


# DEPRECATED: Old make_validation_plots function removed - using enhanced version below
# --- END PATCH ----------------------------------------------------------------


# ----------------- Fossile Mindestprofile (DE & Nachbarn) --------------------
# ------------------------------ Enhanced Validation Functions ---------------






# ------------------------------ Main -----------------------------------------
def main(args):
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Save CLI parameters for reproducibility: only store options actually set on the
    # command line (avoids writing all parser defaults). This inspects sys.argv and
    # the parser's option strings. If the parser is not available, fall back to
    # writing all args (backwards-compatible).
    try:
        import sys
        # If the parser object is available in module scope (build_parser), use it to
        # inspect option strings. Otherwise fall back to writing all args.
        try:
            parser = build_parser()
            # option strings are like '--fleet', map to dest names
            opt_to_dest = {opt: act.dest for opt, act in parser._option_string_actions.items()}
            # Collect which dests were explicitly provided on the command line
            provided_dests = set()
            argv = sys.argv[1:]
            i = 0
            while i < len(argv):
                a = argv[i]
                if a.startswith('--'):
                    # might be --flag=value or --flag value
                    if '=' in a:
                        flag = a.split('=', 1)[0]
                        provided_dests.add(opt_to_dest.get(flag, None))
                        i += 1
                    else:
                        flag = a
                        dest = opt_to_dest.get(flag, None)
                        provided_dests.add(dest)
                        # if next token looks like a value (not another flag) skip it
                        if i + 1 < len(argv) and not argv[i+1].startswith('-'):
                            i += 2
                        else:
                            i += 1
                elif a.startswith('-') and len(a) == 2:
                    # short flag -x (unlikely in this CLI, but handle generically)
                    provided_dests.add(opt_to_dest.get(a, None))
                    if i + 1 < len(argv) and not argv[i+1].startswith('-'):
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

            # Build params dict only with provided dests + required positional/mandatory args
            all_params = vars(args).copy() if hasattr(args, '__dict__') or hasattr(args, '__iter__') else dict()
            params = {}
            for k, v in all_params.items():
                if k in provided_dests or k == 'outdir' or k in ('fleet', 'fuel_prices', 'flows', 'neighbor_gen_dir', 'neighbor_load_dir', 'neighbor_prices'):
                    # include always-required I/O flags and outdir for reproducibility
                    try:
                        json.dumps(v)
                        params[k] = v
                    except Exception:
                        params[k] = str(v)
            
            # Always include ee_surplus_order parameter for consistency tracking
            params["ee_surplus_order"] = getattr(args, "ee_surplus_order", "ee_before_mu")
            # Always include domestic_ee_after_mustrun flag explicitly so runs are reproducible
            params["domestic_ee_after_mustrun"] = bool(getattr(args, "domestic_ee_after_mustrun", False))

        except Exception:
            # Fallback: write all args (previous behaviour)
            params = vars(args).copy() if hasattr(args, '__dict__') or hasattr(args, '__iter__') else dict()
            for k, v in list(params.items()):
                try:
                    json.dumps(v)
                except Exception:
                    params[k] = str(v)
            
            # Always include ee_surplus_order parameter for consistency tracking
            params["ee_surplus_order"] = getattr(args, "ee_surplus_order", "ee_before_mu")
            # Always include domestic_ee_after_mustrun flag explicitly so runs are reproducible
            params["domestic_ee_after_mustrun"] = bool(getattr(args, "domestic_ee_after_mustrun", False))

        params_file = outdir / "run_parameters.json"
        with open(params_file, "w", encoding="utf-8") as fh:
            json.dump(params, fh, ensure_ascii=False, indent=2)

        # Also write a compact human-readable version
        txt_file = outdir / "run_parameters.txt"
        with open(txt_file, "w", encoding="utf-8") as fh:
            fh.write("Run parameters:\n")
            for k in sorted(params.keys()):
                fh.write(f"{k}: {params[k]}\n")

        print(f"[OK] Run parameters written: {params_file} and {txt_file}")
    except Exception as e:
        print(f"[WARN] Could not write run-parameters files: {e}")

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

    # Defer SRMC matrix creation until after time window is defined to avoid memory issues
    srmc_by_unit = compute_unit_srmc_series(fleet, fuel_prices, varom_map)
    units = list(srmc_by_unit.keys())

    fleet_idxed = fleet.set_index("unit_id")
    # Apply non-availability rates if requested
    if getattr(args, 'apply_non_availability', False):
        # Defaults (fraction unavailable)
        default_na = {
            'Kernenergie': 0.07,
            'Braunkohle': 0.13,
            'Steinkohle': 0.20,
            'GuD': 0.13,
            'Gasturbine': 0.13,
            'Ã–l': 0.15,
            'Abfall': 0.15,
            'Other': 0.10
        }
        na_map = default_na.copy()
        if getattr(args, 'non_availability_json', None):
            try:
                with open(args.non_availability_json, 'r', encoding='utf-8') as _fh:
                    import json as _json
                    user_map = _json.load(_fh)
                    for k, v in (user_map.items() if isinstance(user_map, dict) else []):
                        try:
                            na_map[str(k)] = float(v)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[WARN] non_availability_json konnte nicht geladen werden: {e} â€” verwende Defaults.")

        # Map units to plant type and apply
        plant_types = []
        for uid, row in fleet_idxed.iterrows():
            plant_types.append(map_unit_to_plant_type(row.get('fuel_raw', ''), row.get('unit_name', ''), row.get('eta', None)))
        fleet_idxed['plant_type'] = plant_types

        na_pct = [na_map.get(pt, na_map.get('Other', 0.10)) for pt in fleet_idxed['plant_type']]
        fleet_idxed['na_pct'] = [max(0.0, min(1.0, float(x))) for x in na_pct]
        fleet_idxed['available_mw'] = (fleet_idxed['available_mw'].astype(float) * (1.0 - fleet_idxed['na_pct'])).astype('float32')
        print(f"[INFO] Nicht-VerfÃ¼gbarkeit angewendet (apply_non_availability=True). Plant-type counts: {fleet_idxed['plant_type'].value_counts().to_dict()}")
    
    # Note: SRMC matrix alignment will be done after time window is defined

    # 3) Flows & Nachbarpreise
    flows_raw = load_flows(args.flows)
    flows, ambiv_mask_raw = prepare_flows_for_mode(flows_raw, args)
    setattr(args, '_flow_mode_effective', getattr(args, 'flow_mode', 'scheduled'))
    nei_prices = load_neighbor_prices(args.neighbor_prices)
    clusters = cluster_zones_by_price(nei_prices, args.epsilon)
    zones = sorted([
        c.split("price_",1)[1].strip().replace("-", "_")
        for c in nei_prices.columns if str(c).startswith("price_")
    ])
    alias_map = _build_zone_alias_map(zones)
    setattr(args, '_zone_alias_map', alias_map)
    ambiv_mask = _merge_ambiv_masks(ambiv_mask_raw, alias_map)
    setattr(args, '_flow_ambiv_mask', ambiv_mask)
    ambiv_counts = {k: int(v.sum()) for k, v in ambiv_mask.items()}
    if ambiv_counts:
        total_flags = sum(ambiv_counts.values())
        print(f"[INFO] Hybrid flow ambivalence: {total_flags} zone-hours flagged")
    print(f"[INFO] Flow mode aktiv: {getattr(args, 'flow_mode', 'scheduled')}")
    clusters = {t: list(dict.fromkeys(_resolve_zone_name(z, alias_map) for z in zs)) for t, zs in clusters.items()}

    # 3.5) JAO FlowBased Boundaries (optional)
    fb_data = None
    if args.fb_np_csv:
        try:
            print(f"[INFO] Loading JAO FlowBased data: {args.fb_np_csv}")
            fb_data = pd.read_csv(args.fb_np_csv)
            fb_data['timestamp_utc'] = pd.to_datetime(fb_data['timestamp_utc'], utc=True)
            print(f"[OK] JAO FB data loaded: {len(fb_data)} records, "
                  f"{fb_data['fb_boundary'].sum()} boundary hours "
                  f"({100*fb_data['fb_boundary'].mean():.1f}%)")
        except Exception as e:
            print(f"[WARN] Failed to load JAO FB data: {e}")
            fb_data = None

    # 4) OpportunitÃ¤tskosten-Map fÃ¼r Reservoir je Zone & Stunde (Enhanced Realistic Budget Ansatz)
    reservoir_sp_map = {}
    # Realistische monatliche Budgets basierend auf installierten KapazitÃ¤ten und CF
    # Reservoir_E_m â‰ˆ P_installed Ã— CF Ã— hours_month (CF: 0.15-0.4 fÃ¼r Reservoir)
    # Mit Â±20% Carry-over zwischen Monaten fÃ¼r smooth scheduling
    reservoir_budget_params = {
        # Major Alpine zones with large reservoir capacities
        'AT': {'budget_mwh': 45000, 'delta_band': 5.0, 'carry_pct': 0.20},  # AT: ~6GW Ã— 0.25 Ã— 744h â‰ˆ 45 GWh/Monat
        'CH': {'budget_mwh': 85000, 'delta_band': 6.0, 'carry_pct': 0.20},  # CH: ~15GW Ã— 0.3 Ã— 744h â‰ˆ 85 GWh/Monat (Alpenspeicher)
        'NO2': {'budget_mwh': 120000, 'delta_band': 7.0, 'carry_pct': 0.25}, # NO2: ~20GW Ã— 0.25 Ã— 744h â‰ˆ 120 GWh/Monat (Fjordspeicher)
        'SE4': {'budget_mwh': 95000, 'delta_band': 6.0, 'carry_pct': 0.20},  # SE4: ~16GW Ã— 0.25 Ã— 744h â‰ˆ 95 GWh/Monat
        # Moderate reservoir zones
        'FR': {'budget_mwh': 28000, 'delta_band': 4.0, 'carry_pct': 0.15},  # FR: ~5GW Ã— 0.3 Ã— 744h â‰ˆ 28 GWh/Monat (PyrenÃ¤en/Alpen)
        'IT': {'budget_mwh': 22000, 'delta_band': 4.5, 'carry_pct': 0.15},  # IT: ~4GW Ã— 0.3 Ã— 744h â‰ˆ 22 GWh/Monat
        'ES': {'budget_mwh': 18000, 'delta_band': 5.0, 'carry_pct': 0.15},  # ES: ~3.2GW Ã— 0.3 Ã— 744h â‰ˆ 18 GWh/Monat
        'PL': {'budget_mwh': 8500, 'delta_band': 4.0, 'carry_pct': 0.10},   # PL: ~1.5GW Ã— 0.3 Ã— 744h â‰ˆ 8.5 GWh/Monat
        # Limited reservoir zones
        'DE_LU': {'budget_mwh': 6500, 'delta_band': 3.0, 'carry_pct': 0.10}, # DE+LU: ~1.2GW Ã— 0.25 Ã— 744h â‰ˆ 6.5 GWh/Monat
        'BE': {'budget_mwh': 2500, 'delta_band': 3.0, 'carry_pct': 0.05},   # BE: ~0.45GW Ã— 0.3 Ã— 744h â‰ˆ 2.5 GWh/Monat (Coo)
        'NL': {'budget_mwh': 1200, 'delta_band': 2.5, 'carry_pct': 0.05},   # NL: Minimal reservoir
        'CZ': {'budget_mwh': 4800, 'delta_band': 3.5, 'carry_pct': 0.10},   # CZ: ~0.9GW Ã— 0.25 Ã— 744h â‰ˆ 4.8 GWh/Monat
        'DK1': {'budget_mwh': 500, 'delta_band': 2.0, 'carry_pct': 0.05},   # DK1: Minimal
        'DK2': {'budget_mwh': 500, 'delta_band': 2.0, 'carry_pct': 0.05},   # DK2: Minimal
        'default': {'budget_mwh': 15000, 'delta_band': 5.0, 'carry_pct': 0.15}  # Fallback fÃ¼r unbekannte Zonen
    }
    
    for z in zones:
        # Parameter fÃ¼r Zone holen
        params = reservoir_budget_params.get(z, reservoir_budget_params['default'])
        
        try:
            # Neues Energie-Budget System verwenden
            sp = energy_budget_reservoir_prices(
                nei_prices, z,
                reservoir_budget_mwh_per_month=params['budget_mwh'],
                epsilon_coupling=0.075,  # 7.5 cent Kopplung-Toleranz
                delta_band=params['delta_band'],
                smoothing_days=7,
                min_water_value=30.0,
                max_water_value=180.0
            )
            print(f"[HYDRO] {z}: Energie-Budget System aktiviert (Budget={params['budget_mwh']}MWh/Monat, Band=Â±{params['delta_band']}EUR/MWh)")
        except Exception as e:
            print(f"[HYDRO] {z}: Fallback auf altes System - {e}")
            # Fallback auf altes System
            sp = reservoir_shadow_price_series(nei_prices, z)
        
        for tt, val in sp.items():
            v = float(val) if pd.notna(val) else 0.0
            reservoir_sp_map[(z, tt)] = v

    # 5) Nachbar-Gen/Load + DE/LU
    load_by_zone, gen_by_zone = {}, {}
    load_errors: Dict[str, Exception] = {}
    gen_errors: Dict[str, Exception] = {}
    for z in zones:
        try:
            load_by_zone[z] = load_neighbor_load(args.neighbor_load_dir, z)
        except Exception as exc:
            load_errors[z] = exc
            print(f"[WARN] load_{z}: {exc}")
        try:
            gen_by_zone[z] = load_neighbor_gen(args.neighbor_gen_dir, z)
        except Exception as exc:
            gen_errors[z] = exc
            print(f"[WARN] gen_{z}: {exc}")

    if "DE_LU" not in load_by_zone:
        raise RuntimeError(f"load_DE_LU_2024.csv fehlt oder konnte nicht gelesen werden: {load_errors.get('DE_LU', 'unbekannter Fehler')}")
    if "DE_LU" not in gen_by_zone:
        raise RuntimeError(f"actual_gen_DE_LU_2024.csv fehlt oder konnte nicht gelesen werden: {gen_errors.get('DE_LU', 'unbekannter Fehler')}")
    de_load = load_by_zone["DE_LU"]; de_gen = gen_by_zone["DE_LU"]
    de_load, de_gen = robustize_load_gen(de_load, de_gen)

    # Preflight NaN scan: inspect neighbor gen/load and key inputs and fill harmless NaNs
    try:
        preflight_inputs = {
            'neighbor_gen': gen_by_zone,
            'neighbor_load': load_by_zone,
            'fuel_prices': fuel_prices,
            'fleet': fleet_all,
        }
        pf_report = preflight_check_and_fill_nans(preflight_inputs, auto_fill=True)
        if pf_report and (pf_report.get('filled') or pf_report.get('checked')):
            print(f"[PRELIGHT] NaN preflight report: checked={len(pf_report.get('checked', {}))} sections, filled={len(pf_report.get('filled', {}))}")
            # Optionally show details for debugging
            for k, v in pf_report.get('filled', {}).items():
                print(f"[PRELIGHT] Filled in {k}: {v}")
    except Exception as e:
        print(f"[WARN] Preflight NaN check failed: {e}")

    # Defer lignite profile calculation until after SRMC alignment
    auto_lignite_profile = None  # Will be set after SRMC matrix is created
    cap_total_lignite = 0.0  # Will be set after SRMC matrix is created
    if auto_lignite_profile is not None:
        auto_lignite_profile = auto_lignite_profile.reindex(de_gen.index).fillna(0.0)
    else:
        auto_lignite_profile = pd.Series(0.0, index=de_gen.index, dtype='float64')
    auto_lignite_share = None
    if cap_total_lignite > 1e-3:
        share_series = auto_lignite_profile / cap_total_lignite
        share_series = share_series[np.isfinite(share_series) & (share_series > 0)]
        if not share_series.empty:
            auto_lignite_share = float(np.clip(np.nanmedian(share_series), 0.0, 1.0))
    if auto_lignite_share is not None:
        prev_share = float(getattr(args, 'de_mustrun_coal_share', 0.0) or 0.0)
        if auto_lignite_share > prev_share + 1e-6:
            args.de_mustrun_coal_share = auto_lignite_share
        setattr(args, '_auto_de_lignite_mustrun_share', auto_lignite_share)
       

    # Temporarily defer oil mask and profile calculation until ef_keys is available
    oil_mask = None
    cap_total_oil = 0.0  # Will be updated after SRMC creation
    auto_oil_profile = pd.Series(0.0, index=de_gen.index, dtype='float64')  # Placeholder
    auto_oil_share = None
    if cap_total_oil > 1e-3 and not auto_oil_profile.empty:
        share_series_oil = auto_oil_profile / cap_total_oil
        share_series_oil = share_series_oil[np.isfinite(share_series_oil) & (share_series_oil > 0)]
        if not share_series_oil.empty:
            auto_oil_share = float(np.clip(np.nanmedian(share_series_oil), 0.0, 1.0))
    if auto_oil_share is not None:
        setattr(args, '_auto_de_oil_mustrun_share', auto_oil_share)
      

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


    # 7) CRITICAL FIX: Expliziter Voll-Index statt Schnittmenge
    
    # Debug: PrÃ¼fe einzelne Zeitreihen-LÃ¤ngen BEVOR idx_common
    print("[DEBUG] Zeitreihen-LÃ¤ngen vor Index-Intersection:")
    for name, df in [("DE_load", de_load), ("fuel_prices", fuel_prices), 
                     ("flows", flows), ("nei_prices", nei_prices)]:
        idx = df.index
        print(f"  {name}: {len(idx)} Stunden, {idx.min()} bis {idx.max()}")
    
    # Erstelle expliziten 8784h-Index fÃ¼r 2024 (Schaltjahr)
    idx_full = pd.date_range(f"2024-01-01 00:00", f"2024-12-31 23:00", tz=TZ, freq="1h")
    print(f"[INFO] Voll-Index 2024: {len(idx_full)} Stunden (00:00-23:00)")
    
    def _reindex_strict(df, name):
        """Reindexiere auf Voll-Index mit minimaler Interpolation und Memory-Optimierung"""
        if isinstance(df, pd.Series):
            out = df.reindex(idx_full)
            missing_count = out.isna().sum()
        else:
            out = df.reindex(idx_full)
            missing_count = out.isna().sum().sum()
        
        if missing_count > 0:
            print(f"[WARN] {name}: {missing_count} fehlende Werte, interpoliere (max 3h LÃ¼cken)")
            # Kurze LÃ¼cken interpolieren: max 3h
            out = out.interpolate(limit=3).ffill(limit=1).bfill(limit=1)
            
        # Memory optimization: convert to float32 for numerical columns
        if isinstance(out, pd.Series):
            if out.dtype in ['float64', 'int64']:
                out = out.astype('float32')
        else:
            for col in out.columns:
                if out[col].dtype in ['float64', 'int64']:
                    out[col] = out[col].astype('float32')
            
        # Harte Validierung: Muss exakt 8784h haben
        if isinstance(out, pd.Series):
            validate_hourly_index(out.to_frame("temp"), name, expected_hours=8784)
        else:
            validate_hourly_index(out, name, expected_hours=8784)
        return out
    
    # Alle Kernreihen auf Voll-Index zwingen
    de_load = _reindex_strict(de_load, "DE_load_reindexed")
    fuel_prices = _reindex_strict(fuel_prices, "fuel_prices_reindexed") 
    flows = _reindex_strict(flows, "flows_reindexed")
    nei_prices = _reindex_strict(nei_prices, "nei_prices_reindexed")
    
    # Zeitfenster: Jetzt deterministisch von Voll-Index
    def _to_berlin(ts_str: Optional[str]):
        if ts_str is None: return None
        ts = pd.Timestamp(ts_str)
        return ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
    start = _to_berlin(args.start) or idx_full[0]
    end   = _to_berlin(args.end)   or (idx_full[-1] + pd.Timedelta(hours=1))
    idx   = [t for t in idx_full if (t >= start and t < end)]
    print(f"[INFO] Stunden im Lauf: {len(idx)} | Fenster: {start} .. {end} (exkl.)")
    idx_dt = pd.DatetimeIndex(idx)

    # Memory-optimized SRMC matrix creation using chunking
    def create_srmc_matrix_chunked(srmc_by_unit, units, idx_dt, chunk_size_hours=168):
        """Create SRMC matrix in chunks to avoid memory overflow"""
        total_hours = len(idx_dt)
        print(f"[INFO] Creating SRMC matrix in chunks for {total_hours} hours, chunk_size={chunk_size_hours}")
        
        if total_hours <= chunk_size_hours:
            # Small enough - create directly
            SRMC = pd.concat([srmc_by_unit[u].rename(u).reindex(idx_dt) for u in units], axis=1).astype("float32")
            print(f"[INFO] SRMC matrix created directly: {SRMC.shape}")
            return SRMC
        
        # Large dataset - process in chunks
        chunk_dfs = []
        for start_idx in range(0, total_hours, chunk_size_hours):
            end_idx = min(start_idx + chunk_size_hours, total_hours)
            chunk_dt = idx_dt[start_idx:end_idx]
            
            print(f"[INFO] Processing SRMC chunk {start_idx//chunk_size_hours + 1}/{(total_hours-1)//chunk_size_hours + 1}: hours {start_idx}-{end_idx-1}")
            
            # Create chunk matrix
            chunk_matrix = pd.concat(
                [srmc_by_unit[u].rename(u).reindex(chunk_dt) for u in units], 
                axis=1
            ).astype("float32")
            
            chunk_dfs.append(chunk_matrix)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
        
        # Combine all chunks
        print(f"[INFO] Combining {len(chunk_dfs)} SRMC chunks...")
        SRMC = pd.concat(chunk_dfs, axis=0)
        print(f"[INFO] Final SRMC matrix shape: {SRMC.shape}")
        
        # Final garbage collection
        import gc
        del chunk_dfs
        gc.collect()
        
        return SRMC
    
    # Use chunked SRMC creation
    SRMC = create_srmc_matrix_chunked(srmc_by_unit, units, idx_dt)
    
    # Align SRMC matrix and fleet data
    common = [u for u in SRMC.columns if u in fleet_idxed.index]
    SRMC = SRMC.loc[:, common]; fleet_idxed = fleet_idxed.loc[common]
    units    = list(SRMC.columns)
    cap_base = fleet_idxed["available_mw"].astype("float32").to_numpy()
    eta_arr  = fleet_idxed["eta"].astype("float32").to_numpy()
    ef_keys  = fleet_idxed["ef_key"].astype(str).to_numpy()
    
    # Now calculate lignite profile with aligned data
    _tmp_lignite_profile = price_based_lignite_mustrun_profile(
        de_gen=de_gen,
        price_series=nei_prices['price_DE_LU'] if 'price_DE_LU' in nei_prices.columns else None,
        cap_base=cap_base,
        ef_keys=ef_keys,
        price_floor=float(getattr(args, 'lignite_price_floor', 20.0)),
        min_hours=int(getattr(args, 'lignite_price_min_hours', 3)),
        window_hours=int(getattr(args, 'lignite_price_window', 6)),
    )
    # Ensure we never keep a None here (mirror oil-profile handling)
    if _tmp_lignite_profile is not None:
        auto_lignite_profile = _tmp_lignite_profile.reindex(de_gen.index).fillna(0.0)
    else:
        auto_lignite_profile = pd.Series(0.0, index=de_gen.index, dtype='float64')
    cap_total_lignite = float(cap_base[ef_keys == 'Braunkohle'].sum())

    # Now calculate oil profile with aligned data
    oil_mask = np.isin(ef_keys, ['HeizÃ¶l schwer', 'HeizÃ¶l leicht / Diesel', 'Fossil Oil'])
    cap_total_oil = float(cap_base[oil_mask].sum()) if oil_mask.any() else 0.0
    auto_oil_profile = price_based_oil_mustrun_profile(
        de_gen=de_gen,
        price_series=nei_prices['price_DE_LU'] if 'price_DE_LU' in nei_prices.columns else None,
        cap_base=cap_base,
        ef_keys=ef_keys,
        price_floor=float(getattr(args, 'oil_price_floor', 20.0)),
        min_hours=int(getattr(args, 'oil_price_min_hours', 3)),
        window_hours=int(getattr(args, 'oil_price_window', 6)),
    )
    # Respect user flag: dynamic oil must-run heuristic disabled by default to avoid
    # spurious cheap oil MU pushing into export stack (was causing import-side artifacts).
    if not bool(getattr(args, 'enable_oil_mustrun', False)):
        auto_oil_profile = pd.Series(0.0, index=de_gen.index, dtype='float64')
    else:
        if auto_oil_profile is not None:
            auto_oil_profile = auto_oil_profile.reindex(de_gen.index).fillna(0.0)
        else:
            auto_oil_profile = pd.Series(0.0, index=de_gen.index, dtype='float64')

    # 8) Nicht-disponible in DE (nur fÃ¼r Checks; PSP-Gen NICHT abziehen)
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

            fleet_df=fleet_all,   # fÃ¼r Î·-Median je Fuel
        )
    else:
        de_min_total, de_min_by_fuel = compute_fossil_min_profiles(
            gen_df=de_gen.reindex(de_load.index),
            fuels_select=[s.strip() for s in str(args.fossil_mustrun_fuels).split(",") if s.strip()],
            peak_hours=args.mustrun_peak_hours,
            mode=args.fossil_mustrun_mode,
            q=float(args.fossil_mustrun_q),
        )

    de_min_total_idx = (
        de_min_total.reindex(idx_dt, fill_value=0.0)
        if isinstance(de_min_total, pd.Series) and not de_min_total.empty
        else pd.Series(0.0, index=idx_dt, dtype="float64")
    )

    de_min_by_fuel_idx: Dict[str, pd.Series] = {}
    fossil_keys = set(fossil_list)
    if isinstance(de_min_by_fuel, dict):
        fossil_keys.update(de_min_by_fuel.keys())
    for f in sorted(fossil_keys):
        series = de_min_by_fuel.get(f) if isinstance(de_min_by_fuel, dict) else None
        if isinstance(series, pd.Series) and not series.empty:
            de_min_by_fuel_idx[f] = series.reindex(idx_dt, fill_value=0.0).astype(float)
        else:
            de_min_by_fuel_idx[f] = pd.Series(0.0, index=idx_dt, dtype="float64")

    share_based_min_by_fuel: Dict[str, float] = {}
    de_mustrun_shares = fossil_mustrun_shares_for_DE(args)
    if any(share > 0.0 for share in de_mustrun_shares.values()):
        avail_factor = float(getattr(args, "therm_avail", 1.0))
        for ef_name, share in de_mustrun_shares.items():
            if share <= 0.0:
                continue
            mask_f = (ef_keys == ef_name)
            if not np.any(mask_f):
                continue
            cap_ef = float((cap_base[mask_f] * avail_factor).sum())
            share_based_min_by_fuel[ef_name] = share * cap_ef

    # 10) Lignite-MU Profil (falls aktiviert)
    h_start, h_end = [int(x) for x in args.mustrun_peak_hours.split("-")]
    def is_peak(ix: pd.DatetimeIndex):
        return ((ix.hour >= h_start) & (ix.hour < h_end)) if h_start <= h_end else ((ix.hour >= h_start) | (ix.hour < h_end))

    lign_profile = pd.Series(0.0, index=idx_dt, dtype='float64')
    auto_lignite_profile_idx = auto_lignite_profile.reindex(idx_dt).fillna(0.0)
    auto_oil_profile_idx = auto_oil_profile.reindex(idx_dt).fillna(0.0)
    if args.mustrun_mode == "gen_quantile" and "Fossil Brown coal/Lignite" in de_gen.columns:
        lign = de_gen["Fossil Brown coal/Lignite"].reindex(idx_dt)
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
    lign_profile = pd.Series(np.maximum(lign_profile.to_numpy(), auto_lignite_profile_idx.to_numpy()), index=idx_dt)

    # Lignite HilfsgrÃ¶ÃŸen
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

    # 11) Nachbar-Î·-Parameter / KapazitÃ¤tsmaske
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
    if cap_mask:
        cap_mask = {(_resolve_zone_name(z, alias_map), fuel): value for (z, fuel), value in cap_mask.items()}
    nei_dists = {_resolve_zone_name(k, alias_map): v for k, v in nei_dists.items()}


    # 12) MU-Bid Funktion vorbereiten (rolling/default)
    mu_bid_fn_global = precompute_mu_bid_fn(nei_prices, fuel_prices, args, zones)

    # 12.5) Enhanced Realistic Reservoir-Budget Parameter fÃ¼r Band-System (kopiert von oben)
    reservoir_budget_params = {
        # Major Alpine zones with large reservoir capacities
        'AT': {'budget_mwh': 45000, 'delta_band': 5.0, 'carry_pct': 0.20},  # AT: ~6GW Ã— 0.25 Ã— 744h â‰ˆ 45 GWh/Monat
        'CH': {'budget_mwh': 85000, 'delta_band': 6.0, 'carry_pct': 0.20},  # CH: ~15GW Ã— 0.3 Ã— 744h â‰ˆ 85 GWh/Monat (Alpenspeicher)
        'NO2': {'budget_mwh': 120000, 'delta_band': 7.0, 'carry_pct': 0.25}, # NO2: ~20GW Ã— 0.25 Ã— 744h â‰ˆ 120 GWh/Monat (Fjordspeicher)
        'SE4': {'budget_mwh': 95000, 'delta_band': 6.0, 'carry_pct': 0.20},  # SE4: ~16GW Ã— 0.25 Ã— 744h â‰ˆ 95 GWh/Monat
        # Moderate reservoir zones
        'FR': {'budget_mwh': 28000, 'delta_band': 4.0, 'carry_pct': 0.15},  # FR: ~5GW Ã— 0.3 Ã— 744h â‰ˆ 28 GWh/Monat (PyrenÃ¤en/Alpen)
        'IT': {'budget_mwh': 22000, 'delta_band': 4.5, 'carry_pct': 0.15},  # IT: ~4GW Ã— 0.3 Ã— 744h â‰ˆ 22 GWh/Monat
        'ES': {'budget_mwh': 18000, 'delta_band': 5.0, 'carry_pct': 0.15},  # ES: ~3.2GW Ã— 0.3 Ã— 744h â‰ˆ 18 GWh/Monat
        'PL': {'budget_mwh': 8500, 'delta_band': 4.0, 'carry_pct': 0.10},   # PL: ~1.5GW Ã— 0.3 Ã— 744h â‰ˆ 8.5 GWh/Monat
        # Limited reservoir zones
        'DE_LU': {'budget_mwh': 6500, 'delta_band': 3.0, 'carry_pct': 0.10}, # DE+LU: ~1.2GW Ã— 0.25 Ã— 744h â‰ˆ 6.5 GWh/Monat
        'BE': {'budget_mwh': 2500, 'delta_band': 3.0, 'carry_pct': 0.05},   # BE: ~0.45GW Ã— 0.3 Ã— 744h â‰ˆ 2.5 GWh/Monat (Coo)
        'NL': {'budget_mwh': 1200, 'delta_band': 2.5, 'carry_pct': 0.05},   # NL: Minimal reservoir
        'CZ': {'budget_mwh': 4800, 'delta_band': 3.5, 'carry_pct': 0.10},   # CZ: ~0.9GW Ã— 0.25 Ã— 744h â‰ˆ 4.8 GWh/Monat
        'DK1': {'budget_mwh': 500, 'delta_band': 2.0, 'carry_pct': 0.05},   # DK1: Minimal
        'DK2': {'budget_mwh': 500, 'delta_band': 2.0, 'carry_pct': 0.05},   # DK2: Minimal
        'default': {'budget_mwh': 15000, 'delta_band': 5.0, 'carry_pct': 0.15}  # Fallback fÃ¼r unbekannte Zonen
    }

    # 13) Hauptschleife
    results, debug_rows = [], []
    imp_cols = [c for c in flows.columns if c.startswith("imp_") and c != "net_import_total"]
    imp_to_zone = {c: _resolve_zone_name(c.replace("imp_", ""), alias_map) for c in imp_cols}

    # Memory management: process with periodic garbage collection
    total_hours = len(idx)
    print(f"[INFO] Processing {total_hours} hours with memory optimization")

    for i, t in enumerate(idx):
        L  = float(de_load.get(t, np.nan))
        ND = float(de_nondisp.get(t, 0.0))
        if not np.isfinite(L): continue

        lignite_lowprice_mw = float(auto_lignite_profile_idx.get(t, 0.0))
        oil_lowprice_mw = float(auto_oil_profile_idx.get(t, 0.0))

        # JAO FlowBased Boundary Flag
        fb_boundary_flag = False
        fb_slack_min, fb_slack_max = np.nan, np.nan
        fb_min_np, fb_max_np, fb_net_pos = np.nan, np.nan, np.nan
        if fb_data is not None:
            # Convert local time to UTC for FB lookup
            t_utc = t.tz_convert('UTC') if hasattr(t, 'tz_convert') else pd.to_datetime(t, utc=True)
            fb_match = fb_data[fb_data['timestamp_utc'] == t_utc]
            if not fb_match.empty:
                fb_row = fb_match.iloc[0]
                fb_boundary_flag = bool(fb_row['fb_boundary'])
                fb_slack_min = safe_float(fb_row['slack_to_min'])
                fb_slack_max = safe_float(fb_row['slack_to_max'])
                fb_min_np = safe_float(fb_row['minNP'])
                fb_max_np = safe_float(fb_row['maxNP'])
                fb_net_pos = safe_float(fb_row['NetPosition'])

        # FlÃ¼sse
        net_imp = float(flows.loc[t, "net_import_total"]) if t in flows.index else 0.0
        pos_imp_total = 0.0
        pos_imp_by_zone = {}
        neg_exp_total = 0.0
        neg_exp_by_zone = {}
        
        # JAO FlowBased Boundary Flags (default values)
        fb_boundary_flag = False
        fb_slack_min = np.nan
        fb_slack_max = np.nan
        fb_min_np = np.nan
        fb_max_np = np.nan
        fb_net_pos = np.nan
        
        # Load JAO FB data for this timestamp if available
        if fb_data is not None:
            t_utc = t.tz_convert('UTC')
            fb_mask = (fb_data['timestamp_utc'] == t_utc)
            if fb_mask.any():
                fb_row = fb_data[fb_mask].iloc[0]
                fb_boundary_flag = bool(fb_row.get('fb_boundary', False))
                fb_slack_min = safe_float(fb_row.get('slack_to_min', np.nan))
                fb_slack_max = safe_float(fb_row.get('slack_to_max', np.nan))
                fb_min_np = safe_float(fb_row.get('minNP', np.nan))
                fb_max_np = safe_float(fb_row.get('maxNP', np.nan))
                fb_net_pos = safe_float(fb_row.get('NetPosition', np.nan))
        ambiv_masks = getattr(args, '_flow_ambiv_mask', {})
        ambiv_zones_active: list[str] = []
        if imp_cols and (t in flows.index):
            coupled_zones_t = clusters.get(t, [])
            for c in imp_cols:
                if c not in flows.columns:
                    continue
                z = imp_to_zone[c]
                val = flows.at[t, c]
                if not np.isfinite(val):
                    continue
                if coupled_zones_t and (z not in coupled_zones_t):
                    continue
                ambiv_series = ambiv_masks.get(z)
                if ambiv_series is not None:
                    ambiv_flag = bool(ambiv_series.reindex([t], fill_value=False).iloc[0])
                else:
                    ambiv_flag = False
                if ambiv_flag:
                    ambiv_zones_active.append(z)
                    continue
                if val > 0:
                    val_f = float(val)
                    pos_imp_total += val_f
                    pos_imp_by_zone[z] = pos_imp_by_zone.get(z, 0.0) + val_f
                elif val < 0:
                    exp_mw = float(-val)
                    neg_exp_total += exp_mw
                    neg_exp_by_zone[z] = neg_exp_by_zone.get(z, 0.0) + exp_mw

        import_flow_target = pos_imp_total if bool(getattr(args, "coupled_import_anyflow", True)) else max(net_imp, 0.0)
        import_relevant = import_flow_target > 1e-6
        export_flow_target = neg_exp_total if neg_exp_total > 0.0 else max(-net_imp, 0.0)
        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan
        fp_row = fuel_prices.loc[t] if t in fuel_prices.index else None

        import_stack_volume_used = 0.0
        export_stack_volume_used = 0.0
        import_zero_thresh = float(getattr(args, "import_zero_price_threshold", 10.0))

        # --------- RL-Leiter (DE) nach gewÃ¼nschter Reihenfolge ----------
        # NEU: Must-Run vor EE abziehen
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

        # Fossile MU (datengetrieben + optionale Shares als Floor) - VORBERECHNUNG
        oil_mu_t = float(auto_oil_profile_idx.get(t, 0.0))
        de_fossil_mu_cost = float(de_min_total_idx.get(t, 0.0)) if isinstance(de_min_total_idx, pd.Series) else 0.0
        mustrun_de_fuel_req: Dict[str, float] = {}
        for f, series in de_min_by_fuel_idx.items():
            mustrun_de_fuel_req[f] = float(series.get(t, 0.0))
        for f, floor_val in share_based_min_by_fuel.items():
            prev = mustrun_de_fuel_req.get(f, 0.0)
            mustrun_de_fuel_req[f] = max(prev, float(floor_val))
        
        # Oil Must-Run Berechnung
        oil_keys = [key for key in mustrun_de_fuel_req if "HeizÃ¶l" in key or "Fossil Oil" in key]
        baseline_oil_req = sum(mustrun_de_fuel_req.get(k, 0.0) for k in oil_keys)
        oil_mustrun_req = float(max(baseline_oil_req, oil_mu_t))
        if oil_keys:
            if baseline_oil_req > 1e-6 and oil_mustrun_req > 0.0:
                scale = oil_mustrun_req / max(baseline_oil_req, 1e-6)
                for k in oil_keys:
                    mustrun_de_fuel_req[k] = mustrun_de_fuel_req[k] * scale
            elif oil_mustrun_req > 0.0:
                primary_key = oil_keys[0]
                mustrun_de_fuel_req[primary_key] = oil_mustrun_req
                for extra_key in oil_keys[1:]:
                    mustrun_de_fuel_req[extra_key] = 0.0
        
        # Fossiler Must-Run (ohne Oil)
        non_oil_req = sum(
            mustrun_de_fuel_req.get(k, 0.0) for k in mustrun_de_fuel_req if k not in oil_keys
        )
        fossil_mu_req_total = float(max(non_oil_req, 0.0))
        mustrun_de_total = float(oil_mustrun_req + fossil_mu_req_total)

        # EINHEITLICHE RL-BERECHNUNG verwenden statt separate Logik
        # Erstelle konsistente min_by_fuel_zone_t für Deutschland
        min_by_fuel_de = {
            "Müll (nicht biogen)": waste_t,
            "Biomasse": bio_mu_de,
            "Kernenergie": nuc_mu_de,
            "Heizöl schwer": oil_mustrun_req,
            "Heizöl leicht / Diesel": 0.0,
            "Fossil Oil": 0.0
        }
        
        # Verwende einheitliche RL-Leiter
        de_gen_row = de_gen.loc[t] if t in de_gen.index else pd.Series(dtype=float)
        rl_ladder = compute_residual_load_ladder(de_gen_row, L, min_by_fuel_de, mustrun_de_total, args)
        
        # Konsistente RL-Werte aus einheitlicher Berechnung
        RL0, RL1, RL2, RL3, RL4, RL5, RL6, RL7, RL8 = (
            rl_ladder['RL0'], rl_ladder['RL1'], rl_ladder['RL2'], rl_ladder['RL3'], 
            rl_ladder['RL4'], rl_ladder['RL5'], rl_ladder['RL6'], rl_ladder['RL7'], rl_ladder['RL8']
        )
        
        # Extrahiere takes für spätere Verwendung
        takes = rl_ladder['takes']
        take_waste_mu, take_nuc_mu, take_bio_mu, take_oil_mu, take_mu_foss, take_psp, take_res = (
            takes['waste_mu'], takes['nuc_mu'], takes['bio_mu'], takes['oil_mu'], 
            takes['mu_foss'], takes['psp'], takes['res']
        )
        
        # DISABLED: NEU: Nachbarländer-Auffüllung wenn nicht preisgekoppelt aber günstiger
        # PROBLEM: Dies führt zu doppelter Berücksichtigung in der Merit-Order 
        # (einmal price-demand, einmal energy-demand)
        neighbor_fillup_used = 0.0
        RL9 = RL8  # Start with remaining demand after DE reservoir
        
        # COMMENTED OUT due to methodological double-counting issue
        # if RL8 > 1e-6:  # Nur wenn noch Restlast vorhanden
        #     for z in zones:
        #         if z == "DE_LU":
        #             continue
        #             
        #         # Preis vergleichen
        # COMMENTED OUT due to methodological double-counting issue
        #if RL8 > 1e-6:  # Nur wenn noch Restlast vorhanden
        #    for z in zones:
        #        if z == "DE_LU":
        #            continue
        #            
        #        # Preis vergleichen
        #        p_de = float(nei_prices.loc[t, "price_DE_LU"]) if t in nei_prices.index else np.nan
        #        price_col = f"price_{z}"
        #        if price_col in nei_prices.columns and t in nei_prices.index:
        #            p_neighbor = float(nei_prices.loc[t, price_col])
        #            
        #            # Prüfen ob Zone preisgekoppelt ist
        #            cluster_all = clusters.get(t, ["DE_LU"])
        #            coupled_neighbors = [z_c for z_c in cluster_all if z_c != "DE_LU"]
        #            is_coupled = z in coupled_neighbors
        #            
        #            if not is_coupled and np.isfinite(p_neighbor) and np.isfinite(p_de) and p_neighbor < p_de:
        #                # Verfügbare Export-Kapazität des Nachbarn prüfen
        #                if z in gen_by_zone and t in gen_by_zone[z].index:
        #                    gen_row = gen_by_zone[z].loc[t]
        #                    load_z = float(load_by_zone[z].get(t, np.nan)) if z in load_by_zone else np.nan
        #                    
        #                    if np.isfinite(load_z):
        #                        # Vereinfachte Berechnung: Gesamt-Generation minus Last
        #                        total_gen_z = float(gen_row.sum()) if not gen_row.empty else 0.0
        #                        export_potential = max(total_gen_z - load_z, 0.0)
        #                        
        #                        # PATCH 2: Verfügbare Import-Kapazität von Flows richtig lesen
        #                        import_capacity = 0.0
        #                        col = f"imp_{z}"
        #                        if col in flows.columns and t in flows.index:
        #                            val = flows.at[t, col]
        #                            if np.isfinite(val) and val > 0:
        #                                import_capacity += float(val)
        #                        
        #                        # Nutzbaren Auffüll-Betrag bestimmen
        #                        available_fillup = min(export_potential, import_capacity, RL9)
        #                        
        #                        if available_fillup > 1e-6:
        #                            neighbor_fillup_used += available_fillup
        #                            RL9 = max(RL9 - available_fillup, 0.0)
        #                            # Debug-Output optional aktivieren
        #                            # print(f"[DEBUG] {t}: Nachbar-Auffüllung {z}: {available_fillup:.1f}MW (P_neighbor={p_neighbor:.1f} < P_DE={p_de:.1f})")
        #                            
        #                            if RL9 <= 1e-6:  # Alle Restlast abgedeckt
        #                                break
        
        # PATCH 1: Preis-Nachfrage getrennt halten
        # D_price = Preis-Nachfrage (RL8 nach DE-Stufen, ohne non-coupled Fill-up)
        D_price = RL8
        
        # RL_energy = Energie-Bedarfsrechnung (könnte noch non-coupled imports enthalten)
        RL_energy = RL8  # - neighbor_fillup_used  # (non-coupled fillup aktuell deaktiviert)
        
        # Final residual für Merit-Order: nur D_price verwenden
        RL_final = D_price
        
        # Signifikanzregel fÃ¼r Import-Relevanz (mit RL_final)
        for f, series in de_min_by_fuel_idx.items():
            mustrun_de_fuel_req[f] = float(series.get(t, 0.0))
        for f, floor_val in share_based_min_by_fuel.items():
            prev = mustrun_de_fuel_req.get(f, 0.0)
            mustrun_de_fuel_req[f] = max(prev, float(floor_val))
        oil_keys = [key for key in mustrun_de_fuel_req if "HeizÃ¶l" in key or "Fossil Oil" in key]
        baseline_oil_req = sum(mustrun_de_fuel_req.get(k, 0.0) for k in oil_keys)
        oil_mustrun_req = float(max(baseline_oil_req, oil_mu_t))
        if oil_keys:
            if baseline_oil_req > 1e-6 and oil_mustrun_req > 0.0:
                scale = oil_mustrun_req / max(baseline_oil_req, 1e-6)
                for k in oil_keys:
                    mustrun_de_fuel_req[k] = mustrun_de_fuel_req[k] * scale
            elif oil_mustrun_req > 0.0:
                primary_key = oil_keys[0]
                mustrun_de_fuel_req[primary_key] = oil_mustrun_req
                for extra_key in oil_keys[1:]:
                    mustrun_de_fuel_req[extra_key] = 0.0
        non_oil_req = sum(
            mustrun_de_fuel_req.get(k, 0.0) for k in mustrun_de_fuel_req if k not in oil_keys
        )
        fossil_mu_req_total = float(max(non_oil_req, 0.0))
        mustrun_de_total = float(oil_mustrun_req + fossil_mu_req_total)
        # REMOVED: Doppelte RL-Berechnung entfernt - verwende bereits berechnete Werte aus rl_ladder
        # Die RL-Werte (RL0-RL8) und takes sind bereits korrekt in der einheitlichen RL-Ladder berechnet

        # Flex (DE) â€“ PSP / Reservoir
        psp_avail = float(de_gen.get("Hydro Pumped Storage", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Hydro Pumped Storage" in de_gen.columns else 0.0
        psp_avail = max(psp_avail, float(getattr(args, "psp_min_avail_mw", 0.0)))

        de_reservoir_avail = float(de_gen.get("Hydro Water Reservoir", pd.Series(0.0, index=de_gen.index)).reindex([t]).fillna(0.0).iloc[0]) if "Hydro Water Reservoir" in de_gen.columns else 0.0
        de_reservoir_avail = max(de_reservoir_avail, float(getattr(args, "reservoir_min_avail_mw", 0.0)))
        # Signifikanzregel fÃ¼r Import-Relevanz (Ã¼berschreibt anyflow)
        sum_import_cap = pos_imp_total
        import_relevant_signif = (sum_import_cap > 0.0) and (sum_import_cap >= max(0.05 * RL_final, 200.0))
        import_relevant = import_relevant_signif
        # Default: Markierungen
        marginal_side  = None
        marginal_label = None
        marginal_fuel  = None
        marginal_eta   = np.nan
        marginal_srmc  = np.nan
        mef_gpkwh      = np.nan
        DEBUG_rule_psp_price = False
        DEBUG_reservoir_band = ""
        
        # Initialize flag variables for df_res
        flag_psp_price_setting = 0
        flag_psp_scarcity_cap = 0
        flag_reservoir_enhanced = 0
        
        # Initialize hydro_reservoir_marginal variable
        hydro_reservoir_marginal = False

        # Preis-Cluster
        cluster_all = clusters.get(t, ["DE_LU"])
        coupled_neighbors = [z for z in cluster_all if z != "DE_LU"]
        coupling_active = len(coupled_neighbors) > 0

        # PSP Candidate Gate (einheitlich in 2024 & 2030)
        enable_psp_gate = True  # Can be made configurable if needed
        if (RL6 > 0.0) and (psp_avail >= RL6 - 1e-6) and np.isfinite(p_de) and enable_psp_gate:
            price_t = float(p_de)
            w_t = psp_water_value(nei_prices["price_DE_LU"], t,
                                window_h=args.psp_pump_window_h,
                                rte=args.psp_rt_eff,
                                floor=float(getattr(args, "psp_srmc_floor_eur_mwh", 60.0)))
            
            # PSP preissetzend akzeptieren mit dynamischem Price Cap fÃ¼r Scarcity Hours
            # 1. Dynamic price cap: 80EUR normal, 150EUR for scarcity hours (price > 100EUR)
            dynamic_price_cap = args.psp_price_cap if price_t <= 100.0 else 150.0
            # 2. price in [w_t - psp_accept_band, w_t + psp_accept_band] 
            # 3. PSP is discharging
            if (price_t <= dynamic_price_cap) and (abs(price_t - w_t) <= args.psp_accept_band):
                if psp_is_discharging(t, de_gen, threshold_mw=10.0):
                    marginal_side  = "DE"
                    marginal_label = "DE_flex_psp_price_setting"
                    marginal_fuel  = "Hydro Pumped Storage"
                    marginal_eta   = np.nan
                    marginal_srmc  = float(max(0.0, min(price_t, w_t)))
                    mef_gpkwh      = 0.0
                    DEBUG_rule_psp_price = True
                    flag_psp_price_setting = 1
                    # Log dynamic cap usage
                    if dynamic_price_cap > args.psp_price_cap:
                        flag_psp_scarcity_cap = 1

        # Enhanced Reservoir-Hydro with improved water value system
        if (RL7 > 0.0) and (de_reservoir_avail >= RL7 - 1e-6) and np.isfinite(p_de):
            # Enhanced water value calculation with seasonal/time-of-day adjustment
            base_water_value = reservoir_sp_map.get(("DE_LU", t), float(np.clip(p_de, 0.0, float(getattr(args, 'reservoir_max_clip', 300.0)))))
            
            # Time-of-day adjustment for water value (higher in evening peak hours)
            hour = t.hour
            seasonal_factor = 1.0
            if hour >= 18 and hour <= 21:  # Evening peak
                seasonal_factor = 1.15
            elif hour >= 7 and hour <= 9:  # Morning peak  
                seasonal_factor = 1.08
            elif hour >= 0 and hour <= 5:  # Night hours (lower value)
                seasonal_factor = 0.92
                
            reservoir_sp_de = base_water_value * seasonal_factor
            
            # Energie-Budget Band-Check: Ist Preis im Dispatch-Band?
            zone_params = reservoir_budget_params.get('DE_LU', reservoir_budget_params['default'])
            delta_band = zone_params['delta_band']
            
            # Enhanced band check with dynamic tolerance for high prices
            dynamic_delta = delta_band
            if p_de > 80.0:  # Higher tolerance for high price hours
                dynamic_delta = max(delta_band, delta_band * 1.5)
            
            # Band-Check: Dispatch nur wenn Preis im Band [wasserwert-Î”, wasserwert+Î”]
            if is_reservoir_dispatch_candidate(p_de, reservoir_sp_de, dynamic_delta):
                # Nur preissetzend wenn nicht schon durch Merit Order marginal
                if not hydro_reservoir_marginal:
                    marginal_side  = "DE"
                    marginal_label = "DE_flex_reservoir_price_setting_band_enhanced"
                    marginal_fuel  = "Hydro Water Reservoir"
                    # Clip reservoir SRMC conservatively but allow high scarcity values if configured
                    marginal_srmc  = float(np.clip(reservoir_sp_de, 0.0, float(getattr(args, 'reservoir_max_srmc', 300.0))))
                    marginal_eta   = np.nan
                    mef_gpkwh      = 0.0
                    flag_reservoir_enhanced = 1
                DEBUG_reservoir_band = f"âœ“Band[{reservoir_sp_de-dynamic_delta:.1f}-{reservoir_sp_de+dynamic_delta:.1f}], P={p_de:.1f}, W={reservoir_sp_de:.1f}, SF={seasonal_factor:.2f}" + (" [MARGINAL]" if hydro_reservoir_marginal else " [PRICE-SET]")
            else:
                # Reservoir nicht im Band â†’ Kein Dispatch, weiter zu fossilen Kraftwerken
                DEBUG_reservoir_band = f"âœ—Band[{reservoir_sp_de-dynamic_delta:.1f}-{reservoir_sp_de+dynamic_delta:.1f}], P={p_de:.1f}, W={reservoir_sp_de:.1f}, SF={seasonal_factor:.2f}"

        # Residuen (fÃ¼r Reporting)
        residual_domestic_fossil = RL6
        residual_after_trade     = RL_final - max(net_imp, 0.0)  # Nur Import reduziert Residuallast, nicht Export
        residual = max(residual_after_trade, 0.0)

        # ---------- DE-Marginal (Fallback/kein Import oder Rest-Fossil) ----------
        # KapazitÃ¤tsprofil inkl. Lignite-Erzwingung
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

        # DE-Einheit bestimmen (falls spÃ¤ter DE gewÃ¤hlt wird)
        unit_id = None; ef_dom = None; eta_dom = None; srmc_dom = None; mef_dom = np.nan
        # PATCH 4: hydro_reservoir_marginal bereits früh initialisiert - nicht nochmal setzen
        
        # CRITICAL FIX: Auch bei Export/negativer Residuallast Merit Order durchführen
        if (t in SRMC.index):
            # Merit Order Stack: Fossil + Hydro-Reservoir kombiniert
            srmc_t = SRMC.loc[t].to_numpy()
            
            # Bei Export/sehr niedriger Last: Nur billigste Einheit verwenden
            if residual <= 10.0:
                # Bei minimaler/negativer Last: billigstes verfügbares Kraftwerk als marginal
                order = np.argsort(srmc_t, kind="mergesort")
                uidx = order[0]  # Billigstes Kraftwerk
                unit_id = units[uidx]
                ef_dom = ef_keys[uidx] 
                eta_dom = float(eta_arr[uidx])
                srmc_dom = float(srmc_t[uidx])
                mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom,1e-6)) * 1000.0 if ef_dom is not None else np.nan
            # Standard Merit Order für positive Residuallast
            elif residual > 10.0:
                # Hydro-Reservoir als marginale Option hinzufÃ¼gen (wenn verfÃ¼gbar)
                reservoir_srmc = None
                reservoir_cap = 0.0
                if (RL7 > 0.0) and (de_reservoir_avail >= RL7 - 1e-6) and np.isfinite(p_de):
                    reservoir_sp_de = reservoir_sp_map.get(("DE_LU", t), float(np.clip(p_de, 0.0, 60.0)))
                    zone_params = reservoir_budget_params.get('DE_LU', reservoir_budget_params['default'])
                    delta_band = zone_params['delta_band']
                    
                    # Hydro nur wenn im Dispatch-Band
                    if is_reservoir_dispatch_candidate(p_de, reservoir_sp_de, delta_band):
                        reservoir_srmc = float(np.clip(reservoir_sp_de, 0.0, getattr(args, "psp_srmc_floor_eur_mwh", 60.0)))
                        reservoir_cap = float(de_reservoir_avail)
                
                # Erweiterte Merit Order: Fossil + Hydro-Reservoir
                if reservoir_srmc is not None and reservoir_cap > 0.0:
                    # Kombinierter Stack: Fossile Kraftwerke + Hydro-Reservoir
                    extended_srmc = np.append(srmc_t, reservoir_srmc)
                    extended_cap = np.append(cap_t, reservoir_cap)
                    extended_units = list(units) + ["HYDRO_RESERVOIR_DE_LU"]
                    extended_fuel = list(ef_keys) + ["Hydro Water Reservoir"]
                    extended_eta = np.append(eta_arr, np.nan)
                else:
                    # Nur fossile Kraftwerke
                    extended_srmc = srmc_t
                    extended_cap = cap_t
                    extended_units = list(units)
                    extended_fuel = list(ef_keys)
                    extended_eta = eta_arr
                
                order = np.argsort(extended_srmc, kind="mergesort")
                cumcap = np.cumsum(extended_cap[order])
                pos = np.searchsorted(cumcap, residual, side="left")
                if pos >= len(order): pos = len(order) - 1
                
                uidx = order[pos]
                if uidx == len(srmc_t):  # Hydro-Reservoir ist marginal
                    hydro_reservoir_marginal = True
                    unit_id = "HYDRO_RESERVOIR_DE_LU"
                    ef_dom = "Hydro Water Reservoir"
                    eta_dom = np.nan
                    srmc_dom = float(reservoir_srmc)
                    mef_dom = 0.0  # Hydro hat keine CO2-Emissionen
                else:  # Fossiles Kraftwerk ist marginal
                    unit_id = extended_units[uidx]
                    ef_dom = extended_fuel[uidx]
                    eta_dom = float(extended_eta[uidx]) if not np.isnan(extended_eta[uidx]) else np.nan
                    srmc_dom = float(extended_srmc[uidx])
                    mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom,1e-6)) * 1000.0 if ef_dom is not None and not np.isnan(eta_dom) else 0.0
            else:
                # Standard Merit Order: Nur fossile Kraftwerke
                order = np.argsort(srmc_t, kind="mergesort")
                cumcap = np.cumsum(cap_t[order])
                pos = np.searchsorted(cumcap, residual, side="left")
                if pos >= len(order): pos = len(order) - 1
                uidx = order[pos]
                unit_id = units[uidx]
                ef_dom = ef_keys[uidx]
                eta_dom = float(eta_arr[uidx])
                srmc_dom = float(srmc_t[uidx])
                mef_dom = (EF_LOOKUP_T_PER_MWH_TH.get(ef_dom, 0.30) / max(eta_dom,1e-6)) * 1000.0 if ef_dom is not None else np.nan

        # --------- Kopplung: gepoolter Export-Stack aller gekoppelten Zonen (IMPORT-Richtung) ------
        marginal_import_label = None
        import_marg_srmc = None
        import_marg_mef  = None
        import_marg_price_zone = float('nan')

        I_needed = RL_final
        if (I_needed > 1e-6) and coupling_active and import_relevant and (fp_row is not None):
            flow_budget_by_zone = {
                z: float(pos_imp_by_zone.get(z, 0.0))
                for z in coupled_neighbors
                if pos_imp_by_zone.get(z, 0.0) > 1e-6
            }
            total_flow_budget = float(sum(flow_budget_by_zone.values()))
            fallback_budget = max(float(import_flow_target) - total_flow_budget, 0.0)
            I_target = float(I_needed)
            if (total_flow_budget > 1e-6) or (fallback_budget > 1e-6):
                I_target = min(I_target, total_flow_budget + fallback_budget)
            elif import_flow_target > 1e-6:
                I_target = min(I_target, float(import_flow_target))

            if I_target > 1e-6:
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
                I_remaining = I_target
                marg_block = None
                for (_, fuel, srmc, mw, eta, z) in stack_all:
                    if I_remaining <= 1e-6:
                        break
                    zone_budget = flow_budget_by_zone.get(z)
                    if zone_budget is not None:
                        take_cap = min(float(mw), I_remaining, zone_budget)
                    elif fallback_budget > 1e-6:
                        take_cap = min(float(mw), I_remaining, fallback_budget)
                    else:
                        continue
                    if take_cap <= 1e-6:
                        continue
                    I_remaining -= take_cap
                    import_stack_volume_used += take_cap
                    if zone_budget is not None:
                        flow_budget_by_zone[z] = max(zone_budget - take_cap, 0.0)
                    else:
                        fallback_budget = max(fallback_budget - take_cap, 0.0)
                    marg_block = (fuel, srmc, eta, z)

                if marg_block is not None:
                    fuel_m, srmc_m, eta_m, z_m = marg_block
                    price_cols = [f"price_{z_m}", f"price_{z_m.replace('_', '-')}"]
                    price_zone_val = np.nan
                    for col in price_cols:
                        if col in nei_prices.columns and t in nei_prices.index:
                            val = nei_prices.at[t, col]
                            if np.isfinite(val):
                                price_zone_val = float(val)
                                break
                    base_label = f"{z_m}({fuel_m})"
                    srmc_val = float(srmc_m)
                    if fuel_m in ("Reservoir Hydro", "Hydro Pumped Storage", "EE", "Nuclear"):
                        mef_val = 0.0
                    else:
                        ef_th = EF_LOOKUP_T_PER_MWH_TH.get(fuel_m, 0.30)
                        mef_val = float((ef_th / max(eta_m, 1e-6)) * 1000.0)
                    if np.isfinite(price_zone_val):
                        if price_zone_val <= import_zero_thresh:
                            srmc_val = 0.0
                            mef_val = 0.0
                            base_label = f"{z_m}(EE)"
                        else:
                            srmc_val = max(srmc_val, price_zone_val)
                    import_marg_srmc = srmc_val
                    import_marg_mef = mef_val
                    import_marg_price_zone = price_zone_val
                    marginal_import_label = base_label

        export_marg_srmc   = None
        export_marg_mef    = None
        export_marg_label  = None
        export_marg_fuel   = None
        export_marg_eta    = np.nan

        # --------- Neu: Export-Richtung symmetrisch behandeln ---------
        # Bedingung: gekoppelt, relevanter Export (negativer Nettofluss) und DE hat ExportÃ¼berschuss
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
                min_by_fuel_t = {f: float(s.reindex([t]).fillna(0.0).iloc[0]) for f, s in nei_min_by_zone_fuel.get(z, {}).items()}
                demand_importing += residual_need_after_local_steps(
                    t, z, gen_row_z, load_z, min_total_t, min_by_fuel_zone_t=min_by_fuel_t, args=args
                )

            E_target = max(demand_importing, 0.0)
            if export_flow_target > 1e-6:
                E_target = min(E_target, export_flow_target)
            # Fallback: nutze gemessenen Exportbetrag, falls keine RL7 bestimmbar
            if E_target <= 1e-6:
                fallback_export = abs(net_imp)
                if export_flow_target > 1e-6:
                    fallback_export = min(fallback_export, export_flow_target)
                E_target = fallback_export

            if E_target > 1e-6:
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

                E_remaining = E_target
                marg_block_exp = None
                for (_, fuel, srmc, mw, eta, z) in stack_de:
                    if E_remaining <= 1e-6:
                        break
                    take = min(float(mw), E_remaining)
                    if take <= 1e-6:
                        continue
                    E_remaining -= take
                    export_stack_volume_used += take
                    marg_block_exp = (fuel, srmc, eta, z)
                
                # --- EXPORT-KANDIDAT nur vormerken (nicht sofort setzen) ---
                export_marg_srmc   = None
                export_marg_mef    = None
                export_marg_label  = None
                export_marg_fuel   = None
                export_marg_eta    = np.nan
                
                if (marg_block_exp is not None) and (E_remaining <= 1e-3):
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
            else:
                export_marg_srmc   = None
                export_marg_mef    = None
                export_marg_label  = None
                export_marg_fuel   = None
                export_marg_eta    = np.nan
            



        # ------------- Nicht gekoppelt / Restfall in DE -------------
        if (I_needed > 1e-6) and ( (not coupling_active) or (import_marg_srmc is None) ):
            # Rest Waste â†’ 1 EUR
            rest_after_waste = RL1 - take_waste_mu
            if (rest_after_waste <= 1e-6) and (waste_t > 0.0) and (RL1 > 0.0):
                marginal_side  = "DE"; marginal_label = "RestWaste_only"
                marginal_fuel  = "Waste"; marginal_eta = np.nan
                marginal_srmc  = float(getattr(args,"waste_srmc_eur_mwh",1.0))
                mef_gpkwh      = float(getattr(args,"waste_mef_gpkwh",0.0))
            else:
                # Rest Biomasse FLEX â†’ SRMC konservativ
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
            
            # Spezialfall: Hydro-Reservoir Marginal
            if hydro_reservoir_marginal and unit_id == "HYDRO_RESERVOIR_DE_LU":
                de_candidate_srmc = srmc_dom
                de_candidate_label = "DE_flex_reservoir_marginal"
                de_candidate_fuel = "Hydro Water Reservoir"
                de_candidate_eta = np.nan
                de_candidate_mef = 0.0
            
            if (net_imp < -1e-6) and (export_marg_srmc is not None):
                de_candidate_srmc  = export_marg_srmc
                de_candidate_label = "DE_export_pool_price_setting"
                de_candidate_fuel  = export_marg_fuel
                de_candidate_eta   = export_marg_eta
                de_candidate_mef   = export_marg_mef
            # DE-Kandidat auf Basis der Einheitensuche setzen, falls noch nicht gesetzt
            if marginal_label is None and unit_id is not None:
                marginal_side  = "DE"
                if hydro_reservoir_marginal:
                    marginal_label = "DE_flex_reservoir_marginal"
                    marginal_fuel  = "Hydro Water Reservoir"
                    marginal_eta   = np.nan
                    marginal_srmc  = srmc_dom if srmc_dom is not None else np.nan
                    mef_gpkwh      = 0.0  # Hydro hat keine CO2-Emissionen
                else:
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
                    # Enhanced threshold logic for block orders / Euphemia integer solutions
                    # Use dynamic tolerance: higher for high prices (block order territory)
                    dynamic_tol = float(args.price_tol)
                    if p_de > 50.0:  # In higher price ranges, allow more tolerance for block orders
                        dynamic_tol = max(dynamic_tol, max(3.0, p_de * 0.05))  # 5% tolerance, with minimum of 3EUR
                    
                    valid = [c for c in cand if c[1] <= dynamic_tol]
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
                            # Export erfÃ¼llt die Bedingung nicht â†’ setze IMPORT
                            marginal_side  = "IMPORT"
                            marginal_label = marginal_import_label or "import_stack"
                            marginal_fuel  = (marginal_import_label.split("(")[-1].rstrip(")")) if marginal_import_label else "mix"
                            marginal_eta   = np.nan
                            marginal_srmc  = float(import_marg_srmc)
                            mef_gpkwh      = float(import_marg_mef)
                    else:
                        # Normaler DE-Kandidat (z. B. inlÃ¤ndische Einheit)
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
    
        # EE gate: classify hours as EE-driven when price is low and residual after FEE is near-zero
        ee_threshold = float(getattr(args, 'ee_price_threshold', 10.0))
        ee_gate_tol = float(getattr(args, 'ee_gate_tol', 1e-02))
        import_zero_thresh = float(getattr(args, 'import_zero_price_threshold', 10.0))
        rl_after_fee = float(rl_ladder.get('RL1', 0.0))
        # Domestic EE case (net export or zero import) -> DE label
        if np.isfinite(p_de) and (p_de <= ee_threshold) and (net_imp <= 0.0) and (rl_after_fee <= ee_gate_tol):
            marginal_side  = 'DE'
            marginal_label = 'EE_price_setting'
            marginal_fuel  = 'EE'
            marginal_eta   = np.nan
            marginal_srmc  = float(p_de)
            mef_gpkwh      = 0.0
        # Import-side EE override: if price low in neighbor and imports are present and residual after FEE small
        elif np.isfinite(p_de) and (p_de <= import_zero_thresh) and (net_imp > 0.0) and (rl_after_fee <= ee_gate_tol):
            marginal_side  = 'IMPORT'
            marginal_label = 'IMPORT_EE_price_setting'
            marginal_fuel  = 'EE'
            marginal_eta   = np.nan
            marginal_srmc  = float(p_de)
            mef_gpkwh      = 0.0

        

        # --------- Peaker-Override (nur wenn DE plausibel marginal ist) ----------
        if getattr(args, "peak_switch", False):
            thr1, thr2 = _parse_two_floats(getattr(args, "peak_price_thresholds", "180,260"))
            # Voraussetzungen: Preis hoch, Restlast > 0, DE ist gewÃ¤hlt, SRMC preisnah.
            if np.isfinite(p_de) and (RL_final > 0.0) and (marginal_side == "DE") and np.isfinite(marginal_srmc):
                if abs(marginal_srmc - p_de) <= float(args.epsilon) and (p_de >= thr1):
                    co2 = float(fuel_prices.loc[t, "co2_eur_t"])
                    gas_th = float(fuel_prices.loc[t, "gas_eur_mwh_th"])
                    oil_th = float(fuel_prices.loc[t, "oil_eur_mwh_th"])
                    ocgt_eta = float(getattr(args, "peak_eta_ocgt", 0.36))
                    oil_eta  = float(getattr(args, "peak_eta_oil", 0.33))
                    ocgt_srmc = (gas_th + co2 * EF_LOOKUP_T_PER_MWH_TH["Erdgas"]) / max(ocgt_eta, 1e-6)
                    oil_srmc  = (oil_th + co2 * EF_LOOKUP_T_PER_MWH_TH["HeizÃ¶l schwer"]) / max(oil_eta,  1e-6)
                    # Nur ersetzen, wenn die Peaker-SRMC selbst nah am Preis liegen
                    if (p_de >= thr2) and np.isfinite(oil_srmc) and (abs(oil_srmc - p_de) <= float(args.epsilon)):
                        marginal_label = "DE_peaker_override_oil"
                        marginal_fuel  = "HeizÃ¶l schwer"
                        marginal_eta   = oil_eta
                        marginal_srmc  = float(oil_srmc)
                        mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["HeizÃ¶l schwer"] / max(oil_eta,1e-6)) * 1000.0
                    elif np.isfinite(ocgt_srmc) and (abs(ocgt_srmc - p_de) <= float(args.epsilon)):
                        marginal_label = "DE_peaker_override_ocgt"
                        marginal_fuel  = "Erdgas"
                        marginal_eta   = ocgt_eta
                        marginal_srmc  = float(ocgt_srmc)
                        mef_gpkwh      = (EF_LOOKUP_T_PER_MWH_TH["Erdgas"] / max(ocgt_eta,1e-6))  


        # Mustrun-only Floor (falls alles andere weg und Preis existiert)
        fee_total = rl_ladder['takes']['fee']  # Aus einheitlicher RL-Ladder
        mustrun_only = (RL6 <= 1e-6) and (net_imp <= 0.0) and (L > fee_total + 1e-6)
        if mustrun_only and np.isfinite(p_de):
            marginal_side  = "DE"
            marginal_label = "mustrun_floor_price_setting"
            marginal_fuel  = "MustrunMix"
            marginal_eta   = np.nan
            marginal_srmc  = float(np.clip(p_de, 0.0, 60.0))
            # Must-Run MEF: Durchschnitt der verfÃ¼gbaren Must-Run-Kraftwerke
            mustrun_mef_values = []
            if de_fossil_mu_cost > 0 and 'mef_g_per_kwh' in fleet_all.columns:
                # Sammle MEF-Werte der aktiven Must-Run Kraftwerke (nur DE-Kraftwerke)
                for unit_id in fleet_all.index:
                    if fleet_all.loc[unit_id, "is_available"]:
                        fuel_type = fleet_all.loc[unit_id, "fuel"]
                        if fuel_type in ["Braunkohle", "Steinkohle", "Erdgas"]:
                            unit_mef = fleet_all.loc[unit_id, "mef_g_per_kwh"]
                            if pd.notna(unit_mef) and unit_mef > 0:
                                mustrun_mef_values.append(unit_mef)
            
            if mustrun_mef_values:
                mef_gpkwh = np.mean(mustrun_mef_values)
            else:
                # Fallback: Default Must-Run MEF fÃ¼r Deutschland.
                # Set to 0.0 to treat Must-Run as emissions-free instead of using the previous 900 g/kWh placeholder.
                mef_gpkwh = 0.0

        # Falls immer noch nichts gesetzt und DE-Einheit vorhanden â†’ DE wÃ¤hlen
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
            "IMPORT_ambivalent_zones": "|".join(sorted(set(ambiv_zones_active))) if ambiv_zones_active else "",
            "residual_domestic_fossil_MW": residual_domestic_fossil,
            "flag_psp_price_setting": flag_psp_price_setting,
            "flag_psp_scarcity_cap": flag_psp_scarcity_cap,
            "flag_reservoir_enhanced": flag_reservoir_enhanced,
            # JAO FlowBased Boundary Flags
            "fb_boundary": fb_boundary_flag,
            "fb_slack_to_min_MW": fb_slack_min,
            "fb_slack_to_max_MW": fb_slack_max,
            "fb_minNP_MW": fb_min_np,
            "fb_maxNP_MW": fb_max_np,
            "fb_NetPosition_MW": fb_net_pos,
            "FEE_MW": rl_ladder['takes']['fee'],
            "ND_EXTRA_MW": (nuc_t + bio_raw + waste_t),
            "RL_after_FEE_MW": rl_ladder['RL1'],
            "RL_after_MU_WASTE_MW": rl_ladder['RL2'],
            "RL_after_MU_NUC_MW": rl_ladder['RL3'],
            "RL_after_MU_BIO_MW": rl_ladder['RL4'],
            "RL_after_MU_OIL_MW": rl_ladder['RL5'],
            "RL_after_FOSSIL_MU_MW": rl_ladder['RL6'],
            "RL_after_PSP_MW": rl_ladder['RL7'],
            "RL_after_RES_MW": rl_ladder['RL8'],
            "domestic_ee_after_mustrun_applied": bool(rl_ladder.get('domestic_ee_after_mustrun_applied', False)),
            "DE_fossil_mustrun_cost_based_MW": float(de_fossil_mu_cost),
            "DE_fossil_mustrun_required_MW": float(mustrun_de_total),
            "DE_fossil_mustrun_other_required_MW": float(fossil_mu_req_total),
            "DE_oil_mustrun_required_MW": float(oil_mustrun_req),
            "BIO_MW": bio_raw,
            "WASTE_MW": waste_t,
            "OIL_lowprice_profile_MW": float(oil_lowprice_mw),
            "OIL_MU_used_MW": float(take_oil_mu),
            "residual_after_trade_MW": residual,
            "import_flow_target_MW": float(import_flow_target),
            "export_flow_target_MW": float(export_flow_target),
            "import_flow_observed_MW": float(pos_imp_total),
            "export_flow_observed_MW": float(neg_exp_total),
            "stack_import_utilized_MW": float(import_stack_volume_used),
            "stack_export_utilized_MW": float(export_stack_volume_used),
            "IMPORT_price_zone": float(import_marg_price_zone) if np.isfinite(import_marg_price_zone) else np.nan,
            "LIGNITE_lowprice_profile_MW": float(lignite_lowprice_mw),
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
            "IMPORT_ambivalent_zones": "|".join(sorted(set(ambiv_zones_active))) if ambiv_zones_active else "",
            "cluster": "|".join(coupled_neighbors),
            "net_import_total_MW": net_imp,
            "FLOW_import_obs_MW": float(pos_imp_total),
            "FLOW_export_obs_MW": float(neg_exp_total),
            "import_flow_target_MW": float(import_flow_target),
            "export_flow_target_MW": float(export_flow_target),
            "stack_import_utilized_MW": float(import_stack_volume_used),
            "stack_export_utilized_MW": float(export_stack_volume_used),
            "IMPORT_price_zone": float(import_marg_price_zone) if np.isfinite(import_marg_price_zone) else np.nan,
            "price_DE": p_de,
            "ND_MW": ND,
            "Load_MW": L,
            "LIGNITE_MUSTRUN_ENFORCED_MW": lignite_mustrun_enforced,
            "LIGNITE_LOWPRICE_PROFILE_MW": float(lignite_lowprice_mw),
            "DE_FOSSIL_MU_COST_BASED_MW": float(de_fossil_mu_cost),
            "DE_FOSSIL_MU_TOTAL_REQUIRED_MW": float(mustrun_de_total),
            "DE_FOSSIL_MU_OTHER_REQUIRED_MW": float(fossil_mu_req_total),
            "DE_OIL_MU_REQUIRED_MW": float(oil_mustrun_req),
            "DE_FOSSIL_MU_TAKEN_MW": float(take_mu_foss),
            "OIL_lowprice_profile_MW": float(oil_lowprice_mw),
            "OIL_MU_used_MW": float(take_oil_mu),
            "DEBUG_rule_psp_price_setting": DEBUG_rule_psp_price,
            "DEBUG_reservoir_band": DEBUG_reservoir_band,
            "domestic_ee_after_mustrun_applied": bool(rl_ladder.get('domestic_ee_after_mustrun_applied', False)),
        })

        # Memory management: garbage collection every 100 hours
        if (i + 1) % 100 == 0:
            print(f"[INFO] Processed {i+1}/{total_hours} hours, performing garbage collection...")
            import gc
            gc.collect()

    print(f"[INFO] Completed processing all {total_hours} hours")

    # 14) Outputs schreiben
    df_res = pd.DataFrame(results).set_index("timestamp").sort_index()
    df_dbg = pd.DataFrame(debug_rows).set_index("timestamp").sort_index()
    
    # Initialize enhanced feature flags if not present
    enhanced_flags = ["flag_psp_scarcity_cap", "flag_reservoir_enhanced"]
    for flag in enhanced_flags:
        if flag not in df_res.columns:
            df_res[flag] = 0
    
    (outdir / "analysis").mkdir(exist_ok=True, parents=True)
    # Postprocessing: interpolate short NA runs (length 1-4 hours) for selected numeric cols
    try:
        interp_cols = ['price_DE', 'mef_g_per_kwh', 'marginal_srmc_eur_per_mwh']
        if all(c in df_res.columns for c in interp_cols):
            # work on a copy with datetime index
            df_i = df_res.copy()
            # ensure index is tz-aware datetime
            try:
                idx_dt = pd.to_datetime(df_i.index, utc=True)
                df_i.index = idx_dt
            except Exception:
                # if index already fine or parse fails, continue
                pass
            mask_any = df_i[interp_cols].isna().any(axis=1)
            is_na = mask_any.values
            idx = df_i.index
            blocks = []
            start = None
            for i, val in enumerate(is_na):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    blocks.append((start, i-1))
                    start = None
            if start is not None:
                blocks.append((start, len(is_na)-1))
            filled_counts = {c: 0 for c in interp_cols}
            # perform interpolation globally but apply only to blocks of length<=4
            interpolated_all = df_i[interp_cols].interpolate(method='time', limit=4, limit_area='inside')
            for s, e in blocks:
                length = e - s + 1
                if 1 <= length <= 4:
                    rng = idx[s:e+1]
                    for c in interp_cols:
                        mask_fill = df_i[c].isna() & interpolated_all[c].notna()
                        mask_fill = mask_fill.loc[rng]
                        df_i.loc[mask_fill.index, c] = interpolated_all.loc[mask_fill.index, c]
                        filled_counts[c] += int(mask_fill.sum())
            # copy interpolated values back to df_res
            for c in interp_cols:
                df_res.loc[df_i.index, c] = df_i[c].values
            print(f"[INFO] interpolated short NA runs for columns: {filled_counts}")
    except Exception as e:
        print(f"[WARN] interpolation postprocess failed: {e}")

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

    # 16) ZusÃ¤tzliche Residuallast-Plots (wie zuvor)
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
        print("[WARNUNG] FEE/ND_EXTRA nicht gespeichert â†’ diese Kurven fehlen im Plot.")

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
        ax2.set_ylabel("Preis [EUR/MWh]")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

        fig.tight_layout()
        out_path = plots_dir / f"residuallast_month_{month:02d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
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
        "Steinkohle":          "#5F5E5E",
        "Braunkohle":          "#A08269",
        "HeizÃ¶l schwer":       "#DC0C23",
        "HeizÃ¶l leicht / Diesel":"#F5644B",
        "Biomass":             "#D2B900",
        "Waste":               "#F3F0E7",
        "Nuclear":             "#9B0028",
        "MustrunMix":          "#8063A7",
        "NonDisp":             "#518E9F",
        "mix":                 "#E6F0F7",
        "Other":               "#DDDDDD",
    }
    def color_for_share(key): return FUEL_COLOR.get(key, "#BBBBBB")
    
    # Feste Reihenfolge fÃ¼r Brennstoffe (wichtigste zuerst)
    FUEL_ORDER = [
        "Braunkohle", "Steinkohle", "Erdgas", "Nuclear", "MustrunMix",
        "HeizÃ¶l schwer", "HeizÃ¶l leicht / Diesel", "Biomass", "Waste",
        "EE", "Reservoir Hydro", "Hydro Pumped Storage", "NonDisp", "mix"
    ]

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
        
        # Verwende feste Reihenfolge basierend auf FUEL_ORDER
        available_fuels = tab.columns.tolist()
        ordered_fuels = [f for f in FUEL_ORDER if f in available_fuels]
        remaining_fuels = [f for f in available_fuels if f not in FUEL_ORDER]
        
        # Sortiere verbleibende Brennstoffe nach HÃ¤ufigkeit
        remaining_sorted = tab[remaining_fuels].sum(axis=0).sort_values(ascending=False).index.tolist() if remaining_fuels else []
        
        col_order = ordered_fuels + remaining_sorted
        keep_cols = col_order[:TOP_K]
        
        if len(col_order) > TOP_K:
            tab[OTHER_NAME] = tab[[c for c in col_order[TOP_K:]]].sum(axis=1)
            cols_final = keep_cols + [OTHER_NAME]
        else:
            cols_final = keep_cols
        tab = tab[cols_final]
        share = 100.0 * tab.div(total, axis=0).fillna(0.0)

        fig, ax = plt.subplots(figsize=(14, 4.8))  # Breiter fÃ¼r externe Legende
        x = np.arange(24); bottom = np.zeros_like(x, dtype=float)
        for c in cols_final:
            y = share[c].values.astype(float)
            ax.bar(x, y, bottom=bottom, label=c, width=0.9, edgecolor="none", color=color_for_share(c))
            bottom += y
        ax.set_title(f"Anteil marginaler Technologien je Stunde â€“ Monat {m:02d}")
        ax.set_xlabel("Stunde"); ax.set_ylabel("Anteil [%]"); ax.set_xticks(range(24)); ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        
        # FIXED: Externe Legende rechts neben dem Plot
        leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
        out_path = shares_dir / f"marginal_shares_hourly_month_{m:02d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_path, dpi=160, bbox_inches='tight'); plt.close(fig)

    print(f"[OK] Marginal-Share-Plots in {shares_dir} geschrieben.")

    # 18) Enhanced Validation Plots (erweiterte)
    print("[DEBUG] BEFORE make_validation_plots...")
    make_validation_plots(
        outdir, df_res, df_dbg, df_val, nei_prices,
        de_gen=de_gen, de_min_total=de_min_total, de_load=de_load, flows=flows, args=args
    )
    print("[DEBUG] AFTER make_validation_plots - continuing to Enhanced Features...")

    # 19) Enhanced Features - Load Coverage + Enhanced Plots
    print("[DEBUG] Enhanced Features Section erreicht!")
    try:
        print("[INFO] Enhanced Plots mit Stacked Diagrammen werden generiert...")
        
        # Load Coverage Chart (Global Income Distribution Style)
        try:
            print("[DEBUG] Starting create_load_coverage_chart...")
            create_load_coverage_chart(
                outdir=outdir,
                df_res=df_res,
                de_gen=de_gen,
                de_load=de_load,
                args=args
            )
            print("[OK] Load Coverage Chart erstellt")
        except Exception as e:
            print(f"[WARNING] Load Coverage Chart failed: {e}")
        
        # VollstÃ¤ndige Enhanced Plots mit Stacked Diagrammen
        print("[DEBUG] Starting generate_enhanced_plots...")
        enhanced_plots_success = generate_enhanced_plots(
            outdir=outdir,
            df_res=df_res,
            df_dbg=df_dbg,
            df_val=df_val,
            fuel_prices=fuel_prices,
            de_gen=de_gen,
            flows=flows,
            args=args
        )
        
        if enhanced_plots_success:
            print("[OK] Enhanced Plots erfolgreich generiert (inkl. Stacked Diagramme)")
        else:
            print("[WARNING] Enhanced Plots teilweise fehlgeschlagen")
            
    except Exception as e:
        print(f"[ERROR] Enhanced Plots System failed: {e}")
        print("[INFO] Fallback auf Standard-Plots")

    # 20) Gefilterte Korrelation + Offender
    print("[DEBUG] Offender Analysis Section erreicht!")
    try:
        print("[DEBUG] Starting _filtered_corr_and_offenders...")
        corr_filt = _filtered_corr_and_offenders(
            outdir=outdir, df_res=df_res, df_dbg=df_dbg, df_val=df_val,
            fuel_prices=fuel_prices, args=args
        )
        print(f"[VALIDATION] Filtered corr (Preis vs. SRMC) = {corr_filt:.4f}")
        print(f"[VALIDATION] Details: analysis/_corr_offenders.csv und analysis/_corr_offenders_summary.txt")
    except Exception as e:
        print("[VALIDATION] Hinweis â€“ gefilterte Korrelation/Offender konnte nicht berechnet werden:", e)

    # 21) Enhanced Multi-Stage Validation (nach Standard-Validation)
    print("[DEBUG] Enhanced Validation Section erreicht!")
    try:
        print("[DEBUG] Starting run_full_enhanced_validation...")
        # Enhanced Validation ist bereits integriert - nutze lokale Funktion
        enhanced_report = run_full_enhanced_validation(df_res, df_dbg, df_val, args, outdir)
        baseline_ready = enhanced_report['overall_pass']
        print(f"[ENHANCED] {'ðŸŽ¯ BASELINE-TAUGLICH' if baseline_ready else 'ðŸš« NICHT BASELINE-TAUGLICH'}")
        
    except Exception as e:
        print(f"[ENHANCED] Enhanced Validation failed: {e}")
        baseline_ready = False

    print("[INFO] MEF dispatch analysis completed successfully.")
    print(f"[INFO] Results saved to: {outdir}")









if __name__ == "__main__":
    main(build_parser().parse_args())















