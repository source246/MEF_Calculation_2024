from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FUEL_KEY_BY_EF: Dict[str, Tuple[str, str]] = {
    "Erdgas": ("gas", "Erdgas"),
    "Steinkohle": ("coal", "Steinkohle"),
    "Braunkohle": ("lignite", "Braunkohle"),
    "Heizöl schwer": ("oil", "Heizöl schwer"),
    "Heizöl leicht / Diesel": ("oil", "Heizöl leicht / Diesel"),
    # add nuclear mapping (German canonical name and a lowercase variant for robustness)
    "Kernenergie": ("nuclear", "Kernenergie"),
    "kernenergie": ("nuclear", "Kernenergie"),
}

EF_LOOKUP_T_PER_MWH_TH: Dict[str, float] = {
    "Erdgas": 0.201,
    "Steinkohle": 0.335,
    "Braunkohle": 0.383,
    "Heizöl schwer": 0.288,
    "Heizöl leicht / Diesel": 0.266,
    # Nuclear has (essentially) zero direct CO2 per thermal MWh - include keys for robustness
    "Kernenergie": 0.0,
    "kernenergie": 0.0,
}

DEFAULT_NEI_DISTS: Dict[str, Dict[str, float]] = {
    "Erdgas": {"mean": 0.52, "std": 0.043, "min": 0.35, "max": 0.60},
    "Steinkohle": {"mean": 0.41, "std": 0.030, "min": 0.34, "max": 0.45},
    "Braunkohle": {"mean": 0.40, "std": 0.028, "min": 0.33, "max": 0.43},
    "Heizöl schwer": {"mean": 0.36, "std": 0.020, "min": 0.32, "max": 0.40},
}


def truncated_normal(mean: float, std: float, lo: float, hi: float, size: int) -> np.ndarray:
    rnd = np.random.normal
    samples = rnd(mean, std, size=size)
    samples = np.clip(samples, lo, hi)
    return samples


def _eta_repr_for_fuel_DE(fleet_df: pd.DataFrame, ef_name: str, default: float = 0.42) -> float:
    sub = fleet_df[fleet_df["ef_key"] == ef_name]["eta"].dropna().astype(float)
    if sub.empty:
        return float(default)
    median = float(np.nanmedian(sub))
    return float(np.clip(median, 0.20, 0.65))


def _tech_srmc_series_from_eta(fuel_prices: pd.DataFrame, ef_name: str, eta_eff: float) -> pd.Series:
    price_key = FUEL_KEY_BY_EF.get(ef_name, (None, None))[0]
    if price_key is None:
        return pd.Series(np.nan, index=fuel_prices.index)
    # tolerate encoding/case differences: try exact, title, and lowercase lookups before falling back
    ef_th = (
        EF_LOOKUP_T_PER_MWH_TH.get(ef_name)
        if EF_LOOKUP_T_PER_MWH_TH.get(ef_name) is not None
        else EF_LOOKUP_T_PER_MWH_TH.get(ef_name.title())
        if EF_LOOKUP_T_PER_MWH_TH.get(ef_name.title()) is not None
        else EF_LOOKUP_T_PER_MWH_TH.get(ef_name.lower(), 0.0)
    )
    fuel_cost = pd.to_numeric(fuel_prices[f"{price_key}_eur_mwh_th"], errors="coerce")
    co2 = pd.to_numeric(fuel_prices["co2_eur_t"], errors="coerce")
    srmc = (fuel_cost + co2 * ef_th) / max(float(eta_eff), 1e-6)
    return srmc.astype(float)


def _split_peak_mask(ix: pd.DatetimeIndex, peak_hours: str) -> Tuple[pd.Series, pd.Series]:
    start_h, end_h = [int(x) for x in peak_hours.split("-")]
    if start_h <= end_h:
        peak_mask = (ix.hour >= start_h) & (ix.hour < end_h)
    else:
        peak_mask = (ix.hour >= start_h) | (ix.hour < end_h)
    peak = pd.Series(peak_mask, index=ix)
    offpeak = ~peak
    return peak, offpeak


def price_based_lignite_mustrun_profile(
    de_gen: pd.DataFrame,
    price_series: pd.Series,
    cap_base: np.ndarray,
    ef_keys: np.ndarray,
    price_floor: float = 20.0,
    min_hours: int = 3,
    window_hours: int = 6,
) -> Optional[pd.Series]:
    col = "Fossil Brown coal/Lignite"
    if col not in de_gen.columns or price_series is None or price_series.empty:
        return None
    price_aligned = price_series.reindex(de_gen.index)
    if price_aligned.isna().all():
        return None
    lignite_cap = cap_base[ef_keys == "Braunkohle"]
    cap_total = float(lignite_cap.sum()) if lignite_cap.size else 0.0
    if cap_total <= 1e-3:
        return None
    lignite_gen = pd.to_numeric(de_gen[col], errors="coerce").reindex(price_aligned.index)
    if lignite_gen.isna().all():
        return None
    price_neg = price_aligned <= 0.0
    price_low = price_aligned < float(price_floor)
    if price_low.any():
        seq = price_low.astype(int)
        groups = (seq != seq.shift()).cumsum()
        counts = seq.groupby(groups).transform("sum")
        price_low = price_low & (counts >= int(max(min_hours, 1)))
    active_mask = price_neg | price_low
    if not active_mask.any():
        return None
    share_series = (lignite_gen / cap_total).clip(lower=0.0, upper=1.0)
    share_active = share_series.where(active_mask, 0.0)
    if window_hours and window_hours > 1:
        share_active = share_active.rolling(window_hours, min_periods=1).max()
    profile = (share_active * cap_total).fillna(0.0)
    if profile.abs().sum() <= 1e-3:
        return None
    return profile


def price_based_oil_mustrun_profile(
    de_gen: pd.DataFrame,
    price_series: pd.Series,
    cap_base: np.ndarray,
    ef_keys: np.ndarray,
    price_floor: float = 20.0,
    min_hours: int = 3,
    window_hours: int = 6,
) -> Optional[pd.Series]:
    oil_cols = [
        "Fossil Oil",
        "Fossil Oil shale",
        "Heizöl schwer",
        "Heizöl leicht / Diesel",
    ]
    oil_cols = [c for c in oil_cols if c in de_gen.columns]
    if not oil_cols or price_series is None or price_series.empty:
        return None
    price_aligned = price_series.reindex(de_gen.index)
    if price_aligned.isna().all():
        return None
    mask_oil = np.isin(ef_keys, ["Heizöl schwer", "Heizöl leicht / Diesel", "Fossil Oil"])
    cap_total = float(cap_base[mask_oil].sum()) if mask_oil.any() else 0.0
    if cap_total <= 1e-3:
        return None
    oil_gen = de_gen[oil_cols].sum(axis=1).reindex(price_aligned.index)
    oil_gen = pd.to_numeric(oil_gen, errors="coerce").fillna(0.0)
    price_neg = price_aligned <= 0.0
    price_low = price_aligned < float(price_floor)
    if price_low.any():
        seq = price_low.astype(int)
        groups = (seq != seq.shift()).cumsum()
        counts = seq.groupby(groups).transform("sum")
        price_low = price_low & (counts >= int(max(min_hours, 1)))
    active_mask = price_neg | price_low
    if not active_mask.any():
        return None
    share_series = (oil_gen / cap_total).clip(lower=0.0, upper=1.0)
    share_active = share_series.where(active_mask, 0.0)
    if window_hours and window_hours > 1:
        share_active = share_active.rolling(window_hours, min_periods=1).max()
    profile = (share_active * cap_total).fillna(0.0)
    return profile


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
    if gen_df is None or gen_df.empty:
        idx = pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx), {f: pd.Series(0.0, index=idx) for f in fuels_select}

    ix = gen_df.index
    price = pd.to_numeric(price_series.reindex(ix), errors="coerce")
    out_by_fuel = {f: pd.Series(0.0, index=ix, dtype="float64") for f in fuels_select}

    eta_map = {}
    for fuel in fuels_select:
        if eta_source == "DE" and fleet_df is not None:
            eta_map[fuel] = _eta_repr_for_fuel_DE(fleet_df, fuel)
        elif eta_source == "NEI" and nei_dists is not None:
            distribution = nei_dists.get(fuel) or DEFAULT_NEI_DISTS.get(fuel)
            eta_map[fuel] = float(distribution["mean"]) if distribution else 0.42
        else:
            distribution = DEFAULT_NEI_DISTS.get(fuel, {"mean": 0.42})
            eta_map[fuel] = float(distribution["mean"])

    srmc_fuel = {
        fuel: _tech_srmc_series_from_eta(fuel_prices, fuel, eta_map[fuel]).reindex(ix)
        for fuel in fuels_select
    }

    if not monthly:
        segments = [("all", pd.Series(True, index=ix))]
    else:
        segments = []
        for month in sorted(ix.month.unique()):
            month_mask = (ix.month == month)
            if use_peak_split:
                pk, op = _split_peak_mask(ix, peak_hours)
                segments.append((f"m{month:02d}_pk", month_mask & pk))
                segments.append((f"m{month:02d}_op", month_mask & op))
            else:
                segments.append((f"m{month:02d}", month_mask))

    for fuel in fuels_select:
        tech_col = {
            "Erdgas": "Fossil Gas",
            "Steinkohle": "Fossil Hard coal",
            "Heizöl schwer": "Fossil Oil",
            "Heizöl leicht / Diesel": "Fossil Oil",
            "Braunkohle": "Fossil Brown coal/Lignite",
        }.get(fuel)
        if tech_col not in gen_df.columns:
            continue
        generation = pd.to_numeric(gen_df[tech_col].reindex(ix), errors="coerce").fillna(0.0)
        srmc_series = srmc_fuel[fuel]
        threshold = alpha * srmc_series

        mu_series = pd.Series(0.0, index=ix, dtype="float64")
        for _, mask in segments:
            segment_mask = mask & (price <= threshold)
            if segment_mask.any():
                quantile_val = float(np.nanquantile(generation[segment_mask], q)) if generation[segment_mask].size else 0.0
                mu_series.loc[mask] = max(quantile_val, 0.0)
            else:
                mu_series.loc[mask] = 0.0
        out_by_fuel[fuel] = mu_series.clip(lower=0.0)

    total_min = sum(out_by_fuel.values()) if out_by_fuel else pd.Series(0.0, index=ix)
    return total_min, out_by_fuel


def compute_fossil_min_profiles(
    gen_df: pd.DataFrame,
    fuels_select: List[str],
    peak_hours: str,
    mode: str,
    q: float = 0.10,
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    if mode == "off" or gen_df is None or gen_df.empty:
        idx = gen_df.index if (gen_df is not None and isinstance(gen_df.index, pd.DatetimeIndex)) else pd.DatetimeIndex([])
        return pd.Series(0.0, index=idx), {f: pd.Series(0.0, index=idx) for f in fuels_select}

    tech_map = {"Fossil Gas": "Erdgas", "Fossil Hard coal": "Steinkohle", "Fossil Oil": "Heizöl schwer"}
    tech_cols = [c for c in tech_map if c in gen_df.columns]

    peak_mask, offpeak_mask = _split_peak_mask(gen_df.index, peak_hours)
    idx = gen_df.index
    min_by_fuel = {fuel: pd.Series(0.0, index=idx, dtype="float64") for fuel in fuels_select}

    if mode == "min_all":
        for tech in tech_cols:
            fuel = tech_map[tech]
            if fuel not in fuels_select:
                continue
            minimum = float(pd.to_numeric(gen_df[tech], errors="coerce").min(skipna=True))
            min_by_fuel[fuel] = pd.Series(max(minimum, 0.0), index=idx, dtype="float64")

    elif mode == "min_peak":
        pk_mask = peak_mask
        op_mask = offpeak_mask
        for tech in tech_cols:
            fuel = tech_map[tech]
            if fuel not in fuels_select:
                continue
            series = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            result = pd.Series(0.0, index=idx, dtype="float64")
            result[pk_mask] = float(series[pk_mask].min()) if pk_mask.any() else 0.0
            result[op_mask] = float(series[op_mask].min()) if op_mask.any() else 0.0
            min_by_fuel[fuel] = result

    elif mode == "min_peak_monthly":
        pk_global = peak_mask
        for tech in tech_cols:
            fuel = tech_map[tech]
            if fuel not in fuels_select:
                continue
            series = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            result = pd.Series(0.0, index=idx, dtype="float64")
            for month in sorted(idx.month.unique()):
                month_mask = (idx.month == month)
                pk_mask = month_mask & pk_global
                op_mask = month_mask & (~pk_global)
                result[pk_mask] = float(series[pk_mask].min()) if pk_mask.any() else 0.0
                result[op_mask] = float(series[op_mask].min()) if op_mask.any() else 0.0
            min_by_fuel[fuel] = result

    elif mode == "q_all":
        for tech in tech_cols:
            fuel = tech_map[tech]
            if fuel not in fuels_select:
                continue
            series = pd.to_numeric(gen_df[tech], errors="coerce").fillna(0.0)
            quantile_val = float(np.nanquantile(series, q)) if len(series) else 0.0
            min_by_fuel[fuel] = pd.Series(max(quantile_val, 0.0), index=idx, dtype="float64")

    for fuel, series in min_by_fuel.items():
        min_by_fuel[fuel] = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)

    total_min = sum(min_by_fuel.values()) if min_by_fuel else pd.Series(0.0, index=idx)
    return total_min, min_by_fuel


def fossil_mustrun_shares_for_DE(args) -> Dict[str, float]:
    return {
        "Erdgas": float(getattr(args, "de_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "de_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,
        "Heizöl schwer": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
        "Heizöl leicht / Diesel": float(getattr(args, "de_mustrun_oil_share", 0.0) or 0.0),
    }


def fossil_mustrun_shares_for_NEI(args) -> Dict[str, float]:
    return {
        "Erdgas": float(getattr(args, "nei_mustrun_gas_share", 0.0) or 0.0),
        "Steinkohle": float(getattr(args, "nei_mustrun_coal_share", 0.0) or 0.0),
        "Braunkohle": 0.0,
        "Heizöl schwer": float(getattr(args, "nei_mustrun_oil_share", 0.0) or 0.0),
        "Heizöl leicht / Diesel": float(getattr(args, "nei_mustrun_oil_share", 0.0) or 0.0),
    }
