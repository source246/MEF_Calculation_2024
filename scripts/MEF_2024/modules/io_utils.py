from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import re
import numpy as np
import pandas as pd


def _norm(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower().strip()
    t = t.replace("-", " ").replace("/", " ").replace(",", " ")
    return " ".join(t.split())


def _norm_zone(z: str) -> str:
    return str(z or "").strip().replace("-", "_").upper()


def _zone_variants(zone: str) -> list[str]:
    base = _norm_zone(zone)
    variants = {base}
    variants.update({base.replace('-', '_'), base.replace('_', '-')})
    m = re.match(r'([A-Z]+)(\d+)$', base)
    if m:
        prefix, digits = m.groups()
        variants.add(f"{prefix}_{digits}")
        variants.add(f"{prefix}-{digits}")
    m2 = re.match(r'([A-Z]+)[_-](\d+)$', base)
    if m2:
        prefix, digits = m2.groups()
        variants.add(f"{prefix}{digits}")
    return [v for v in variants if v]


def _map_neighbor_fuel(s: str) -> Optional[str]:
    t = _norm(s)
    if "gas" in t:
        return "Erdgas"
    if "hard" in t or "stein" in t:
        return "Steinkohle"
    if "braun" in t or "lignite" in t:
        return "Braunkohle"
    if "öl" in t or "oel" in t or "oil" in t or "diesel" in t:
        return "Heizöl schwer" if "schwer" in t else "Heizöl leicht / Diesel"
    if "nuclear" in t or "kern" in t:
        return "Nuclear"
    if "biom" in t:
        return "Biomass"
    if "waste" in t or "abfall" in t:
        return "Waste"
    return None


def _eta_from_row(r) -> Optional[float]:
    eta_cols = [
        c for c in r.index
        if "eta" in c.lower() or "wirkungsgrad" in c.lower()
    ]
    for c in eta_cols:
        val = pd.to_numeric(r[c], errors="coerce")
        if pd.notna(val):
            return float(val)
    return None


# ------------------------------ Time helpers ---------------------------------

def parse_ts(series: pd.Series) -> pd.DatetimeIndex:
    ser_utc = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.DatetimeIndex(ser_utc).tz_convert("Europe/Berlin")


def validate_hourly_index(df: pd.DataFrame, name: str, expected_hours: int = 8784) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: Index ist kein DatetimeIndex")
    if df.index.freq is None:
        df = df.asfreq("H")
    if len(df.index) != expected_hours:
        raise ValueError(f"{name}: erwartet {expected_hours} Stunden, erhalten {len(df.index)}")
    return df


def force_hourly(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("force_hourly: kein DatetimeIndex")
    if df.index.freq == "H":
        return df
    
    # Convert all columns to numeric first, then filter
    df_converted = df.copy()
    for col in df.columns:
        df_converted[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only aggregate numeric columns
    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns
    df_numeric = df_converted[numeric_cols]
    
    if how == "sum":
        return df_numeric.resample("h").sum()
    return df_numeric.resample("h").mean()


def read_csv_auto_time(path: str, time_cols: List[str], expected_hours: int = 8784) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in time_cols if c in df.columns), df.columns[0])
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    df = force_hourly(df)
    return validate_hourly_index(df, Path(path).name, expected_hours)


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
            except Exception as err:  # pragma: no cover - best effort fallback
                last_err = err
                continue
    raise RuntimeError(f"CSV nicht lesbar: {path} – letzter Fehler: {last_err}")


# ------------------------------- Fleet / fuels -------------------------------

def map_fuel_to_price_and_ef(raw: str):
    t = _norm(raw)
    if any(k in t for k in ["erdgas", "erdölgas", "erdolgas", "fossilgas", " gas"]):
        return ("gas", "Erdgas")
    if "steinkohle" in t or "hard coal" in t:
        return ("coal", "Steinkohle")
    if "braunkohle" in t or "lignite" in t:
        return ("lignite", "Braunkohle")
    if "heizöl" in t or "heizoel" in t or "diesel" in t or " öl" in t or "oel" in t or " oil" in t:
        return ("oil", "Heizöl leicht / Diesel" if ("leicht" in t or "diesel" in t) else "Heizöl schwer")
    return (None, None)


def load_fleet(path: str, eta_col: Optional[str]) -> pd.DataFrame:
    df = read_csv_smart(path, min_cols=5)
    pcol = next((c for c in [
        "MW Nettonennleistung der Einheit",
        "Leistung_MW",
        "Nettonennleistung der Einheit",
        "Nettonennleistung",
        "p_mw",
        "P_MW",
    ] if c in df.columns), None)
    fcol = next((c for c in [
        "Hauptbrennstoff der Einheit",
        "Energieträger",
        "Hauptbrennstoff",
        "Brennstoff",
        "fuel",
        "Fuel",
    ] if c in df.columns), None)
    idcol = next((c for c in df.columns if "mastr" in c.lower() or "unit_id" in c.lower()), df.columns[0])
    namecol = next((c for c in df.columns if "anzeige" in c.lower() or "name" in c.lower()), idcol)
    if not pcol or not fcol:
        raise ValueError("Fleet: Leistungs- oder Brennstoffspalte fehlt.")

    eta_cols = [eta_col] if eta_col else []
    eta_cols += [
        "Effizienz",
        "Effizienz_imputiert",
        "eta",
        "Eta",
        "wirkungsgrad",
        "Imputed_Effizienz_binned",
    ]
    use_eta = next((c for c in eta_cols if c in df.columns), None)
    if use_eta is None:
        raise ValueError(f"Keine Effizienzspalte gefunden: {eta_cols}")

    out = pd.DataFrame({
        "unit_id": df[idcol].astype(str),
        "unit_name": df[namecol].astype(str),
        "fuel_raw": df[fcol].astype(str),
        "eta": pd.to_numeric(df[use_eta], errors="coerce"),
        "p_mw": pd.to_numeric(df[pcol], errors="coerce"),
    })

    price_key, ef_key = [], []
    for fuel in out["fuel_raw"]:
        pk, ek = map_fuel_to_price_and_ef(fuel)
        price_key.append(pk)
        ef_key.append(ek)
    out["price_key"] = price_key
    out["ef_key"] = ef_key
    out = out[(out["price_key"].notna()) & (out["ef_key"].notna())].copy()

    eta_clean = pd.to_numeric(out["eta"], errors="coerce").to_numpy()
    if np.nanmedian(eta_clean) > 1.5:  # Prozent → Anteil
        eta_clean = eta_clean / 100.0
    eta_clean = np.clip(eta_clean, 0.20, 0.65)
    out["eta"] = eta_clean
    out["available_mw"] = pd.to_numeric(out["p_mw"], errors="coerce").fillna(0.0).clip(lower=0).astype("float32")
    return out.dropna(subset=["eta"])


def map_unit_to_plant_type(fuel_raw: str, unit_name: str, eta: float) -> str:
    fr = str(fuel_raw or "").lower()
    un = str(unit_name or "").lower()
    if any(k in fr for k in ("nuclear", "kern", "atom")):
        return "Kernenergie"
    if "braun" in fr or "lignite" in fr:
        return "Braunkohle"
    if "stein" in fr or "coal" in fr:
        return "Steinkohle"
    if any(k in fr for k in ("heizöl", "heizoel", "oil", "diesel", "ö")):
        return "Öl"
    if "waste" in fr or "abfall" in fr or "müll" in un:
        return "Abfall"
    if "gas" in fr:
        if any(k in un for k in ("gasturb", "ocgt", "gt ")):
            return "Gasturbine"
        try:
            if eta and float(eta) >= 0.50:
                return "GuD"
        except Exception:
            pass
        return "GuD"
    return "Other"


def load_fuel_prices(path: str) -> pd.DataFrame:
    df = read_csv_auto_time(
        path,
        [
            "timestamp",
            "time",
            "datetime",
            "MTU",
        ],
    )
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = force_hourly(df, "mean")
    if "net_import_total" not in df.columns:
        imp_cols = [c for c in df.columns if c.startswith("imp_")]
        df["net_import_total"] = df[imp_cols].sum(axis=1) if imp_cols else 0.0
    return df


def prepare_flows_for_mode(flows_sched: pd.DataFrame, args) -> Tuple[pd.DataFrame, dict]:
    mode = getattr(args, "flow_mode", "scheduled")
    # Accept both 'schedule' and 'scheduled' (parser uses 'schedule' by default).
    # Treat them identically to avoid requiring --flows_physical when using schedule.
    if isinstance(mode, str) and mode.strip().lower() in ("schedule", "scheduled"):
        mode = "scheduled"
    flows_sched = flows_sched.copy()
    import_cols = [c for c in flows_sched.columns if c.startswith("imp_")]
    ambiv_mask: dict[str, pd.Series] = {}

    def _strip_net(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None:
            return None
        df = df.copy()
        if "net_import_total" in df.columns:
            df = df.drop(columns=["net_import_total"])
        return df

    if mode == "scheduled":
        flows_final = _strip_net(flows_sched)
        if flows_final is None:
            raise ValueError("scheduled flows missing")
        if import_cols:
            flows_final["net_import_total"] = flows_final[import_cols].sum(axis=1)
        else:
            flows_final["net_import_total"] = 0.0
        return flows_final, ambiv_mask

    phys_path = getattr(args, "flows_physical", None)
    if phys_path is None:
        raise ValueError(f"--flow_mode={mode} benötigt --flows_physical")

    flows_phys = load_flows(phys_path)
    flows_sched = _strip_net(flows_sched)
    flows_phys = _strip_net(flows_phys)
    if flows_sched is None or flows_phys is None:
        raise ValueError("Flows konnten nicht vorbereitet werden")

    union_index = flows_sched.index.union(flows_phys.index)
    flows_sched = flows_sched.reindex(union_index)
    flows_phys = flows_phys.reindex(union_index)

    for col in import_cols:
        if col not in flows_phys.columns:
            flows_phys[col] = 0.0
    flows_phys = flows_phys.reindex(columns=flows_sched.columns, fill_value=0.0)

    if mode == "physical":
        flows_final = flows_phys.copy()
        if import_cols:
            flows_final["net_import_total"] = flows_final[import_cols].sum(axis=1)
        else:
            flows_final["net_import_total"] = 0.0
        return flows_final, ambiv_mask

    # hybrid mode
    threshold = float(getattr(args, "flow_hybrid_ambiv_threshold", 0.40))
    flows_final = flows_phys.copy()
    eps = 1e-6
    for col in import_cols:
        s = flows_sched[col].astype(float)
        p = flows_phys[col].astype(float)
        new = p.copy()
        mask_missing = p.isna()
        new.loc[mask_missing] = s.loc[mask_missing]
        mask_import = (s > 0.0) & (p > 0.0)
        new.loc[mask_import] = np.minimum(s.loc[mask_import], p.loc[mask_import])
        mask_conflict = (s > 0.0) & (~mask_import) & (~mask_missing)
        new.loc[mask_conflict] = 0.0
        new = new.fillna(0.0)
        flows_final[col] = new
        abs_p = p.abs()
        valid = abs_p > eps
        delta = (p - s).abs()
        ambiv = valid & (delta > threshold * abs_p)
        zone = col.replace("imp_", "")
        ambiv_mask[zone] = ambiv.fillna(False)

    if import_cols:
        flows_final["net_import_total"] = flows_final[import_cols].sum(axis=1)
    else:
        flows_final["net_import_total"] = 0.0
    return flows_final, ambiv_mask


def load_neighbor_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()), df.columns[0])
    df.index = parse_ts(df[tcol])
    df = df.drop(columns=[tcol])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return force_hourly(df, "mean")


def load_neighbor_load(path_dir: str, zone: str) -> pd.Series:
    candidates: list[Path] = []
    for variant in _zone_variants(zone):
        candidates.extend(Path(path_dir).glob(f"load_{variant}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Load-CSV fehlt: load_{zone}_2024*.csv in {path_dir}")
    df = read_csv_auto_time(
        str(sorted(candidates)[0]),
        [
            "timestamp_cec",
            "timestamp",
            "time",
            "timestamp_brussels",
            "timestamp_utc",
        ],
    )
    load_col = next((c for c in df.columns if "actualtotalload" in c.lower() or "load" in c.lower()), df.columns[0])
    return pd.to_numeric(df[load_col], errors="coerce")


def load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    candidates: list[Path] = []
    for variant in _zone_variants(zone.strip()):
        candidates.extend(Path(path_dir).glob(f"actual_gen_{variant}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Gen-CSV fehlt: actual_gen_{zone}_2024*.csv in {path_dir}")

    df_raw = read_csv_smart(str(sorted(candidates)[0]), min_cols=2)
    tcol = next((c for c in [
        "timestamp_cec",
        "timestamp",
        "time",
        "datetime",
        "MTU",
    ] if c in df_raw.columns), df_raw.columns[0])
    df_raw.index = parse_ts(df_raw[tcol])
    df_raw = df_raw.drop(columns=[tcol])

    def _is_tech_col(col_name: str) -> bool:
        lc = str(col_name).lower()
        keywords = [
            "fossil",
            "nuclear",
            "wind",
            "solar",
            "hydro",
            "biomass",
            "waste",
            "geothermal",
            "other",
        ]
        return any(k in lc for k in keywords)

    wide_cols = [c for c in df_raw.columns if _is_tech_col(c)]
    if len(wide_cols) >= 2:
        df_wide = df_raw.copy()
    else:
        cols_lc = {c.lower(): c for c in df_raw.columns}
        tech_col = next((cols_lc[c] for c in cols_lc if any(k in c for k in ("productiontype", "type", "technology", "tech", "fuel"))), None)
        val_col = next((cols_lc[c] for c in cols_lc if any(k in c for k in ("actual", "generation", "gen", "mw", "value"))), None)
        if tech_col is None or val_col is None:
            raise ValueError(f"Unbekanntes Gen-Format in {candidates[0].name}: brauche Tech- und Wertspalte")
        df_wide = (
            df_raw
            .assign(**{val_col: pd.to_numeric(df_raw[val_col], errors="coerce")})
            .pivot_table(index=df_raw.index, columns=tech_col, values=val_col, aggfunc="sum")
        )

    alias_map = {
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
    }
    df_wide = df_wide.rename(columns=lambda c: alias_map.get(str(c), c))
    df_wide = df_wide.sort_index()
    df_wide = force_hourly(df_wide, "mean")
    return df_wide


def load_neighbor_fleet(path: str) -> Tuple[dict, dict]:
    df = read_csv_smart(path, min_cols=3)
    cols = {c.lower(): c for c in df.columns}
    zcol = cols.get("zone") or cols.get("bidding_zone") or cols.get("country") or list(df.columns)[0]
    fcol = cols.get("fuel") or cols.get("brennstoff") or cols.get("energieträger") or cols.get("energietraeger")
    pcol = cols.get("capacity_mw") or cols.get("leistung_mw") or cols.get("mw") or None
    if fcol is None:
        raise ValueError("neighbor_fleet: 'fuel' fehlt.")
    if zcol is None:
        raise ValueError("neighbor_fleet: 'zone' fehlt.")

    df["_zone"] = df[zcol].map(_norm_zone)
    df["_fuel"] = df[fcol].map(_map_neighbor_fuel)
    df["_cap"] = pd.to_numeric(df[pcol], errors="coerce").fillna(0.0) if pcol is not None else 0.0
    df["_eta"] = df.apply(_eta_from_row, axis=1)
    df = df[df["_fuel"].notna() & df["_zone"].notna()].copy()

    cap_mask = {(z, f): float(sub["_cap"].sum()) for (z, f), sub in df.groupby(["_zone", "_fuel"], dropna=True)}
    nei_dists_zonal: dict = {}
    for (zone, fuel), sub in df.groupby(["_zone", "_fuel"], dropna=True):
        etas = pd.to_numeric(sub["_eta"], errors="coerce").dropna()
        if len(etas) == 0:
            continue
        mean_eta = float(etas.mean())
        std_eta = float(np.std(etas)) if len(etas) > 1 else max(0.02, mean_eta / 12.0)
        lo = float(np.quantile(etas, 0.05)) if len(etas) >= 5 else max(0.20, mean_eta - 2 * std_eta)
        hi = float(np.quantile(etas, 0.95)) if len(etas) >= 5 else min(0.65, mean_eta + 2 * std_eta)
        nei_dists_zonal.setdefault(zone, {})[fuel] = {
            "mean": mean_eta,
            "std": std_eta,
            "min": lo,
            "max": hi,
        }
    return nei_dists_zonal, cap_mask



