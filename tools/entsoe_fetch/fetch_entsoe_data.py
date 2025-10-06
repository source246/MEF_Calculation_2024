
#!/usr/bin/env python3
"""Fetch ENTSO-E transparency data (flows, load, generation) and format like Track-C inputs.

Requirements
------------
- Python 3.9+
- `requests`, `pandas`, `numpy`
- An ENTSO-E Transparency API token in environment variable `ENTSOE_API_TOKEN`.

Example
-------
python tools/entsoe_fetch/fetch_entsoe_data.py     --base-zone DE_LU     --neighbors AT BE CH CZ DK_1 DK_2 FR NL NO_2 PL SE_4     --start 2024-01-01 --end 2025-01-01     --outdir entsoe_2024

Outputs
-------
- flows/flows_scheduled_<base>_<year>_net.csv (hourly import flows, positive = import)
- flows/flows_actual_<base>_<year>_net.csv (optional, physical flows)
- load/load_<zone>_<year>.csv (hourly actual total load per zone)
- gen/actual_gen_<zone>_<year>.csv (hourly actual generation per technology)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET

LOGGER = logging.getLogger("entsoe_fetch")
API_URL = "https://web-api.tp.entsoe.eu/api"

EIC_AREA = {
    "DE_LU": "10Y1001A1001A82H",
    "DE": "10Y1001A1001A83F",
    "AT": "10YAT-APG------L",
    "BE": "10YBE----------2",
    "CH": "10YCH-SWISSGRIDZ",
    "CZ": "10YCZ-CEPS-----N",
    "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M",
    "FR": "10YFR-RTE------C",
    "NL": "10YNL----------L",
    "NO_2": "10YNO-2--------T",
    "PL": "10YPL-AREA-----S",
    "SE_4": "10Y1001A1001A47J",
    "LU": "10YLU-CEGEDEL-NQ",
    "SK": "10YSK-SEPS-----K",
    "ES": "10YES-REE------0",
    "IT": "10YIT-GRTN-----B",
    "HU": "10YHU-MAVIR----U",
    "RO": "10YRO-TEL------P",
    "BG": "10YCA-BULGARIA-R",
    "SI": "10YSI-ELES-----O",
    "LT": "10YLT-1001A0008Q",
    "LV": "10YLV-1001A00074",
    "EE": "10Y1001A1001A39I",
    "GB": "10YGB----------A",
    "IE": "10YIE-1001A00010",
}

# Production type mapping (ENTSO-E code -> friendly column name)
PRODUCTION_MAP = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B03": "Fossil Coal-derived gas",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B07": "Fossil Oil shale",
    "B08": "Fossil Peat",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and poundage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
    "B21": "AC Link",
    "B22": "Transformer",
    "B23": "DC Link",
    "B24": "Substation",
}

class EntsoeClient:
    def __init__(self, token: str, sleep: float = 1.0) -> None:
        self.token = token
        self.sleep = sleep
        self.session = requests.Session()

    def _request(self, params: Dict[str, str]) -> ET.Element:
        params = {k: v for k, v in params.items() if v is not None}
        params["securityToken"] = self.token
        resp = self.session.get(API_URL, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"ENTSO-E error {resp.status_code}: {resp.text[:200]}")
        time.sleep(self.sleep)
        return ET.fromstring(resp.content)

    def fetch_exchange(self, start: datetime, end: datetime, in_domain: str, out_domain: str, document: str) -> pd.Series:
        params = {
            "documentType": document,
            "in_Domain": in_domain,
            "out_Domain": out_domain,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        root = self._request(params)
        values: Dict[datetime, float] = {}
        for ts in root.findall("{*}TimeSeries"):
            for period in ts.findall("{*}Period"):
                start_text = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution") or "PT60M"
                dt = datetime.fromisoformat(start_text)
                if resolution == "PT15M":
                    step = timedelta(minutes=15)
                elif resolution == "PT30M":
                    step = timedelta(minutes=30)
                elif resolution == "PT60M":
                    step = timedelta(hours=1)
                else:
                    raise ValueError(f"Unsupported resolution {resolution}")
                for point in period.findall("{*}Point"):
                    pos = int(point.findtext("{*}position"))
                    qty = float(point.findtext("{*}quantity"))
                    ts_dt = dt + step * (pos - 1)
                    values.setdefault(ts_dt, qty)
        if not values:
            LOGGER.warning("Empty exchange series for %s -> %s", in_domain, out_domain)
            return pd.Series(dtype="float64")
        ser = pd.Series(values).sort_index()
        idx = pd.DatetimeIndex(ser.index)
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        idx = idx.tz_convert("Europe/Berlin").tz_localize(None)
        ser.index = idx
        ser = ser.groupby(ser.index).mean()
        ser = ser.resample('1h').mean()
        return ser

    def fetch_load(self, start: datetime, end: datetime, domain: str) -> pd.Series:
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": domain,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        root = self._request(params)
        values: Dict[datetime, float] = {}
        for ts in root.findall("{*}TimeSeries"):
            for period in ts.findall("{*}Period"):
                start_text = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution") or "PT60M"
                dt = datetime.fromisoformat(start_text)
                if resolution == "PT15M":
                    step = timedelta(minutes=15)
                elif resolution == "PT30M":
                    step = timedelta(minutes=30)
                elif resolution == "PT60M":
                    step = timedelta(hours=1)
                else:
                    raise ValueError(f"Unsupported load resolution {resolution}")
                for point in period.findall("{*}Point"):
                    pos = int(point.findtext("{*}position"))
                    qty = float(point.findtext("{*}quantity"))
                    ts_dt = dt + step * (pos - 1)
                    values.setdefault(ts_dt, qty)
        ser = pd.Series(values).sort_index()
        if ser.empty:
            return ser
        idx = pd.DatetimeIndex(ser.index)
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        idx = idx.tz_convert("Europe/Berlin").tz_localize(None)
        ser.index = idx
        ser = ser.groupby(ser.index).mean()
        ser = ser.resample("1h").mean()
        return ser

    def fetch_generation(self, start: datetime, end: datetime, domain: str) -> pd.DataFrame:
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": domain,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        root = self._request(params)
        records: Dict[str, Dict[datetime, float]] = {}
        for ts in root.findall("{*}TimeSeries"):
            prod_type = ts.findtext("{*}MktPSRType/{*}psrType")
            label = PRODUCTION_MAP.get(prod_type, prod_type)
            for period in ts.findall("{*}Period"):
                start_text = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution") or "PT60M"
                dt = datetime.fromisoformat(start_text)
                if resolution == "PT15M":
                    step = timedelta(minutes=15)
                elif resolution == "PT30M":
                    step = timedelta(minutes=30)
                elif resolution == "PT60M":
                    step = timedelta(hours=1)
                else:
                    raise ValueError(f"Generation resolution {resolution} not supported")
                data = records.setdefault(label, {})
                for point in period.findall("{*}Point"):
                    pos = int(point.findtext("{*}position"))
                    qty_text = point.findtext("{*}quantity")
                    if qty_text is None:
                        continue
                    qty = float(qty_text)
                    ts_dt = dt + step * (pos - 1)
                    data[ts_dt] = qty
        if not records:
            LOGGER.warning("Empty generation series for %s", domain)
            return pd.DataFrame()
        df = pd.DataFrame(records)
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        idx = idx.tz_convert("Europe/Berlin").tz_localize(None)
        df.index = idx
        df = df.sort_index()
        return df

    def fetch_day_ahead_price(self, start: datetime, end: datetime, domain: str) -> pd.Series:
        params = {
            "documentType": "A44",
            "in_Domain": domain,
            "out_Domain": domain,
            "periodStart": start.strftime("%Y%m%d%H%M"),
            "periodEnd": end.strftime("%Y%m%d%H%M"),
        }
        root = self._request(params)
        values: Dict[datetime, float] = {}
        for ts in root.findall("{*}TimeSeries"):
            for period in ts.findall("{*}Period"):
                start_text = period.findtext("{*}timeInterval/{*}start")
                resolution = period.findtext("{*}resolution") or "PT60M"
                dt = datetime.fromisoformat(start_text)
                if resolution == "PT15M":
                    step = timedelta(minutes=15)
                elif resolution == "PT30M":
                    step = timedelta(minutes=30)
                elif resolution == "PT60M":
                    step = timedelta(hours=1)
                else:
                    raise ValueError(f"Unsupported price resolution {resolution}")
                for point in period.findall("{*}Point"):
                    pos = int(point.findtext("{*}position"))
                    amt_text = point.findtext("{*}price.amount")
                    if amt_text is None:
                        continue
                    qty = float(amt_text)
                    ts_dt = dt + step * (pos - 1)
                    values.setdefault(ts_dt, qty)
        if not values:
            LOGGER.warning("Empty price series for %s", domain)
            return pd.Series(dtype="float64")
        ser = pd.Series(values).sort_index()
        idx = pd.DatetimeIndex(ser.index)
        if idx.tz is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        idx = idx.tz_convert("Europe/Berlin").tz_localize(None)
        ser.index = idx
        ser = ser.groupby(ser.index).mean()
        ser = ser.resample('1h').mean()
        return ser


def build_pairwise_flow_dataframe(flows: Dict[tuple[str, str], pd.Series]) -> pd.DataFrame:
    valid = {k: ser for k, ser in flows.items() if ser is not None and not ser.empty}
    if not valid:
        return pd.DataFrame(columns=['time'])
    union_index = None
    for ser in valid.values():
        union_index = ser.index if union_index is None else union_index.union(ser.index)
    df = pd.DataFrame(index=union_index)
    for (src, dst), ser in flows.items():
        col = f"flow_{src}_to_{dst}"
        if ser is None or ser.empty:
            df[col] = np.nan
        else:
            df[col] = ser.reindex(union_index)
    df.insert(0, 'time', df.index)
    return df.reset_index(drop=True)


def compute_net_positions(zones: Iterable[str], flows: Dict[tuple[str, str], pd.Series]) -> pd.DataFrame:
    valid = {k: ser for k, ser in flows.items() if ser is not None and not ser.empty}
    if not valid:
        return pd.DataFrame(columns=['time', 'zone', 'net_position_mw', 'position'])
    union_index = None
    for ser in valid.values():
        union_index = ser.index if union_index is None else union_index.union(ser.index)
    if union_index is None or union_index.empty:
        return pd.DataFrame(columns=['time', 'zone', 'net_position_mw', 'position'])
    union_index = union_index.sort_values()

    def _zero_series() -> pd.Series:
        return pd.Series(0.0, index=union_index, dtype='float64')

    net_map: Dict[str, pd.Series] = {zone: _zero_series() for zone in zones}
    for (src, dst), ser in valid.items():
        aligned = ser.reindex(union_index).fillna(0.0).astype(float)
        net_map[src] = net_map.get(src, _zero_series()).add(aligned, fill_value=0.0)
        net_map[dst] = net_map.get(dst, _zero_series()).add(-aligned, fill_value=0.0)

    df_wide = pd.DataFrame(net_map, index=union_index).sort_index()
    df_wide.insert(0, 'time', df_wide.index)
    df_long = df_wide.melt(id_vars='time', var_name='zone', value_name='net_position_mw')
    tol = 1e-6
    df_long['position'] = np.where(
        df_long['net_position_mw'] > tol,
        'Export',
        np.where(df_long['net_position_mw'] < -tol, 'Import', 'Balanced')
    )
    return df_long


def build_flows_dataframe(base_zone: str, flows: Dict[str, pd.Series]) -> pd.DataFrame:
    valid = {z: ser for z, ser in flows.items() if ser is not None and not ser.empty}
    if not valid:
        return pd.DataFrame(columns=['time', 'net_import_total'])
    union_index = None
    for ser in valid.values():
        union_index = ser.index if union_index is None else union_index.union(ser.index)
    df = pd.DataFrame(index=union_index)
    for zone, ser in flows.items():
        col = f"imp_{zone}"
        if ser is None or ser.empty:
            df[col] = np.nan
        else:
            df[col] = ser.reindex(union_index)
    df.insert(0, 'time', df.index)
    df['net_import_total'] = df.filter(like='imp_').fillna(0.0).sum(axis=1)
    return df.reset_index(drop=True)



def ensure_dirs(outdir: Path) -> Dict[str, Path]:
    flows_dir = outdir / "flows"
    load_dir = outdir / "load"
    gen_dir = outdir / "gen"
    price_dir = outdir / "prices"
    pair_dir = outdir / "flows_pairwise"
    netpos_dir = outdir / "net_positions"
    flows_dir.mkdir(parents=True, exist_ok=True)
    load_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    price_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    netpos_dir.mkdir(parents=True, exist_ok=True)
    return {"flows": flows_dir, "load": load_dir, "gen": gen_dir, "prices": price_dir, "flows_pairwise": pair_dir, "netpos": netpos_dir}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch ENTSO-E flows/load/gen for Track-C inputs")
    p.add_argument("--base-zone", required=True, help="E.g. DE_LU")
    p.add_argument("--neighbors", nargs="+", required=True, help="Neighbor zones (e.g. NL FR AT)")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, exclusive)")
    p.add_argument("--outdir", required=True, help="Output base directory")
    p.add_argument("--include-actual-flows", action="store_true", help="Also store physical flows (A11)")
    p.add_argument("--all-pairs-flows", action="store_true", help="Fetch scheduled/physical flows for all ordered zone pairs")
    p.add_argument("--write-netpos", action="store_true", help="Store derived net position tables (requires --all-pairs-flows).")
    p.add_argument("--no-load", action="store_true", help="Skip load fetch")
    p.add_argument("--no-generation", action="store_true", help="Skip generation fetch")
    p.add_argument("--no-prices", action="store_true", help="Skip day-ahead price fetch")
    p.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between API calls")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def to_datetime(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    token = os.environ.get("ENTSOE_API_TOKEN")
    if not token:
        LOGGER.error("Environment variable ENTSOE_API_TOKEN missing")
        return 1

    base_code = EIC_AREA.get(args.base_zone)
    if base_code is None:
        LOGGER.error("Unknown base zone %s", args.base_zone)
        return 1

    neighbors = []
    for zone in args.neighbors:
        code = EIC_AREA.get(zone)
        if code is None:
            LOGGER.error("Unknown neighbor zone %s", zone)
            return 1
        neighbors.append(zone)

    start = to_datetime(args.start)
    end = to_datetime(args.end)
    if start >= end:
        raise ValueError("start must be < end")

    out_paths = ensure_dirs(Path(args.outdir))
    client = EntsoeClient(token=token, sleep=args.sleep)

    # Scheduled flows
    flow_series: Dict[str, pd.Series] = {}
    actual_flow_series: Dict[str, pd.Series] = {}
    pair_flow_series: Dict[tuple[str, str], pd.Series] = {}
    pair_flow_series_phys: Dict[tuple[str, str], pd.Series] = {}
    for zone in neighbors:
        zone_code = EIC_AREA[zone]
        LOGGER.info("Fetching scheduled flows %s -> %s", zone, args.base_zone)
        try:
            ser = client.fetch_exchange(start, end, in_domain=zone_code, out_domain=base_code, document="A09")
        except Exception as exc:
            LOGGER.warning("Scheduled flow fetch failed for %s -> %s: %s", zone, args.base_zone, exc)
            ser = pd.Series(dtype="float64")
        flow_series[zone] = ser

        if args.include_actual_flows:
            if ser.empty:
                LOGGER.warning("Skip physical flows for %s -> %s (no scheduled data)", zone, args.base_zone)
                continue
            LOGGER.info("Fetching physical flows %s -> %s", zone, args.base_zone)
            try:
                ser_phys = client.fetch_exchange(start, end, in_domain=zone_code, out_domain=base_code, document="A11")
            except Exception as exc:
                LOGGER.warning("Physical flow fetch failed for %s -> %s: %s", zone, args.base_zone, exc)
            else:
                actual_flow_series[zone] = ser_phys

    if not flow_series:
        LOGGER.error("No flow data fetched")
        return 1

    df_flows = build_flows_dataframe(args.base_zone, flow_series)

    if args.all_pairs_flows:
        zones_all = [args.base_zone, *neighbors]
        for src in zones_all:
            src_code = EIC_AREA[src]
            for dst in zones_all:
                if src == dst:
                    continue
                dst_code = EIC_AREA[dst]
                LOGGER.info("[pairs] Fetching scheduled flows %s -> %s", src, dst)
                try:
                    ser_pair = client.fetch_exchange(start, end, in_domain=src_code, out_domain=dst_code, document="A09")
                except Exception as exc:
                    LOGGER.warning("[pairs] Scheduled flow fetch failed for %s -> %s: %s", src, dst, exc)
                    ser_pair = pd.Series(dtype="float64")
                if not ser_pair.empty:
                    pair_flow_series[(src, dst)] = ser_pair
                if args.include_actual_flows:
                    if ser_pair.empty:
                        LOGGER.warning("[pairs] Skip physical flows for %s -> %s (no scheduled data)", src, dst)
                        continue
                    LOGGER.info("[pairs] Fetching physical flows %s -> %s", src, dst)
                    try:
                        ser_pair_phys = client.fetch_exchange(start, end, in_domain=src_code, out_domain=dst_code, document="A11")
                    except Exception as exc:
                        LOGGER.warning("[pairs] Physical flow fetch failed for %s -> %s: %s", src, dst, exc)
                    else:
                        if not ser_pair_phys.empty:
                            pair_flow_series_phys[(src, dst)] = ser_pair_phys

    year = start.year
    flow_file = out_paths["flows"] / f"flows_scheduled_{args.base_zone}_{year}_net.csv"
    df_flows.to_csv(flow_file, index=False)
    LOGGER.info("Saved scheduled flows to %s", flow_file)

    if args.include_actual_flows and actual_flow_series:
        df_flows_phys = build_flows_dataframe(args.base_zone, actual_flow_series)
        phys_file = out_paths["flows"] / f"flows_actual_{args.base_zone}_{year}_net.csv"
        df_flows_phys.to_csv(phys_file, index=False)
        LOGGER.info("Saved physical flows to %s", phys_file)
    elif args.include_actual_flows:
        LOGGER.warning("No physical flow data stored")

    if args.all_pairs_flows:
        if pair_flow_series:
            df_pair = build_pairwise_flow_dataframe(pair_flow_series)
            pair_file = out_paths["flows_pairwise"] / f"flows_scheduled_pairs_{args.base_zone}_{year}.csv"
            df_pair.to_csv(pair_file, index=False)
            LOGGER.info("Saved pairwise scheduled flows to %s", pair_file)
        else:
            LOGGER.warning("No pairwise scheduled flows stored")

        if args.include_actual_flows:
            if pair_flow_series_phys:
                df_pair_phys = build_pairwise_flow_dataframe(pair_flow_series_phys)
                pair_phys_file = out_paths["flows_pairwise"] / f"flows_actual_pairs_{args.base_zone}_{year}.csv"
                df_pair_phys.to_csv(pair_phys_file, index=False)
                LOGGER.info("Saved pairwise physical flows to %s", pair_phys_file)
            else:
                LOGGER.warning("No pairwise physical flows stored")

    # Load
    if args.write_netpos:
        if not args.all_pairs_flows:
            LOGGER.warning("--write-netpos requires --all-pairs-flows; skipping net position export.")
        else:
            zones_all = [args.base_zone, *neighbors]
            if pair_flow_series:
                netpos_sched = compute_net_positions(zones_all, pair_flow_series)
                if not netpos_sched.empty:
                    net_file = out_paths["netpos"] / f"net_positions_scheduled_{args.base_zone}_{year}.csv"
                    netpos_sched.to_csv(net_file, index=False)
                    LOGGER.info("Saved scheduled net positions to %s", net_file)
                else:
                    LOGGER.warning("No scheduled net positions stored (empty result).")
            else:
                LOGGER.warning("No pairwise scheduled flows available for net positions.")
            if args.include_actual_flows:
                if pair_flow_series_phys:
                    netpos_phys = compute_net_positions(zones_all, pair_flow_series_phys)
                    if not netpos_phys.empty:
                        net_file = out_paths["netpos"] / f"net_positions_physical_{args.base_zone}_{year}.csv"
                        netpos_phys.to_csv(net_file, index=False)
                        LOGGER.info("Saved physical net positions to %s", net_file)
                    else:
                        LOGGER.warning("No physical net positions stored (empty result).")
                else:
                    LOGGER.warning("No pairwise physical flows available for net positions.")

    # Load
    if not args.no_load:
        for zone in [args.base_zone, *neighbors]:
            zone_code = EIC_AREA[zone]
            LOGGER.info("Fetching actual load for %s", zone)
            try:
                ser_load = client.fetch_load(start, end, zone_code)
            except Exception as exc:
                LOGGER.warning("Load fetch failed for %s: %s", zone, exc)
                continue
            if ser_load.empty:
                LOGGER.warning("No load data for %s", zone)
                continue
            ser_load = ser_load.resample("1h").mean()
            df_load = pd.DataFrame({"timestamp": ser_load.index, "ActualTotalLoad_MW": ser_load.values})
            load_file = out_paths["load"] / f"load_{zone}_{year}.csv"
            df_load.to_csv(load_file, index=False)
            LOGGER.info("Saved load to %s", load_file)

    # Generation
    if not args.no_generation:
        for zone in [args.base_zone, *neighbors]:
            zone_code = EIC_AREA[zone]
            LOGGER.info("Fetching generation mix for %s", zone)
            try:
                df_gen = client.fetch_generation(start, end, zone_code)
            except Exception as exc:
                LOGGER.warning("Generation fetch failed for %s: %s", zone, exc)
                continue
            if df_gen.empty:
                LOGGER.warning("No generation data for %s", zone)
                continue
            df_gen = df_gen.resample("1h").mean()
            df_gen.insert(0, "timestamp", df_gen.index)
            gen_file = out_paths["gen"] / f"actual_gen_{zone}_{year}.csv"
            df_gen.to_csv(gen_file, index=False)
            LOGGER.info("Saved generation to %s", gen_file)

    price_series: Dict[str, pd.Series] = {}
    if not args.no_prices:
        for zone in [args.base_zone, *neighbors]:
            zone_code = EIC_AREA[zone]
            LOGGER.info("Fetching day-ahead price for %s", zone)
            try:
                ser_price = client.fetch_day_ahead_price(start, end, zone_code)
            except Exception as exc:
                LOGGER.warning("Price fetch failed for %s: %s", zone, exc)
                continue
            if ser_price.empty:
                LOGGER.warning("No price data for %s", zone)
                continue
            price_series[zone] = ser_price

        if price_series:
            union_index = None
            for ser in price_series.values():
                union_index = ser.index if union_index is None else union_index.union(ser.index)
            df_price = pd.DataFrame(index=union_index)
            for zone, ser in price_series.items():
                df_price[f"price_{zone}"] = ser.reindex(union_index)
            df_price.insert(0, "time", df_price.index)
            price_file = out_paths["prices"] / f"prices_{args.base_zone}_neighbors.csv"
            df_price.to_csv(price_file, index=False)
            LOGGER.info("Saved prices to %s", price_file)
        else:
            LOGGER.warning("No price data stored")

    LOGGER.info("ENTSO-E fetch complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
