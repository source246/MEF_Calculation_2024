
#!/usr/bin/env python3
"""Analyse ENTSO-E datasets for hourly balance and flow consistency.

- For each requested zone compute hourly balance: generation - load + net_import.
- Flag hours where balance > 0 and price <= threshold (if price data available).
- Optionally compare scheduled vs. physical flows for base zone.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("balance")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse ENTSO-E balances and flows")
    p.add_argument("--data-dir", required=True, help="Directory with entsoe_fetch outputs")
    p.add_argument("--base-zone", required=True, help="Base bidding zone (e.g. DE_LU)")
    p.add_argument("--zones", nargs="*", default=None,
                   help="Additional zones to analyse (default: only base zone)")
    p.add_argument("--year", type=int, default=2024, help="Target year (default: 2024)")
    p.add_argument("--price-threshold", type=float, default=5.0,
                   help="Preis-Schwelle in €/MWh für Low-Price-Markierung")
    p.add_argument("--out-dir", default=None,
                   help="Optionaler Output-Ordner (default: data-dir)")
    p.add_argument("--compare-flows", action="store_true",
                   help="Vergleiche geplante und physische Flüsse für die Basiszone")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def _read_csv_timeindex(path: Path, time_col_opts: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in time_col_opts:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: "time"})
            return df.set_index("time").sort_index()
    raise ValueError(f"Keine Zeitspalte in {path}")


def read_flow_file(base_dir: Path, zone: str, year: int, prefix: str = "flows_scheduled") -> pd.DataFrame:
    path = base_dir / "flows" / f"{prefix}_{zone}_{year}_net.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv_timeindex(path, ["time", "timestamp"])
    if "net_import_total" not in df.columns:
        raise ValueError(f"Spalte 'net_import_total' fehlt in {path}")
    df["net_import_total"] = pd.to_numeric(df["net_import_total"], errors="coerce")
    return df


def read_load_series(base_dir: Path, zone: str, year: int) -> pd.Series:
    path = base_dir / "load" / f"load_{zone}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv_timeindex(path, ["timestamp", "timestamp_cec", "time"])
    val_col = next((c for c in df.columns if c not in ("time",)), None)
    if val_col is None:
        raise ValueError(f"Keine Lastspalte in {path}")
    ser = pd.to_numeric(df[val_col], errors="coerce")
    ser = ser.resample("1h").mean()
    return ser


def read_generation_series(base_dir: Path, zone: str, year: int) -> pd.Series:
    path = base_dir / "gen" / f"actual_gen_{zone}_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv_timeindex(path, ["timestamp", "time"])
    if df.empty or not len(df.columns):
        raise ValueError(f"Keine Generationsspalten in {path}")
    df_numeric = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    total = df_numeric.sum(axis=1, min_count=1)
    total = total.resample("1h").mean()
    return total


def read_price_series(base_dir: Path, zone: str) -> Optional[pd.Series]:
    price_file = base_dir / "prices" / f"prices_{zone}_neighbors.csv"
    if not price_file.exists():
        return None
    df = _read_csv_timeindex(price_file, ["time", "timestamp"])
    col = f"price_{zone}"
    if col not in df.columns:
        LOGGER.warning("Preis-Spalte %s nicht gefunden in %s", col, price_file)
        return None
    ser = pd.to_numeric(df[col], errors="coerce")
    ser = ser.resample("1h").mean()
    return ser


def analyse_zone(base_dir: Path, zone: str, year: int, price_threshold: float, out_dir: Path) -> None:
    try:
        flows = read_flow_file(base_dir, zone, year)
    except FileNotFoundError:
        LOGGER.warning("Flows für %s %s nicht gefunden", zone, year)
        return
    try:
        load = read_load_series(base_dir, zone, year)
    except FileNotFoundError:
        LOGGER.warning("Load-Datei für %s %s nicht gefunden", zone, year)
        load = pd.Series(dtype="float64")
    try:
        gen = read_generation_series(base_dir, zone, year)
    except FileNotFoundError:
        LOGGER.warning("Generation-Datei für %s %s nicht gefunden", zone, year)
        gen = pd.Series(dtype="float64")
    price = read_price_series(base_dir, zone)

    idx = flows.index.union(load.index).union(gen.index)
    df = pd.DataFrame(index=idx)
    df["generation_MW"] = gen.reindex(idx)
    df["load_MW"] = load.reindex(idx)
    df["net_import_MW"] = flows["net_import_total"].reindex(idx)
    df = df.sort_index()
    df = df.interpolate(limit=1)
    df["balance_MW"] = df["generation_MW"] - df["load_MW"] + df["net_import_MW"]
    if price is not None:
        price_idx = price.reindex(df.index)
        df["price_EUR_MWh"] = price_idx
        df["low_price_surplus"] = (df["balance_MW"] > 0.0) & (price_idx <= price_threshold)
    else:
        df["low_price_surplus"] = df["balance_MW"] > 0.0

    out_csv = out_dir / f"balance_{zone}_{year}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=True, date_format="%Y-%m-%dT%H:%M:%S")
    LOGGER.info("[%s] Balance-Datei geschrieben: %s", zone, out_csv)
    surplus_hours = df[df["low_price_surplus"]]
    LOGGER.info("[%s] Stunden mit Balance>0 und Preis <= %.2f €/MWh: %d", zone, price_threshold, surplus_hours.shape[0])
    if not surplus_hours.empty:
        display_cols = ["generation_MW", "load_MW", "net_import_MW", "balance_MW"]
        if "price_EUR_MWh" in surplus_hours:
            display_cols.append("price_EUR_MWh")
        top5 = surplus_hours[display_cols].nlargest(5, "balance_MW")
        LOGGER.info("[%s] Top 5 Surplus-Stunden:\n%s", zone, top5)


def compare_flows(base_dir: Path, base_zone: str, year: int, out_dir: Path) -> None:
    try:
        sched = read_flow_file(base_dir, base_zone, year)
    except FileNotFoundError:
        LOGGER.warning("Keine geplanten Flüsse für %s %s", base_zone, year)
        return
    try:
        actual = read_flow_file(base_dir, base_zone, year, prefix="flows_actual")
    except FileNotFoundError:
        LOGGER.warning("Keine physischen Flüsse für %s %s", base_zone, year)
        return
    idx = sched.index.union(actual.index)
    df = pd.DataFrame(index=idx)
    for col in sched.columns:
        if col == "net_import_total":
            df[f"scheduled_{col}"] = sched[col].reindex(idx)
            act_col = actual.get(col)
            if act_col is not None:
                df[f"actual_{col}"] = act_col.reindex(idx)
                df[f"diff_{col}"] = df[f"actual_{col}"] - df[f"scheduled_{col}"]
        elif col.startswith("imp_"):
            df[col] = sched[col].reindex(idx)
            if col in actual:
                df[f"actual_{col}"] = actual[col].reindex(idx)
                df[f"diff_{col}"] = df[f"actual_{col}"] - df[col]
    out_file = out_dir / f"flows_compare_{base_zone}_{year}.csv"
    df.to_csv(out_file, index=True, date_format="%Y-%m-%dT%H:%M:%S")
    LOGGER.info("Flows scheduled vs. actual gespeichert: %s", out_file)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    base_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base_dir

    zones = args.zones if args.zones else [args.base_zone]
    for zone in zones:
        analyse_zone(base_dir, zone, args.year, args.price_threshold, out_dir)

    if args.compare_flows:
        compare_flows(base_dir, args.base_zone, args.year, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
