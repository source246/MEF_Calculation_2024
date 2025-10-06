# -*- coding: utf-8 -*-
"""
download_neighbors_load.py
Zieht Actual Total Load [6.1.A] für alle dt. Nachbar-BZNs via ENTSO-E API.
Speichert als: input/neighbors/out_load/<YEAR>/load_<ZONE>_<YEAR>.csv

Nutzung (PowerShell):
  py .\scripts\download_neighbors_load.py --start-year 2020 --end-year 2025
Optional:
  --zones "AT,BE,CH,CZ,DK_1,DK_2,FR,NL,NO_2,PL,SE_4"
  --out-root ".\input\neighbors\out_load"
  --token "DEIN-KEY"   (sonst aus ENV ENTSOE_TOKEN)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from time import sleep
from typing import List

import pandas as pd
from entsoe import EntsoePandasClient
from requests.exceptions import HTTPError

DEFAULT_ZONES = ["AT","BE","CH","CZ","DK_1","DK_2","FR","NL","NO_2","PL","SE_4"]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--zones", type=str, default=",".join(DEFAULT_ZONES),
                    help="Kommagetrennt, z.B. 'AT,BE,CH,...'")
    ap.add_argument("--out-root", type=str, default=str(Path("input")/"neighbors"/"out_load"))
    ap.add_argument("--token", type=str, default=os.environ.get("ENTSOE_TOKEN") or os.environ.get("ENTSOE_API_KEY"),
                    help="Falls nicht gesetzt, wird ENTSOE_TOKEN aus der Umgebung genutzt.")
    ap.add_argument("--tz", type=str, default="Europe/Brussels", help="API-Zeitzone (ENTSO-E erwartet Brussels).")
    ap.add_argument("--retry", type=int, default=5, help="Max. Retry-Versuche pro Request.")
    ap.add_argument("--cooldown", type=float, default=0.5, help="Sekunden Schlaf zwischen Requests (Rate-Limit-Schonung).")
    return ap.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fetch_year_zone(client: EntsoePandasClient, zone: str, year: int, tz: str, retries: int, cooldown: float) -> pd.DataFrame:
    """
    Holt 6.1.A Actual Total Load für einen BZN/Jahr und gibt IMMER ein DataFrame zurück.
    Normalisiert heterogene entsoe-py-Rückgaben (Series vs DataFrame).
    """
    start = pd.Timestamp(f"{year}-01-01 00:00", tz=tz)
    end   = pd.Timestamp(f"{year+1}-01-01 00:00", tz=tz)

    attempt = 0
    while True:
        try:
            obj = client.query_load(zone, start=start, end=end)  # Series ODER DataFrame
            if obj is None or (hasattr(obj, "empty") and obj.empty):
                return pd.DataFrame(columns=["timestamp_brussels","ActualTotalLoad_MW"]).set_index("timestamp_brussels")

            # Normalisieren auf DataFrame mit EINER Spalte "ActualTotalLoad_MW"
            if isinstance(obj, pd.Series):
                df = obj.to_frame(name="ActualTotalLoad_MW")
            elif isinstance(obj, pd.DataFrame):
                df = obj.copy()
                # Falls mehrere Spalten zurückkommen, heuristisch passende wählen
                # 1) Spalte mit "load" im Namen, 2) erste Spalte
                candidates = [c for c in df.columns if "load" in c.lower()]
                col = candidates[0] if candidates else df.columns[0]
                # sicherstellen, dass es numerisch ist
                df = df[[col]].rename(columns={col: "ActualTotalLoad_MW"})
            else:
                # Fallback – alles in einen leeren DF
                df = pd.DataFrame(columns=["ActualTotalLoad_MW"])

            # Index säubern: sortiert, eindeutig, tz-aware
            df = df.sort_index()
            if df.index.tz is None:
                # entsoe-py sollte tz-aware liefern; falls nicht, auf Brussels setzen
                df.index = df.index.tz_localize(tz)
            df = df[~df.index.duplicated(keep="first")]
            return df

        except HTTPError:
            attempt += 1
            if attempt > retries:
                raise
            sleep(cooldown * attempt)

def main():
    args = parse_args()
    zones: List[str] = [z.strip() for z in args.zones.split(",") if z.strip()]
    if not args.token:
        raise SystemExit("Kein ENTSO-E API Token gefunden. Setze ENTSOE_TOKEN oder nutze --token.")

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    client = EntsoePandasClient(api_key=args.token)

    print(f"Zonen: {zones}")
    print(f"Jahre: {args.start_year}–{args.end_year}")
    for year in range(args.start_year, args.end_year + 1):
        year_dir = out_root / f"{year}"
        ensure_dir(year_dir)

        for z in zones:
            out_file = year_dir / f"load_{z}_{year}.csv"
            if out_file.exists():
                print(f"[SKIP] {out_file} existiert bereits.")
                continue
            try:
                print(f"[DL ] {z} {year} ...")
                sdf = fetch_year_zone(client, z, year, args.tz, args.retry, args.cooldown)
                if sdf is None or sdf.empty:
                    print(f"[WARN] Keine Daten für {z} {year}.")
                    pd.DataFrame(columns=["timestamp_brussels","timestamp_cec","timestamp_utc","ActualTotalLoad_MW"]).to_csv(out_file, index=False)
                    continue
                
                # Index -> Spalte(n) + zusätzliche Timestamps
                df = sdf.copy()
                df["timestamp_brussels"] = df.index
                df["timestamp_cec"] = df.index.tz_convert("Europe/Berlin")
                df["timestamp_utc"] = df.index.tz_convert("UTC")
                df.reset_index(drop=True, inplace=True)
                df = df[["timestamp_brussels","timestamp_cec","timestamp_utc","ActualTotalLoad_MW"]]
                
                df.to_csv(out_file, index=False)
                print(f"[OK ] {out_file}")
                sleep(args.cooldown)

            except Exception as e:
                print(f"[FAIL] {z} {year}: {e}")
                # Fehler auch als Logdatei ablegen
                with open(year_dir / f"load_{z}_{year}.error.txt", "w", encoding="utf-8") as f:
                    f.write(str(e))

    print("Fertig.")

if __name__ == "__main__":
    main()
