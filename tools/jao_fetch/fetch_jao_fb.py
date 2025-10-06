#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAO Publication Tool (Core + Nordic) -> FB-Gate für DE_LU

Erzeugt CSV mit:
- minNP_DE_LU, maxNP_DE_LU (aus Core/maxNetPos)
- maxBEX_NEIGHBOR->DE_LU (aus Core/maxExchanges bzw. Nordic/maxExchanges)
- optional: netpos_mw + fb_ok_netpos (Check: NetPos in [minNP, maxNP])

Konventionen:
- Canonical Hubs: DE_LU, AT, BE, CZ, FR, NL, PL, DK_1, DK_2, SE_4, NO_2, ...
- Zeit: UTC (timestamp_utc)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import pandas as pd
import requests

BASE_CORE   = "https://publicationtool.jao.eu/core/api/data"
BASE_NORDIC = "https://publicationtool.jao.eu/nordic/api/data"

UA_HEADERS = {"User-Agent": "mef-jao-fetch/1.1"}

# --------------------------
# Hub-Normalisierung
# --------------------------

def _canonize(h: str) -> str:
    """JAO-Hubstring -> kanonisch (DE_LU, DK_1, SE_4, NO_2, ...)"""
    if h is None:
        return ""
    s = str(h).strip().upper().replace(" ", "").replace("-", "_")
    # Nordic Kurzformen zu Canon:
    s = s.replace("DK1", "DK_1").replace("DK2", "DK_2").replace("SE4", "SE_4").replace("NO2", "NO_2")
    # Core 'DE' repräsentiert DE-LU
    if s == "DE":
        return "DE_LU"
    return s

def _pretty(h: str) -> str:
    return _canonize(h)

# --------------------------
# HTTP Helper
# --------------------------

def _get(base: str, endpoint: str, start_utc: str, end_utc: str, take: int = 40000) -> pd.DataFrame:
    out = []
    skip = 0
    while True:
        params = {"FromUtc": start_utc, "ToUtc": end_utc, "skip": skip, "take": take, "filter": []}
        r = requests.get(f"{base}/{endpoint}", params=params, headers=UA_HEADERS, timeout=60)
        r.raise_for_status()
        j = r.json()
        data = j.get("data") if isinstance(j, dict) else j
        if not data:
            break
        out.extend(data)
        if len(data) < take:
            break
        skip += take
    return pd.DataFrame(out)

def _find_time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        lc = c.lower()
        if "utc" in lc or "time" in lc or "mtu" in lc or "date" in lc:
            return c
    raise ValueError("Zeitspalte in JAO-DF nicht gefunden.")

# --------------------------
# maxNetPos: wide -> long
# --------------------------

def _maxnp_wide_to_long(df: pd.DataFrame, timecol: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[timecol], utc=True, errors="coerce")
    cols = list(df.columns)
    min_cols = [c for c in cols if c.lower().startswith("min")]
    max_cols = [c for c in cols if c.lower().startswith("max")]
    if not min_cols or not max_cols:
        # könnte bereits long sein
        return _maxnp_long(df, timecol)

    def suffix(c): return c[len("min"):] if c.lower().startswith("min") else c[len("max"):]
    hubs = sorted(set(suffix(c) for c in (min_cols + max_cols)))

    rows = []
    for h in hubs:
        minc = next((c for c in min_cols if c.lower() == ("min"+h).lower()), None)
        maxc = next((c for c in max_cols if c.lower() == ("max"+h).lower()), None)
        if minc is None or maxc is None:
            continue
        tmp = pd.DataFrame({
            "timestamp_utc": ts,
            "hub": [h]*len(ts),
            "min_np_mw": pd.to_numeric(df[minc], errors="coerce"),
            "max_np_mw": pd.to_numeric(df[maxc], errors="coerce"),
        })
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp_utc","hub","min_np_mw","max_np_mw"])
    out["hub"] = out["hub"].map(_canonize)
    out = out.dropna(subset=["timestamp_utc"])
    return out

def _maxnp_long(df: pd.DataFrame, timecol: str) -> pd.DataFrame:
    # Versuche generische Spaltennamen zu erraten
    cols_lc = {c.lower(): c for c in df.columns}
    hubcol = cols_lc.get("hub") or cols_lc.get("bidhub") or cols_lc.get("zone") or cols_lc.get("areaname")
    mincol = next((c for c in df.columns if "min" in c.lower() and "pos" in c.lower()), None)
    maxcol = next((c for c in df.columns if "max" in c.lower() and "pos" in c.lower()), None)
    if not all([hubcol, mincol, maxcol]):
        raise ValueError(f"Unbekannte Spalten in maxNetPos (long): {df.columns.tolist()}")
    out = df[[timecol, hubcol, mincol, maxcol]].copy()
    out.columns = ["timestamp_utc", "hub", "min_np_mw", "max_np_mw"]
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["hub"] = out["hub"].map(_canonize)
    out = out.dropna(subset=["timestamp_utc"])
    return out

def fetch_maxnp_core(start_utc: str, end_utc: str) -> pd.DataFrame:
    df = _get(BASE_CORE, "maxNetPos", start_utc, end_utc)
    if df.empty:
        return df
    tcol = _find_time_col(df)
    return _maxnp_wide_to_long(df, tcol)

def fetch_maxnp_nordic(start_utc: str, end_utc: str) -> pd.DataFrame:
    df = _get(BASE_NORDIC, "maxNetPos", start_utc, end_utc)
    if df.empty:
        return df
    tcol = _find_time_col(df)
    # Nordic liefert teils schon long; handle dennoch wide robust
    return _maxnp_wide_to_long(df, tcol)

# --------------------------
# maxExchanges / MaxBEX
# --------------------------

def _parse_maxex(df: pd.DataFrame, timecol: str) -> pd.DataFrame:
    lc = {c.lower(): c for c in df.columns}
    # mögliche Namensvarianten
    fromcol = next((v for k,v in lc.items() if ("from" in k and "hub" in k) or k in ("hubfrom", "fromhub")), None)
    tocol   = next((v for k,v in lc.items() if ("to"   in k and "hub" in k) or k in ("hubto", "tohub")), None)
    valcol  = next((v for k,v in lc.items() if ("max" in k and ("ex" in k or "bex" in k or "flow" in k))), None)
    if not all([fromcol, tocol, valcol]):
        raise ValueError(f"Unbekannte Spalten in maxExchanges: {df.columns.tolist()}")
    out = df[[timecol, fromcol, tocol, valcol]].copy()
    out.columns = ["timestamp_utc", "hub_from", "hub_to", "max_bex_mw"]
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out["hub_from"] = out["hub_from"].map(_canonize)
    out["hub_to"]   = out["hub_to"].map(_canonize)
    out["max_bex_mw"] = pd.to_numeric(out["max_bex_mw"], errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])
    return out

def fetch_maxbex_core(start_utc: str, end_utc: str) -> pd.DataFrame:
    df = _get(BASE_CORE, "maxExchanges", start_utc, end_utc)
    if df.empty: return df
    tcol = _find_time_col(df)
    return _parse_maxex(df, tcol)

def fetch_maxbex_nordic(start_utc: str, end_utc: str) -> pd.DataFrame:
    df = _get(BASE_NORDIC, "maxExchanges", start_utc, end_utc)
    if df.empty: return df
    tcol = _find_time_col(df)
    return _parse_maxex(df, tcol)

# --------------------------
# Pivoting & Gate-Bau
# --------------------------

def _pivot_maxnp_for_base(maxnp: pd.DataFrame, base_hub_canon: str) -> pd.DataFrame:
    sub = maxnp[maxnp["hub"] == base_hub_canon].copy()
    if sub.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    sub = sub.sort_values("timestamp_utc").drop_duplicates(["timestamp_utc","hub"], keep="last")
    sub = sub.set_index("timestamp_utc")[["min_np_mw","max_np_mw"]]
    sub.columns = [f"minNP_{base_hub_canon}", f"maxNP_{base_hub_canon}"]
    return sub

def _pivot_maxbex_pairs(maxbex: pd.DataFrame, base_hub_canon: str, neighbors_canon: List[str]) -> pd.DataFrame:
    recs = []
    for n in neighbors_canon:
        pair = maxbex[(maxbex["hub_from"] == n) & (maxbex["hub_to"] == base_hub_canon)].copy()
        if pair.empty:
            continue
        pair = pair.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
        pair["col"] = f"maxBEX_{n}->{base_hub_canon}"
        recs.append(pair[["timestamp_utc","col","max_bex_mw"]])
    if not recs:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    df = None
    for p in recs:
        wide = p.pivot(index="timestamp_utc", columns="col", values="max_bex_mw")
        df = wide if df is None else df.join(wide, how="outer")
    return df

def _read_netpos_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    # finde Zeit- & NetPos-Spalte heuristisch
    tcol = next((c for c in df.columns if "time" in c.lower() or "stamp" in c.lower() or "date" in c.lower()), df.columns[0])
    cand_np = [c for c in df.columns if "netpos" in c.lower() or "net_position" in c.lower() or c.upper().replace("-","_") in ("DE_LU","DE")]
    vcol = cand_np[0] if cand_np else df.columns[-1]
    out = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(df[tcol], utc=True, errors="coerce"),
        "netpos_mw": pd.to_numeric(df[vcol], errors="coerce")
    }).dropna(subset=["timestamp_utc"])
    return out

def build_fb_gate(base_hub: str,
                  neighbors: List[str],
                  start_utc: str,
                  end_utc: str,
                  outdir: str,
                  netpos_csv: Optional[str] = None) -> Path:
    base = _canonize(base_hub)
    neis = [_canonize(z) for z in neighbors]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Fetch ---
    maxnp_core   = fetch_maxnp_core(start_utc, end_utc)
    maxbex_core  = fetch_maxbex_core(start_utc, end_utc)
    maxnp_nordic = fetch_maxnp_nordic(start_utc, end_utc)
    maxbex_nord  = fetch_maxbex_nordic(start_utc, end_utc)

    # optional: Rohdaten ablegen
    (outdir / "raw").mkdir(exist_ok=True, parents=True)
    if not maxnp_core.empty:  maxnp_core.to_csv(outdir/"raw/core_maxNetPos.csv", index=False, encoding="utf-8-sig")
    if not maxbex_core.empty: maxbex_core.to_csv(outdir/"raw/core_maxExchanges.csv", index=False, encoding="utf-8-sig")
    if not maxnp_nordic.empty:  maxnp_nordic.to_csv(outdir/"raw/nordic_maxNetPos.csv", index=False, encoding="utf-8-sig")
    if not maxbex_nord.empty:   maxbex_nord.to_csv(outdir/"raw/nordic_maxExchanges.csv", index=False, encoding="utf-8-sig")

    # --- Pivot ---
    # min/max NP kommt für DE_LU nur aus Core (Nordic hat DE_LU nicht)
    tbl_np = _pivot_maxnp_for_base(maxnp_core, base)

    # MaxBEX: Core + Nordic je nach Nachbar
    tbl_bex_core = _pivot_maxbex_pairs(maxbex_core, base, neis) if not maxbex_core.empty else pd.DataFrame()
    tbl_bex_nord = _pivot_maxbex_pairs(maxbex_nord, base, neis) if not maxbex_nord.empty else pd.DataFrame()

    # Merge MaxBEX (Core priorisiert; wo Core fehlt, Nordic verwenden)
    if tbl_bex_core.empty and not tbl_bex_nord.empty:
        tbl_bex = tbl_bex_nord
    elif not tbl_bex_core.empty and tbl_bex_nord.empty:
        tbl_bex = tbl_bex_core
    elif tbl_bex_core.empty and tbl_bex_nord.empty:
        tbl_bex = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    else:
        # gleiche Spaltennamen -> combine_first
        tbl_bex = tbl_bex_core.combine_first(tbl_bex_nord)

    # --- Join NP + BEX ---
    fb_gate = tbl_np.join(tbl_bex, how="outer").sort_index()
    fb_gate.index.name = "timestamp_utc"

    # --- Optional: NetPos & fb_ok ---
    df_np = _read_netpos_csv(netpos_csv)
    if df_np is not None and not df_np.empty:
        fb_gate = fb_gate.join(df_np.set_index("timestamp_utc"), how="left")
        if f"minNP_{base}" in fb_gate.columns and f"maxNP_{base}" in fb_gate.columns:
            fb_gate["fb_ok_netpos"] = (
                (fb_gate["netpos_mw"] >= fb_gate[f"minNP_{base}"]) &
                (fb_gate["netpos_mw"] <= fb_gate[f"maxNP_{base}"])
            )

    # --- Output ---
    out_csv = outdir / f"fb_gate_{base}.csv"
    fb_gate.to_csv(out_csv, encoding="utf-8-sig")

    # Meta: welche Nachbarn hatten BEX (Core/Nordic)?
    meta = []
    for n in neis:
        col = f"maxBEX_{n}->{base}"
        src = []
        if not tbl_bex_core.empty and col in tbl_bex_core.columns: src.append("Core")
        if not tbl_bex_nord.empty and col in tbl_bex_nord.columns: src.append("Nordic")
        meta.append({"neighbor": n, "column": col, "source": "+".join(src) if src else ""})
    pd.DataFrame(meta).to_csv(outdir / f"fb_gate_sources_{base}.csv", index=False, encoding="utf-8-sig")

    return out_csv

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-hub", default="DE_LU", help="Basis-Hub (default DE_LU).")
    ap.add_argument("--neighbors", nargs="+", required=True,
                   help="Nachbarn (AT BE CZ FR NL PL DK_1 DK_2 SE_4 NO_2 ...). CH wird (mangels FB) ignoriert.")
    ap.add_argument("--start-utc", required=True, help="Start (UTC, ISO), z. B. 2023-12-31T23:00:00Z")
    ap.add_argument("--end-utc",   required=True, help="Ende (UTC, ISO), z. B. 2024-12-31T23:00:00Z")
    ap.add_argument("--outdir", required=True, help="Zielordner für CSVs.")
    ap.add_argument("--netpos-csv", default=None, help="Optional: DE_LU Net-Positionen (für fb_ok).")
    args = ap.parse_args()

    out = build_fb_gate(args.base_hub, args.neighbors, args.start_utc, args.end_utc, args.outdir, args.netpos_csv)
    print(f"[OK] FB-Gate gespeichert: {out}")

if __name__ == "__main__":
    main()
