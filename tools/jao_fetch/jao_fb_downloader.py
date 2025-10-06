#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAO Publication Tool Downloader (Core/Nordic):
- Lädt Max/Min Net Positions und Net Positions als CSV
- Chunked (tageweise), robust mit Retry
- Baut fb_boundary-Flag (slack-to-min/max mit Toleranz)

Usage:
  python jao_fb_downloader.py \
    --region core \
    --hub DE \
    --start 2024-01-01 --end 2025-01-01 \
    --maxnp_endpoint "https://publicationtool.jao.eu/core/api/data/maxNetPositions" \
    --netpos_endpoint "https://publicationtool.jao.eu/core/api/data/netPosition" \
    --out fb_core_DE_2024.csv

Notes:
- Endpoints variieren leicht; kopiere die exakte URL aus dem API-Test-Tab der JAO-Seite.
- Für Nordic analog: .../nordic/api/data/minMaxNetPositions, .../nordic/api/data/netPosition (siehe Handbook).
"""

import argparse
import sys
import time
import io
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

DEF_TIMEOUT = 30
DEF_RETRY = 4

def get_csv(url, params, timeout=DEF_TIMEOUT, retries=DEF_RETRY):
    """Download CSV with retry logic"""
    last_err = None
    for k in range(retries):
        try:
            print(f"[INFO] GET {url} with params {params} (attempt {k+1}/{retries})")
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200 and r.text.strip():
                return pd.read_csv(io.StringIO(r.text))

            # If we got a 404, try a few common alternative endpoint names (JAO API naming varies)
            if r.status_code == 404:
                alt_urls = []
                if 'minMaxNetPositions' in url:
                    alt_urls.append(url.replace('minMaxNetPositions', 'maxNetPositions'))
                if 'maxNetPositions' in url:
                    alt_urls.append(url.replace('maxNetPositions', 'minMaxNetPositions'))
                # Generic alternative: try replacing 'minMax' with 'max' if present
                if 'minMax' in url and 'minMaxNetPositions' not in url:
                    alt_urls.append(url.replace('minMax', 'max'))

                for alt in alt_urls:
                    try:
                        print(f"[INFO] Trying alternative endpoint {alt}")
                        r2 = requests.get(alt, params=params, timeout=timeout)
                        if r2.status_code == 200 and r2.text.strip():
                            return pd.read_csv(io.StringIO(r2.text))
                    except Exception as e2:
                        print(f"[WARN] Alternative endpoint {alt} failed: {e2}")

            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            print(f"[WARN] Attempt {k+1} failed: {e}")
        time.sleep(1.2*(k+1))
    raise last_err

def daterange_utc(start_dt, end_dt, step_days=7):
    """Generate date chunks for API calls"""
    t = start_dt
    while t < end_dt:
        u = min(t + timedelta(days=step_days), end_dt)
        yield t, u
        t = u

def norm_time(s):
    """JAO nutzt Business Day/MTU. API liefert DateTime-Spalte; wir übernehmen UTC."""
    return pd.to_datetime(s, utc=True, errors="coerce")

def find_column(df, keywords, description="column"):
    """Robust column finder"""
    for keyword in keywords:
        matches = [c for c in df.columns if keyword.lower() in c.lower()]
        if matches:
            return matches[0]
    raise RuntimeError(f"Cannot find {description} in columns: {df.columns.tolist()[:10]}")

def main():
    ap = argparse.ArgumentParser(description="JAO FlowBased Boundary Downloader")
    ap.add_argument("--region", choices=["core","nordic"], required=True,
                    help="JAO region (core or nordic)")
    ap.add_argument("--hub", required=True, 
                    help="Bidding zone code (z.B. DE, FR, NL, ...)")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    ap.add_argument("--maxnp_endpoint", required=True,
                    help="API URL für Min/Max Net Positions (aus JAO API Test kopieren)")
    ap.add_argument("--netpos_endpoint", required=True,
                    help="API URL für Net Position (aus JAO API Test kopieren)")
    ap.add_argument("--step_days", type=int, default=2, 
                    help="Chunk size in days (default: 2). JAO API limits ranges to 2 days.")
    ap.add_argument("--out", required=True, help="Output CSV file")
    ap.add_argument("--tau_mw", type=float, default=None,
                    help="Fixed tolerance in MW (Default: max(100 MW, 2%% of NP-Range))")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = ap.parse_args()

    if args.verbose:
        print(f"[INFO] JAO FB Downloader - Region: {args.region}, Hub: {args.hub}")
        print(f"[INFO] Period: {args.start} to {args.end}")
        print(f"[INFO] MaxNP endpoint: {args.maxnp_endpoint}")
        print(f"[INFO] NetPos endpoint: {args.netpos_endpoint}")

    tz_utc = timezone.utc
    t0 = datetime.fromisoformat(args.start).replace(tzinfo=tz_utc)
    t1 = datetime.fromisoformat(args.end).replace(tzinfo=tz_utc)

    rows = []
    total_chunks = len(list(daterange_utc(t0, t1, step_days=args.step_days)))
    
    for i, (a, b) in enumerate(daterange_utc(t0, t1, step_days=args.step_days), 1):
        print(f"[INFO] Processing chunk {i}/{total_chunks}: {a.date()} to {b.date()}")
        
        # Base parameters for JAO API
        # Note: Some JAO endpoints expect FromUtc/ToUtc (error shows this),
        # older examples used FromDateTime/ToDateTime. Try FromUtc first and
        # fall back to FromDateTime if the server rejects the request.
        base_params = {
            "FromUtc": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ToUtc":   b.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Format": "CSV"
        }

        try:
            # 1) Download Max/Min Net Positions
            print(f"[INFO] Downloading Max/Min Net Positions...")
            df_maxnp = get_csv(args.maxnp_endpoint, base_params).copy()
            
            if df_maxnp.empty:
                print(f"[WARN] No MaxNP data for chunk {i}")
                continue
                
            # Find columns robustly
            tcol = find_column(df_maxnp, ["Date", "Time", "DateTime"], "timestamp")
            hub_col = find_column(df_maxnp, ["Hub", "BZ", "Area", "BiddingZone", "Zone"], "hub")
            min_col = find_column(df_maxnp, ["min", "Min"], "min position")
            max_col = find_column(df_maxnp, ["max", "Max"], "max position")
            
            # Process MaxNP data
            df_maxnp[tcol] = norm_time(df_maxnp[tcol])
            df_maxnp = df_maxnp[df_maxnp[hub_col].astype(str).str.upper().eq(args.hub.upper())]
            df_maxnp = df_maxnp[[tcol, min_col, max_col]].rename(
                columns={tcol:"timestamp_utc", min_col:"minNP", max_col:"maxNP"}
            )
            
            if df_maxnp.empty:
                print(f"[WARN] No MaxNP data for hub {args.hub} in chunk {i}")
                continue

            # 2) Download Net Position  
            print(f"[INFO] Downloading Net Positions...")
            df_np = get_csv(args.netpos_endpoint, base_params).copy()
            
            if df_np.empty:
                print(f"[WARN] No NetPos data for chunk {i}")
                continue
                
            # Find columns for NetPos
            tcol2 = find_column(df_np, ["Date", "Time", "DateTime"], "timestamp")
            hub_col2 = find_column(df_np, ["Hub", "BZ", "Area", "BiddingZone", "Zone"], "hub") 
            np_col = find_column(df_np, ["net", "Net", "Position"], "net position")
            
            # Process NetPos data
            df_np[tcol2] = norm_time(df_np[tcol2])
            df_np = df_np[df_np[hub_col2].astype(str).str.upper().eq(args.hub.upper())]
            df_np = df_np[[tcol2, np_col]].rename(
                columns={tcol2:"timestamp_utc", np_col:"NetPosition"}
            )
            
            if df_np.empty:
                print(f"[WARN] No NetPos data for hub {args.hub} in chunk {i}")
                continue

            # 3) Merge data with time tolerance
            df_chunk = pd.merge_asof(
                df_np.sort_values("timestamp_utc"),
                df_maxnp.sort_values("timestamp_utc"),
                on="timestamp_utc", 
                direction="nearest", 
                tolerance=pd.Timedelta("30min")
            ).dropna(subset=["minNP","maxNP"])
            
            if not df_chunk.empty:
                rows.append(df_chunk)
                print(f"[INFO] Chunk {i}: {len(df_chunk)} valid records")
            else:
                print(f"[WARN] Chunk {i}: No valid merged data")
                
        except Exception as e:
            print(f"[ERROR] Failed to process chunk {i}: {e}")
            continue

    if not rows:
        raise SystemExit("No data downloaded - check endpoints/parameters/time range.")

    # Combine all chunks
    print(f"[INFO] Combining {len(rows)} chunks...")
    all_df = pd.concat(rows, axis=0, ignore_index=True).drop_duplicates(subset=["timestamp_utc"])
    print(f"[INFO] Total records after deduplication: {len(all_df)}")

    # Calculate slacks & boundary flags
    print(f"[INFO] Calculating boundary flags...")
    rng = (all_df["maxNP"] - all_df["minNP"]).clip(lower=1.0)
    
    if args.tau_mw is None:
        tau = (0.02 * rng).clip(lower=100.0)  # max(100 MW, 2% der NP-Range)
        print(f"[INFO] Using adaptive tolerance: min=100MW, avg={tau.mean():.1f}MW")
    else:
        tau = float(args.tau_mw)
        print(f"[INFO] Using fixed tolerance: {tau}MW")
    
    slack_min = (all_df["NetPosition"] - all_df["minNP"]).clip(lower=0.0)
    slack_max = (all_df["maxNP"] - all_df["NetPosition"]).clip(lower=0.0)
    
    all_df["slack_to_min"] = slack_min
    all_df["slack_to_max"] = slack_max
    all_df["fb_boundary"] = (slack_min <= tau) | (slack_max <= tau)
    
    # Statistics
    boundary_hours = all_df["fb_boundary"].sum()
    boundary_pct = 100.0 * boundary_hours / len(all_df)
    print(f"[INFO] Boundary hours: {boundary_hours}/{len(all_df)} ({boundary_pct:.1f}%)")
    
    # Save results
    all_df.sort_values("timestamp_utc").to_csv(args.out, index=False)
    print(f"[OK] Wrote {args.out} with {len(all_df)} rows")
    
    # Summary statistics
    print(f"\n[SUMMARY] JAO FlowBased Boundary Analysis:")
    print(f"  Region: {args.region}")
    print(f"  Hub: {args.hub}")
    print(f"  Period: {all_df['timestamp_utc'].min()} to {all_df['timestamp_utc'].max()}")
    print(f"  NetPosition range: {all_df['NetPosition'].min():.0f} to {all_df['NetPosition'].max():.0f} MW")
    print(f"  MinNP range: {all_df['minNP'].min():.0f} to {all_df['minNP'].max():.0f} MW")
    print(f"  MaxNP range: {all_df['maxNP'].min():.0f} to {all_df['maxNP'].max():.0f} MW")
    print(f"  Boundary flag: {boundary_pct:.1f}% of hours")

if __name__ == "__main__":
    sys.exit(main())
