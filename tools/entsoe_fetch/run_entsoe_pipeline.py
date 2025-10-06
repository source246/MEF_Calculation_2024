
#!/usr/bin/env python3
"""Convenience wrapper to fetch ENTSO-E data for multiple zones and run balance analysis."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


FETCH_SCRIPT = Path(__file__).with_name("fetch_entsoe_data.py")
ANALYZE_SCRIPT = Path(__file__).with_name("analyze_balance.py")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch flows/load/gen/prices for multiple zones and analyse balances")
    p.add_argument("--zones", nargs="+", required=True,
                   help="Bidding zones (first zone will be used as analysis base unless --analysis-base is set)")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD, exclusive)")
    p.add_argument("--outdir", required=True, help="Output directory for fetch results")
    p.add_argument("--include-actual-flows", action="store_true",
                   help="Request physical flows (A11) in addition to scheduled flows")
    p.add_argument("--analysis-base", default=None,
                   help="Zone to use as base for balance analysis (default: first zone in --zones)")
    p.add_argument("--price-threshold", type=float, default=5.0,
                   help="Preis-Schwelle in €/MWh für Low-Price-Surplus")
    p.add_argument("--compare-flows", action="store_true",
                   help="Vergleiche geplante vs. physische Flüsse für Analyse-Basiszone")
    p.add_argument("--extra-fetch-args", nargs=argparse.REMAINDER,
                   help="Zusätzliche Argumente, die an fetch_entsoe_data.py weitergereicht werden (nach '--')")
    return p.parse_args(argv)


def run_cmd(cmd: List[str]) -> None:
    print(f"[PIPELINE] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    zones = args.zones
    if len(zones) < 1:
        print("Need at least one zone", file=sys.stderr)
        return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extra_fetch = args.extra_fetch_args or []

    # Fetch for each zone with all other zones as neighbours
    for zone in zones:
        neighbours = [z for z in zones if z != zone]
        cmd = [sys.executable, str(FETCH_SCRIPT),
               "--base-zone", zone,
               "--start", args.start,
               "--end", args.end,
               "--outdir", str(outdir)]
        if neighbours:
            cmd.extend(["--neighbors", *neighbours])
        if args.include_actual_flows:
            cmd.append("--include-actual-flows")
        if extra_fetch:
            cmd.extend(extra_fetch)
        run_cmd(cmd)

    analysis_base = args.analysis_base or zones[0]
    other_zones = [z for z in zones if z != analysis_base]

    analyze_cmd = [sys.executable, str(ANALYZE_SCRIPT),
                   "--data-dir", str(outdir),
                   "--base-zone", analysis_base,
                   "--year", str(Path(args.start).name[:4] if args.start else 2024),
                   "--price-threshold", str(args.price_threshold)]

    if other_zones:
        analyze_cmd.extend(["--zones", *other_zones])
    if args.compare_flows:
        analyze_cmd.append("--compare-flows")

    run_cmd(analyze_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
