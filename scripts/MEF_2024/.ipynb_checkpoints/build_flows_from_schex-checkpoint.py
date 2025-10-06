# scripts/track_c/build_flows_from_schex.py
import argparse
from pathlib import Path
import pandas as pd

# alle Zonen, inkl. zusammengesetzter
KNOWN_ZONES = {
    "DE", "DE_LU", "AT", "BE", "CH", "CZ", "FR", "NL", "PL",
    "DK_1", "DK_2", "DK1", "DK2", "NO_2", "SE_4", "NO2", "SE4"
}

def normalize_zone(tokens):
    """
    Nimmt eine Token-Liste (z.B. ['DE','LU','DK','1']) und erzeugt
    ['DE_LU','DK_1'] bzw. ['FR','DE_LU'] etc.
    """
    out = []
    i = 0
    while i < len(tokens):
        # 3er-Kombis zuerst (z.B. DE LU XX) -> DE_LU
        if i+1 < len(tokens) and tokens[i] == "DE" and tokens[i+1] == "LU":
            out.append("DE_LU"); i += 2; continue
        # DK_1 / DK_2
        if i+1 < len(tokens) and tokens[i] == "DK" and tokens[i+1] in {"1","2"}:
            out.append(f"DK_{tokens[i+1]}"); i += 2; continue
        # NO_2 / SE_4
        if i+1 < len(tokens) and tokens[i] in {"NO","SE"} and tokens[i+1].isdigit():
            out.append(f"{tokens[i]}_{tokens[i+1]}"); i += 2; continue
        # Einzeltoken (AT, BE, FR, NL, PL, CH, CZ, DE, DK1/DK2/NO2/SE4)
        cand = tokens[i]
        # normalize DK1 -> DK_1 etc.
        if cand in {"DK1","DK2","NO2","SE4"}:
            cand = cand[:-1] + "_" + cand[-1]
        out.append(cand)
        i += 1
    return out

def parse_side_tokens(stem: str):
    """
    stem 'schex_DE_LU_DK_1_2024' -> ('DE_LU','DK_1','2024')
    stem 'schex_FR_DE_LU_2024'   -> ('FR','DE_LU','2024')
    """
    if not stem.startswith("schex_"):
        raise ValueError(f"{stem}: expected to start with 'schex_'")
    parts = stem.replace("schex_","",1).split("_")
    year = parts[-1]             # letztes Element ist das Jahr
    body_tokens = parts[:-1]     # Rest: FROM und TO (evtl. mehrteilig)
    norm = normalize_zone(body_tokens)
    if len(norm) != 2:
        raise ValueError(f"{stem}: cannot infer FROM/TO (tokens after normalize: {norm})")
    from_side, to_side = norm[0], norm[1]
    if from_side not in KNOWN_ZONES or to_side not in KNOWN_ZONES:
        # kein harter Fehler; wir lassen es trotzdem durch
        pass
    return from_side, to_side, year

def load_schex_csv(path: Path, timecol: str):
    df = pd.read_csv(path)

    if timecol not in df.columns:
        raise ValueError(f"{path.name}: expected a time column '{timecol}', got {list(df.columns)}")

    # robust gegen CET/CEST + Offsets: direkt UTC-parsen
    t = pd.to_datetime(df[timecol], errors='coerce', utc=True)
    # als naive UTC-Zeit (wir arbeiten UTC-naiv)
    t_utc = t.dt.tz_convert('UTC').dt.tz_localize(None)

    # erste Spalte, die mit 'scheduled' beginnt
    pcols = [c for c in df.columns if str(c).lower().startswith('scheduled')]
    if not pcols:
        raise ValueError(f"{path.name}: no 'scheduled_*' column found")
    mw = pd.to_numeric(df[pcols[0]], errors='coerce')

    out = pd.DataFrame({'time': t_utc, 'mw': mw}) \
            .dropna(subset=['time']) \
            .sort_values('time') \
            .groupby('time', as_index=False)['mw'].sum()
    return out

def main():
    ap = argparse.ArgumentParser(description="Aggregate bilateral Scheduled Commercial Exchanges into a single wide flows CSV")
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timecol", default="timestamp_cec")
    args = ap.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob("schex_*.csv"))
    if not files:
        raise SystemExit(f"No files under {indir} matching schex_*.csv")

    wide = None
    for f in files:
        from_side, to_side, year = parse_side_tokens(f.stem)
        col = f"{from_side}->{to_side}"
        df = load_schex_csv(f, args.timecol).rename(columns={'mw': col})
        wide = df if wide is None else pd.merge(wide, df, on='time', how='outer')

    wide = wide.sort_values('time').reset_index(drop=True)
    wide.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with columns: {list(wide.columns)}")

if __name__ == "__main__":
    main()

