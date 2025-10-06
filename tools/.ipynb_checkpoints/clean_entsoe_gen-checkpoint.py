import sys, os, glob, pandas as pd, numpy as np

ENC_TRY = ["utf-8-sig","utf-16","utf-16le","cp1252","latin1","utf-8"]
DELIMS  = [",",";","\t","|"]

# Aliase -> auf Standardnamen mappen
FUEL_ALIAS_MAP = {
    "Fossil Gas": ["Fossil Gas","Gas","Natural gas","Erdgas","Fossil gas"],
    "Fossil Hard coal": ["Fossil Hard coal","Hard coal","Steinkohle","Coal","Fossil Hard Coal"],
    "Fossil Brown coal": ["Fossil Brown coal","Brown coal","Braunkohle","Lignite","Fossil Brown coal/Lignite","Brown coal/Lignite"],
    "Fossil Oil": ["Fossil Oil","Oil","Öl","Fossil oil"],
    "Wind Onshore": ["Wind Onshore","Wind onshore","Onshore wind"],
    "Wind Offshore": ["Wind Offshore","Wind offshore","Offshore wind"],
    "Solar": ["Solar","Photovoltaic","PV"],
    "Hydro run-of-river": ["Hydro Run-of-river","Hydro run-of-river","Hydro run-of-river and poundage","Run-of-river","Hydro Run-of-River"],
}

TIME_ALIASES = ["time","Time","Datetime","datetime","MTU","timestamp","timestamp_utc","timestamp_cec"]

def read_loose(path):
    last=None
    for enc in ENC_TRY:
        for sep in DELIMS:
            try:
                df = pd.read_csv(path, sep=sep, header=None, encoding=enc, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last=e
    raise RuntimeError(f"Cannot read {path}: {last}")

def detect_header_rows(raw: pd.DataFrame):
    """
    Variante A (ENTSO-E 3-Zeilen-Header):
      row 0: Brennstoffe (erste Zelle oft leer)
      row 1: 'Actual Aggregated' / 'Actual Consumption' ...
      row 2: 'timestamp_...'
    Variante B (flach):
      row 0: 'timestamp_cec,...'
    -> wir suchen die 'timestamp'-Zeile (t_idx).
       Wenn die Zeile darüber plausibel wie Fuel-Namen aussieht, nehmen wir diese als Header.
    """
    t_idx = None
    for i in range(min(10, len(raw))):
        row = raw.iloc[i].astype(str).str.lower().tolist()
        if any(("timestamp" in c) or (c.strip() in ("time","mtu","datetime")) for c in row):
            t_idx = i
            break
    if t_idx is None:
        # flacher Header? nimm Zeile 0
        return 0, None, "flat"

    # Prüfe Zeile darüber (Header-Kandidaten)
    if t_idx >= 1:
        hdr_row = raw.iloc[t_idx-1].astype(str).tolist()
        # Heuristik: viele nicht-leere Strings -> Headerzeile mit Fuelnamen
        non_empty = sum(1 for x in hdr_row if str(x).strip()!="")
        if non_empty >= 3:
            return t_idx-1, t_idx, "multi"
    # sonst: wir nehmen die timestamp-Zeile als Header (Variante B)
    return t_idx, t_idx, "flat"

def canonicalize_columns(cols):
    # konsistente, eindeutige Namen
    out=[]
    seen=set()
    for c in cols:
        s = "" if pd.isna(c) else str(c).strip()
        if s == "": s = "unnamed"
        base = s
        idx = 1
        while s in seen:
            idx += 1
            s = f"{base}_{idx}"
        seen.add(s)
        out.append(s)
    return out

def pick_time_col(df):
    for t in TIME_ALIASES:
        for col in df.columns:
            if col == t or col.lower()==t.lower():
                if col != "time":
                    df = df.rename(columns={col:"time"})
                return df
    # kein Alias gefunden -> nimm erste Spalte als Zeit
    if df.columns[0] != "time":
        df = df.rename(columns={df.columns[0]:"time"})
    return df

def rename_fuels(df):
    rename={}
    for std, cands in FUEL_ALIAS_MAP.items():
        for c in df.columns:
            cl = c.strip().lower()
            for cand in cands:
                if cl == cand.strip().lower():
                    rename[c]=std
    return df.rename(columns=rename)

def clean_one(src, outdir):
    raw = read_loose(src)
    hdr_idx, t_idx, mode = detect_header_rows(raw)

    # mit erkannten Headern neu einlesen
    df = None
    for enc in ENC_TRY:
        try:
            df = pd.read_csv(src, header=hdr_idx, sep=None, engine="python", encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"Cannot re-read {src} with header row {hdr_idx}")

    df.columns = canonicalize_columns(df.columns.tolist())

    if mode == "multi":
        # erste Datenzeile ist t_idx+1 -> drop die 'Actual ...'-Zeile (die Zeile direkt unter dem Header)
        # Häufig steht in Zeile 0 nach Header „Actual Aggregated“ in vielen Spalten
        # Wir erkennen das und droppen genau 1 Zeile
        # Falls die erste Zeile nach dem Header doch Daten ist, schadet ein einmaliges Droppen nicht (sie wird eh durch timestamp-Zeile neu indiziert).
        pass  # Header bereits richtig

        # Bei multi: die Timestamp-Zeile ist nicht Teil des DataFrames → wir lesen ab hdr_idx,
        # Pandas hat bereits Spaltennamen gesetzt und erste Datenzeile ist die „Actual“-Zeile → entfernen:
        first = df.iloc[0].astype(str).str.lower()
        if (first.str.contains("actual").sum() > len(df.columns)//4) or (first.str.contains("consumption").sum() > len(df.columns)//4):
            df = df.iloc[1:].reset_index(drop=True)

    # Zeitspalte bestimmen
    df = pick_time_col(df)

    # Fuelnamen normalisieren
    df = rename_fuels(df)

    # Zeit parsen + TZ weg
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(None)

    # Numerik
    for c in df.columns:
        if c=="time": continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # relevante Spalten (falls vorhanden)
    keep = ["time","Fossil Gas","Fossil Hard coal","Fossil Brown coal","Fossil Oil",
            "Wind Onshore","Wind Offshore","Solar","Hydro run-of-river"]
    present = [c for c in keep if c in df.columns]
    if present == ["time"]:
        # nichts erkannt? dann lass alles (zum Debuggen)
        present = df.columns.tolist()

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, os.path.basename(src))
    df[present].to_csv(out, index=False)
    return out

def main():
    if len(sys.argv)<3:
        print("Usage: python clean_entsoe_gen.py <input_dir> <output_dir>")
        sys.exit(1)
    indir, outdir = sys.argv[1], sys.argv[2]
    files = glob.glob(os.path.join(indir, "actual_gen_*_2024*"))
    outs=[]
    for f in files:
        try:
            o = clean_one(f, outdir)
            outs.append(o)
            print("OK:", o)
        except Exception as e:
            print("FAIL:", f, "->", e)
    print(f"Cleaned {len(outs)} files -> {outdir}")

if __name__=="__main__":
    main()
