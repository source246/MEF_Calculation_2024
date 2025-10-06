#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scale_by_peak_from_gen.py
=========================

Berechnet peak-basierte Skalierungsfaktoren je (Bidding-Zone × Fuel) aus
ENTSO-E-Generationsdateien (z. B. actual_gen_<ZONE>_2024.csv) und skaliert
eine bereits imputierte Kraftwerksliste entsprechend. Schreibt:
  - eine skalierte Anlagenliste (CSV)
  - transparente Skalierungsfaktoren (CSV)

Beispiel (PowerShell):
  py scale_by_peak_from_gen.py `
    --plants-imputed "C:\\...\\plants_imputed_eta_bz_2024_fix.csv" `
    --gen-glob "C:\\...\\gen_2024\\actual_gen_*_2024.csv" `
    --out-scaled "C:\\...\\out\\plants_scaled_peak_2024.csv" `
    --out-factors "C:\\...\\out\\scaling_factors_peak_2024.csv"

Unterstützte ENTSO-E-Dateiformate:
  A) wide + 2 Headerzeilen (Zeile1: Production Types, Zeile2: Actual Aggregated/Consumption)
     -> Es werden ausschließlich "Actual Aggregated" ausgewertet.
  B) wide + 1 Headerzeile (Spalten = Production Types + Timestamp)
  C) Fallback long (Production Type + Actual Aggregated)
"""

import argparse, glob, os, re, sys
import pandas as pd
import numpy as np

FUELS = ["gas", "hardcoal", "lignite", "oil"]
# im Skript oben bei FUEL_PATTERNS ersetzen/ergänzen:
FUEL_PATTERNS = {
    "gas": [
        r"fossil\s+gas",
        r"natural\s+gas",
        r"coal[-\s]*derived\s+gas",   # z.B. "Fossil Coal-derived gas" → als gas behandeln
        r"cogeneration\s+gas",        # falls mal auftaucht
    ],
    "hardcoal": [
        r"fossil\s+hard\s*coal",
        r"fossil\s+coal(?!.*lignite)",  # „Coal“ ohne „Lignite“ dahinter
    ],
    "lignite": [
        r"fossil\s+brown\s*coal\s*/?\s*lignite",
        r"\blignite\b",
    ],
    "oil": [
        r"fossil\s+oil",
        r"oil\s+shale",               # falls in manchen Exports so auftaucht
    ],
}


def _to_num(series_like) -> pd.Series:
    s = pd.Series(series_like).astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def detect_zone_from_filename(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"actual_gen_(.+?)_20\\d{2}", base, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"_(DK_1|DK_2|NO_2|SE_4|DE_LU|[A-Z]{2})", base)
    return m.group(1).upper() if m else os.path.splitext(base)[0].upper()

def _norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9/ ]+", " ", str(s or "").lower()).strip()

def parse_gen_peaks(path: str) -> dict:
    import pandas as pd, re
    # 1) Versuch: 2-zeilig wide (PT-Zeile + "Actual Aggregated")
    try:
        probe = pd.read_csv(path, header=None, nrows=3, sep=None, engine="python")
        row1_has_agg = probe.shape[0] >= 2 and probe.iloc[1].astype(str).str.contains("Actual Aggregated", case=False, na=False).any()
        if row1_has_agg:
            raw = pd.read_csv(path, header=None, sep=None, engine="python")
            pt_row, meas_row = raw.iloc[0].fillna(""), raw.iloc[1].fillna("")
            data = raw.iloc[3:].reset_index(drop=True)
            sel_idx = [i for i,m in enumerate(meas_row) if isinstance(m,str) and ("actual aggregated" in m.lower())]
            cols = [0] + sel_idx
            data = data.iloc[:, cols]
            new_cols = ["timestamp"] + [str(pt_row[i]).strip() for i in sel_idx]
            data.columns = new_cols
            for c in new_cols[1:]:
                data[c] = _to_num(data[c])
            pt_cols = new_cols[1:]
            return _peaks_from_wide(data, pt_cols)
    except Exception:
        pass

    # 2) Versuch: 1-zeilig wide (Timestamp-Spalte + PT-Spalten)
    try:
        hdr = pd.read_csv(path, nrows=0, sep=None, engine="python")
        ts_cols = [c for c in hdr.columns if c.lower().startswith("timestamp")]
        if ts_cols:
            df = pd.read_csv(path, sep=None, engine="python")
            ts = ts_cols[0]; pt_cols = [c for c in df.columns if c != ts]
            for c in pt_cols: df[c] = _to_num(df[c])
            data = df.rename(columns={ts:"timestamp"})
            return _peaks_from_wide(data, pt_cols)
    except Exception:
        pass

    # 3) Fallback: long (Production Type + Actual Aggregated)
    df = pd.read_csv(path, sep=None, engine="python")
    pt_col  = [c for c in df.columns if re.search(r"(?i)production\s*type", c)]
    val_col = [c for c in df.columns if re.search(r"(?i)actual\s+aggregated", c)]
    if not (pt_col and val_col):
        raise ValueError("Unbekanntes Kopf-/Spaltenformat (weder wide noch long).")
    df["production_type_n"] = df[pt_col[0]].apply(_norm_token)
    df["value"] = _to_num(df[val_col[0]])
    df["row_id"] = range(len(df))
    piv = df.pivot_table(index="row_id", columns="production_type_n", values="value", aggfunc="sum")
    return _peaks_from_pivot(piv)

def _peaks_from_wide(data, pt_cols):
    piv = data[pt_cols]  # rows=time, cols=PT
    # normiere Spaltennamen
    cols_map = {c: _norm_token(c) for c in pt_cols}
    peaks = {k: np.nan for k in FUELS}
    for fuel, pats in FUEL_PATTERNS.items():
        cols_for_fuel = [c for c,n in cols_map.items() if any(re.search(p, n) for p in pats)]
        if cols_for_fuel:
            series = piv[cols_for_fuel].sum(axis=1)
            peaks[fuel] = float(series.max()) if len(series) else np.nan
    return peaks

def _peaks_from_pivot(piv):
    peaks = {k: np.nan for k in FUELS}
    for fuel, pats in FUEL_PATTERNS.items():
        cols_for_fuel = [c for c in piv.columns if any(re.search(p, str(c), flags=re.IGNORECASE) for p in pats)]
        if cols_for_fuel:
            series = piv[cols_for_fuel].sum(axis=1)
            peaks[fuel] = float(series.max()) if len(series) else np.nan
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plants-imputed", required=True)
    ap.add_argument("--gen-glob", required=True)
    ap.add_argument("--out-scaled", required=True)
    ap.add_argument("--out-factors", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.gen_glob, recursive=True))
    print(f"[INFO] Matched {len(paths)} files for --gen-glob: {args.gen_glob}")
    for p in paths[:10]: print("   ->", p)
    if not paths: sys.exit(f"[ERROR] No files matched pattern: {args.gen_glob}")

    plants = pd.read_csv(args.plants_imputed)
    # Case-insensitive remap
    lc = {c.lower(): c for c in plants.columns}
    for need in ["zone","fuel","capacity_mw"]:
        if need not in lc: sys.exit(f"[ERROR] Column '{need}' missing in plants file.")
    plants = plants.rename(columns={lc["zone"]:"zone", lc["fuel"]:"fuel", lc["capacity_mw"]:"capacity_mw"})
    plants["capacity_mw"] = _to_num(plants["capacity_mw"])
    plants["fuel"] = plants["fuel"].astype(str).str.lower()
    plants["zone"] = plants["zone"].astype(str)
    # Zonen-Alias: passe Plants-Zonen an die Gen-Datei-Zonen an
    ZONE_ALIAS = {
        "NO": "NO_2",   # wir haben Gen-Datei für NO_2
        "SE": "SE_4",   # wir haben Gen-Datei für SE_4
        "DE": "DE_LU",  # falls noch einzelne 'DE'/'LU' auftauchen
        "LU": "DE_LU",
    }
    plants["zone"] = plants["zone"].replace(ZONE_ALIAS)

    baseline = (plants[plants["fuel"].isin(FUELS)]
                .dropna(subset=["zone","fuel","capacity_mw"])
                .groupby(["zone","fuel"], as_index=False)["capacity_mw"]
                .sum().rename(columns={"capacity_mw":"baseline_mw"}))

    factor_rows = []
    for path in paths:
        zone = detect_zone_from_filename(path)
        try:
            peaks = parse_gen_peaks(path)
        except Exception as e:
            print(f"[WARN] {os.path.basename(path)}: {e}")
            peaks = {}
        for fuel in FUELS:
            factor_rows.append({"zone": zone, "fuel": fuel, "peak_mw": peaks.get(fuel, np.nan)})

    df_fr = pd.DataFrame(factor_rows, columns=["zone","fuel","peak_mw"])
    if df_fr.empty: sys.exit("[ERROR] No peaks parsed. Check headers/format of actual_gen files.")
    peaks_df = df_fr.groupby(["zone","fuel"], as_index=False)["peak_mw"].max()

    fac = peaks_df.merge(baseline, on=["zone","fuel"], how="left")
    fac["baseline_mw"] = fac["baseline_mw"].fillna(0.0)
    fac["scale"] = np.where(fac["baseline_mw"]>0, fac["peak_mw"]/fac["baseline_mw"], np.nan)

    out_cols = ["name","country","zone","fuel","tech","commissioned","capacity_mw","eta"]
    for c in out_cols:
        if c not in plants.columns: plants[c] = np.nan
    scaled = plants[out_cols].merge(fac[["zone","fuel","scale"]], on=["zone","fuel"], how="left")
    scaled["scaled_capacity_mw"] = np.where(scaled["scale"].notna(),
                                            _to_num(scaled["capacity_mw"]) * _to_num(scaled["scale"]),
                                            _to_num(scaled["capacity_mw"]))
    scaled["scaled_capacity_mw"] = scaled["scaled_capacity_mw"].clip(lower=0)

    os.makedirs(os.path.dirname(args.out_scaled), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_factors), exist_ok=True)
    scaled.to_csv(args.out_scaled, index=False)
    fac.to_csv(args.out_factors, index=False)
    print(f"[OK] Scaled plants  → {args.out_scaled}")
    print(f"[OK] Factors (peaks)→ {args.out_factors}")

if __name__ == "__main__":
    main()
