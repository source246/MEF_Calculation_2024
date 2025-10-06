# scripts/track_a/mef_a_price_anchor_opt_raw.py
import argparse, json, math, os
import pandas as pd
import numpy as np

THERM_MAP = {
    # ENTSO-E Namen -> interne Fuelkeys
    "Fossil Gas": "gas",
    "Fossil Hard coal": "hardcoal",
    "Fossil Brown coal/Lignite": "lignite",
    "Fossil Oil": "oil",
    # Varianten/Schreibweisen
    "Fossil Brown coal": "lignite",
    "Fossil Hard Coal": "hardcoal",
    "Fossil Coal": "hardcoal",
}
EE_KEYS = {
    "Wind Onshore": "wind_on",
    "Wind Offshore": "wind_off",
    "Solar": "solar",
    "Hydro Run-of-river and poundage": "ror",
    "Hydro Water Reservoir": "hydro_res",
}

EF_TH = { # tCO2/MWh_th
    "gas": 0.202, "hardcoal": 0.340, "lignite": 0.364, "oil": 0.267
}
ETA = { # default Wirkungsgrade elektrisch
    "gas": 0.55, "hardcoal": 0.40, "lignite": 0.35, "oil": 0.38
}
VOM = {"gas": 2.0, "hardcoal": 3.0, "lignite": 3.0, "oil": 4.0}  # €/MWh_el Heuristik
EF_EL = {k: EF_TH[k]/ETA[k] for k in EF_TH}  # tCO2/MWh_el

def _read_csv_any(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    # sehr tolerantes Einlesen
    df = pd.read_csv(path)
    # Kandidaten-Zeitspalten
    for zc in ["time","timestamp","timestamp_cec","Datetime","datetime","TIMESTAMP","date"]:
        if zc in df.columns:
            t = pd.to_datetime(df[zc], errors="coerce", utc=True)
            if t.notna().any():
                df = df.loc[t.notna()].copy()
                df["time"] = t.loc[t.notna()]
                df = df.drop(columns=[c for c in df.columns if c != "time" and c.startswith("Unnamed")], errors="ignore")
                df = df.set_index("time").sort_index()
                return df
    # Manche ENTSO-E CSVs haben 2 Header-Zeilen (Label/Qualifier)
    # Versuche erneut mit header=[0,1] und flache Spaltennamen
    df2 = pd.read_csv(path, header=[0,1])
    df2.columns = [a if b.startswith("Unnamed") else a for a,b in df2.columns]
    # Zeit suchen
    for zc in df2.columns:
        if "time" in zc.lower() or "timestamp" in zc.lower():
            t = pd.to_datetime(df2[zc], errors="coerce", utc=True)
            df2 = df2.loc[t.notna()].copy()
            df2["time"] = t.loc[t.notna()]
            df2 = df2.set_index("time").sort_index()
            # evtl zweite Kopfzeile loswerden
            return df2
    raise RuntimeError(f"CSV unlesbar: {path}")

def read_entsoe_gen_csv(path):
    df = _read_csv_any(path)
    # Spalten bereinigen: wir behalten nur uns wichtige (therm + EE)
    keep = []
    for c in df.columns:
        base = c.split("(")[0].strip()
        if base in THERM_MAP or base in EE_KEYS or base in ["Total Load","Actual Total Load","Actual Consumption"]:
            keep.append(c)
    if not keep:
        # Fallback: nimm alle Spalten und filtere später per Map
        keep = df.columns.tolist()
    df = df[keep].copy()
    # Duplikatspalten (z.B. Solar und Solar (Consumption)) -> sum
    df = df.groupby(level=0, axis=1).sum()
    # 15-min → 60-min mitteln
    if (df.index.to_series().diff().dropna().dt.total_seconds().mode().iloc[0] < 3600):
        df = df.resample("1H").mean()
    return df

def read_timeseries(path, timecol="time"):
    df = pd.read_csv(path)
    if timecol not in df.columns:
        # toleranter Versuch
        for tcol in ["time","Datetime","timestamp","date"]:
            if tcol in df.columns:
                timecol = tcol; break
    t = pd.to_datetime(df[timecol], errors="coerce", utc=True)
    df = df.loc[t.notna()].copy()
    df["time"] = t
    df = df.set_index("time").sort_index()
    return df

def read_fuels(path, timecol="time"):
    df = pd.read_csv(path)
    t = pd.to_datetime(df[timecol], errors="coerce", utc=True)
    df = df.loc[t.notna()].copy()
    df["time"] = t
    df = df.set_index("time").sort_index()
    return df[["gas_eur_mwh_th","coal_eur_mwh_th","lignite_eur_mwh_th","oil_eur_mwh_th","co2_eur_t"]].astype(float)

def read_neighbor_prices(path, idx):
    df = pd.read_csv(path)
    t = pd.to_datetime(df.iloc[:,0], errors="coerce", utc=True)  # erste Spalte = Zeit
    df = df.loc[t.notna()].copy()
    df["time"]=t
    df = df.set_index("time").sort_index()
    # alle Preis-Spalten durchlassen (price_* oder *_eur_mwh)
    price_cols = [c for c in df.columns if c != "time"]
    out = df[price_cols].copy()
    # reindex auf Hour
    out = out.reindex(idx).interpolate(limit_direction="both")
    return out

def read_flows_net(path, idx):
    df = pd.read_csv(path)
    t = pd.to_datetime(df.iloc[:,0], errors="coerce", utc=True)
    df = df.loc[t.notna()].copy()
    df["time"]=t
    df = df.set_index("time").sort_index()
    # Netto = Summe(XX->DE_LU) - Summe(DE_LU->XX)
    imp_cols = [c for c in df.columns if "_DE_LU_" in c and not c.startswith("schex_DE_LU_")]
    exp_cols = [c for c in df.columns if c.startswith("schex_DE_LU_")]
    imp = df[imp_cols].sum(axis=1) if imp_cols else 0.0
    exp = df[exp_cols].sum(axis=1) if exp_cols else 0.0
    net = (imp - exp).rename("net_import_mw").astype(float)
    return net.reindex(idx).fillna(0.0)

def srmc_series(fuels):
    # €/MWh_el
    co2 = fuels["co2_eur_t"]
    out = pd.DataFrame(index=fuels.index)
    out["gas"]      = (fuels["gas_eur_mwh_th"]/ETA["gas"]) + (co2*EF_TH["gas"]/ETA["gas"]) + VOM["gas"]
    out["hardcoal"] = (fuels["coal_eur_mwh_th"]/ETA["hardcoal"]) + (co2*EF_TH["hardcoal"]/ETA["hardcoal"]) + VOM["hardcoal"]
    out["lignite"]  = (fuels["lignite_eur_mwh_th"]/ETA["lignite"]) + (co2*EF_TH["lignite"]/ETA["lignite"]) + VOM["lignite"]
    out["oil"]      = (fuels["oil_eur_mwh_th"]/ETA["oil"]) + (co2*EF_TH["oil"]/ETA["oil"]) + VOM["oil"]
    return out

def monthly_mustrun(gen_by_fuel, q=0.10):
    # gen_by_fuel: DataFrame[time, fuels...] MW
    if not isinstance(gen_by_fuel.index, pd.DatetimeIndex):
        raise ValueError("gen_by_fuel index muss DatetimeIndex sein")
    mr = gen_by_fuel.groupby(gen_by_fuel.index.to_period("M")).transform(lambda x: x.quantile(q))
    return mr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--de_gen_file", required=True)
    ap.add_argument("--fuels", required=True)
    ap.add_argument("--neighbor_prices", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price_band", type=float, default=8.0)
    ap.add_argument("--mustrun_q", type=float, default=0.10)
    args = ap.parse_args()

    ts = read_timeseries(args.timeseries)  # erwartet price_da_eur_mwh drin; sonst später check
    idx = ts.index

    fuels = read_fuels(args.fuels)
    fuels = fuels.reindex(idx).interpolate(limit_direction="both")
    price = None
    # Preisspalte suchen
    for c in ["price_da_eur_mwh","price","da_price","EPEX_DA","DE_LU_DA"]:
        if c in ts.columns:
            price = ts[c].astype(float).rename("price_da_eur_mwh")
            break
    if price is None:
        raise KeyError("Day-Ahead-Preisspalte (z.B. price_da_eur_mwh) nicht in timeseries gefunden.")

    de_raw = read_entsoe_gen_csv(args.de_gen_file)
    # auf selben Index
    de_raw = de_raw.reindex(idx).interpolate(limit_direction="both")

    # Technologie-Buckets bilden (MW)
    de_fuel = pd.DataFrame(index=idx, columns=["gas","hardcoal","lignite","oil"]).fillna(0.0)
    for col in de_raw.columns:
        base = col.split("(")[0].strip()
        if base in THERM_MAP:
            de_fuel[THERM_MAP[base]] = de_fuel[THERM_MAP[base]].add(pd.to_numeric(de_raw[col], errors="coerce").fillna(0.0), fill_value=0.0)

    # Must-run je Monat
    must = monthly_mustrun(de_fuel, q=float(args.mustrun_q))
    de_above = (de_fuel - must).clip(lower=0.0)

    # SRMC je Fuel
    cost = srmc_series(fuels)

    # Preisanker → marginaler Fuel (domestic)
    # Heuristik: wähle Fuel dessen SRMC dem Preis am nächsten ist, vorausgesetzt de_above[fuel] > 0
    diffs = (cost.sub(price, axis=0)).abs()
    cand = diffs.copy()
    # unzulässige (kein „above mustrun“) vermeiden durch große Differenz
    for f in ["gas","hardcoal","lignite","oil"]:
        cand.loc[de_above[f] <= 0.0, f] = np.nan
    dom_fuel = cand.idxmin(axis=1)
    # wenn alles NaN → non_thermal
    dom_fuel = dom_fuel.fillna("non_thermal")
    dom_ef = dom_fuel.map(lambda f: EF_EL.get(f, 0.0)).astype(float)

    # Importe
    net_import = read_flows_net(args.flows, idx)
    # sehr simple Import-EF: wenn Nettoimport>0 → setze EF_import ~ 0.15 t/MWh (gemischt),
    # Du kannst hier später weiter verfeinern (Nachbarpreise/Genmix)
    ef_import = pd.Series(0.0, index=idx)
    ef_import.loc[net_import > 0.0] = 0.15

    # MEF Logik:
    # - wenn domestic marginal (dom_fuel != non_thermal): nehme dom_ef
    # - wenn non_thermal & Nettoimport>0 → Import-EF
    mef = pd.Series(0.0, index=idx, dtype=float)
    mef.loc[dom_fuel != "non_thermal"] = dom_ef.loc[dom_fuel != "non_thermal"]
    mef.loc[(dom_fuel=="non_thermal") & (net_import>0)] = ef_import.loc[(dom_fuel=="non_thermal") & (net_import>0)]

    out = pd.DataFrame({
        "time": idx.tz_convert(None),
        "ef_de": dom_ef.values,
        "import_share": (net_import.clip(lower=0.0) / (net_import.abs().max() or 1.0)).values,
        "ef_import": ef_import.values,
        "mef_tco2_per_mwh": mef.values,
        "net_import_mw": net_import.values,
        "dom_marg_fuel": dom_fuel.values,
        "price_da_eur_mwh": price.values
    })
    out.to_csv(args.out, index=False)
    print(f"[Track A RAW] geschrieben: {args.out}")

if __name__ == "__main__":
    main()
