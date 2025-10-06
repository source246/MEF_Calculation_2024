#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np

TZ = "Europe/Berlin"

# ----------------- helpers: time & resample -----------------
def parse_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    cand = [c for c in df.columns if c.lower() in {"timestamp","time","datetime","mtu (utc)","mtu_start","date_time","datum"}]
    col = cand[0] if cand else df.columns[0]
    ts = pd.to_datetime(df[col], errors="coerce", utc=True)
    if ts.isna().all():
        ts = pd.to_datetime(df[col], errors="coerce")
        ts = ts.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    else:
        ts = ts.dt.tz_convert(TZ)
    return ts

def as_hourly(x, how="mean"):
    if isinstance(x, pd.Series):
        return (x.resample("1h").mean() if how=="mean" else x.resample("1h").sum())
    return (x.resample("1h").mean() if how=="mean" else x.resample("1h").sum())

# ----------------- subtype mapping (eff table) --------------
def clean_subtype(name: str) -> str:
    if name is None: return ""
    s = str(name)
    s = re.sub(r"\s*\[.*?\]\s*$", "", s)
    return s.strip()

def build_eff_map_from_json(path: str, known_subtypes: list[str]) -> pd.DataFrame:
    """
    Liest TYNDP JSON (Liste von Einträgen mit fuel/type/eta_std_ncv/co2_ef_t_per_mwh_th/vom_eur_per_mwh_el)
    und erzeugt eine Mapping-Tabelle auf deine Subtypen (Spaltennamen aus Gen-CSV).
    key = clean_subtype(subtype); cols: fuel, eta, varom_eur_mwh, co2_th_t_per_mwh, mef_gpkwh (aus co2_th/eta)
    """
    p = Path(path)
    if p.suffix.lower() != ".json":
        # Falls doch eine CSV/XLS kam, kannst du hier zur Not deinen alten Loader rufen
        raise ValueError("Bitte JSON-Datei an --eff übergeben (tyndp_2024_efficiencies_full.json).")

    raw = pd.read_json(path)

    # Index nach (fuel, type)
    # Normieren (lower)
    raw["fuel_n"] = raw["fuel"].astype(str).str.strip().str.lower()
    raw["type_n"] = raw["type"].astype(str).str.strip().str.lower()
    raw = raw.set_index(["fuel_n","type_n"])

    # kleine Helfer zum Suchen
    def pick_row(fuel_n, type_n):
        if (fuel_n, type_n) in raw.index:
            return raw.loc[(fuel_n, type_n)]
        # Fallbacks: CCS → enthält "ccs"; present/old/new Varianten tolerant
        # z.B. wenn type_n = "ccgt present 2" existiert, sonst "ccgt present 1"
        # Wir versuchen ein paar einfache Heuristiken:
        # 1) exakter fuel, irgendein type, der alle tokens von type_n enthält
        tokens = [t for t in type_n.split() if t]
        cand = raw.loc[fuel_n]
        mask = np.ones(len(cand), dtype=bool)
        for t in tokens:
            mask &= cand.index.get_level_values("type_n").str.contains(rf"\b{re.escape(t)}\b")
        if mask.any():
            return cand[mask].iloc[0]
        # 2) exakter fuel, erster Eintrag
        return cand.iloc[0]

    # Parser der Subtypen aus Spaltennamen
    def parse_subtype(s: str):
        s0 = clean_subtype(s).lower()

        # Fuel
        if "natural gas" in s0 or "gas" in s0:
            fuel = "natural gas"
        elif "hard coal" in s0 or "coal" in s0:
            fuel = "hard coal"
        elif "lignite" in s0 or "braunkohle" in s0:
            fuel = "lignite"
        elif "light oil" in s0 or "diesel" in s0:
            fuel = "light oil"
        elif "heavy oil" in s0 or "oil" in s0:
            fuel = "heavy oil"
        elif "oil shale" in s0 or "shale" in s0:
            fuel = "oil shale"
        elif "nuclear" in s0:
            fuel = "nuclear"
        elif "hydrogen" in s0 or "h2" in s0:
            # falls du H2-Generatoren als Subtypen hast
            fuel = "hydrogen"  # nicht in deiner JSON – dann wird ef_th=0, varom wie Gas angenommen
        else:
            fuel = None

        # Type
        tkn = []
        if "ccgt" in s0: tkn.append("ccgt")
        if "ocgt" in s0: tkn.append("ocgt")
        if "conventional" in s0: tkn.append("conventional")
        if "present 2" in s0: tkn += ["present","2"]
        if "present 1" in s0: tkn += ["present","1"]
        if "present" in s0 and "present 1" not in s0 and "present 2" not in s0: tkn.append("present")
        if "old 2" in s0: tkn += ["old","2"]
        if "old 1" in s0: tkn += ["old","1"]
        if re.search(r"\bold\b", s0) and "old 1" not in s0 and "old 2" not in s0: tkn.append("old")
        if "new" in s0: tkn.append("new")
        if "ccs" in s0: tkn.append("ccs")

        # Default-Typ, falls keiner erkannt:
        if not tkn:
            if fuel in ("natural gas","hard coal","lignite"): tkn = ["old"]  # konservativ
            else: tkn = ["-"]

        type_guess = " ".join(tkn).strip()
        return fuel, type_guess

    rows = []
    for col in known_subtypes:
        key = clean_subtype(col)
        fuel, type_guess = parse_subtype(key)
        if fuel is None:
            continue  # Subtyp ohne fossilen Fuel → ignorieren

        # Normalisiere fuel auf JSON-Keys
        fuel_map = {
            "natural gas":"natural gas",
            "hard coal":"hard coal",
            "lignite":"lignite",
            "light oil":"light oil",
            "heavy oil":"heavy oil",
            "oil shale":"oil shale",
            "nuclear":"nuclear",
            "hydrogen":"natural gas",  # für H2 nehmen wir Gas-VOM; EF_th=0 behandeln wir später
        }
        fuel_n = fuel_map.get(fuel, fuel)

        row = pick_row(fuel_n, type_guess)
        eta = float(row.get("eta_std_ncv", np.nan))
        vom = float(row.get("vom_eur_per_mwh_el", np.nan))
        co2_th = float(row.get("co2_ef_t_per_mwh_th", 0.0))  # t/MWh_th

        # MEF (g/kWh) aus thermischer EF: (t/MWh_th)/eta * 1000
        mef_gpkwh = (co2_th/max(eta,1e-6))*1000.0 if np.isfinite(co2_th) and np.isfinite(eta) else np.nan

        # Hydrogen & Nuclear: direkte EF = 0
        if "hydrogen" in fuel or "nuclear" in fuel:
            mef_gpkwh = 0.0

        rows.append({
            "key": key,
            "fuel": fuel.title() if fuel != "hydrogen" else "Hydrogen",
            "eta": eta,
            "varom_eur_mwh": vom,
            "co2_th_t_per_mwh": co2_th,
            "mef_gpkwh": mef_gpkwh
        })

    df = pd.DataFrame(rows).drop_duplicates("key").set_index("key")
    if df.empty:
        raise ValueError("Konnte aus JSON keine Subtypen matchen – prüfe Spaltennamen der Gen-CSV.")
    return df


# ----------------- prices, flows, gen, fuel TS -----------------
def load_prices_neighbors(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = parse_time_index(df)
    if df.columns[0].lower() in {"timestamp","time","datetime"}:
        df = df.drop(columns=[df.columns[0]])
    df = df.apply(pd.to_numeric, errors="coerce")
    return as_hourly(df,"mean").sort_index()

def load_flows(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = parse_time_index(df)
    if df.columns[0].lower() in {"timestamp","time","datetime"}:
        df = df.drop(columns=[df.columns[0]])
    df = df.apply(pd.to_numeric, errors="coerce")
    if "net_import_total" not in df.columns:
        imp_cols = [c for c in df.columns if c.startswith("imp_")]
        if imp_cols: df["net_import_total"] = df[imp_cols].sum(axis=1)
    return as_hourly(df,"mean").sort_index()

def load_gen_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = parse_time_index(df)
    for c in df.columns:
        if df[c].dtype==object: df[c] = df[c].str.replace(",",".",regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return as_hourly(df,"mean").sort_index()

def load_zone_gen(nei_dir: str, zone: str) -> pd.DataFrame|None:
    p = Path(nei_dir); c = list(p.glob(f"actual_gen_{zone}_2030*.csv"))
    if not c: return None
    return load_gen_file(str(c[0]))

def load_fuel_ts(path: str|None) -> pd.DataFrame|None:
    if not path: return None
    df = pd.read_csv(path)
    df.index = parse_time_index(df)
    if df.columns[0].lower() in {"timestamp","time","datetime"}:
        df = df.drop(columns=[df.columns[0]])
    df = df.apply(pd.to_numeric, errors="coerce")
    return as_hourly(df,"mean").sort_index()

# -------------- SRMC/MEF core (defaults if no offer) --------------
CONST_CO2_2030 = 113.4
CONST_FUEL_2030 = {"Erdgas": 6.3*3.6, "Steinkohle": 1.8*3.6, "Braunkohle":1.8*3.6,
                   "Heizöl schwer":9.6*3.6, "Heizöl leicht / Diesel":11.7*3.6,
                   "Wasserstoff":17.6*3.6, "Nuclear":1.7*3.6}
EF_TH_T_PER_MWH_TH = {"Erdgas":57.0*3.6/1000.0, "Steinkohle":94.0*3.6/1000.0,
                      "Braunkohle":101.0*3.6/1000.0, "Heizöl schwer":78.0*3.6/1000.0,
                      "Heizöl leicht / Diesel":78.0*3.6/1000.0, "Wasserstoff":0.0, "Nuclear":0.0}
VAROM_DEFAULT = {"Erdgas":1.6,"Steinkohle":3.3,"Braunkohle":3.3,"Heizöl schwer":3.3,
                 "Heizöl leicht / Diesel":1.1,"Wasserstoff":1.6,"Nuclear":9.0}

def srmc_from_fuel(fuel: str, eta: float, ts_fuel: pd.Series|None, varom: float|None) -> float:
    if not np.isfinite(eta) or eta<=0: return np.nan
    if ts_fuel is not None:
        fmap = {"Erdgas":"gas_eur_mwh_th","Steinkohle":"coal_eur_mwh_th","Braunkohle":"lignite_eur_mwh_th",
                "Heizöl schwer":"oil_eur_mwh_th","Heizöl leicht / Diesel":"oil_eur_mwh_th","Wasserstoff":"h2_eur_mwh_th",
                "Nuclear":"nuclear_eur_mwh_th"}
        fuel_th = float(ts_fuel.get(fmap.get(fuel,""), np.nan))
        co2_eur_t = float(ts_fuel.get("co2_eur_t", np.nan))
    else:
        fuel_th = CONST_FUEL_2030.get(fuel,np.nan); co2_eur_t = CONST_CO2_2030
    if not np.isfinite(fuel_th): return np.nan
    if not np.isfinite(co2_eur_t): co2_eur_t = CONST_CO2_2030
    var = varom if (varom is not None and np.isfinite(varom)) else VAROM_DEFAULT.get(fuel,0.0)
    ef_th = EF_TH_T_PER_MWH_TH.get(fuel,0.30)
    co2_term = 0.0 if fuel in ("Wasserstoff","Nuclear") else co2_eur_t*ef_th
    return (fuel_th + co2_term)/max(eta,1e-6) + var

def mef_from_eta_co2(eta: float, co2: float, basis: str) -> float:
    """g/kWh; basis='el' → co2 [t/MWh_el]; basis='th' → co2 [t/MWh_th]"""
    if not np.isfinite(eta) or eta<=0 or not np.isfinite(co2): return np.nan
    if basis=="el":
        return float(co2*1000.0)                 # already per MWh_el
    else:
        return float((co2/max(eta,1e-6))*1000.0) # per MWh_th → per MWh_el

# -------------- Other_nonres dispatched loader ----------------
def load_dispatched_dir(path: str) -> dict[str, pd.DataFrame]:
    """Liest Dateien dispatch_<ZONE>_other_nonres.csv mit Spalten:
       timestamp,zone,block,offer,cap_mw,gen_mw,eta,co2
       Rückgabe: {zone: DataFrame indexed by timestamp}"""
    out = {}
    p = Path(path)
    files = list(p.glob("dispatch_*_other_nonres.csv"))
    for f in files:
        df = pd.read_csv(f)
        df.index = parse_time_index(df)
        # keep only needed cols robustly
        cols = {c.lower():c for c in df.columns}
        need = ["zone","offer","gen_mw","cap_mw","eta","co2"]
        for n in need:
            if n not in cols: raise ValueError(f"{f.name}: Spalte '{n}' fehlt.")
        df = df[[cols["zone"], cols["offer"], cols["gen_mw"], cols["cap_mw"], cols["eta"], cols["co2"]]].copy()
        # numeric
        for c in [cols["offer"], cols["gen_mw"], cols["cap_mw"], cols["eta"], cols["co2"]]:
            if df[c].dtype==object: df[c]=df[c].str.replace(",",".",regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # split by zone
        for z, d in df.groupby(cols["zone"]):
            d = as_hourly(d, "mean").sort_index()
            out.setdefault(z, d.rename(columns={cols["offer"]:"offer", cols["gen_mw"]:"gen_mw",
                                                cols["cap_mw"]:"cap_mw", cols["eta"]:"eta", cols["co2"]:"co2"}))
    return out

# -------------- coupling & stacks ----------------
def coupled_zones(pr_row: pd.Series, eps: float) -> list[str]:
    p_de = float(pr_row.get("price_DE_LU", np.nan))
    zs = [c.replace("price_","") for c in pr_row.index if c.startswith("price_") and c!="price_DE_LU"]
    return sorted([z for z in zs if np.isfinite(pr_row.get(f"price_{z}", np.nan)) and abs(pr_row[f"price_{z}"]-p_de)<=eps+1e-9])

def stack_from_gen_row(gen_row: pd.Series, eff_map: pd.DataFrame, ts_fuel: pd.Series|None):
    blocks = []
    for col, val in gen_row.items():
        if not np.isfinite(val) or float(val)<=0: continue
        key = clean_subtype(col)
        if key not in eff_map.index: continue
        eta = float(eff_map.loc[key,"eta"]); fuel = str(eff_map.loc[key,"fuel"])
        varom = float(eff_map.loc[key,"varom_eur_mwh"])
        srmc = srmc_from_fuel(fuel, eta, ts_fuel, varom)
        mef = mef_from_eta_co2(eta, co2=0.0, basis="el")  # direct emissions 0 unless fuelled; override via eff if needed
        if fuel in ("Erdgas","Steinkohle","Braunkohle","Heizöl schwer","Heizöl leicht / Diesel","Wasserstoff","Nuclear"):
            # derive MEF from standard EF_th when fossil (H2/Nuclear -> 0)
            ef_th = {"Erdgas":57.0*3.6/1000.0,"Steinkohle":94.0*3.6/1000.0,"Braunkohle":101.0*3.6/1000.0,"Heizöl schwer":78.0*3.6/1000.0,"Heizöl leicht / Diesel":78.0*3.6/1000.0,"Wasserstoff":0.0,"Nuclear":0.0}[fuel]
            mef = (ef_th/max(eta,1e-6))*1000.0
        blocks.append({"subtype": key, "fuel": fuel, "eta": eta, "srmc": srmc, "mef_gpkwh": mef, "mw": float(val)})
    return blocks

# --------------------------- MAIN ----------------------------
def main():
    ap = argparse.ArgumentParser(description="MEF – marginal candidates via price window incl. dispatched stacks")
    ap.add_argument("--eff", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--de_gen", required=True)
    ap.add_argument("--nei_gen_dir", required=True)
    ap.add_argument("--fuel_ts", required=True)
    ap.add_argument("--dispatched_dir", default=None, help="Folder with dispatch_<ZONE>_other_nonres.csv")
    ap.add_argument("--co2_basis", choices=["el","th"], default="el", help="co2 column basis in dispatched files")

    ap.add_argument("--eps_coupling", type=float, default=2.0)
    ap.add_argument("--tol_window", type=float, default=2.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    de_gen  = load_gen_file(args.de_gen)
    eff     = build_eff_map_from_json(args.eff, known_subtypes=list(de_gen.columns))
    prices  = load_prices_neighbors(args.prices)
    flows   = load_flows(args.flows)
    fuel_ts= load_fuel_ts(args.fuel_ts)
    disp_map = load_dispatched_dir(args.dispatched_dir) if args.dispatched_dir else {}

    common = sorted(set(prices.index) & set(flows.index) & set(de_gen.index))
    if not common: raise RuntimeError("No common hourly index among prices/flows/de_gen.")
    def to_berlin(x):
        if x is None: return None
        t = pd.Timestamp(x)
        return t.tz_localize(TZ) if t.tz is None else t.tz_convert(TZ)
    t0 = to_berlin(args.start) or common[0]
    t1 = to_berlin(args.end) or (common[-1] + pd.Timedelta(hours=1))
    idx = [t for t in common if (t>=t0 and t<t1)]

    zones = [c.replace("price_","") for c in prices.columns if c.startswith("price_") and c!="price_DE_LU"]
    gen_cache = {"DE_LU": de_gen}

    cand_rows = []; band_rows = []

    for t in idx:
        p_de = float(prices.loc[t,"price_DE_LU"])
        eps, tol = float(args.eps_coupling), float(args.tol_window)
        coupled = coupled_zones(prices.loc[t], eps)
        net_imp = float(flows.loc[t,"net_import_total"]) if "net_import_total" in flows.columns else 0.0
        ts_fuel = fuel_ts.loc[t] if t in fuel_ts.index else None

        # Domestic from gen
        de_blocks = stack_from_gen_row(gen_cache["DE_LU"].loc[t], eff, ts_fuel)
        cands_de = [ {"timestamp":t,"mode":"coupled" if (coupled and net_imp>0) else "domestic_only","price_DE_eur_mwh":p_de,
                      "zone":"DE_LU","subtype":b["subtype"],"fuel":b["fuel"],"eta":b["eta"],
                      "srmc_eur_mwh":b["srmc"],"dist_to_price_eur":abs(b["srmc"]-p_de),
                      "gen_mw":b["mw"],"mef_gpkwh":b["mef_gpkwh"]}
                     for b in de_blocks if np.isfinite(b["srmc"]) and abs(b["srmc"]-p_de)<=tol+1e-9 and b["mw"]>0 ]

        # Domestic dispatched (other_nonres) if available
        if "DE_LU" in disp_map and t in disp_map["DE_LU"].index:
            row = disp_map["DE_LU"].loc[[t]]
            for _, r in row.iterrows():
                if r["gen_mw"]>0 and np.isfinite(r["offer"]) and abs(r["offer"]-p_de)<=tol+1e-9:
                    mef = mef_from_eta_co2(r["eta"], r["co2"], args.co2_basis)
                    cands_de.append({"timestamp":t,"mode":"coupled" if (coupled and net_imp>0) else "domestic_only",
                                     "price_DE_eur_mwh":p_de,"zone":"DE_LU","subtype":"other_nonres_dispatch",
                                     "fuel":"Other","eta":r["eta"],"srmc_eur_mwh":r["offer"],
                                     "dist_to_price_eur":abs(r["offer"]-p_de),"gen_mw":r["gen_mw"],
                                     "mef_gpkwh":mef})

        # Import candidates from gen + dispatched (only if coupled & net import > 0 & imp_Z>0)
        cands_imp = []
        if coupled and net_imp>0:
            imp_cols = [c for c in flows.columns if c.startswith("imp_")]
            importing = [c.replace("imp_","") for c in imp_cols if c.replace("imp_","") in coupled and float(flows.loc[t,c])>1e-6]

            for z in importing:
                # gen-based
                if z not in gen_cache:
                    g = load_zone_gen(args.nei_gen_dir, z)
                    if g is not None: gen_cache[z]=g
                if z in gen_cache and t in gen_cache[z].index:
                    z_blocks = stack_from_gen_row(gen_cache[z].loc[t], eff, ts_fuel)
                    for b in z_blocks:
                        if b["mw"]>0 and np.isfinite(b["srmc"]) and abs(b["srmc"]-p_de)<=tol+1e-9:
                            cands_imp.append({"timestamp":t,"mode":"coupled","price_DE_eur_mwh":p_de,
                                              "zone":z,"subtype":b["subtype"],"fuel":b["fuel"],"eta":b["eta"],
                                              "srmc_eur_mwh":b["srmc"],"dist_to_price_eur":abs(b["srmc"]-p_de),
                                              "gen_mw":b["mw"],"mef_gpkwh":b["mef_gpkwh"]})
                # dispatched-based
                if z in disp_map and t in disp_map[z].index:
                    row = disp_map[z].loc[[t]]
                    for _, r in row.iterrows():
                        if r["gen_mw"]>0 and np.isfinite(r["offer"]) and abs(r["offer"]-p_de)<=tol+1e-9:
                            mef = mef_from_eta_co2(r["eta"], r["co2"], args.co2_basis)
                            cands_imp.append({"timestamp":t,"mode":"coupled","price_DE_eur_mwh":p_de,
                                              "zone":z,"subtype":"other_nonres_dispatch","fuel":"Other","eta":r["eta"],
                                              "srmc_eur_mwh":r["offer"],"dist_to_price_eur":abs(r["offer"]-p_de),
                                              "gen_mw":r["gen_mw"],"mef_gpkwh":mef})

        hour_cands = (cands_de + cands_imp) if (coupled and net_imp>0) else cands_de
        for r in hour_cands: cand_rows.append(r)
        if hour_cands:
            mefs = [r["mef_gpkwh"] for r in hour_cands if np.isfinite(r["mef_gpkwh"])]
            band_rows.append({"timestamp":t,"mef_min_gpkwh":float(np.min(mefs)) if mefs else np.nan,
                              "mef_max_gpkwh":float(np.max(mefs)) if mefs else np.nan,
                              "n_candidates":len(hour_cands),
                              "mode":"coupled" if (coupled and net_imp>0) else "domestic_only"})
        else:
            band_rows.append({"timestamp":t,"mef_min_gpkwh":np.nan,"mef_max_gpkwh":np.nan,
                              "n_candidates":0,"mode":"domestic_only"})

    df_c = pd.DataFrame(cand_rows).set_index("timestamp").sort_index()
    df_b = pd.DataFrame(band_rows).set_index("timestamp").sort_index()
    out_c = Path(args.outdir)/"marginal_candidates.csv"
    out_b = Path(args.outdir)/"mef_band_by_hour.csv"
    out_c.parent.mkdir(parents=True, exist_ok=True)
    df_c.to_csv(out_c, index=True); df_b.to_csv(out_b, index=True)
    print(f"[OK] candidates -> {out_c}  ({len(df_c)} rows)")
    print(f"[OK] band       -> {out_b}  ({len(df_b)} hours)")

if __name__ == "__main__":
    main()
