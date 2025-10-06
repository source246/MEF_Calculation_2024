# dispatch_backcast_c.py
# Backcast stündlicher MEF (DE_LU) inkl. Import-Attribution über Nachbar-Merit.
# Erwartet: ENTSO-E-ähnliche Erzeugungsdateien (2-zeilige Header), MaStR-Fleet mit Effizienz_imputiert,
#            weite Scheduled-Exchanges (A->DE_LU, DE_LU->A), optional Nachbarpreise, optional EF_import_t.

import argparse, os, math
from typing import Dict
import numpy as np
import pandas as pd
import csv
import codecs

# ---------------------- Konstanten: Kosten & Emissionsfaktoren ----------------------
CO2_TH_T_PER_MWH_TH = {"lignite":0.364, "coal":0.340, "gas":0.202, "oil_light":0.267, "oil_heavy":0.320}
TRANSPORT_EUR_PER_MWH_TH = {"lignite":0.0, "coal":1.25, "gas":0.5, "oil":0.3}
VOM_EUR_PER_MWH_EL = {"lignite":1.7, "coal":1.3, "gas_ccgt":1.5, "gas_ocgt":1.0, "oil":1.0}

# ---------------------- Utilities ---------------------------------------------------
def _to_float(x):
    if isinstance(x,(int,float,np.floating)): return float(x)
    s=str(x).replace('\u00a0','').replace('\u202f','').replace('.','').replace(',','.')
    return pd.to_numeric(s, errors='coerce')
def _parse_time_col(df: pd.DataFrame) -> pd.Series:
    """
    Sucht eine Zeitspalte (time/timestamp/timestamp_cec/Datetime/…) und gibt UTC-naiv zurück.
    Falls nichts explizit genannt ist: probiert jede Spalte heuristisch.
    """
    # 1) nach Namen suchen
    cand_names = []
    for c in df.columns:
        name = ""
        if isinstance(c, tuple):
            name = " ".join([str(x) for x in c if pd.notna(x)]).strip().lower()
        else:
            name = str(c).strip().lower()
        if any(k in name for k in ["timestamp_cec","timestamp","zeitstempel","datetime","time"]):
            cand_names.append(c)

    if cand_names:
        t = pd.to_datetime(df[cand_names[0]], errors="coerce", utc=True)
        return t.dt.tz_convert("UTC").dt.tz_localize(None)

    # 2) heuristisch
    for c in df.columns:
        t = pd.to_datetime(df[c], errors="coerce", utc=True)
        if t.notna().sum() >= max(3, int(0.5*len(df))):
            return t.dt.tz_convert("UTC").dt.tz_localize(None)

    raise ValueError("Keine Zeitspalte gefunden (time/timestamp/timestamp_cec/Datetime/…).")

def _parse_time_series_any_df(df):
    """
    Robust: finde Zeitspalte in beliebigen CSV-Varianten.
    1) Name enthält: timestamp_cec / timestamp / zeitstempel / datetime / time
    2) sonst: erste Spalte, die sich zu >=50% als Datum parsen lässt
    Rückgabe: pandas.DatetimeIndex (UTC-naiv)
    """
    # Kandidaten per Name
    name_hits = []
    for c in df.columns:
        label = ""
        if isinstance(c, tuple):
            label = " ".join([str(x) for x in c if pd.notna(x)]).strip().lower()
        else:
            label = str(c).strip().lower()
        if any(k in label for k in ["timestamp_cec","timestamp","zeitstempel","datetime","time"]):
            name_hits.append(c)
    # 1) Bevorzugt der erste Namenskandidat
    if name_hits:
        s = df[name_hits[0]]
        t = pd.to_datetime(s, errors="coerce", utc=True)
        if t.notna().sum() >= max(3, int(0.5*len(s))):
            return t.dt.tz_convert("UTC").dt.tz_localize(None)

    # 2) Heuristik über alle Spalten
    for c in df.columns:
        s = df[c]
        t = pd.to_datetime(s, errors="coerce", utc=True)
        if t.notna().sum() >= max(3, int(0.5*len(s))):
            return t.dt.tz_convert("UTC").dt.tz_localize(None)

    raise ValueError(f"Keine Zeitspalte erkennbar. Spalten gesehen: {list(df.columns)[:8]}")

def read_entsoe_gen_csv(path: str) -> pd.DataFrame:
    """
    Liest ENTSO-E/SMARD-ähnliche Erzeugungsdateien sehr robust:
    - versucht [Multi-Header; ,] -> [Single-Header; ,] -> [keine Header; ,] -> dieselben mit ';'
    - erkennt Zeitspalte heuristisch und konvertiert auf UTC-naiv
    - liefert Spalten: Fossil Gas / Fossil Hard coal / Fossil Brown coal/Lignite / Fossil Oil / Load
    """
    import pandas as pd
    import numpy as np

    attempts = [
        dict(header=[0,1], sep=","),
        dict(header=0,     sep=","),
        dict(header=None,  sep=","),
        dict(header=[0,1], sep=";"),
        dict(header=0,     sep=";"),
        dict(header=None,  sep=";"),
    ]

    last_err = None
    for opt in attempts:
        try:
            raw = pd.read_csv(path, **opt)
            # wenn ohne Header: künstliche Namen
            if opt["header"] is None:
                raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
            # Zeitspalte erkennen
            t = _parse_time_series_any_df(raw)

            # Header flatten
            if isinstance(raw.columns, pd.MultiIndex):
                cols = {}
                for c in raw.columns:
                    if isinstance(c, tuple):
                        top = (c[0] or "").strip()
                        sub = (c[1] or "").strip()
                        cols[c] = f"{top}|{sub}" if sub else top
                    else:
                        cols[c] = str(c)
                df = raw.rename(columns=cols)
            else:
                df = raw.copy()

            out = pd.DataFrame({"time": t})

            def _series_from_candidates(cands):
                # nimmt den ersten Treffer; toleriert Duplikate (nimmt erste Spalte)
                for cand in cands:
                    if cand in df.columns:
                        s = df[cand]
                        if isinstance(s, pd.DataFrame):
                            s = s.iloc[:, 0]
                        return pd.to_numeric(s, errors="coerce")
                # fuzzy: alle Spalten, die mit "base|" anfangen und "Actual Aggregated" enthalten
                base = cands[0].split("|")[0]
                fuzzy = [c for c in df.columns if str(c).startswith(base + "|") and "Actual Aggregated" in str(c)]
                if fuzzy:
                    s = df[fuzzy[0]]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    return pd.to_numeric(s, errors="coerce")
                return pd.Series(np.nan, index=out.index)

            # Fossile
            out["Fossil Gas"] = _series_from_candidates([ "Fossil Gas|Actual Aggregated", "Fossil Gas" ])
            out["Fossil Hard coal"] = _series_from_candidates([ "Fossil Hard coal|Actual Aggregated", "Fossil Hard coal" ])
            out["Fossil Brown coal/Lignite"] = _series_from_candidates([ "Fossil Brown coal/Lignite|Actual Aggregated", "Fossil Brown coal/Lignite" ])
            out["Fossil Oil"] = _series_from_candidates([ "Fossil Oil|Actual Aggregated", "Fossil Oil" ])

            # Load (mehrere mögliche Labels)
            load_candidates = [
                "Total Load|Actual Aggregated", "Actual Consumption",
                "Total Load", "Load"
            ]
            out["Load"] = _series_from_candidates(load_candidates)

            return out.dropna(subset=["time"]).set_index("time").sort_index()

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Datei '{path}' konnte nicht gelesen werden: letzter Fehler: {last_err}")




def read_flows_wide(path: str, zone="DE_LU") -> pd.Series:
    """weite Scheduled-Exchanges: Spalten 'A->B' (MW); gibt Serie net_imports_MW (Import – Export) zurück"""
    df = pd.read_csv(path)
    t = _parse_time_col(df)
    df = df.drop(columns=[c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()])
    df.insert(0,"time",t); df=df.set_index("time")
    imp=[]; exp=[]
    for c in df.columns:
        if "->" not in c: continue
        a,b=[s.strip() for s in c.split("->")]
        if b==zone: imp.append(pd.to_numeric(df[c], errors="coerce"))
        if a==zone: exp.append(pd.to_numeric(df[c], errors="coerce"))
    imp_sum = sum(imp) if imp else pd.Series(0.0,index=df.index)
    exp_sum = sum(exp) if exp else pd.Series(0.0,index=df.index)
    return (imp_sum - exp_sum).rename("net_imports_MW")

def read_neighbor_prices(path: str, index) -> pd.DataFrame:
    """liest neighbor_prices – akzeptiert Spalten mit Präfix 'price_' (price_FR → FR)"""
    df = pd.read_csv(path)
    t = _parse_time_col(df)
    df = df.drop(columns=[c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower()])
    new_cols={}
    for c in df.columns:
        name=str(c)
        if name.lower().startswith("price_"): name=name.split("_",1)[1]
        new_cols[c]=name.upper()
    out = df.rename(columns=new_cols)
    out.insert(0,"time",t)
    return out.set_index("time").reindex(index).ffill()

def srmc_el(fuel: str, eta: float, p_fuel_th: float, p_co2: float, oil_heavy=False) -> float:
    trans = TRANSPORT_EUR_PER_MWH_TH.get(fuel,0.0)
    ef_th = CO2_TH_T_PER_MWH_TH["oil_heavy" if (fuel=="oil" and oil_heavy) else ("oil_light" if fuel=="oil" else fuel)]
    vom = VOM_EUR_PER_MWH_EL.get("gas_ccgt" if fuel=="gas" else fuel, 0.0)
    return (p_fuel_th+trans)/eta + (ef_th/eta)*p_co2 + vom

def ef_el(fuel: str, eta: float, oil_heavy=False) -> float:
    ef_th = CO2_TH_T_PER_MWH_TH["oil_heavy" if (fuel=="oil" and oil_heavy) else ("oil_light" if fuel=="oil" else fuel)]
    return ef_th/eta

def derive_import_ef_from_neighbors(t, need_mw, neighbors: Dict[str,dict]):
    """baut exportfähige Merit über Nachbarn und liefert gewichteten Import-EF"""
    if need_mw<=0 or not neighbors: return (np.nan, 0.0, {})
    offers=[]
    for land,info in neighbors.items():
        gen_f={k:max(0.0,float(info["gen"].get(k,0.0))) for k in ["lignite","coal","gas","oil"]}
        load=max(0.0,float(info.get("load",0.0)))
        total=sum(gen_f.values())+float(info.get("gen_ee",0.0))
        export=max(0.0,total-load)
        if export<=0: continue
        order=sorted(gen_f.items(), key=lambda kv: float(info["srmc"].get(kv[0], np.inf)))
        for fuel,gmw in order:
            if gmw<=0: continue
            srmc=float(info["srmc"].get(fuel,np.inf)); ef=float(info["ef"].get(fuel,np.nan))
            avail=min(gmw, export)
            if avail>0 and np.isfinite(srmc) and np.isfinite(ef):
                offers.append((land,srmc,avail,ef,fuel)); export-=avail
                if export<=0: break
    if not offers: return (np.nan,0.0,{})
    offers.sort(key=lambda x:x[1])
    take=[]; remain=need_mw
    for land,srmc,avail,ef,fuel in offers:
        q=min(avail,remain)
        if q<=0: break
        take.append((land,srmc,q,ef,fuel)); remain-=q
        if remain<=1e-6: break
    mw=sum(x[2] for x in take)
    w_price=sum(x[1]*x[2] for x in take)/mw
    w_ef=sum(x[3]*x[2] for x in take)/mw
    details={land:{"share_mw":q,"marg_fuel":fuel,"marg_price":srmc} for land,srmc,q,ef,fuel in take}
    return (w_price,w_ef,details)
def read_csv_mastr_robust(path):
    """
    Liest deutsche MaStR-CSV robust:
    - probiert ; und , als Separator
    - probiert Encodings: utf-8-sig, utf-8, latin1
    - behandelt BOM/Anführungszeichen sauber
    Gibt immer einen DataFrame zurück oder wirft einen klaren Fehler.
    """
    seps = [';', ',']
    encs = ['utf-8-sig', 'utf-8', 'latin1']
    last_err = None
    for sep in seps:
        for enc in encs:
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=enc,
                    engine='python',      # toleranter bei Quotes
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                    escapechar='\\',
                    na_values=['', 'NA', 'NaN', 'nan']
                )
                # heuristik: wenn nur ~3 Spalten rauskamen, aber 1000+ Zeichen in der 1. Zeile, war der sep falsch
                if df.shape[1] <= 3 and df.shape[0] > 2:
                    # wahrscheinlich falscher Separator — weiter probieren
                    last_err = Exception(f"Parsed {df.shape[1]} columns with sep='{sep}', likely wrong sep.")
                    continue
                return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Fleet-Datei konnte nicht robust gelesen werden: {last_err}")

import re, csv

def _normalize_colnames(cols):
    out = {}
    for c in cols:
        s = str(c)
        s = (s
             .replace('\ufeff','')
             .replace('\u00a0',' ')   # NBSP
             .replace('\u202f',' ')   # NNBSP
             .strip())
        out[c] = s
    return out

def _to_num_de(s):
    # deutsche Zahlen "1.234,56" -> "1234.56"
    return (s.astype(str)
             .str.replace('\ufeff','', regex=False)
             .str.replace('\u00a0','', regex=False)
             .str.replace('\u202f','', regex=False)
             .str.replace('.', '', regex=False)
             .str.replace(',', '.', regex=False)
             .str.strip()
             .pipe(pd.to_numeric, errors='coerce'))

def read_csv_mastr_robust(path):
    seps = [';', ',']
    encs = ['utf-8-sig', 'utf-8', 'latin1']
    last = None
    for sep in seps:
        for enc in encs:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, engine='python',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                 escapechar='\\', na_values=['', 'NA', 'NaN', 'nan'])
                return df
            except Exception as e:
                last = e
                continue
    raise RuntimeError(f"Fleet-Datei nicht lesbar: {last}")

def mastr_to_fleet_units(fleet_csv: str) -> pd.DataFrame:
    df = read_csv_mastr_robust(fleet_csv)
    df = df.rename(columns=_normalize_colnames(df.columns))

    # ---- Spalten finden (robust) ----
    # Brennstoff:
    fuel_col = None
    fuel_candidates = [
        "Hauptbrennstoff der Einheit", "Energieträger", "EnergietrÃ¤ger",
        "Brennstoff", "Brennstoffart"
    ]
    for c in fuel_candidates:
        if c in df.columns:
            fuel_col = c; break
    if fuel_col is None:
        # Heuristik: Spalte mit vielen typischen Brennstoff-Treffern
        best, bestc = 0, None
        for c in df.columns:
            vals = df[c].astype(str).str.lower()
            hits = vals.str.contains("gas|kohl|braunkohl|heizöl|heizoel|öl|oel|diesel|fuel", regex=True, na=False).sum()
            if hits > best:
                best, bestc = hits, c
        fuel_col = bestc

    # Leistung:
    cap_col = None
    for c in ["MW Nettonennleistung der Einheit","Leistung_MW","Nettonennleistung der Einheit","MW Nettonennleistung"]:
        if c in df.columns:
            cap_col = c; break

    # Effizienz:
    eta_col = "Effizienz_imputiert" if "Effizienz_imputiert" in df.columns else ("Effizienz" if "Effizienz" in df.columns else None)

    if fuel_col is None or cap_col is None or eta_col is None:
        raise ValueError(f"Fleet: Spalten nicht gefunden. Gefunden={list(df.columns)[:15]}")

    # ---- Mapping Fuel -> {gas, coal, lignite, oil, oil_heavy, other} ----
    fraw = df[fuel_col].astype(str).str.lower()

    def map_fuel(s):
        # explizit erst Nicht-preissetzende abfangen
        if re.search(r"biogas|klärgas|deponiegas|raffineriegas|hochofengas|konvertergas|abfall|biogen|holz|pellet|biodiesel|wasserstoff|wärme|dampf|lauge|rest", s):
            return "other"
        if "braunkohl" in s:
            return "lignite"
        if "steinkohl" in s or "wirbelschicht" in s or "trockenkohl" in s or "brikett" in s:
            return "coal"
        if "erdgas" in s or re.search(r"\bgas\b", s):
            return "gas"
        if "heizöl" in s or "heizoel" in s or re.search(r"\böl\b", s) or "oel" in s or "diesel" in s or "fuel" in s:
            return "oil_heavy" if "schwer" in s else "oil"
        return "other"

    df["fuel_type"] = fraw.map(map_fuel)

    # ---- Zahlen konvertieren ----
    df["capacity_MW"] = _to_num_de(df[cap_col])
    df["efficiency"]  = _to_num_de(df[eta_col])

    # Diagnose vor Filter:
    diag = df.groupby("fuel_type")["capacity_MW"].sum(min_count=1)
    print("Fleet Capacity by fuel_type [MW]:")
    print(diag.to_string())

    valid = df["fuel_type"].isin(["gas","coal","lignite","oil","oil_heavy"])
    cap_ok = df["capacity_MW"].gt(0)
    eta_ok = df["efficiency"].between(0.15, 0.70, inclusive="both")

    fleet = df.loc[valid & cap_ok & eta_ok, ["fuel_type","capacity_MW","efficiency"]].copy()

    # Wenn leer, versuche weicher zu filtern (nur >0 und >0.1):
    if fleet.empty:
        print("WARN: Fleet leer nach striktem Filter – wende weichen Filter an (cap>0 & eta>0.1).")
        fleet = df.loc[valid & df["capacity_MW"].gt(0) & df["efficiency"].gt(0.1), ["fuel_type","capacity_MW","efficiency"]].copy()

    # Wenn immer noch leer: als letzte Rettung ohne Filter, aber nur valid fuels + dropna
    if fleet.empty:
        print("WARN: Fleet weiterhin leer – gebe valid fuels mit dropna zurück (letzte Rettung).")
        fleet = df.loc[valid, ["fuel_type","capacity_MW","efficiency"]].dropna().copy()

    if fleet.empty:
        raise ValueError("Fleet bleibt leer – bitte Fleet-Datei posten (erste 5 Zeilen), dann passe ich das Mapping gezielt an.")

    return fleet
def fleet_df_to_units(fleet_df: pd.DataFrame):
    """
    Konvertiert den Fleet-DataFrame (fuel_type, capacity_MW, efficiency)
    in das vom Backcast erwartete Dict: {fuel: [ {unit...}, ... ] }.
    - 'oil_heavy' wird als 'oil' mit Flag oil_heavy=True abgebildet.
    - unit_id/unit_name sind Platzhalter (für MEF reicht das).
    """
    out = {"gas": [], "coal": [], "lignite": [], "oil": []}
    for i, r in fleet_df.reset_index(drop=True).iterrows():
        f_raw = str(r["fuel_type"]).lower()
        fuel = "oil" if f_raw in ("oil", "oil_heavy") else f_raw  # heavy → oil, Flag separat
        if fuel not in out:
            continue
        unit = {
            "unit_id": f"unit_{fuel}_{i}",
            "unit_name": f"{fuel.upper()}_{i}",
            "fuel": fuel,
            "eta": float(r["efficiency"]),
            "capacity_mw": float(r["capacity_MW"]),
            "vom_key": "gas_ccgt" if fuel == "gas" else fuel,
            "oil_heavy": (f_raw == "oil_heavy"),
        }
        out[fuel].append(unit)
    # Effizienteste zuerst (wie bisher)
    for f in out:
        out[f] = sorted(out[f], key=lambda u: (-u["eta"], u["capacity_mw"]))
    return out


# ---------------------- Hauptlauf ---------------------------------------------------
def run_backcast(timeseries_csv, fuels_csv, fleet_csv, out_csv,
                 flows_csv=None, zone="DE_LU",
                 neighbor_dir=None, neighbor_prices_csv=None,
                 import_ef_csv=None, oil_heavy=False,
                 consumption_based=True, price_accept_band=8.0):

    # 1) Inlands-Zeitreihen (Erzeugung)
    ts = read_entsoe_gen_csv(timeseries_csv); idx = ts.index

    # 2) Brennstoff- & CO2-Preise (UTC)
    fuels = pd.read_csv(fuels_csv)
    ft = pd.to_datetime(fuels.iloc[:,0], errors="coerce")
    if ft.dt.tz is None: ft = ft.dt.tz_localize("UTC")
    fuels.index = ft.dt.tz_convert("UTC").dt.tz_localize(None)
    fuels = fuels.reindex(idx).ffill()
    pf = {k: pd.to_numeric(fuels[f"{k}_eur_mwh_th"], errors="coerce").reindex(idx).ffill() for k in ["gas","coal","lignite","oil"]}
    pco2 = pd.to_numeric(fuels["co2_eur_t"], errors="coerce").reindex(idx).ffill()

    # 3) Fleet (MaStR)
    fleet_df = mastr_to_fleet_units(fleet_csv)
    fleet = fleet_df_to_units(fleet_df)
    sorted_units = fleet  # bereits vorsortiert
    fossil_cols={"Fossil Gas":"gas","Fossil Hard coal":"coal","Fossil Brown coal/Lignite":"lignite","Fossil Oil":"oil"}

    # 4) Flüsse -> Net Imports
    if flows_csv:
        net_imports = read_flows_wide(flows_csv, zone=zone).reindex(idx).fillna(0.0)
    else:
        net_imports = pd.Series(0.0, index=idx, name="net_imports_MW")
    net_exports = (-net_imports).clip(lower=0.0)

    # 5) Import-EF direkt?
    if import_ef_csv:
        ief = pd.read_csv(import_ef_csv)
        it = pd.to_datetime(ief.iloc[:,0], errors="coerce")
        if it.dt.tz is None: it = it.dt.tz_localize("UTC")
        ief.index = it.dt.tz_convert("UTC").dt.tz_localize(None)
        ef_import_direct = pd.to_numeric(ief.iloc[:,1], errors="coerce").reindex(idx)
    else:
        ef_import_direct = None

    # 6) Nachbarn (falls kein direkter Import-EF)
    neighbors_data=None
    if neighbor_dir and ef_import_direct is None:
        neighbors_data={}
        nprice = read_neighbor_prices(neighbor_prices_csv, idx) if (neighbor_prices_csv and os.path.exists(neighbor_prices_csv)) else pd.DataFrame(index=idx)
        eta_typ={"gas":0.55,"coal":0.40,"lignite":0.35,"oil":0.38}
        for fn in os.listdir(neighbor_dir):
            if not fn.lower().endswith(".csv"): continue
            path=os.path.join(neighbor_dir, fn)
            df=read_entsoe_gen_csv(path).reindex(idx).fillna(0.0)
            gen={"gas":df.get("Fossil Gas",0.0),"coal":df.get("Fossil Hard coal",0.0),
                 "lignite":df.get("Fossil Brown coal/Lignite",0.0),"oil":df.get("Fossil Oil",0.0)}
            gen_ee=df[[c for c in df.columns if c in ["Wind Onshore","Wind Offshore","Solar","Hydro Water Reservoir","Hydro Run-of-river and poundage"]]].sum(axis=1)
            load=pd.to_numeric(df.get("Load",0.0), errors="coerce").fillna(0.0)
            srmc_tbl=pd.DataFrame(index=idx)
            for f in ["gas","coal","lignite","oil"]:
                srmc_tbl[f]=(pf[f]/eta_typ[f]) + ((CO2_TH_T_PER_MWH_TH["oil_light" if f=="oil" else f]/eta_typ[f]) * pco2) + VOM_EUR_PER_MWH_EL.get("gas_ccgt" if f=="gas" else f,0.0)
            ef_tbl=pd.DataFrame({f:[CO2_TH_T_PER_MWH_TH["oil_light" if f=="oil" else f]/eta_typ[f]]*len(idx) for f in ["gas","coal","lignite","oil"]}, index=idx)
            land = os.path.splitext(fn)[0].split("_")[-2].upper() if "actual_gen_" in fn else os.path.splitext(fn)[0].upper()
            neighbors_data[land]={"gen":gen,"gen_ee":gen_ee,"load":load,"srmc":srmc_tbl,"ef":ef_tbl,
                                  "price":pd.to_numeric(nprice.get(land, pd.Series(np.nan,index=idx)), errors="coerce")}

    # 7) Backcast je Stunde
    out=[]
    for t in idx:
        used=[]
        # inländische fossile Blöcke nach SRMC
        for col,fuel in fossil_cols.items():
            gen=float(ts.at[t,col]) if col in ts.columns and pd.notna(ts.at[t,col]) else 0.0
            if gen<=0 or not sorted_units.get(fuel): continue
            rows=[]
            for u in sorted_units[fuel]:
                srmc = srmc_el(fuel, u["eta"], float(pf[fuel].at[t]), float(pco2.at[t]), oil_heavy=bool(u.get("oil_heavy", False)))
                rows.append({"eta":u["eta"],"capacity_mw":u["capacity_mw"],"srmc":srmc,
                             "unit_id":u["unit_id"],"unit_name":u["unit_name"],"fuel":fuel})
            dfu=pd.DataFrame(rows).sort_values(["srmc","eta"], ascending=[True,False])
            dfu["cumcap"]=dfu["capacity_mw"].cumsum()
            prev=0.0
            for j,u in dfu.reset_index(drop=True).iterrows():
                prev=0.0 if j==0 else float(dfu.iloc[j-1]["cumcap"])
                q=max(0.0, min(gen-prev, float(u["capacity_mw"])))
                if q<=0: continue
                used.append({"fuel":fuel,"srmc":float(u["srmc"]),"mef":ef_el(fuel,float(u["eta"])),
                             "unit_id":u["unit_id"],"unit_name":u["unit_name"],"marginal_output_mw":q})

        # Import-Block
        imp_mw=float(net_imports.at[t]) if pd.notna(net_imports.at[t]) else 0.0
        import_price=np.nan
        ef_import=np.nan
        if imp_mw>0:
            if ef_import_direct is not None and pd.notna(ef_import_direct.at[t]):
                ef_import=float(ef_import_direct.at[t])
            elif neighbors_data:
                # Snapshot je Stunde
                snap={}
                for land,info in neighbors_data.items():
                    snap[land]={"gen":{k:float(info["gen"][k].at[t]) for k in ["lignite","coal","gas","oil"]},
                                "gen_ee":float(info["gen_ee"].at[t]),
                                "load":float(info["load"].at[t]),
                                "srmc":{k:float(info["srmc"].at[t,k]) for k in ["lignite","coal","gas","oil"]},
                                "ef":{k:float(info["ef"].at[t,k]) for k in ["lignite","coal","gas","oil"]}}
                import_price, ef_import, _ = derive_import_ef_from_neighbors(t, imp_mw, snap)
        
            # Fallbacks
            if not np.isfinite(import_price):
                import_price = (np.nanmean([u["srmc"] for u in used]) if used else np.nan)
            if not np.isfinite(ef_import):
                ef_import = np.nan
        
            used.append({
                "fuel":"IMPORT",
                "srmc": float(import_price),
                "mef":  float(ef_import),
                "unit_id":"__IMPORT__",
                "unit_name":"__IMPORT__",
                "marginal_output_mw":imp_mw
            })

            if not note:
                note = "consumption_based: export_relief_zero"
        

        marginal=max(used, key=lambda x: x["srmc"] if np.isfinite(x["srmc"]) else -1e9)
        out.append({"time":t,"price_from_dispatch":float(marginal["srmc"]),
                    "marginal_unit":marginal["unit_name"],"marginal_fuel":marginal["fuel"],
                    "MEF_t_per_MWh":float(marginal["mef"]), "notes":note})

    res=pd.DataFrame(out).set_index("time").sort_index()
    res.to_csv(out_csv)
    return res

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--timeseries", required=True)
    ap.add_argument("--fuels", required=True)
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flows", default=None)
    ap.add_argument("--zone", default="DE_LU")
    ap.add_argument("--neighbor_dir", default=None)
    ap.add_argument("--neighbor_prices", default=None)
    ap.add_argument("--import_ef_csv", default=None)
    ap.add_argument("--oil_heavy", action="store_true")
    ap.add_argument("--consumption_based", action="store_true")
    ap.add_argument("--price_accept_band", type=float, default=8.0)
    args=ap.parse_args()

    run_backcast(args.timeseries, args.fuels, args.fleet, args.out,
                 flows_csv=args.flows, zone=args.zone,
                 neighbor_dir=args.neighbor_dir, neighbor_prices_csv=args.neighbor_prices,
                 import_ef_csv=args.import_ef_csv, oil_heavy=args.oil_heavy,
                 consumption_based=args.consumption_based,
                 price_accept_band=args.price_accept_band)

if __name__ == "__main__":
    main()
