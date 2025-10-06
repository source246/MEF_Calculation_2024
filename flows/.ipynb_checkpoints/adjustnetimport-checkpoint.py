#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
adjustnetimport.py
...
Aufrufbeispiel (PowerShell):
    cd C:\Users\schoenmeiery\MEF\MEF_2024\flows
    python .\adjustnetimport.py `
      --flows .\flows_scheduled_DE_LU_2024_net.csv `
      --netpos .\Net_Positions_2024.csv `
      --out   .\flows_scheduled_DE_LU_2024_net_reconciled.csv `
      --zone  DE_LU
"""


import argparse
from pathlib import Path
import pandas as pd

TZ = "Europe/Berlin"

# ---------- Helpers ----------
def to_berlin(ts_like) -> pd.DatetimeIndex:
    """Gibt immer tz-aware Europe/Berlin zurück; robust gg. ENTSO-E-Intervalle und DST."""
    # ENTSO-E: "01/01/2024 00:00 - 01/01/2024 01:00" -> Beginn
    if isinstance(ts_like, (pd.Series, list)) and len(ts_like) and isinstance(ts_like[0], str) and " - " in str(ts_like[0]):
        starts = [s.split(" - ")[0].strip() if isinstance(s, str) else s for s in ts_like]
        dt = pd.to_datetime(starts, errors="coerce", dayfirst=True)
        idx = pd.DatetimeIndex(dt)
    else:
        dt = pd.to_datetime(ts_like, errors="coerce", utc=False)
        idx = pd.DatetimeIndex(dt)

    if idx.tz is None:
        try:
            idx = idx.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
        except Exception:
            # Fallback z. B. bei DST-Ende: erste 02:00 als Sommerzeit akzeptieren
            idx = idx.tz_localize(TZ, ambiguous=True, nonexistent="shift_forward")
    else:
        idx = idx.tz_convert(TZ)
    return idx


def autodetect_sep_decimal(csv_path: str) -> tuple[str, str]:
    """Erkennt ;/,(Semikolon/Dezimalkomma) vs ,/. (Komma/Dezimalpunkt)."""
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = "".join([next(f) for _ in range(5)])
    if head.count(";") > head.count(","):
        return ";", ","
    return ",", "."


def normalize_zone(z: str) -> str:
    """Normalisiert Zonencodes wie 'BZN|DE-LU' -> 'DE_LU'."""
    s = str(z).upper().strip()
    s = s.replace("BZN|", "").replace("CTY|", "")
    s = s.replace("|", "_").replace("-", "_").replace(" ", "_")
    s = s.replace("GERMANY/LUXEMBOURG", "DE_LU")
    return s


# ---------- IO ----------
def read_flows(path_flows: str) -> pd.DataFrame:
    """Liest die Flows-Datei mit imp_* Spalten ein und gibt stündlich gemittelte imp_* zurück."""
    df = pd.read_csv(path_flows, sep=None, engine="python")
    # Zeitspalte heuristisch finden
    tcol = next((c for c in df.columns if any(k in c.lower() for k in ["mtu", "timestamp", "time", "date"])), df.columns[0])
    df.index = to_berlin(df[tcol])
    df = df.drop(columns=[tcol], errors="ignore")

    imp_cols = [c for c in df.columns if str(c).startswith("imp_")]
    if not imp_cols:
        raise ValueError("Flows: Keine 'imp_*' Spalten gefunden.")

    # numeric + >=0
    for c in imp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    # ggf. 15min -> 1h
    df = df.resample("1h").mean()
    return df[imp_cols].sort_index()


def read_netpos(path_netpos: str, zone: str = "DE_LU") -> pd.Series:
    """ENTSO-E Net Positions einlesen, Zone filtern, Richtung anwenden und stündlich auf Importziel (>=0) mappen."""
    import re

    # --- robustes Sep/Decimal-Autodetect ---
    with open(path_netpos, "r", encoding="utf-8-sig", errors="ignore") as f:
        head_lines = [next(f) for _ in range(5)]
    head = "".join(head_lines)
    sep = ";" if head.count(";") >= head.count(",") else ","

    # provisorisch lesen um Dezimal zu erkennen
    tmp = pd.read_csv(path_netpos, sep=sep, encoding="utf-8-sig", nrows=5, low_memory=False)
    # suche die Werte-Spalte
    cand_val = next((c for c in tmp.columns if "net" in c.lower() and "position" in c.lower()), tmp.columns[-1])
    # nimm erstes Nicht-NA und prüfe Zeichen
    sample = None
    for v in tmp[cand_val].astype(str):
        v = v.strip()
        if v and v.lower() != "na":
            sample = v; break
    # Dezimal bestimmen
    if sample and ("," in sample) and not ("." in sample):
        decimal = ","
    else:
        decimal = "."

    df = pd.read_csv(path_netpos, sep=sep, decimal=decimal, encoding="utf-8-sig", low_memory=False)

    # --- Zeitspalte (ENTSO-E: "MTU (CET/CEST)" mit Intervallen) ---
    tcol = next((c for c in df.columns if any(k in c.lower() for k in ["mtu","timestamp","time","date"])), None)
    if tcol is None:
        raise ValueError("NetPositions: keine Zeitspalte gefunden.")
    idx_all = to_berlin(df[tcol])  # nimmt Intervall-Beginn

    # --- Zonenspalte normalisieren ---
    def norm_zone(z: str) -> str:
        s = str(z).upper().strip()
        s = s.replace("BZN|", "").replace("CTY|", "")
        s = s.replace("|", "_").replace("-", "_").replace(" ", "_")
        s = s.replace("GERMANY/LUXEMBOURG", "DE_LU")
        return s

    zcol = next((c for c in df.columns if any(k in c.lower() for k in ["area","zone","bidding","code"])), None)
    if zcol is not None:
        zn = df[zcol].map(norm_zone)
        mask = zn.eq(zone.upper()) | zn.str.contains(rf"\b{re.escape(zone.upper())}\b", na=False)
        df = df[mask]; idx = idx_all[mask]
    else:
        print("[WARN] NetPositions: keine Zonenspalte gefunden; es wird nicht gefiltert.")
        idx = idx_all

    # --- Werte- & Richtungs-Spalten finden ---
    vcol = next((c for c in df.columns if "net" in c.lower() and "position" in c.lower()), None)
    if vcol is None:
        numcols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        if not numcols:
            raise ValueError("NetPositions: keine numerische Spalte erkennbar.")
        vcol = numcols[-1]

    dcol = next((c for c in df.columns if "export" in c.lower() and "import" in c.lower()), None)
    # Werte als float
    val = pd.to_numeric(df[vcol], errors="coerce")

    # Richtung zu Vorzeichen: Export -> negativ, Import -> positiv
    if dcol is not None:
        diru = df[dcol].astype(str).str.strip().str.lower()
        sign = diru.map(lambda s: -1.0 if s.startswith("export") else (1.0 if s.startswith("import") else 0.0))
    else:
        # Falls keine Richtungsspalte vorhanden ist, nehmen wir an: positives Vorzeichen = bereits signiert
        sign = 1.0

    signed = val * sign
    s_signed = pd.Series(signed.values, index=idx).sort_index()

    # Ziel: Import-Anteil (>=0). Netto-Exportstunden -> 0
    s_target = s_signed.resample("1h").mean()
    s_target = s_target.clip(lower=0.0).rename("target_import_MW")

    # Sanity
    print("[NETPOS] parsed:", path_netpos)
    print("  n (nicht-NaN):", int(s_target.notna().sum()))
    print("  Anteil >0 MW:", float((s_target > 0).mean()))
    print("  Beispielwerte:", s_target.dropna().head(3).tolist())
    return s_target



# ---------- Core ----------
def reconcile(flows_path: str, netpos_path: str, zone: str, out_path: str | None):
    flows = read_flows(flows_path)                # imp_* Spalten stündlich
    target = read_netpos(netpos_path, zone=zone)  # Serie target_import_MW (>=0)

    # Gemeinsamer Index
    idx = flows.index.intersection(target.index)
    flows = flows.reindex(idx).fillna(0.0)
    target = target.reindex(idx).fillna(0.0)

    imp_cols = list(flows.columns)
    current_total = flows[imp_cols].sum(axis=1)

    # Skalierung je Stunde (nur reduzieren)
    scale = pd.Series(1.0, index=idx)
    mask_reduce = (current_total > 0) & (target < current_total)
    scale.loc[mask_reduce] = (target[mask_reduce] / current_total[mask_reduce]).clip(lower=0.0, upper=1.0)
    # wenn aktuelle Summe == 0 -> nicht erhöhen
    scale.loc[current_total <= 0] = 0.0

    flows_scaled = flows.mul(scale, axis=0).clip(lower=0.0)
    flows_scaled["net_import_total"] = flows_scaled.sum(axis=1)

    out = out_path or str(Path(flows_path).with_name(Path(flows_path).stem + "_reconciled.csv"))
    flows_scaled.reset_index().rename(columns={"index": "timestamp"}).to_csv(out, index=False)

    # Mini-Report
    print(f"[OK] Reconciled geschrieben: {out}")
    print("Mittelwerte (MW):")
    print("  Summe imp_* vorher :", float(current_total.mean()))
    print("  Ziel (Net Position):", float(target.mean()))
    print("  Summe imp_* nachher:", float(flows_scaled['net_import_total'].mean()))


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", required=True, help="Pfad zu flows_scheduled_DE_LU_2024_net.csv")
    ap.add_argument("--netpos", required=True, help="Pfad zu Net_Positions_2024.csv (ENTSO-E Export)")
    ap.add_argument("--zone", default="DE_LU", help="Zielzone, z. B. DE_LU")
    ap.add_argument("--out", default=None, help="Output-CSV (optional)")
    args = ap.parse_args()
    reconcile(args.flows, args.netpos, args.zone, args.out)


if __name__ == "__main__":
    main()
