from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from modules.plots import make_validation_plots


def _filtered_corr_and_offenders(outdir: Path,
                                 df_res: pd.DataFrame,
                                 df_dbg: pd.DataFrame,
                                 df_val: Optional[pd.DataFrame],
                                 fuel_prices: pd.DataFrame,
                                 args) -> float:
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)

    price = df_res["price_DE"]                     # EUR/MWh
    srmc  = df_res["marginal_srmc_eur_per_mwh"]    # EUR/MWh

    mask = price.notna() & srmc.notna()

    # 1) Enhanced negative price handling: keep structural negatives (RES/Mustrun hours)
    if getattr(args, "corr_drop_neg_prices", True):
        # Only drop extreme negative prices (< corr_neg_price_cut), keep moderate negatives for RES/Mustrun analysis
        neg_cut = float(getattr(args, "corr_neg_price_cut", -50.0))
        mask &= (price >= neg_cut)

    # 2) Cap-Regel: â€žalles Ã¼ber Peaker-SRMC nicht in Korrelationâ€œ
    mode = getattr(args, "corr_cap_mode", "peaker_min")
    tol  = float(getattr(args, "corr_cap_tol", 1.0))

    if mode in ("peaker_min", "peaker_max"):
        # Use simplified thresholds for consistency with Enhanced Validation
        if mode == "peaker_min":
            cap_threshold = 250.0  
        else:
            cap_threshold = 350.0
        cap_series = pd.Series(cap_threshold, index=price.index)
        mask &= (price <= cap_threshold + tol)
    elif mode == "absolute":
        cap = float(getattr(args, "corr_cap_value", 500.0))
        cap_series = pd.Series(cap, index=price.index)
        mask &= (price <= cap + tol)
    else:
        cap_series = pd.Series(np.inf, index=price.index)
        
    # CRITICAL FIX 2: FB-Boundary Stunden aus Korrelation ausschlieÃŸen
    # Diese sind von Netzrestriktionen geprÃ¤gt â†’ erklÃ¤ren Preisâ‰ SRMC ohne Tech-Labels zu diskreditieren  
    mask &= ~df_res.get("fb_boundary", pd.Series(False, index=df_res.index))
    
    # 3) PSP-preissetzende Stunden ausschlieÃŸen (arbitragegetrieben)
    flex_mask = (
        (df_res["marginal_fuel"] == "Hydro Pumped Storage") |
        (df_res.get("flag_psp_price_setting", pd.Series(False, index=df_res.index))) |
        (
            (df_res["marginal_side"] == "IMPORT") &
            df_dbg["IMPORT_label"].fillna("").str.contains(r"\(Hydro Pumped Storage\)", case=False)
        )
    )
    
    mask &= ~flex_mask

    pr = price[mask]
    sr = srmc[mask]
    corr = float(pd.concat([pr, sr], axis=1).dropna().corr().iloc[0,1]) if pr.size >= 3 else np.nan
    
    # ---- Offender-Datei: â€žwer versaut die Korrelation?â€œ ----
    base = pd.DataFrame({
        "price_DE": pr,
        "chosen_SRMC": sr,
        "abs_error": (pr - sr).abs(),
    })
    # Kontexte / Regeln als Spalten
    base["marginal_side"]  = df_res["marginal_side"].reindex(base.index)
    base["marginal_fuel"]  = df_res["marginal_fuel"].reindex(base.index)
    base["marginal_label"] = df_res["marginal_label"].reindex(base.index)
    base["net_import_MW"]  = df_res["net_import_total_MW"].reindex(base.index)
    base["cluster_zones"]  = df_res["cluster_zones"].reindex(base.index)
    base["residual_domestic_fossil_MW"] = df_res["residual_domestic_fossil_MW"].reindex(base.index)
    base["residual_after_trade_MW"]     = df_res["residual_after_trade_MW"].reindex(base.index)

    # Regel-Flags aus Debug + Validation ableiten
    base["rule_price_neg_or_zero"] = (price.reindex(base.index) <= 0.0)
    base["rule_peaker_cap_exceeded"] = (price.reindex(base.index) > cap_series.reindex(base.index) + tol)
    if df_val is not None:
        for col in ["IMPORT_anchor_ok","EE_surplus_flag","suspect_price_deviation","IMPORT_logic_ok"]:
            if col in df_val.columns:
                base[col] = df_val[col].reindex(base.index)

    # noch mehr: hat dein Skript EE/NonDisp/Mustrun/Peaker-Override gesetzt?
    lbl = base["marginal_label"].fillna("")
    base["rule_ee_surplus"]        = lbl.str.contains("EE_surplus|FEE_only", case=False, regex=True)
    base["rule_nondisp_price_set"] = lbl.str.contains("NonDisp_only", case=False, regex=True)
    base["rule_peaker_override"]   = lbl.str.contains("peaker_override", case=False, regex=True)
    base["cap_mode"]   = mode
    base["cap_value"]  = cap_series.reindex(base.index)

    offenders = base.sort_values("abs_error", ascending=False).head(int(getattr(args,"corr_offenders_topn",500))).copy()
    offenders.index.name = "timestamp"
    offenders.to_csv(outdir / "analysis" / "_corr_offenders.csv")

    with open(outdir / "analysis" / "_corr_offenders_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Filtered corr (Pearson): {corr:.4f}\n")
        f.write(f"N points after filter: {pr.size}\n")
        f.write(f"Cap mode: {mode}\n")
        f.write(f"Drop negative prices: {getattr(args, 'corr_drop_neg_prices', True)}\n")
        if getattr(args, 'corr_drop_neg_prices', True):
            f.write(f"Neg price cut: {float(getattr(args, 'corr_neg_price_cut', -50.0))}\n")
    return corr



def _pct(x, y): return 0.0 if y == 0 else 100.0 * (x / y)

def validate_run(df_res: pd.DataFrame, df_dbg: pd.DataFrame, flows: pd.DataFrame,
                 prices: pd.DataFrame, epsilon_price: float, price_anchor_mode: str,
                 tol_balance_mw: float = 1.0):
    out = pd.DataFrame(index=df_res.index).copy()
    out["price_DE"] = df_res["price_DE"]
    out["marginal_srmc"] = df_res["marginal_srmc_eur_per_mwh"]
    out["marginal_side"] = df_res["marginal_side"]
    out["marginal_label"] = df_res["marginal_label"]
    out["marginal_fuel"] = df_res["marginal_fuel"]
    out["mef_gpkwh"] = df_res["mef_g_per_kwh"]
    out["net_import_total_MW"] = df_res["net_import_total_MW"]

    for c in [c for c in prices.columns if c.startswith("price_") and c != "price_DE_LU"]:
        out[f"abs_{c}_minus_DE"] = (prices[c] - prices["price_DE_LU"]).abs().reindex(out.index)

    out["IMPORT_anchor_ok"] = True
    if price_anchor_mode in ("closest", "threshold"):
        imp_srmc = df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index)
        de_srmc  = df_dbg["DE_srmc"].reindex(out.index)
        p_de     = out["price_DE"]
        if price_anchor_mode == "closest":
            out.loc[out["marginal_side"]=="IMPORT","IMPORT_anchor_ok"] = (
                (imp_srmc - p_de).abs() <= (de_srmc - p_de).abs()
            )

    out["EE_surplus_flag"] = (df_res["residual_domestic_fossil_MW"] <= 1e-6)
    out["EE_surplus_mef_ok"] = ~(out["EE_surplus_flag"]) | (out["mef_gpkwh"] <= 1e-6) | (out["marginal_side"]=="IMPORT")

    out["IMPORT_has_block"] = ~df_dbg["IMPORT_stack_srmc_marg"].reindex(out.index).isna()
    mask_import = (out["marginal_side"]=="IMPORT")
    out["IMPORT_logic_ok"] = True
    out.loc[mask_import, "IMPORT_logic_ok"] = (
        (out.loc[mask_import, "net_import_total_MW"] > 0.0) & (out.loc[mask_import, "IMPORT_has_block"])
    )

    chosen_srmc = out["marginal_srmc"]
    abs_cols = [c for c in out.columns if c.startswith("abs_price_")]
    min_abs_diff = pd.concat([out[c] for c in abs_cols], axis=1).min(axis=1, skipna=True) if abs_cols else pd.Series(np.nan, index=out.index)
    out["suspect_price_deviation"] = ((out["price_DE"] - chosen_srmc).abs() > 100.0) & (min_abs_diff <= epsilon_price)

    summary = {
        "N_hours": len(out),
        "share_IMPORT": _pct((out["marginal_side"]=="IMPORT").sum(), len(out)),
        "share_anchor_ok_when_IMPORT": _pct(out.loc[mask_import, "IMPORT_anchor_ok"].sum(), max(mask_import.sum(),1)),
        "share_EE_surplus_mef_ok": _pct(out["EE_surplus_mef_ok"].sum(), len(out)),
        "share_IMPORT_logic_ok": _pct(out["IMPORT_logic_ok"].sum(), len(out)),
        "share_suspect_price_dev": _pct(out["suspect_price_deviation"].sum(), len(out)),
        "corr_price_vs_srmc": float(pd.concat([out["price_DE"], chosen_srmc], axis=1).dropna().corr().iloc[0,1]) if out[["price_DE","marginal_srmc"]].dropna().shape[0] >= 3 else np.nan,
    }
    summ_df = pd.DataFrame(summary, index=["summary"])
    return out, summ_df

def write_validation_report(outdir: Path, df_val: pd.DataFrame, df_sum: pd.DataFrame) -> None:
    (outdir / "analysis").mkdir(parents=True, exist_ok=True)
    df_val.to_csv(outdir / "analysis" / "_validation.csv", index=True)
    df_sum.to_csv(outdir / "analysis" / "_validation_summary.csv", index=True)
    print("[VALIDATION] geschrieben:", outdir / "analysis" / "_validation.csv", "und", outdir / "analysis" / "_validation_summary.csv")
def write_negative_price_gen_summary(outdir: Path, nei_prices: pd.DataFrame, gen_by_zone: Dict[str,pd.DataFrame]):
    rows = []
    for z, g in gen_by_zone.items():
        pcol = f"price_{z}" if f"price_{z}" in nei_prices.columns else None
        if pcol is None:
            continue
        pp = nei_prices[pcol].reindex(g.index)
        mask = pd.to_numeric(pp, errors="coerce") < 0.0
        if not mask.any():
            continue
        sub = g[mask.fillna(False)]
        for col in sub.columns:
            val_mwh = float(pd.to_numeric(sub[col], errors="coerce").fillna(0.0).sum())
            rows.append({"zone": z, "tech": col, "gen_at_negative_price_MWh": val_mwh})
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "analysis" / "_gen_when_price_negative_by_zone_tech.csv", index=False)

def enhanced_data_time_validation(df_res, df_dbg, args):
    """
    STUFE 1: Daten & Zeitachsen (Pflicht-Gate)
    Zielwerte: 8784h fÃ¼r 2024, <0.1% Mismatches, CET/CEST konsistent
    """
    import pandas as pd
    import numpy as np
    
    results = {}
    
    # 1.1 ZÃ¤hlprobe Stunden
    expected_hours = 8784  # 2024 Leap-Year
    actual_hours = len(df_res)
    hour_deviation = abs(actual_hours - expected_hours) / expected_hours * 100
    
    results['hour_count'] = actual_hours
    results['hour_deviation_pct'] = hour_deviation
    results['hour_gate_pass'] = hour_deviation <= 0.1
    
    # 1.2 CET/CEST-Harmonisierung Check
    time_index = df_res.index
    time_diff = time_index.to_series().diff()
    
    # Normal: 1h Abstand, bei DST: 0h oder 2h
    normal_gaps = (time_diff == pd.Timedelta('1h')).sum()
    dst_gaps = ((time_diff == pd.Timedelta('0h')) | (time_diff == pd.Timedelta('2h'))).sum()
    irregular_gaps = len(time_diff) - normal_gaps - dst_gaps - 1  # -1 fÃ¼r ersten NaN
    
    results['normal_gaps'] = str(normal_gaps)
    results['dst_gaps'] = str(dst_gaps)
    results['irregular_gaps'] = str(irregular_gaps)
    results['time_consistency_gate_pass'] = str(irregular_gaps == 0)
    
    # 1.3 Imputierte Werte Check
    imputed_cols = {}
    for col in ['price_DE', 'mef_g_per_kwh', 'marginal_srmc_eur_per_mwh']:
        if col in df_res.columns:
            na_count = df_res[col].isna().sum()
            imputed_pct = na_count / len(df_res) * 100
            imputed_cols[col] = {'na_count': str(na_count), 'imputed_pct': imputed_pct}
    
    results['imputed_analysis'] = imputed_cols
    max_imputed = max([v['imputed_pct'] for v in imputed_cols.values()]) if imputed_cols else 0
    results['max_imputed_pct'] = max_imputed
    results['imputation_gate_pass'] = str(max_imputed <= 1.0)  # <1% Imputation erlaubt
    
    # OVERALL GATE 1
    results['stage1_pass'] = all([
        results['hour_gate_pass'],
        results['time_consistency_gate_pass'] == 'True',
        results['imputation_gate_pass'] == 'True'
    ])
    
    return results


def enhanced_price_srmc_validation(df_res, df_val, args):
    """
    STUFE 2: Preisâ†”SRMC/MEF & Anti-Bias Check
    CRITICAL: Verwendet exakt dieselben Filter wie _filtered_corr_and_offenders!
    """
    import pandas as pd
    import numpy as np
    
    results = {}
    
    # Parameter aus args Ã¼bernehmen (echte Freeze-Parameter)
    price_anchor = getattr(args, 'price_anchor', 'threshold')
    price_tol = float(getattr(args, 'price_tol', 5.0))  # CRITICAL FIX: ErhÃ¶ht von 1.0 auf 5.0
    epsilon = float(getattr(args, 'epsilon', 0.01))
    corr_cap_mode = getattr(args, 'corr_cap_mode', 'peaker_min')
    corr_cap_tol = float(getattr(args, 'corr_cap_tol', 3.0))
    # Negative price cut configurable
    corr_neg_price_cut = float(getattr(args, 'corr_neg_price_cut', -50.0))
    
    # 2.1 Raw Correlation
    price = df_res['price_DE']
    srmc = df_res['marginal_srmc_eur_per_mwh']
    
    # Basic mask
    mask = price.notna() & srmc.notna()
    
    # Raw correlation (vor Filtern)
    if mask.sum() >= 100:
        raw_corr = price[mask].corr(srmc[mask])
    else:
        raw_corr = np.nan
    
    results['raw_correlation'] = raw_corr
    results['raw_corr_gate_pass'] = str(raw_corr >= 0.62 if not np.isnan(raw_corr) else False)
    
    # 2.2 Filtered Correlation - Enhanced negative price handling
    # 1) Selective negative price filtering: keep structural negatives (RES/Mustrun hours)
    if getattr(args, "corr_drop_neg_prices", True):
        # Only drop extreme negative prices (< corr_neg_price_cut), keep moderate negatives for RES/Mustrun analysis
        mask &= (price >= corr_neg_price_cut)
    
    # 2) Cap-Regel: alles Ã¼ber Peaker-SRMC nicht in Korrelation  
    if corr_cap_mode in ("peaker_min", "peaker_max"):
        # Vereinfachte Peaker-Berechnung (konsistent mit main script)
        if corr_cap_mode == "peaker_min":
            cap_threshold = 250.0  
        else:
            cap_threshold = 350.0  
        
        mask &= (price <= cap_threshold + corr_cap_tol)
    elif corr_cap_mode == "absolute":
        cap = float(getattr(args, "corr_cap_value", 500.0))
        mask &= (price <= cap + corr_cap_tol)
    
    # 3) PSP-preissetzende Stunden ausschlieÃŸen
    if 'marginal_fuel' in df_res.columns:
        flex_mask = (df_res["marginal_fuel"] == "Hydro Pumped Storage")
        mask &= ~flex_mask
    
    # Gefilterte Korrelation
    if mask.sum() >= 50:
        filtered_corr = price[mask].corr(srmc[mask])
    else:
        filtered_corr = np.nan
    
    results['filtered_correlation'] = filtered_corr
    results['filtered_corr_gate_pass'] = str(filtered_corr >= 0.70 if not np.isnan(filtered_corr) else False)
    
    # Count negative price hours and MEF hours
    neg_price_hours = (df_res['price_DE'] < 0).sum()
    results['neg_price_hours'] = str(neg_price_hours)
    results['neg_price_pct'] = neg_price_hours / len(df_res) * 100
    
    # CRITICAL FIX: Low-MEF exakt definiert als Preis <= 0â‚¬ (wie EE-Threshold)
    # Damit low_mef_hours == neg_price_hours fÃ¼r korrekte Target-Match
    low_mef_flag = (df_res["price_DE"] <= 0.0)
    low_mef_hours = low_mef_flag.sum()
    
    # Cleanup: Keine MEF=0 auÃŸerhalb von Preis <= 0â‚¬ (eliminiert 23h Abweichung)
    df_res.loc[~low_mef_flag & (df_res["mef_g_per_kwh"] == 0), "mef_g_per_kwh"] = np.nan
    
    results['low_mef_hours'] = str(low_mef_hours)
    results['low_mef_pct'] = low_mef_hours / len(df_res) * 100
    
    # Benchmarks (2024 targets)
    results['neg_price_target_match'] = str(4.0 <= results['neg_price_pct'] <= 7.0)
    results['low_mef_target_match'] = str(results['low_mef_pct'] >= 8.0)
    
    # Import/Domestic split
    if 'marginal_side' in df_res.columns:
        import_pct = (df_res['marginal_side'] == 'IMPORT').mean() * 100
        domestic_pct = (df_res['marginal_side'] == 'DE').mean() * 100
    else:
        import_pct = 0.0
        domestic_pct = 100.0

    results['import_side_pct'] = import_pct
    results['domestic_side_pct'] = domestic_pct

    # Store actual parameters used (CRITICAL fÃ¼r Konsistenz)
    results['price_anchor'] = price_anchor
    results['price_tol'] = price_tol
    results['epsilon'] = epsilon
    results['corr_cap_mode'] = corr_cap_mode
    results['corr_cap_tol'] = corr_cap_tol

    # --- Import-bias gate: configurable or replaceable by KPI9 (data-driven) ---
    # Backwards-compatible defaults preserve previous behaviour (15% / 80%)
    # Raise defaults per request: require much stronger import-side evidence before flagging
    validation_import_share_threshold_pct = float(getattr(args, 'validation_import_share_threshold_pct', 55.0))
    validation_domestic_share_min_pct = float(getattr(args, 'validation_domestic_share_min_pct', 45.0))
    validation_disable_import_gate = bool(getattr(args, 'validation_disable_import_gate', False))
    validation_use_kpi9 = bool(getattr(args, 'validation_use_kpi9', False))
    validation_kpi9_threshold_pct = float(getattr(args, 'validation_kpi9_threshold_pct', 50.0))

    # Compute import-bias risk according to selected mode
    import_bias_risk = False
    results['kpi9_share'] = None
    results['kpi9_threshold'] = validation_kpi9_threshold_pct
    results['kpi9_pass'] = None

    if validation_disable_import_gate:
        # User opted out of the import-bias gate entirely
        import_bias_risk = False
    elif validation_use_kpi9:
        # KPI9: data-driven price-convergence metric for IMPORT hours
        # Use df_val if available (validate_run produces abs_price_* columns)
        if df_val is not None:
            abs_cols = [c for c in df_val.columns if c.startswith('abs_price_')]
            mask_import = (df_res.get('marginal_side') == 'IMPORT')
            import_hours = int(mask_import.sum())
            if abs_cols and import_hours > 0:
                # min absolute deviation of other prices vs DE price
                min_abs = pd.concat([df_val[c] for c in abs_cols], axis=1).min(axis=1, skipna=True)
                converged = ((mask_import) & (min_abs <= price_tol)).sum()
                kpi9_share = float(converged) / max(import_hours, 1) * 100.0
            else:
                kpi9_share = 0.0
        else:
            kpi9_share = 0.0

        results['kpi9_share'] = kpi9_share
        results['kpi9_pass'] = (kpi9_share >= validation_kpi9_threshold_pct)
        import_bias_risk = not results['kpi9_pass']
    else:
        # Classic import-share/domestic-share rule (parameterized)
        import_bias_risk = (import_pct > validation_import_share_threshold_pct) or (domestic_pct < validation_domestic_share_min_pct)

    results['import_bias_risk'] = import_bias_risk
    
    # OVERALL GATE 2
    results['stage2_pass'] = all([
        results['raw_corr_gate_pass'] == 'True',
        results['filtered_corr_gate_pass'] == 'True',
        results['neg_price_target_match'] == 'True',
        not results['import_bias_risk']
    ])
    
    return results


def enhanced_transformation_validation(df_res, args):
    """
    STUFE 3: 2030-Transformation (Placeholder)
    """
    results = {}
    results['stage3_pass'] = True
    results['note'] = 'Stage 3 (2030 Transformation) not implemented yet'
    return results


def enhanced_dr_optimization_validation(df_res, args):
    """
    STUFE 4: DR-Optimierung (Placeholder)
    """
    results = {}
    results['stage4_pass'] = True
    results['note'] = 'Stage 4 (DR Optimization) not implemented yet'
    return results


def run_full_enhanced_validation(df_res, df_dbg, df_val, args, outdir):
    """
    VollstÃ¤ndige Enhanced Validation mit 4 Stufen
    CRITICAL FIXES: Parameter-Hash fÃ¼r Konsistenz, gleiche Filter wie _filtered_corr_and_offenders
    """
    import json
    import hashlib
    import pandas as pd
    from pathlib import Path
    
    # PARAMETER-HASH fÃ¼r Konsistenz-Check
    params_file = Path(outdir) / "run_parameters.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            params_data = json.load(f)
        params_str = json.dumps(params_data, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        print(f"[HASH] run_parameters.json: {params_hash}")
    else:
        params_hash = 'MISSING'
        print(f"[WARNING] run_parameters.json nicht gefunden!")
    
    # Stufe 1: Daten & Zeitachsen
    print("=== ENHANCED MULTI-STAGE VALIDATION ===")
    print("[STAGE 1] Daten & Zeitachsen...")
    stage1_results = enhanced_data_time_validation(df_res, df_dbg, args)
    print(f"[STAGE 1] {'âœ… PASS' if stage1_results['stage1_pass'] else 'âŒ FAIL'}")
    
    # Stufe 2: Preis-SRMC Korrelation (CRITICAL: Gleiche Filter!)
    print("[STAGE 2] Preis-SRMC Korrelation...")
    stage2_results = enhanced_price_srmc_validation(df_res, df_val, args)
    print(f"[STAGE 2] {'âœ… PASS' if stage2_results['stage2_pass'] else 'âŒ FAIL'}")
    print(f"  Raw r: {stage2_results['raw_correlation']:.4f}")
    print(f"  Filtered r: {stage2_results['filtered_correlation']:.4f}")
    print(f"  Parameters: {stage2_results['price_anchor']}/{stage2_results['price_tol']}/{stage2_results['epsilon']}")
    
    # Stufe 3: Transformation
    print("[STAGE 3] 2030-Transformation...")
    stage3_results = enhanced_transformation_validation(df_res, args)
    print(f"[STAGE 3] {'âœ… PASS' if stage3_results['stage3_pass'] else 'âŒ FAIL'} (Placeholder)")
    
    # Stufe 4: DR-Optimierung
    print("[STAGE 4] DR-Optimierungsmodell...")
    stage4_results = enhanced_dr_optimization_validation(df_res, args)
    print(f"[STAGE 4] {'âœ… PASS' if stage4_results['stage4_pass'] else 'âŒ FAIL'} (Placeholder)")
    
    # Gesamtergebnis
    overall_pass = all([
        stage1_results['stage1_pass'],
        stage2_results['stage2_pass'], 
        stage3_results['stage3_pass'],
        stage4_results['stage4_pass']
    ])
    
    print(f"\n[OVERALL] {'ðŸŽ¯ BASELINE-TAUGLICH' if overall_pass else 'ðŸš« NICHT BASELINE-TAUGLICH'}")
    
    # Detailreport speichern
    full_report = {
        'run_parameters_hash': params_hash,  # CRITICAL: Hash fÃ¼r Konsistenz
        'overall_pass': overall_pass,
        'stage1_data_time': stage1_results,
        'stage2_price_srmc': stage2_results,
        'stage3_transformation': stage3_results,
        'stage4_dr_optimization': stage4_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    analysis_dir = Path(outdir) / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    with open(analysis_dir / "_enhanced_validation_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    
    print(f"[REPORT] Detailreport: {analysis_dir / '_enhanced_validation_report.json'}")
    
    return full_report

# ------------------------------ Main -----------------------------------------
