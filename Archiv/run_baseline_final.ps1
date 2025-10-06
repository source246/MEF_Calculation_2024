# =================== BASELINE FINAL RUN (VOLLSTÄNDIG) ===================
# Implementiert alle 5 kritischen Fixes für Korrelations-Verbesserung

& ".venv/Scripts/python.exe" "scripts/track_c/mef_dispatch_2024_Final_Version.py" `
    --fleet "input/de/fleet/Kraftwerke_eff_binned.csv" `
    --eta_col "Imputed_Effizienz_binned" `
    --fuel_prices "input/de/fuels/prices_2024.csv" `
    --flows "flows/flows_scheduled_DE_LU_2024_net.csv" `
    --neighbor_gen_dir "input/neighbors/gen_2024" `
    --neighbor_load_dir "input/neighbors/out_load/2024" `
    --neighbor_prices "input/neighbors/prices/neighbor_prices_2024.csv" `
    --year 2024 `
    --start "2024-01-01 00:00" `
    --end "2025-01-01 00:00" `
    --outdir "out/baseline_final_corrected_2024" `
    --price_anchor "closest" `
    --price_tol 30.0 `
    --epsilon 0.5 `
    --corr_drop_neg_prices `
    --corr_cap_mode "absolute" `
    --corr_cap_value 500.0 `
    --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 `
    --psp_rt_eff 0.78 `
    --psp_pump_window_h 48 `
    --psp_accept_band 5.0 `
    --psp_price_cap 180.0 `
    --ee_price_threshold 5.0 `
    --nei_eta_mode "bounds" `
    --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --mu_cost_mode "q_vs_cost" `
    --mu_cost_q 0.20 `
    --mu_cost_alpha 0.70 `
    --mu_cost_monthly `
    --mu_cost_use_peak `
    --fossil_mustrun_mode "off" `
    --mustrun_lignite_q 0 `
    --mustrun_mode "capacity" `
    --mustrun_monthly `
    --mustrun_peak_hours "00-24" `
    --mustrun_quantile 0.08 `
    --mustrun_bid_eur_mwh -10 `
    --peak_switch `
    --peak_price_thresholds "300,500"

Write-Host "=== BASELINE FINAL RUN COMPLETE ===" -ForegroundColor Green
Write-Host "Erwartete Verbesserungen:" -ForegroundColor Yellow
Write-Host "  - filtered_correlation > 0.65 (war 0.6169)" -ForegroundColor Cyan
Write-Host "  - low_mef_target_match = True (war False, 519h vs 457h)" -ForegroundColor Cyan  
Write-Host "  - stage2_pass = true (war false)" -ForegroundColor Cyan
Write-Host "  - Öl-Artefakte eliminiert (peak_thresholds 300,500 EUR)" -ForegroundColor Cyan
Write-Host "  - EE-Threshold 5 EUR für realistische Low-MEF Korridore" -ForegroundColor Cyan