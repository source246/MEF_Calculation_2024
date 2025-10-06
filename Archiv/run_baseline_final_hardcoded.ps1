# =====================================================================
# FINAL BASELINE RUN - KOMPROMISSLOSE KORRELATIONS-FIXES
# Implementiert alle wissenschaftlich belegten Baseline-Korrekturen
# =====================================================================

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
    --outdir "out/baseline_final_hardcoded_2024" `
    --nei_eta_mode "bounds" `
    --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --price_anchor "closest" `
    --price_tol 30.0 `
    --epsilon 0.5 `
    --corr_cap_mode "absolute" `
    --corr_cap_value 500.0 `
    --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 `
    --ee_price_threshold 5.0 `
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
    --peak_switch `
    --peak_price_thresholds "300,500" `
    --corr_drop_neg_prices

Write-Host "=== BASELINE FINAL RUN COMPLETE ===" -ForegroundColor Green
Write-Host "VALIDIERUNGS-GATES (alle müssen grün sein):" -ForegroundColor Yellow
Write-Host "  Gate 1: r_raw >= 0.70 und r_filt >= 0.75" -ForegroundColor Cyan
Write-Host "  Gate 2: Öl-Anteil DE-LU ≈ 0 in Top-500 Offendern" -ForegroundColor Cyan  
Write-Host "  Gate 3: Low-MEF-Stunden signifikant ↑ (>457h bei 5EUR threshold)" -ForegroundColor Cyan
Write-Host "  Gate 4: Spread |Preis-SRMC| median deutlich ↓" -ForegroundColor Cyan

# =====================================================================
# MINI-BATCH SENSITIVITÄT (3 kritische Achsen)
# =====================================================================

Write-Host "=== STARTE MINI-BATCH SENSITIVITÄT ===" -ForegroundColor Magenta

# 1. EE-Threshold Sensitivität
Write-Host "Sensitivität 1: EE-Threshold 2 EUR" -ForegroundColor Yellow
& ".venv/Scripts/python.exe" "scripts/track_c/mef_dispatch_2024_Final_Version.py" `
    --fleet "input/de/fleet/Kraftwerke_eff_binned.csv" `
    --eta_col "Imputed_Effizienz_binned" `
    --fuel_prices "input/de/fuels/prices_2024.csv" `
    --flows "flows/flows_scheduled_DE_LU_2024_net.csv" `
    --neighbor_gen_dir "input/neighbors/gen_2024" `
    --neighbor_load_dir "input/neighbors/out_load/2024" `
    --neighbor_prices "input/neighbors/prices/neighbor_prices_2024.csv" `
    --year 2024 --start "2024-01-01 00:00" --end "2025-01-01 00:00" `
    --outdir "out/sens_ee_threshold_2eur" `
    --nei_eta_mode "bounds" --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --price_anchor "closest" --price_tol 30.0 --epsilon 0.5 `
    --corr_cap_mode "absolute" --corr_cap_value 500.0 --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 --ee_price_threshold 2.0 `
    --mu_cost_mode "q_vs_cost" --mu_cost_q 0.20 --mu_cost_alpha 0.70 --mu_cost_monthly --mu_cost_use_peak `
    --fossil_mustrun_mode "off" --mustrun_lignite_q 0 --mustrun_mode "capacity" --mustrun_monthly `
    --mustrun_peak_hours "00-24" --mustrun_quantile 0.08 --peak_switch --peak_price_thresholds "300,500" --corr_drop_neg_prices

Write-Host "Sensitivität 2: EE-Threshold 10 EUR" -ForegroundColor Yellow
& ".venv/Scripts/python.exe" "scripts/track_c/mef_dispatch_2024_Final_Version.py" `
    --fleet "input/de/fleet/Kraftwerke_eff_binned.csv" `
    --eta_col "Imputed_Effizienz_binned" `
    --fuel_prices "input/de/fuels/prices_2024.csv" `
    --flows "flows/flows_scheduled_DE_LU_2024_net.csv" `
    --neighbor_gen_dir "input/neighbors/gen_2024" `
    --neighbor_load_dir "input/neighbors/out_load/2024" `
    --neighbor_prices "input/neighbors/prices/neighbor_prices_2024.csv" `
    --year 2024 --start "2024-01-01 00:00" --end "2025-01-01 00:00" `
    --outdir "out/sens_ee_threshold_10eur" `
    --nei_eta_mode "bounds" --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --price_anchor "closest" --price_tol 30.0 --epsilon 0.5 `
    --corr_cap_mode "absolute" --corr_cap_value 500.0 --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 --ee_price_threshold 10.0 `
    --mu_cost_mode "q_vs_cost" --mu_cost_q 0.20 --mu_cost_alpha 0.70 --mu_cost_monthly --mu_cost_use_peak `
    --fossil_mustrun_mode "off" --mustrun_lignite_q 0 --mustrun_mode "capacity" --mustrun_monthly `
    --mustrun_peak_hours "00-24" --mustrun_quantile 0.08 --peak_switch --peak_price_thresholds "300,500" --corr_drop_neg_prices

# 2. Anchor-Pack Sensitivität 
Write-Host "Sensitivität 3: Threshold Anchor" -ForegroundColor Yellow
& ".venv/Scripts/python.exe" "scripts/track_c/mef_dispatch_2024_Final_Version.py" `
    --fleet "input/de/fleet/Kraftwerke_eff_binned.csv" `
    --eta_col "Imputed_Effizienz_binned" `
    --fuel_prices "input/de/fuels/prices_2024.csv" `
    --flows "flows/flows_scheduled_DE_LU_2024_net.csv" `
    --neighbor_gen_dir "input/neighbors/gen_2024" `
    --neighbor_load_dir "input/neighbors/out_load/2024" `
    --neighbor_prices "input/neighbors/prices/neighbor_prices_2024.csv" `
    --year 2024 --start "2024-01-01 00:00" --end "2025-01-01 00:00" `
    --outdir "out/sens_threshold_anchor" `
    --nei_eta_mode "bounds" --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --price_anchor "threshold" --price_tol 5.0 --epsilon 0.01 `
    --corr_cap_mode "absolute" --corr_cap_value 500.0 --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 --ee_price_threshold 5.0 `
    --mu_cost_mode "q_vs_cost" --mu_cost_q 0.20 --mu_cost_alpha 0.70 --mu_cost_monthly --mu_cost_use_peak `
    --fossil_mustrun_mode "off" --mustrun_lignite_q 0 --mustrun_mode "capacity" --mustrun_monthly `
    --mustrun_peak_hours "00-24" --mustrun_quantile 0.08 --peak_switch --peak_price_thresholds "300,500" --corr_drop_neg_prices

# 3. Cap Sensitivität
Write-Host "Sensitivität 4: Cap 700 EUR" -ForegroundColor Yellow
& ".venv/Scripts/python.exe" "scripts/track_c/mef_dispatch_2024_Final_Version.py" `
    --fleet "input/de/fleet/Kraftwerke_eff_binned.csv" `
    --eta_col "Imputed_Effizienz_binned" `
    --fuel_prices "input/de/fuels/prices_2024.csv" `
    --flows "flows/flows_scheduled_DE_LU_2024_net.csv" `
    --neighbor_gen_dir "input/neighbors/gen_2024" `
    --neighbor_load_dir "input/neighbors/out_load/2024" `
    --neighbor_prices "input/neighbors/prices/neighbor_prices_2024.csv" `
    --year 2024 --start "2024-01-01 00:00" --end "2025-01-01 00:00" `
    --outdir "out/sens_cap_700eur" `
    --nei_eta_mode "bounds" --nei_eta_json "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json" `
    --varom_json "input/neighbors/neighbors_varom_from_maf2020.json" `
    --price_anchor "closest" --price_tol 30.0 --epsilon 0.5 `
    --corr_cap_mode "absolute" --corr_cap_value 700.0 --corr_cap_tol 3.0 `
    --psp_srmc_floor_eur_mwh 60.0 --ee_price_threshold 5.0 `
    --mu_cost_mode "q_vs_cost" --mu_cost_q 0.20 --mu_cost_alpha 0.70 --mu_cost_monthly --mu_cost_use_peak `
    --fossil_mustrun_mode "off" --mustrun_lignite_q 0 --mustrun_mode "capacity" --mustrun_monthly `
    --mustrun_peak_hours "00-24" --mustrun_quantile 0.08 --peak_switch --peak_price_thresholds "300,500" --corr_drop_neg_prices

Write-Host "=== MINI-BATCH COMPLETE - PRÜFE VALIDIERUNGS-GATES ===" -ForegroundColor Green