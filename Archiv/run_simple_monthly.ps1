# Einfaches monatsweises MEF-Script für 2024

Write-Host "Starte monatsweise MEF-Berechnung 2024..." -ForegroundColor Green

$Months = @(
    @{Name="Januar"; Start="2024-01-01 00:00"; End="2024-02-01 00:00"},
    @{Name="Februar"; Start="2024-02-01 00:00"; End="2024-03-01 00:00"},
    @{Name="März"; Start="2024-03-01 00:00"; End="2024-04-01 00:00"},
    @{Name="April"; Start="2024-04-01 00:00"; End="2024-05-01 00:00"},
    @{Name="Mai"; Start="2024-05-01 00:00"; End="2024-06-01 00:00"},
    @{Name="Juni"; Start="2024-06-01 00:00"; End="2024-07-01 00:00"}
)

foreach ($Month in $Months) {
    Write-Host "`nStarte $($Month.Name)..." -ForegroundColor Cyan
    
    $OutDir = "out\modified_2024_$($Month.Name.ToLower())"
    
    python scripts\track_c\mef_dispatch_2024_Final_Version.py --fleet "input\de\fleet\Kraftwerke_eff_binned.csv" --eta_col "Imputed_Effizienz_binned" --fuel_prices "input\de\fuels\prices_2024.csv" --flows "flows\flows_scheduled_DE_LU_2024_net.csv" --neighbor_gen_dir "input\neighbors\gen_2024" --neighbor_load_dir "input\neighbors\out_load\2024" --neighbor_prices "input\neighbors\prices\neighbor_prices_2024.csv" --year 2024 --start "$($Month.Start)" --end "$($Month.End)" --outdir "$OutDir" --price_anchor "threshold" --price_tol 3.0 --epsilon 0.01 --corr_drop_neg_prices --corr_cap_mode "peaker_min" --corr_cap_tol 3.0 --psp_srmc_floor_eur_mwh 60.0 --psp_rt_eff 0.78 --psp_pump_window_h 48 --psp_accept_band 5.0 --psp_price_cap 180.0 --nei_eta_mode "bounds" --nei_eta_json "input\neighbors\neighbors_eta_bounds_from_jrc_2024.json" --varom_json "input\neighbors\neighbors_varom_from_maf2020.json" --mu_cost_mode "q_vs_cost" --mu_cost_q 0.20 --mu_cost_alpha 0.70 --mu_cost_monthly --mu_cost_use_peak --mustrun_mode "capacity" --mustrun_monthly --mustrun_peak_hours "00-24" --mustrun_quantile 0.08 --mustrun_lignite_q 0 --mustrun_neg_pricing_enable --mustrun_neg_price_threshold_pct 0.75 --mustrun_neg_price_value -10 --mustrun_bid_eur_mwh -10 --fb_np_csv "input\fb\fb_core_DE_2024.csv" --jao_fb_enable --jao_fb_tolerance 100
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $($Month.Name) erfolgreich" -ForegroundColor Green
    } else {
        Write-Host "❌ $($Month.Name) fehlgeschlagen" -ForegroundColor Red
    }
}

Write-Host "`nMonatsweise Berechnung abgeschlossen!" -ForegroundColor Green