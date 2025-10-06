# Monatsweise MEF-Berechnung für 2024 mit modifizierter Dispatch-Logik
# Must-Run vor EE + Neighbor fillup

$BaseParams = @(
    "--fleet", "input\de\fleet\Kraftwerke_eff_binned.csv",
    "--eta_col", "Imputed_Effizienz_binned",
    "--fuel_prices", "input\de\fuels\prices_2024.csv",
    "--flows", "flows\flows_scheduled_DE_LU_2024_net.csv",
    "--neighbor_gen_dir", "input\neighbors\gen_2024",
    "--neighbor_load_dir", "input\neighbors\out_load\2024",
    "--neighbor_prices", "input\neighbors\prices\neighbor_prices_2024.csv",
    "--year", "2024",
    "--price_anchor", "threshold",
    "--price_tol", "3.0",
    "--epsilon", "0.01",
    "--corr_drop_neg_prices",
    "--corr_cap_mode", "peaker_min",
    "--corr_cap_tol", "3.0",
    "--psp_srmc_floor_eur_mwh", "60.0",
    "--psp_rt_eff", "0.78",
    "--psp_pump_window_h", "48",
    "--psp_accept_band", "5.0",
    "--psp_price_cap", "180.0",
    "--nei_eta_mode", "bounds",
    "--nei_eta_json", "input\neighbors\neighbors_eta_bounds_from_jrc_2024.json",
    "--varom_json", "input\neighbors\neighbors_varom_from_maf2020.json",
    "--mu_cost_mode", "q_vs_cost",
    "--mu_cost_q", "0.20",
    "--mu_cost_alpha", "0.70",
    "--mu_cost_monthly",
    "--mu_cost_use_peak",
    "--mustrun_mode", "capacity",
    "--mustrun_monthly",
    "--mustrun_peak_hours", "00-24",
    "--mustrun_quantile", "0.08",
    "--mustrun_lignite_q", "0",
    "--mustrun_neg_pricing_enable",
    "--mustrun_neg_price_threshold_pct", "0.75",
    "--mustrun_neg_price_value", "-10",
    "--mustrun_bid_eur_mwh", "-10",
    "--fb_np_csv", "input\fb\fb_core_DE_2024.csv",
    "--jao_fb_enable",
    "--jao_fb_tolerance", "100"
)

# Monats-Definitionen
$Months = @(
    @{Name="Januar"; Start="2024-01-01 00:00"; End="2024-02-01 00:00"},
    @{Name="Februar"; Start="2024-02-01 00:00"; End="2024-03-01 00:00"},
    @{Name="März"; Start="2024-03-01 00:00"; End="2024-04-01 00:00"},
    @{Name="April"; Start="2024-04-01 00:00"; End="2024-05-01 00:00"},
    @{Name="Mai"; Start="2024-05-01 00:00"; End="2024-06-01 00:00"},
    @{Name="Juni"; Start="2024-06-01 00:00"; End="2024-07-01 00:00"},
    @{Name="Juli"; Start="2024-07-01 00:00"; End="2024-08-01 00:00"},
    @{Name="August"; Start="2024-08-01 00:00"; End="2024-09-01 00:00"},
    @{Name="September"; Start="2024-09-01 00:00"; End="2024-10-01 00:00"},
    @{Name="Oktober"; Start="2024-10-01 00:00"; End="2024-11-01 00:00"},
    @{Name="November"; Start="2024-11-01 00:00"; End="2024-12-01 00:00"},
    @{Name="Dezember"; Start="2024-12-01 00:00"; End="2025-01-01 00:00"}
)

$StartTime = Get-Date
Write-Host "========================================" -ForegroundColor Green
Write-Host "Monatsweise MEF-Berechnung 2024 GESTARTET" -ForegroundColor Green
Write-Host "Mit modifizierter Dispatch-Logik:" -ForegroundColor Yellow
Write-Host "  - Must-Run vor EE" -ForegroundColor Yellow
Write-Host "  - Neighbor fillup Logik" -ForegroundColor Yellow
Write-Host "Startzeit: $StartTime" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

$SuccessfulMonths = @()
$FailedMonths = @()

foreach ($Month in $Months) {
    $MonthStart = Get-Date
    Write-Host "`n--- STARTE $($Month.Name) ---" -ForegroundColor Cyan
    Write-Host "Zeitraum: $($Month.Start) bis $($Month.End)" -ForegroundColor Gray
    
    $OutDir = "out\modified_2024_$($Month.Name.ToLower())"
    
    # Kombiniere alle Parameter
    $AllParams = $BaseParams + @(
        "--start", $Month.Start,
        "--end", $Month.End,
        "--outdir", $OutDir
    )
    
    try {
        # Führe MEF-Berechnung aus
        & python scripts\track_c\mef_dispatch_2024_Final_Version.py @AllParams
        
        if ($LASTEXITCODE -eq 0) {
            $MonthEnd = Get-Date
            $Duration = $MonthEnd - $MonthStart
            Write-Host "✅ $($Month.Name) ERFOLGREICH in $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            $SuccessfulMonths += $Month.Name
            
            # Prüfe MEF-Outputs
            if (Test-Path "$OutDir\mef_track_c_2024.csv") {
                $MefData = Get-Content "$OutDir\mef_track_c_2024.csv" | Select-Object -First 3
                Write-Host "MEF-Output gefunden:" -ForegroundColor Gray
                $MefData | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
            }
        } else {
            Write-Host "❌ $($Month.Name) FEHLGESCHLAGEN (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
            $FailedMonths += $Month.Name
        }
    }
    catch {
        Write-Host "❌ $($Month.Name) FEHLER: $($_.Exception.Message)" -ForegroundColor Red
        $FailedMonths += $Month.Name
    }
}

$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "MONATSWEISE BERECHNUNG ABGESCHLOSSEN" -ForegroundColor Green
Write-Host "Gesamtdauer: $($TotalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`n📊 ZUSAMMENFASSUNG:" -ForegroundColor White
Write-Host "Erfolgreich: $($SuccessfulMonths.Count)/12 Monate" -ForegroundColor Green
if ($SuccessfulMonths.Count -gt 0) {
    Write-Host "✅ Erfolgreich: $($SuccessfulMonths -join ', ')" -ForegroundColor Green
}

if ($FailedMonths.Count -gt 0) {
    Write-Host "❌ Fehlgeschlagen: $($FailedMonths -join ', ')" -ForegroundColor Red
}

# Erstelle Gesamtübersicht der MEF-Ergebnisse
if ($SuccessfulMonths.Count -gt 0) {
    Write-Host "`n📈 MEF-ERGEBNISSE SAMMELN..." -ForegroundColor Cyan
    
    $AllMefResults = @()
    foreach ($MonthName in $SuccessfulMonths) {
        $OutDir = "out\modified_2024_$($MonthName.ToLower())"
        $MefFile = "$OutDir\mef_track_c_2024.csv"
        
        if (Test-Path $MefFile) {
            $MefContent = Get-Content $MefFile
            if ($MefContent.Count -gt 1) {
                # Header nur beim ersten Mal hinzufügen
                if ($AllMefResults.Count -eq 0) {
                    $AllMefResults += $MefContent[0]  # Header
                }
                # Daten ohne Header hinzufügen
                $AllMefResults += $MefContent[1..($MefContent.Count-1)]
            }
        }
    }
    
    # Schreibe zusammengefasste MEF-Ergebnisse
    $CombinedFile = "out\mef_2024_modified_complete.csv"
    $AllMefResults | Out-File -FilePath $CombinedFile -Encoding UTF8
    Write-Host "📄 Kombinierte MEF-Ergebnisse geschrieben: $CombinedFile" -ForegroundColor Green
    
    # Zeige erste und letzte Zeilen
    if ($AllMefResults.Count -gt 1) {
        Write-Host "`nErste 3 Zeilen:" -ForegroundColor Gray
        $AllMefResults[0..2] | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        
        if ($AllMefResults.Count -gt 5) {
            Write-Host "Letzte 2 Zeilen:" -ForegroundColor Gray
            $AllMefResults[(-2)..-1] | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
        }
    }
}

Write-Host "`n🎯 BEREIT FÜR ANALYSE!" -ForegroundColor Magenta