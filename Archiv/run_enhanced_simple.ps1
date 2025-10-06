# ENHANCED MEF 2024 - CRITICAL BUGS FIXED
param([string]$PythonExe = "python")

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptRoot

$PyScript = Join-Path $ScriptRoot "scripts\track_c\mef_dispatch_2024_Final_Version.py"
$OutDir = Join-Path $ScriptRoot "out\ENHANCED_CORRECTED_2024"

if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

Write-Host "=== ENHANCED MEF 2024 - CRITICAL FIXES APPLIED ===" -ForegroundColor Green

function Abs([string]$rel){ 
    if ([System.IO.Path]::IsPathRooted($rel)) { return $rel } 
    else { return (Resolve-Path -LiteralPath (Join-Path $ScriptRoot $rel)).Path } 
}

$MefArgs = @(
    "--fleet", (Abs ".\input\de\fleet\Kraftwerke_eff_binned.csv")
    "--eta_col", "Imputed_Effizienz_binned"
    "--vres", (Abs ".\input\de\generation\dena_erneuerbaren_einspeisegang.csv")
    "--varom", (Abs ".\input\de\fleet\Kraftwerke_VAROM_must_run.csv")
    "--load", (Abs ".\input\de\load\dena_verbrauch_2024.csv")
    "--year", "2024"
    "--neighbors", (Abs ".\input\neighbors\neighbors_2024.csv")
    "--nei_prices", (Abs ".\entsoe_full_2024\prices\prices_2024.csv")
    "--nei_eta_json", (Abs ".\input\neighbors\nei_efficiency_bounds.json")
    "--nei_eta_mode", "bounds"
    "--jao_fb_enable"
    "--fb_np_csv", (Abs ".\flows\Net_Positions_2024.csv")
    "--jao_fb_tolerance", "100.0"
    "--price_anchor", "threshold"
    "--price_tol", "5.0"
    "--epsilon", "0.01"
    "--enable_reservoir_hydro"
    "--reservoir_budget_mwh", "75000"
    "--reservoir_max_water_value", "150.0"
    "--enable_enhanced_psp"
    "--psp_dynamic_pricing"
    "--psp_price_cap", "180.0"
    "--psp_accept_band", "5.0"
    "--ee_price_threshold", "0.0"
    "--enable_enhanced_mustrun"
    "--lig_mustrun_hours", "3500"
    "--nuc_mustrun_hours", "8000"
    "--enable_selective_negative_handling"
    "--neg_price_mustrun_threshold", "-20.0"
    "--neg_price_vres_curtail", "0.15"
    "--enable_memory_optimization"
    "--chunk_size_hours", "168"
    "--output_dir", $OutDir
    "--enable_enhanced_validation"
    "--save_detailed_debug"
)

Write-Host "Running Enhanced MEF with CRITICAL FIXES..." -ForegroundColor Cyan

$start_time = Get-Date
try {
    $process = Start-Process -FilePath $PythonExe -ArgumentList @($PyScript) + $MefArgs -Wait -PassThru -NoNewWindow
    $exit_code = $process.ExitCode
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$end_time = Get-Date
$duration = $end_time - $start_time

if ($exit_code -eq 0) {
    Write-Host ""
    Write-Host "=== SUCCESS ===" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString())" -ForegroundColor Green
    Write-Host "Output: $OutDir" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "=== FAILED ===" -ForegroundColor Red
    Write-Host "Exit Code: $exit_code" -ForegroundColor Red
    exit $exit_code
}