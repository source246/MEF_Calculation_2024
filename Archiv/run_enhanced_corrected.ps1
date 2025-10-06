# ENHANCED MEF 2024 - CRITICAL BUGS FIXED
# Fixes 6 kritische Systemfehler basierend auf wissenschaftlicher Analyse

param(
    [string]$PythonExe = "python"
)

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $ScriptRoot

$PyScript = Join-Path $ScriptRoot "scripts\track_c\mef_dispatch_2024_Final_Version.py"
$OutDir = Join-Path $ScriptRoot "out\ENHANCED_CORRECTED_2024"

if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

Write-Host "=== ENHANCED MEF 2024 - CRITICAL FIXES APPLIED ===" -ForegroundColor Green
Write-Host "FIX 1: EE-Threshold 0€ (statt -50€) für korrekte Low-Price-MEF" -ForegroundColor Yellow
Write-Host "FIX 2: Price-Tolerance 5€ (statt 1€) für robuste Korrelation" -ForegroundColor Yellow  
Write-Host "FIX 3: Reservoir-Budget MW-basiert (statt Stunden-zählen)" -ForegroundColor Yellow
Write-Host "FIX 4: PSP-Cap 180€ (statt 80€) für realistische Knappheitspreise" -ForegroundColor Yellow
Write-Host "FIX 5: JAO-FB aktiviert für unexplained correlation outliers" -ForegroundColor Yellow
Write-Host ""

function Abs([string]$rel){ 
    if ([System.IO.Path]::IsPathRooted($rel)) { return $rel } 
    else { return (Resolve-Path -LiteralPath (Join-Path $ScriptRoot $rel)).Path } 
}

# ENHANCED ARGUMENTS - ALLE BUGS FIXED
$MefArgs = @(
    # === CORE DATA ===
    "--fleet",              (Abs ".\input\de\fleet\Kraftwerke_eff_binned.csv"),
    "--eta_col",            "Imputed_Effizienz_binned",
    "--vres",               (Abs ".\input\de\generation\dena_erneuerbaren_einspeisegang.csv"),
    "--varom",              (Abs ".\input\de\fleet\Kraftwerke_VAROM_must_run.csv"),
    "--load",               (Abs ".\input\de\load\dena_verbrauch_2024.csv"),
    "--year",               2024,

    # === NEIGHBOR INTEGRATION ===
    "--neighbors",          (Abs ".\input\neighbors\neighbors_2024.csv"),
    "--nei_prices",         (Abs ".\entsoe_full_2024\prices\prices_2024.csv"),
    "--nei_eta_json",       (Abs ".\input\neighbors\nei_efficiency_bounds.json"),
    "--nei_eta_mode",       "bounds",

    # === FLOW-BASED BOUNDARIES (CRITICAL FIX) ===
    "--jao_fb_enable",      # FIX 4: Aktiviert JAO FlowBased boundaries
    "--fb_np_csv",          (Abs ".\flows\Net_Positions_2024.csv"),
    "--jao_fb_tolerance",   100.0,

    # === ENHANCED PRICE ANCHORING (CRITICAL FIX) ===
    "--price_anchor",       "threshold",
    "--price_tol",          5.0,        # FIX 2: Erhöht von 1.0 auf 5.0€ 
    "--epsilon",            0.01,

    # === ENHANCED RESERVOIR HYDRO ===
    "--enable_reservoir_hydro",
    "--reservoir_budget_mwh", 75000,    # Realistic monthly budget
    "--reservoir_max_water_value", 150.0,

    # === ENHANCED PSP LOGIC (CRITICAL FIX) ===
    "--enable_enhanced_psp",
    "--psp_dynamic_pricing",
    "--psp_price_cap",      180.0,      # FIX 5: Erhöht von 80€ auf 180€
    "--psp_accept_band",    5.0,

    # === ENHANCED EE-THRESHOLD (CRITICAL FIX) ===
    "--ee_price_threshold", 0.0,        # FIX 1: Korrekt 0€ statt -50€

    # === ENHANCED MUSTRUN LOGIC ===
    "--enable_enhanced_mustrun",
    "--lig_mustrun_hours",  3500,
    "--nuc_mustrun_hours",  8000,

    # === ENHANCED NEGATIVE PRICE HANDLING ===
    "--enable_selective_negative_handling",
    "--neg_price_mustrun_threshold", -20.0,
    "--neg_price_vres_curtail", 0.15,

    # === MEMORY OPTIMIZATION ===
    "--enable_memory_optimization",
    "--chunk_size_hours",   168,        # 1 week chunks

    # === OUTPUT ===
    "--output_dir",         $OutDir,
    "--enable_enhanced_validation",
    "--save_detailed_debug"
)

Write-Host "Running Enhanced MEF with CRITICAL FIXES..." -ForegroundColor Cyan
Write-Host "Command: $PythonExe $PyScript $($MefArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# EXECUTE
$start_time = Get-Date
try {
    & $PythonExe @($PyScript) @MefArgs
    $exit_code = $LASTEXITCODE
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

$end_time = Get-Date
$duration = $end_time - $start_time

if ($exit_code -eq 0) {
    Write-Host ""
    Write-Host "=== ENHANCED MEF CORRECTED - SUCCESS ===" -ForegroundColor Green
    Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    Write-Host "Output: $OutDir" -ForegroundColor Green
    Write-Host ""
    Write-Host "CRITICAL FIXES APPLIED:" -ForegroundColor Yellow
    Write-Host "✓ EE-Threshold: 0€ (Low-Price-Hours korrekt klassifiziert)" -ForegroundColor Green
    Write-Host "✓ Price-Tolerance: 5€ (Robuste Korrelation)" -ForegroundColor Green  
    Write-Host "✓ Reservoir-Budget: MW-basiert (Realistische Energie-Integration)" -ForegroundColor Green
    Write-Host "✓ PSP-Cap: 180€ (Realistische Knappheitspreise)" -ForegroundColor Green
    Write-Host "✓ JAO-FB: Aktiviert (Boundary-Detection für unexplained outliers)" -ForegroundColor Green
    Write-Host "✓ Validation: Enhanced für wissenschaftliche Validierung" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "=== ENHANCED MEF CORRECTED - FAILED ===" -ForegroundColor Red
    Write-Host "Duration: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Red
    Write-Host "Exit Code: $exit_code" -ForegroundColor Red
    exit $exit_code
}