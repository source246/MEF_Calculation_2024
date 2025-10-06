param(
  [string]$PythonExe = "python",
  [switch]$NoParallel
)

# ---------- Root & Pfade robust bestimmen ----------
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
Set-Location -Path $ScriptRoot

# Python-Skript absolut
$PyScript = Join-Path $ScriptRoot "scripts\track_c\mef_dispatch_v5.6.py"
if (!(Test-Path $PyScript)) { throw "Python-Skript nicht gefunden: $PyScript" }

# Out-Root absolut
$OutRoot = Join-Path $ScriptRoot "out\sweep"
if (!(Test-Path $OutRoot)) { New-Item -ItemType Directory -Path $OutRoot | Out-Null }

# ---------- Python prüfen ----------
try { $pyver = & $PythonExe -V 2>&1 } catch { throw "Python nicht gefunden unter '$PythonExe'." }
Write-Host "[OK] Python: $pyver"
Write-Host "[OK] Skript: $PyScript"
Write-Host "[OK] OutRoot: $OutRoot"

# ---------- Helper: absolut machen ----------
function Abs([string]$rel){ if ([System.IO.Path]::IsPathRooted($rel)) { return $rel } else { return (Resolve-Path -LiteralPath (Join-Path $ScriptRoot $rel)).Path } }

# ---------- BASE-Argumente (ALLE Pfade ABSOLUT) ----------
$BASE = @(
  "--fleet",              (Abs ".\input\de\fleet\Kraftwerke_eff_binned.csv"),
  "--eta_col",            "Imputed_Effizienz_binned",
  "--fuel_prices",        (Abs ".\input\de\fuels\prices_2024.csv"),
  "--flows",              (Abs ".\flows\flows_scheduled_DE_LU_2024_net.csv"),
  "--neighbor_gen_dir",   (Abs ".\input\neighbors\gen_2024"),
  "--neighbor_load_dir",  (Abs ".\input\neighbors\out_load\2024"),
  "--neighbor_prices",    (Abs ".\input\neighbors\prices\neighbor_prices_2024.csv"),
  "--nei_eta_mode", "mean",
  "--mu_cost_mode", "q_vs_cost",
  "--mu_cost_alpha", "0.75",
  "--mu_cost_q", "0.50",
  "--mu_cost_monthly",
  "--mu_cost_use_peak",
  "--de_fossil_mustrun_from_cost",
  "--corr_drop_neg_prices",
  "--corr_cap_mode", "peaker_min",
  "--corr_cap_tol", "1",
  "--psp_srmc_floor_eur_mwh", "60",
  "--price_anchor", "closest",
  "--price_tol", "2"
)

# ---------- Sweep-Design ----------
$EPS_LIST = @(2,5,8)
$DRAWS    = @(10,20,50)

# ---------- Parallelität ----------
$MaxParallel = if ($NoParallel) { 1 } else { [Math]::Max(1, [Environment]::ProcessorCount - 1) }
$jobs = @()

# ---------- Runner (kein $using:, absolute Pfade) ----------
function Start-MefRun {
  param(
    [string]$PyExe,
    [string]$PyFile,
    [string[]]$BaseArgs,
    [int]$Epsilon,
    [int]$Draws,
    [string]$OutDir,
    [int]$MaxParallel
  )
  if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
  $logDir  = Join-Path $OutDir "logs"
  if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
  $logFile = Join-Path $logDir "run.log"

  $args = $BaseArgs + @("--epsilon","$Epsilon","--nei_mc_draws","$Draws","--outdir",$OutDir)
  $cmd  = @($PyExe, "-B", $PyFile) + $args

  if ($MaxParallel -gt 1) {
    return Start-Job -Name ("eps{0}_d{1}" -f $Epsilon,$Draws) -ArgumentList @($cmd,$OutDir,$logFile) -ScriptBlock {
      param($cmd,$OutDir,$logFile)
      $sw = [System.Diagnostics.Stopwatch]::StartNew()
      try {
        if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }
        if (!(Test-Path (Split-Path $logFile -Parent))) { New-Item -ItemType Directory -Path (Split-Path $logFile -Parent) | Out-Null }

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName  = $cmd[0]
        $psi.Arguments = ($cmd[1..($cmd.Count-1)] -join " ")
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError  = $true
        $psi.UseShellExecute = $false

        $p = [System.Diagnostics.Process]::Start($psi)
        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()
        $p.WaitForExit()

        $stdout | Out-File -Encoding UTF8 -FilePath $logFile
        if ($stderr) { "`n--- STDERR ---`n$stderr" | Out-File -Append -Encoding UTF8 -FilePath $logFile }
        if ($p.ExitCode -ne 0) { "ExitCode: $($p.ExitCode)" | Out-File -Append -Encoding UTF8 -FilePath $logFile; throw "Run failed." }
      } finally {
        $sw.Stop()
        "Duration: {0:n1} s" -f ($sw.Elapsed.TotalSeconds) | Out-File -Append -Encoding UTF8 -FilePath $logFile
      }
    }
  }
  else {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $cmd[0] $cmd[1..($cmd.Count-1)] *>> $logFile
    $code = $LASTEXITCODE
    $sw.Stop()
    "Duration: {0:n1} s" -f ($sw.Elapsed.TotalSeconds) | Out-File -Append -Encoding UTF8 -FilePath $logFile
    if ($code -ne 0) { throw "Run failed with ExitCode $code. See $logFile" }
    return $null
  }
}

# ---------- Sweeps ----------
foreach($e in $EPS_LIST){
  foreach($d in $DRAWS){
    $out = Join-Path $OutRoot ("eps{0}_a075_q50_clo_tol2_mean_draws{1}" -f $e,$d)
    $job = Start-MefRun -PyExe $PythonExe -PyFile $PyScript `
                        -BaseArgs $BASE -Epsilon $e -Draws $d -OutDir $out `
                        -MaxParallel $MaxParallel
    if ($job) { $jobs += $job }
    if ($jobs.Count -ge $MaxParallel -and $MaxParallel -gt 1) {
      $done = Wait-Job -Any $jobs
      Receive-Job $done | Out-Null
      $jobs = $jobs | Where-Object { $_.Id -ne $done.Id }
    }
  }
}
if ($jobs.Count -gt 0) { Wait-Job $jobs | Receive-Job | Out-Null }

Write-Host "[DONE] Sweep abgeschlossen. Ergebnisse unter: $OutRoot"
