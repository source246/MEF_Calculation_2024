$python   = ".venv\Scripts\python.exe"
$runner   = "scripts\track_c\run_sensitivity_grid.py"
$baseline = "out/trackC_run_v1_final_2024_final"
$concur   = 3

$months = @(
    @{ tag = "JAN"; start = "2024-01-01"; end = "2024-02-01" },
    @{ tag = "MAR"; start = "2024-03-01"; end = "2024-04-01" },
    @{ tag = "JUL"; start = "2024-07-01"; end = "2024-08-01" }
)

function Run-Block($name, $extraArgs) {
    foreach ($m in $months) {
        $outdir = "out/sensitivity_{0}_{1}" -f $name, $m.tag
        Write-Host ">>> Block $name – Monat $($m.tag) → $outdir"
        & $python $runner `
            --baseline $baseline `
            --concurrency $concur `
            --outbase $outdir `
            --var ("start={0}" -f $m.start) `
            --var ("end={0}" -f $m.end) `
            @($extraArgs)
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Block $name ($($m.tag)) endete mit Exitcode $LASTEXITCODE"
        }
    }
}

Run-Block "coupling" @(
    "--var", "epsilon=0.01,0.10,1.0",
    "--var", "price_anchor=closest,threshold",
    "--var", "price_tol=10,20,40"
)

Run-Block "flows" @(
    "--var", "flow_mode=scheduled,hybrid"
)

Run-Block "fossil_mustrun" @(
    "--var", "fossil_mustrun_q=0.05,0.10,0.20",
    "--var", "mustrun_quantile=0.15,0.20,0.25"
)

Run-Block "mucost" @(
    "--var", "mu_cost_mode=q_vs_cost",
    "--var", "mu_cost_q=0.25,0.50",
    "--var", "mu_cost_alpha=0.70,0.80,0.85",
    "--var", "de_fossil_mustrun_from_cost=True"
)

Run-Block "corr" @(
    "--var", "corr_drop_neg_prices=True",
    "--var", "corr_cap_mode=peaker_min,absolute",
    "--var", "corr_cap_value=300,500",
    "--var", "corr_cap_tol=0.5,1.0"
)

Run-Block "peak" @(
    "--var", "peak_switch=True,False",
    "--var", "peak_price_thresholds=160,220;180,260"
)
