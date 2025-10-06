# Datei: run_matrix.py
import os, subprocess, csv, json, re
from pathlib import Path

PY = "python"
SCRIPT = r".\scripts\track_c\mef_dispatch_v5.4.py"  # anpassen, falls nötig

BASE = [
  "--fleet", r".\input\de\fleet\Kraftwerke_eff_binned.csv",
  "--eta_col", "Imputed_Effizienz_binned",
  "--fuel_prices", r".\input\de\fuels\prices_2024.csv",
  "--flows", r".\flows\flows_scheduled_DE_LU_2024_net.csv",
  "--neighbor_gen_dir", r".\input\neighbors\gen_2024",
  "--neighbor_load_dir", r".\input\neighbors\out_load\2024",
  "--neighbor_prices", r".\input\neighbors\prices\neighbor_prices_2024.csv",
  "--neighbor_fleet", r".\input\neighbors\neighbor_fleet_summary.csv",
]
BASE_FLAGS = [
  "--nei_eta_mode", "mean",
  "--nei_mc_draws", "50",
  "--price_anchor", "closest",
  "--price_tol", "5",
  "--epsilon", "0.01",
  "--fossil_mustrun_mode", "q_all",
  "--fossil_mustrun_q", "0.10",
  "--nei_nuclear_mustrun_share", "0.5",
  "--mustrun_neg_pricing_enable",
]

SCENARIOS = [
  ("base_eps001_paClosest_tol5_q010_nucMU50_negMU_on", []),
  ("eps2",                ["--epsilon","2"]),
  ("eps5",                ["--epsilon","5"]),
  ("paThreshold_tol5",    ["--price_anchor","threshold","--price_tol","5"]),
  ("paThreshold_tol20",   ["--price_anchor","threshold","--price_tol","20"]),
  ("eta_bounds",          ["--nei_eta_mode","bounds"]),
  ("eta_mc50",            ["--nei_eta_mode","mc","--nei_mc_draws","50"]),
  ("q005",                ["--fossil_mustrun_q","0.05"]),
  ("q020",                ["--fossil_mustrun_q","0.20"]),
  ("negMU_off",           []),  # wir lassen das Flag einfach weg
  ("nucMU0",              ["--nei_nuclear_mustrun_share","0.0"]),
  ("peak_on",             ["--peak_switch","--peak_price_thresholds","180,260"]),
  ("psp_floor_40",        ["--psp_srmc_floor_eur_mwh","40"]),
  ("psp_floor_80",        ["--psp_srmc_floor_eur_mwh","80"]),
]

def run_one(name, extra_flags, drop_flags=()):
    outdir = Path(f".\\out\\2024_MEF_{name}")
    outdir.mkdir(parents=True, exist_ok=True)
    flags = [f for f in BASE_FLAGS if f not in drop_flags]
    cmd = [PY, "-B", SCRIPT] + BASE + flags + extra_flags + ["--outdir", str(outdir)]
    (outdir/"_run_cmd.txt").write_text(" ".join(cmd), encoding="utf-8")
    print("RUN:", name)
    subprocess.run(cmd, check=True)
    return outdir

def parse_validation(outdir: Path):
    # 1) validation_summary
    summary_csv = outdir/"analysis"/"_validation_summary.csv"
    vals = {}
    if summary_csv.exists():
        import csv
        with summary_csv.open("r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            row = next(rd, {})
            for k,v in row.items():
                try: vals[k] = float(v)
                except: vals[k] = v

    # 2) filtered corr (aus _corr_offenders_summary.txt)
    off_sum = outdir/"analysis"/"_corr_offenders_summary.txt"
    if off_sum.exists():
        txt = off_sum.read_text(encoding="utf-8")
        m = re.search(r"Filtered corr \(Pearson\):\s*([-\d\.]+)", txt)
        n = re.search(r"N points after filter:\s*(\d+)", txt)
        if m: vals["corr_filtered"] = float(m.group(1))
        if n: vals["corr_filtered_n"] = int(n.group(1))

    # 3) einfache Kennzahlen aus MEF-Datei
    import pandas as pd
    mef_csv = outdir/"mef_track_c_2024.csv"
    if mef_csv.exists():
        df = pd.read_csv(mef_csv, parse_dates=[0], index_col=0)
        vals["mef_mean_gpkwh"] = float(df["mef_g_per_kwh"].mean())
        vals["share_IMPORT_%"] = 100.0*float((df["marginal_side"]=="IMPORT").mean())
    return vals

def main():
    results = []
    for name, extra in SCENARIOS:
        drop = ("--mustrun_neg_pricing_enable",) if name=="negMU_off" else ()
        outdir = run_one(name, extra, drop_flags=drop)
        vals = parse_validation(outdir)
        vals["scenario"] = name
        results.append(vals)
    # Matrix-Summary schreiben
    import pandas as pd
    pd.DataFrame(results).to_csv(".\\out\\matrix_summary.csv", index=False)
    print("→ geschrieben: .\\out\\matrix_summary.csv")

if __name__ == "__main__":
    main()
