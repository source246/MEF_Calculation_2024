#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for downloading JAO FlowBased boundary data and integrating it into MEF analysis

This script demonstrates:
1. Downloading JAO Core region data for DE hub
2. Running MEF dispatch with FB boundary integration
3. Analyzing the impact of boundary constraints on marginal pricing

Usage:
  python example_jao_fb_integration.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"\n[INFO] {description}")
    print(f"[CMD] {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[OK] {description} completed successfully")
        if result.stdout.strip():
            print(f"[OUTPUT] {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with exit code {e.returncode}")
        if e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False

def main():
    """Main execution function"""
    
    # Configuration
    region = "core"
    hub = "DE"
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    
    # JAO API endpoints (these are examples - get actual URLs from JAO API test tab)
    maxnp_endpoint = "https://publicationtool.jao.eu/core/api/data/maxNetPositions"
    netpos_endpoint = "https://publicationtool.jao.eu/core/api/data/netPosition"
    
    # File paths
    project_root = Path(__file__).parent.parent
    jao_downloader = project_root / "tools" / "jao_fetch" / "jao_fb_downloader.py"
    mef_script = project_root / "scripts" / "track_c" / "mef_dispatch_2024_Final_Version.py"
    fb_output = project_root / "inputs" / "fb_core_DE_2024.csv"
    mef_output = project_root / "out" / "mef_with_fb_boundaries_2024"
    
    # Ensure directories exist
    fb_output.parent.mkdir(exist_ok=True)
    mef_output.mkdir(exist_ok=True)
    
    print("=== JAO FlowBased Boundary Integration Example ===")
    print(f"Region: {region}")
    print(f"Hub: {hub}")
    print(f"Period: {start_date} to {end_date}")
    print(f"FB output: {fb_output}")
    print(f"MEF output: {mef_output}")
    
    # Step 1: Download JAO FlowBased data
    print("\n" + "="*60)
    print("STEP 1: Download JAO FlowBased Boundary Data")
    print("="*60)
    
    fb_cmd = [
        sys.executable, str(jao_downloader),
        "--region", region,
        "--hub", hub,
        "--start", start_date,
        "--end", end_date,
        "--maxnp_endpoint", maxnp_endpoint,
        "--netpos_endpoint", netpos_endpoint,
        "--out", str(fb_output),
        "--verbose"
    ]
    
    if not run_command(fb_cmd, "JAO FlowBased data download"):
        print("[ERROR] Cannot proceed without FB data")
        return 1
    
    # Step 2: Run MEF dispatch with FB integration
    print("\n" + "="*60)
    print("STEP 2: Run MEF Dispatch with FlowBased Integration")
    print("="*60)
    
    mef_cmd = [
        sys.executable, str(mef_script),
        "--fleet", "input/de/fleet/Kraftwerke_eff_binned.csv",
        "--eta_col", "Imputed_Effizienz_binned",
        "--fuel_prices", "input/de/fuels/prices_2024.csv",
        "--flows", "flows/flows_scheduled_DE_LU_2024_net.csv",
        "--neighbor_gen_dir", "input/neighbors/gen_2024",
        "--neighbor_load_dir", "input/neighbors/out_load/2024",
        "--neighbor_prices", "input/neighbors/prices/neighbor_prices_2024.csv",
        "--fb_np_csv", str(fb_output),  # JAO FB integration
        "--year", "2024",
        "--start", "2024-01-01 00:00",
        "--end", "2024-01-08 00:00",  # Test with 1 week first
        "--outdir", str(mef_output),
        "--price_anchor", "threshold",
        "--price_tol", "1.0",
        "--epsilon", "0.01",
        "--corr_drop_neg_prices",
        "--corr_cap_mode", "peaker_min",
        "--corr_cap_tol", "2.5",
        "--psp_srmc_floor_eur_mwh", "60",
        "--ee_price_threshold", "0.0",
        "--nei_eta_mode", "bounds",
        "--nei_eta_json", "input/neighbors/neighbors_eta_bounds_from_jrc_2024.json",
        "--varom_json", "input/neighbors/neighbors_varom_from_maf2020.json",
        "--mu_cost_mode", "q_vs_cost",
        "--mu_cost_q", "0.20",
        "--mu_cost_alpha", "0.70",
        "--mu_cost_monthly",
        "--mu_cost_use_peak",
        "--fossil_mustrun_mode", "off",
        "--mustrun_lignite_q", "0",
        "--mustrun_mode", "capacity",
        "--mustrun_monthly",
        "--mustrun_peak_hours", "00-24",
        "--mustrun_quantile", "0.08",
        "--mustrun_neg_pricing_enable",
        "--mustrun_neg_share", "0.8",
        "--mustrun_bid_eur_mwh", "-10"
    ]
    
    if not run_command(mef_cmd, "MEF dispatch with FB boundaries"):
        print("[ERROR] MEF dispatch failed")
        return 1
    
    # Step 3: Quick analysis
    print("\n" + "="*60)
    print("STEP 3: Quick Analysis")
    print("="*60)
    
    try:
        import pandas as pd
        
        # Load results
        results_file = mef_output / "mef_track_c_2024.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            print(f"[INFO] Loaded {len(df)} result rows")
            
            # FB boundary analysis
            if 'fb_boundary' in df.columns:
                boundary_hours = df['fb_boundary'].sum()
                total_hours = len(df)
                boundary_pct = 100.0 * boundary_hours / total_hours
                
                print(f"\n[ANALYSIS] JAO FlowBased Boundary Impact:")
                print(f"  Total hours: {total_hours}")
                print(f"  Boundary hours: {boundary_hours} ({boundary_pct:.1f}%)")
                
                if boundary_hours > 0:
                    # Compare MEF at boundary vs non-boundary hours
                    df_boundary = df[df['fb_boundary'] == True]
                    df_normal = df[df['fb_boundary'] == False]
                    
                    avg_mef_boundary = df_boundary['mef_g_per_kwh'].mean()
                    avg_mef_normal = df_normal['mef_g_per_kwh'].mean()
                    
                    print(f"  Average MEF at boundary: {avg_mef_boundary:.1f} g/kWh")
                    print(f"  Average MEF normal: {avg_mef_normal:.1f} g/kWh")
                    print(f"  MEF difference: {avg_mef_boundary - avg_mef_normal:.1f} g/kWh")
                    
                    # Marginal technology at boundary
                    boundary_marginals = df_boundary['marginal_label'].value_counts()
                    print(f"\n  Marginal technologies at boundary:")
                    for tech, count in boundary_marginals.head(5).items():
                        pct = 100.0 * count / len(df_boundary)
                        print(f"    {tech}: {count} hours ({pct:.1f}%)")
            else:
                print("[WARN] No fb_boundary column found - FB integration may have failed")
                
        else:
            print(f"[WARN] Results file not found: {results_file}")
            
    except ImportError:
        print("[WARN] pandas not available for analysis")
    except Exception as e:
        print(f"[WARN] Analysis failed: {e}")
    
    print("\n" + "="*60)
    print("JAO FlowBased Integration Example Complete!")
    print("="*60)
    print(f"Results saved to: {mef_output}")
    print(f"FB data saved to: {fb_output}")
    print("\nNext steps:")
    print("- Review the fb_boundary flag in the results CSV")
    print("- Analyze correlation between boundary hours and marginal technologies")
    print("- Consider boundary context when interpreting price deviations")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())