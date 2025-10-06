#!/usr/bin/env python3
"""
Create hourly FB NetPosition CSV by merging:
- Daily maxNetPos (minNP/maxNP) from fb_core_DE_2024.csv
- Hourly NetPosition from scheduled flows (flows_scheduled_DE_LU_2024_net.csv)

Maps daily minNP/maxNP to each hour and calculates:
- slack_to_min = NetPosition - minNP
- slack_to_max = maxNP - NetPosition  
- fb_boundary = (abs(slack_to_min) <= tol) or (abs(slack_to_max) <= tol)
where tol = max(100 MW, 0.02*(maxNP-minNP))
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_numeric_column(series):
    """Clean numeric columns that may have German decimal format (comma) or quotes"""
    if series.dtype == 'object':
        # Remove quotes and replace comma with dot
        cleaned = series.astype(str).str.replace('"', '').str.replace(',', '.')
        return pd.to_numeric(cleaned, errors='coerce')
    return series

def load_daily_fb_data(fb_path):
    """Load and clean daily FB data"""
    print(f"Loading daily FB data from {fb_path}")
    df = pd.read_csv(fb_path)
    
    # Clean timestamp column
    df['timestamp_utc'] = df['timestamp_utc'].astype(str).str.replace('"', '')
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
    
    # Clean numeric columns
    for col in ['minNP', 'maxNP', 'NetPosition', 'slack_to_min', 'slack_to_max']:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])
    
    # Filter valid rows with non-null timestamp and minNP/maxNP
    df = df.dropna(subset=['timestamp_utc', 'minNP', 'maxNP'])
    
    # Extract date for merging
    df['date'] = df['timestamp_utc'].dt.date
    
    print(f"Loaded {len(df)} daily FB records")
    return df[['date', 'minNP', 'maxNP']].drop_duplicates()

def load_hourly_flows(flows_path):
    """Load hourly scheduled flows"""
    print(f"Loading hourly flows from {flows_path}")
    df = pd.read_csv(flows_path)
    
    # Parse time column
    df['time'] = pd.to_datetime(df['time'])
    
    # Use net_import_total as NetPosition (negative = net export, positive = net import)
    # JAO convention: positive = import into DE
    df['NetPosition'] = df['net_import_total'] * -1  # Flip sign to match JAO convention
    
    # Extract date for merging
    df['date'] = df['time'].dt.date
    
    print(f"Loaded {len(df)} hourly flow records")
    return df[['time', 'date', 'NetPosition']]

def merge_and_calculate(daily_fb, hourly_flows, tolerance_mw=100):
    """Merge daily FB limits with hourly NetPosition and calculate slacks"""
    print("Merging daily FB limits with hourly flows...")
    
    # Merge on date
    merged = hourly_flows.merge(daily_fb, on='date', how='inner')
    
    print(f"Merged {len(merged)} hourly records with FB limits")
    
    # Calculate slacks
    merged['slack_to_min'] = merged['NetPosition'] - merged['minNP']
    merged['slack_to_max'] = merged['maxNP'] - merged['NetPosition']
    
    # Calculate adaptive tolerance: max(100 MW, 2% of range)
    range_2pct = 0.02 * (merged['maxNP'] - merged['minNP'])
    tolerance = np.maximum(tolerance_mw, range_2pct)
    
    # FB boundary detection
    near_min = np.abs(merged['slack_to_min']) <= tolerance
    near_max = np.abs(merged['slack_to_max']) <= tolerance
    merged['fb_boundary'] = near_min | near_max
    
    # Rename time to timestamp_utc for consistency
    merged = merged.rename(columns={'time': 'timestamp_utc'})
    
    # Select final columns
    result = merged[['timestamp_utc', 'minNP', 'maxNP', 'NetPosition', 
                    'slack_to_min', 'slack_to_max', 'fb_boundary']].copy()
    
    # Sort by time
    result = result.sort_values('timestamp_utc').reset_index(drop=True)
    
    return result

def main():
    # Paths
    base_dir = Path("c:/Users/schoenmeiery/Lastgangmanagement/MEF_Berechnung_2024")
    fb_daily_path = base_dir / "input/fb/fb_core_DE_2024.csv"
    flows_path = base_dir / "flows/flows_scheduled_DE_LU_2024_net.csv"
    output_path = base_dir / "input/fb/fb_core_DE_2024_hourly.csv"
    
    try:
        # Load data
        daily_fb = load_daily_fb_data(fb_daily_path)
        hourly_flows = load_hourly_flows(flows_path)
        
        # Merge and calculate
        result = merge_and_calculate(daily_fb, hourly_flows)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        
        # Report statistics
        print(f"\n=== Created {output_path} ===")
        print(f"Records: {len(result):,}")
        print(f"Time range: {result['timestamp_utc'].min()} → {result['timestamp_utc'].max()}")
        print(f"FB boundary hours: {result['fb_boundary'].sum():,} ({result['fb_boundary'].mean()*100:.1f}%)")
        print(f"NetPosition NaN: {result['NetPosition'].isna().sum()}")
        print(f"minNP ≤ maxNP violations: {(result['minNP'] > result['maxNP']).sum()}")
        
        # Sample rows
        print(f"\nFirst 5 rows:")
        print(result.head().to_string(index=False))
        
        print(f"\nLast 5 rows:")
        print(result.tail().to_string(index=False))
        
        print(f"\nSample FB boundary hours:")
        boundary_samples = result[result['fb_boundary']].head(3)
        if len(boundary_samples) > 0:
            print(boundary_samples[['timestamp_utc', 'NetPosition', 'minNP', 'maxNP', 'slack_to_min', 'slack_to_max']].to_string(index=False))
        else:
            print("No FB boundary hours found")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()