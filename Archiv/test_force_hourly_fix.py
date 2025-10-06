import sys
sys.path.append('scripts/track_c')
import pandas as pd
import numpy as np

from modules.io_utils import read_csv_smart, parse_ts

def force_hourly_fixed(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """Fixed version that converts data to numeric first"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("force_hourly: kein DatetimeIndex")
    if df.index.freq == "H":
        return df
    
    # Convert all columns to numeric first, then filter
    df_converted = df.copy()
    for col in df.columns:
        df_converted[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Only aggregate numeric columns
    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns
    df_numeric = df_converted[numeric_cols]
    
    if how == "sum":
        return df_numeric.resample("h").sum()
    return df_numeric.resample("h").mean()

# Test the fix
df_raw = read_csv_smart('input/neighbors/gen_2024/actual_gen_DE_LU_2024.csv', min_cols=2)
tcol = 'Unnamed: 0'
df_raw.index = parse_ts(df_raw[tcol])
df_raw = df_raw.drop(columns=[tcol])

print("Testing force_hourly_fixed:")
result = force_hourly_fixed(df_raw, "mean")
print(f"Result shape: {result.shape}")
print(f"Result columns: {list(result.columns)}")
print(f"Sample data:\n{result.head(3)}")

# Check if the expected renewable columns are there
fee_cols = ["Solar","Wind Onshore","Wind Offshore","Hydro Run-of-river and poundage"]
for col in fee_cols:
    if col in result.columns:
        print(f"{col}: {result[col].iloc[0]:.2f} MW")
    else:
        print(f"{col}: NOT FOUND")