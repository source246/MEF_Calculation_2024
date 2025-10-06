import sys
sys.path.append('scripts/track_c')
import pandas as pd
import numpy as np

from modules.io_utils import read_csv_smart, parse_ts

# Load the file
df_raw = read_csv_smart('input/neighbors/gen_2024/actual_gen_DE_LU_2024.csv', min_cols=2)
print(f"Raw shape: {df_raw.shape}")
print(f"First few rows:\n{df_raw.head()}")

# Drop timestamp 
tcol = 'Unnamed: 0'
df_raw.index = parse_ts(df_raw[tcol])
df_raw = df_raw.drop(columns=[tcol])

print(f"\nData types:")
print(df_raw.dtypes)

print(f"\nFirst few values of Solar column:")
print(df_raw['Solar'].head(10))

print(f"\nChecking for numeric data:")
numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
print(f"Numeric columns: {list(numeric_cols)}")

print(f"\nTrying to convert Solar to numeric:")
solar_converted = pd.to_numeric(df_raw['Solar'], errors='coerce')
print(f"Solar after conversion: {solar_converted.head(10)}")
print(f"NaN count in Solar: {solar_converted.isna().sum()}")

print(f"\nLooking at rows with actual timestamp data:")
valid_idx = df_raw.index.notna()
print(f"Valid timestamps: {valid_idx.sum()}")
if valid_idx.any():
    valid_data = df_raw[valid_idx]
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Valid data sample:\n{valid_data.head(3)}")
    
    # Try numeric conversion on valid data
    solar_valid = pd.to_numeric(valid_data['Solar'], errors='coerce')
    print(f"Solar valid data: {solar_valid.head(5)}")