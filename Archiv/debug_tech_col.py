import sys
sys.path.append('scripts/track_c')
from modules.io_utils import load_neighbor_gen
import pandas as pd

def _is_tech_col(col_name):
    lc = str(col_name).lower()
    keywords = [
        "fossil",
        "nuclear", 
        "wind",
        "solar",
        "hydro",
        "biomass",
        "waste", 
        "geothermal",
        "other",
    ]
    return any(k in lc for k in keywords)

# Test raw loading first
from modules.io_utils import read_csv_smart
df_raw = read_csv_smart('input/neighbors/gen_2024/actual_gen_DE_LU_2024.csv', min_cols=2)

print("RAW COLUMNS:")
for i, col in enumerate(df_raw.columns):
    print(f"{i}: {repr(col)} -> _is_tech_col: {_is_tech_col(col)}")

# Filter columns
tech_cols = [c for c in df_raw.columns if _is_tech_col(c)]
print(f"\nTECH COLUMNS FOUND: {len(tech_cols)}")
for col in tech_cols:
    print(f"  - {repr(col)}")

# Drop timestamp column
tcol = next((c for c in [
    "timestamp_cec",
    "timestamp", 
    "time",
    "datetime",
    "MTU",
] if c in df_raw.columns), df_raw.columns[0])

print(f"\nTIMESTAMP COLUMN: {repr(tcol)}")

df_clean = df_raw.drop(columns=[tcol])
print(f"COLUMNS AFTER TIMESTAMP DROP: {len(df_clean.columns)}")

# Now apply tech filter
wide_cols = [c for c in df_clean.columns if _is_tech_col(c)]
print(f"WIDE COLUMNS: {len(wide_cols)} -> {wide_cols}")