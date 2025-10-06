import sys
sys.path.append('scripts/track_c')
import pandas as pd

from modules.io_utils import read_csv_smart, parse_ts, force_hourly
from pathlib import Path

def _zone_variants(zone: str):
    """Internal function to mimic _zone_variants"""
    yield zone

def _is_tech_col(col_name: str) -> bool:
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

def debug_load_neighbor_gen(path_dir: str, zone: str) -> pd.DataFrame:
    candidates = []
    for variant in _zone_variants(zone.strip()):
        candidates.extend(Path(path_dir).glob(f"actual_gen_{variant}_2024*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Gen-CSV fehlt: actual_gen_{zone}_2024*.csv in {path_dir}")

    print(f"Loading file: {candidates[0]}")
    df_raw = read_csv_smart(str(sorted(candidates)[0]), min_cols=2)
    print(f"Raw shape: {df_raw.shape}")
    print(f"Raw columns: {list(df_raw.columns)}")
    
    tcol = next((c for c in [
        "timestamp_cec",
        "timestamp",
        "time",
        "datetime",
        "MTU",
    ] if c in df_raw.columns), df_raw.columns[0])
    
    print(f"Using timestamp column: {tcol}")
    
    # Parse timestamps
    df_raw.index = parse_ts(df_raw[tcol])
    print(f"Index after parsing: {df_raw.index[:3]}")
    
    df_raw = df_raw.drop(columns=[tcol])
    print(f"Columns after dropping timestamp: {list(df_raw.columns)}")

    wide_cols = [c for c in df_raw.columns if _is_tech_col(c)]
    print(f"Wide columns found: {len(wide_cols)}")
    print(f"Wide columns: {wide_cols}")
    
    if len(wide_cols) >= 2:
        print("Using wide format")
        df_wide = df_raw.copy()
    else:
        print("Converting to wide format (shouldn't happen)")
        cols_lc = {c.lower(): c for c in df_raw.columns}
        tech_col = next((cols_lc[c] for c in cols_lc if any(k in c for k in ("productiontype", "type", "technology", "tech", "fuel"))), None)
        val_col = next((cols_lc[c] for c in cols_lc if any(k in c for k in ("actual", "generation", "gen", "mw", "value"))), None)
        if tech_col is None or val_col is None:
            raise ValueError(f"Unbekanntes Gen-Format in {candidates[0].name}: brauche Tech- und Wertspalte")
        df_wide = (
            df_raw
            .assign(**{val_col: pd.to_numeric(df_raw[val_col], errors="coerce")})
            .pivot_table(index=df_raw.index, columns=tech_col, values=val_col, aggfunc="sum")
        )

    print(f"df_wide shape before processing: {df_wide.shape}")
    print(f"df_wide columns: {list(df_wide.columns)}")
    
    # Apply aliases
    alias_map = {
        "Hydro PumpedStorage": "Hydro Pumped Storage",
        "Pumped Storage": "Hydro Pumped Storage",
        "Run-of-river": "Hydro Run-of-river and poundage",
        "Hydro Run-of-river": "Hydro Run-of-river and poundage",
        "Hard coal": "Fossil Hard coal",
        "Brown coal": "Fossil Brown coal/Lignite",
        "Lignite": "Fossil Brown coal/Lignite",
        "Oil": "Fossil Oil",
        "Biomasse": "Biomass",
        "Abfall": "Waste",
    }
    df_wide = df_wide.rename(columns=lambda c: alias_map.get(str(c), c))
    print(f"df_wide columns after aliases: {list(df_wide.columns)}")
    
    df_wide = df_wide.sort_index()
    print(f"df_wide shape after sort: {df_wide.shape}")
    
    df_wide = force_hourly(df_wide, "mean")
    print(f"Final shape: {df_wide.shape}")
    print(f"Final columns: {list(df_wide.columns)}")
    
    return df_wide

# Test the function
result = debug_load_neighbor_gen('input/neighbors/gen_2024', 'DE_LU')
print(f"\nFINAL RESULT:")
print(f"Shape: {result.shape}")
print(f"Columns: {list(result.columns)}")
print(result.head(2))