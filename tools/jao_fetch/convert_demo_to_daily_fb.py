#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert demo hourly JAO CSV to daily FB gate CSV matching required schema.
Writes: input/fb/fb_core_DE_2024.csv
"""
import pandas as pd
from pathlib import Path

IN = Path('inputs/demo_fb_core_DE_2024.csv')
OUT_DIR = Path('input/fb')
OUT = OUT_DIR / 'fb_core_DE_2024.csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Reading demo file: {IN}")
df = pd.read_csv(IN)
# normalize timestamp column name
ts_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower() or 'stamp' in c.lower()), df.columns[0])
print(f"Using timestamp column: {ts_col}")
df['timestamp_utc'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
# ensure numeric
for col in ['minNP','maxNP','NetPosition']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# compute range and date
df = df.dropna(subset=['timestamp_utc']).copy()
df['range'] = (df['maxNP'] - df['minNP']).abs()
df['date'] = df['timestamp_utc'].dt.floor('D')

# for each date, pick the row with minimal range ("strengster/engster Bereich")
idx = df.groupby('date')['range'].idxmin()
sel = df.loc[idx].sort_values('date')

# build output
out = pd.DataFrame()
out['timestamp_utc'] = sel['date'].dt.strftime('%Y-%m-%d 00:00:00')
# ensure numeric columns exist
out['minNP'] = sel['minNP'].astype('float')
out['maxNP'] = sel['maxNP'].astype('float')
if 'NetPosition' in sel.columns:
    out['NetPosition'] = sel['NetPosition'].astype('float')
else:
    out['NetPosition'] = pd.NA

# compute slacks
out['slack_to_min'] = out['NetPosition'] - out['minNP']
out['slack_to_max'] = out['maxNP'] - out['NetPosition']

# fb_boundary with tolerance 100 MW: TRUE if abs(NetPosition - minNP) <=100 or abs(NetPosition - maxNP) <=100
import numpy as np
cond = (~out['NetPosition'].isna()) & (
    (out['NetPosition'].sub(out['minNP']).abs() <= 100.0) | (out['NetPosition'].sub(out['maxNP']).abs() <= 100.0)
)
out['fb_boundary'] = cond.fillna(False)

# final ordering and types
out = out[['timestamp_utc','minNP','maxNP','NetPosition','slack_to_min','slack_to_max','fb_boundary']]

# Validate min<=max
bad = (out['minNP'] > out['maxNP']).sum()
if bad:
    raise SystemExit(f"Validation failed: {bad} rows with minNP>maxNP")

# write CSV (UTF-8, comma)
out.to_csv(OUT, index=False, float_format='%.3f')
print(f"Wrote {OUT} with {len(out)} rows")

# Smoke checks
print('\n--- Smoke checks ---')
print('Head:')
print(out.head().to_string(index=False))
print('\nTail:')
print(out.tail().to_string(index=False))
print('\nminNP<=maxNP valid:', (out['minNP'] <= out['maxNP']).all())
print('fb_boundary TRUE count:', int(out['fb_boundary'].sum()))
