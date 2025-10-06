import pandas as pd
from pathlib import Path
import sys

A = Path('out/trackC_run_v1_final_2024/mef_track_c_2024.csv')
B = Path('out/trackC_run_v1_final_2024_repro_FinalVersion/mef_track_c_2024.csv')
if not A.exists() or not B.exists():
    print(f'Missing files: A={A.exists()}, B={B.exists()}')
    sys.exit(2)

print('Reading CSVs (this may take a few seconds)...')
dfA = pd.read_csv(A, parse_dates=[0], dayfirst=False, infer_datetime_format=True)
dfB = pd.read_csv(B, parse_dates=[0], dayfirst=False, infer_datetime_format=True)

# align on first column name
ts_col = dfA.columns[0]
if ts_col != dfB.columns[0]:
    dfB.rename(columns={dfB.columns[0]: ts_col}, inplace=True)

# set index if possible
try:
    dfA.set_index(ts_col, inplace=True)
    dfB.set_index(ts_col, inplace=True)
except Exception:
    pass

common_cols = [c for c in dfA.columns if c in dfB.columns]
print(f'Rows: A={len(dfA)}, B={len(dfB)}. Common columns: {len(common_cols)}')

num_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(dfA[c])]
obj_cols = [c for c in common_cols if c not in num_cols]

# identical rows count
identical_rows = (dfA[common_cols].fillna('<<NA>>') == dfB[common_cols].fillna('<<NA>>')).all(axis=1).sum()
print('Rows identical across all common columns:', identical_rows)

summary = []
for c in num_cols:
    a = pd.to_numeric(dfA[c], errors='coerce').fillna(0).astype(float)
    b = pd.to_numeric(dfB[c], errors='coerce').fillna(0).astype(float)
    diff = (a - b).abs()
    n_diff = int((diff > 1e-9).sum())
    mean = float(diff.mean())
    median = float(diff.median())
    maxd = float(diff.max())
    summary.append((c, 'numeric', n_diff, mean, median, maxd))

for c in obj_cols:
    a = dfA[c].fillna('<<NA>>').astype(str)
    b = dfB[c].fillna('<<NA>>').astype(str)
    neq = (a != b)
    n_diff = int(neq.sum())
    samples = []
    if n_diff > 0:
        merged = pd.concat([a[neq].rename(c+'_A'), b[neq].rename(c+'_B')], axis=1)
        samples = merged.head(10).to_dict(orient='records')
    summary.append((c, 'object', n_diff, None, None, samples))

print('\nPer-column summary:')
for r in summary:
    if r[1] == 'numeric':
        print(f"{r[0]:60s} | numeric | n_diff={r[2]:6d} | mean={r[3]:.6g} | median={r[4]:.6g} | max={r[5]:.6g}")
    else:
        print(f"{r[0]:60s} | object  | n_diff={r[2]:6d} | samples={r[5]}")

# overall differing rows
diff_mask = (dfA[common_cols].fillna('<<NA>>') != dfB[common_cols].fillna('<<NA>>')).any(axis=1)
n_any = int(diff_mask.sum())
print(f'\nRows with any difference: {n_any}')

out_dir = Path('out/diffs')
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / 'column_diff_summary.csv', 'w', encoding='utf8') as f:
    f.write('column,type,n_diff,mean_abs_diff,median_abs_diff,max_or_samples\n')
    for r in summary:
        f.write('"%s",%s,%s,%s,%s,"%s"\n' % (r[0], r[1], r[2], r[3] if r[3] is not None else '', r[4] if r[4] is not None else '', str(r[5]).replace('\n',' ')))

if n_any > 0:
    sampleA = dfA[common_cols][diff_mask].head(200).reset_index()
    sampleB = dfB[common_cols][diff_mask].head(200).reset_index()
    merged = sampleA.add_suffix('_A').join(sampleB.add_suffix('_B'))
    merged.to_csv(out_dir / 'sample_diffs_A_vs_B.csv', index=False)
    print('Wrote sample diffs to', out_dir / 'sample_diffs_A_vs_B.csv')
else:
    print('No differing rows; nothing written')

print('Wrote column summary to', out_dir / 'column_diff_summary.csv')
print('Done')
