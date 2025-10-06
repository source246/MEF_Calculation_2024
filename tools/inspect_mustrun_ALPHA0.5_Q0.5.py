import pandas as pd, numpy as np, os,sys
p='out/trackC_ALPHA0.5_Q0.5/mef_track_c_2024.csv'
if not os.path.exists(p):
    print('MISSING_FILE',p); sys.exit(1)

df=pd.read_csv(p)
print('FILE',p,'shape=',df.shape)
if 'timestamp' in df.columns:
    ts=pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    mask=ts.notna()
    df=df.loc[mask].copy()
    df.index = ts[mask].tz_convert('Europe/Berlin')
else:
    raise SystemExit('no timestamp column')

candidate_cols = [c for c in df.columns if any(k in c.lower() for k in ['de_fossil','mustrun','lowprice_profile','oil','lignite','rl_after_mu','rl_after_fossil'])]
explicit = ['DE_fossil_mustrun_cost_based_MW','DE_fossil_mustrun_required_MW','DE_oil_mustrun_required_MW','OIL_MU_used_MW','OIL_lowprice_profile_MW','LIGNITE_lowprice_profile_MW','RL_after_FOSSIL_MU_MW']
for e in explicit:
    if e in df.columns and e not in candidate_cols:
        candidate_cols.append(e)
candidate_cols = sorted(set(candidate_cols))
print('\nDetected MU-related columns:', candidate_cols)

if not candidate_cols:
    print('No MU-related columns found for monthly summary')
    sys.exit(0)

monthly = df[candidate_cols].resample('M').mean()
print('\nMonthly means (MW) for candidate MU columns:')
with pd.option_context('display.float_format','{:.3f}'.format):
    print(monthly.to_string())

print('\nAnnual summary:')
for c in candidate_cols:
    arr = df[c].fillna(0.0).astype(float)
    print(f"{c}: annual mean={arr.mean():.3f} MW, annual sum={arr.sum():.1f} MWh, hours>0={int((arr>0).sum())}")

if 'mef_g_per_kwh' in df.columns:
    mef=df['mef_g_per_kwh'].dropna()
    print('\nMEF summary: count',len(mef),' mean={:.3f} median={:.3f} min={:.3f} max={:.3f}'.format(mef.mean(),mef.median(),mef.min(),mef.max()))

if 'marginal_srmc_eur_per_mwh' in df.columns:
    s=df['marginal_srmc_eur_per_mwh'].dropna()
    print('\nSRMC summary: count',len(s),' mean={:.2f} median={:.2f} min={:.2f} max={:.2f}'.format(s.mean(),s.median(),s.min(),s.max()))

if 'marginal_side' in df.columns:
    vc=df['marginal_side'].fillna('NA').value_counts()
    print('\nMarginal side distribution:')
    for k,v in vc.items(): print(f'  {k}: {v} ({v/len(df)*100:.1f}%)')

outdir='out/trackC_ALPHA0.5_Q0.5/analysis'
import pathlib
pathlib.Path(outdir).mkdir(parents=True,exist_ok=True)
monthly.to_csv(pathlib.Path(outdir)/'_monthly_mustrun_summary.csv')
print('\nSaved monthly summary to', pathlib.Path(outdir)/'_monthly_mustrun_summary.csv')
