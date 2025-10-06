import pandas as pd
import numpy as np
import os

os.chdir('C:/Users/schoenmeiery/Lastgangmanagement/MEF_Berechnung_2024')

# Load data from recent run
outdir = 'out/enhanced_validation_test'
df_res = pd.read_csv(f'{outdir}/mef_track_c_2024.csv', index_col=0, parse_dates=True)

print('🔍 SRMC vs PREIS ABWEICHUNGSANALYSE')
print('=' * 50)

# Calculate deviations
df_res['price_srmc_diff'] = df_res['price_DE'] - df_res['marginal_srmc_eur_per_mwh']
df_res['price_srmc_diff_abs'] = np.abs(df_res['price_srmc_diff'])

# Basic stats
corr = df_res[['price_DE', 'marginal_srmc_eur_per_mwh']].corr().iloc[0,1]
mean_diff = df_res['price_srmc_diff'].mean()
std_diff = df_res['price_srmc_diff'].std()
mean_abs_diff = df_res['price_srmc_diff_abs'].mean()

print(f'📊 GRUNDSTATISTIKEN:')
print(f'Korrelation: {corr:.3f}')
print(f'Mean Abweichung: {mean_diff:.2f} EUR/MWh')
print(f'Std Abweichung: {std_diff:.2f} EUR/MWh')
print(f'Mean |Abweichung|: {mean_abs_diff:.2f} EUR/MWh')
print()

# Top 20 worst deviations
print('🚨 TOP 20 GRÖSSTE ABWEICHUNGEN:')
cols = ['price_DE', 'marginal_srmc_eur_per_mwh', 'price_srmc_diff', 'marginal_fuel', 'marginal_side']
worst = df_res.nlargest(20, 'price_srmc_diff_abs')[cols]

for i, (ts, row) in enumerate(worst.iterrows(), 1):
    price = row['price_DE']
    srmc = row['marginal_srmc_eur_per_mwh']
    diff = row['price_srmc_diff']
    fuel = row['marginal_fuel']
    side = row['marginal_side']
    time_str = ts.strftime('%Y-%m-%d %H:%M')
    print(f'{i:2d}. {time_str} | Preis: {price:7.2f} | SRMC: {srmc:7.2f} | Diff: {diff:8.2f} | {fuel} | {side}')

print()

# Analysis by fuel type
print('⛽ ABWEICHUNG NACH BRENNSTOFF:')
fuel_stats = df_res.groupby('marginal_fuel').agg({
    'price_srmc_diff': ['mean', 'std', 'count'],
    'price_srmc_diff_abs': 'mean'
}).round(2)
fuel_stats.columns = ['Mean_Diff', 'Std_Diff', 'Count', 'Mean_Abs_Diff']
print(fuel_stats.sort_values('Mean_Abs_Diff', ascending=False))
print()

# Analysis by marginal side
print('🌍 ABWEICHUNG NACH MARGINAL SIDE:')
side_stats = df_res.groupby('marginal_side').agg({
    'price_srmc_diff': ['mean', 'std', 'count'],
    'price_srmc_diff_abs': 'mean'
}).round(2)
side_stats.columns = ['Mean_Diff', 'Std_Diff', 'Count', 'Mean_Abs_Diff']
print(side_stats)
print()

# Monthly pattern
print('📅 MONATLICHE ABWEICHUNGEN:')
df_res_with_dt = df_res.copy()
df_res_with_dt.index = pd.to_datetime(df_res_with_dt.index)
df_res_with_dt['month'] = df_res_with_dt.index.month
monthly_stats = df_res_with_dt.groupby('month').agg({
    'price_srmc_diff': 'mean',
    'price_srmc_diff_abs': 'mean'
}).round(2)
monthly_stats.columns = ['Mean_Diff', 'Mean_Abs_Diff']
print(monthly_stats)
print()

# Price range analysis
print('💰 ABWEICHUNG NACH PREISBEREICH:')
df_res_with_dt['price_range'] = pd.cut(df_res_with_dt['price_DE'], 
                               bins=[-1000, 0, 50, 100, 150, 1000], 
                               labels=['Negativ', '0-50', '50-100', '100-150', '>150'])
price_range_stats = df_res_with_dt.groupby('price_range').agg({
    'price_srmc_diff': 'mean',
    'price_srmc_diff_abs': 'mean',
    'price_DE': 'count'
}).round(2)
price_range_stats.columns = ['Mean_Diff', 'Mean_Abs_Diff', 'Count']
print(price_range_stats)