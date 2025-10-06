#!/usr/bin/env python3
"""
Detaillierte Offender-Analyse für 72.86% Korrelations-Run
Analysiert die größten Preisabweichungen und deren Ursachen
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lade die Offender-Daten
df = pd.read_csv('out/FinalModel/Run_only_braunkohle_mustrun_jan/analysis/_corr_offenders.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

print("=== DETAILLIERTE OFFENDER-ANALYSE ===")
print(f"Analysiere Top-{len(df)} Offender aus 72.86% Korrelations-Run\n")

# === KATEGORIE 1: PRICE RANGE ANALYSIS ===
print("1. PRICE RANGE VERTEILUNG")
price_bins = [0, 50, 75, 100, 125, 200]
df['price_category'] = pd.cut(df['price_DE'], bins=price_bins, 
                             labels=['<50', '50-75', '75-100', '100-125', '>125'])
price_stats = df.groupby('price_category').agg({
    'abs_error': ['count', 'mean', 'std', 'min', 'max'],
    'price_DE': 'mean',
    'chosen_SRMC': 'mean'
}).round(2)
print(price_stats)
print()

# === KATEGORIE 2: FUEL TYPE ANALYSIS ===
print("2. MARGINAL FUEL VERTEILUNG")
fuel_stats = df.groupby('marginal_fuel').agg({
    'abs_error': ['count', 'mean', 'std'],
    'price_DE': 'mean',
    'chosen_SRMC': 'mean',
    'net_import_MW': 'mean'
}).round(2)
print(fuel_stats)
print()

# === KATEGORIE 3: TIME PATTERN ANALYSIS ===
print("3. ZEITLICHE MUSTER")
hourly_stats = df.groupby('hour').agg({
    'abs_error': ['count', 'mean'],
    'price_DE': 'mean',
    'net_import_MW': 'mean'
}).round(2)
print("Stunden-Verteilung (Top-10 Problemstunden):")
print(hourly_stats.sort_values(('abs_error', 'count'), ascending=False).head(10))
print()

# === KATEGORIE 4: IMPORT/EXPORT ANALYSIS ===
print("4. IMPORT/EXPORT ANALYSE")
df['import_category'] = pd.cut(df['net_import_MW'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
import_stats = df.groupby('import_category').agg({
    'abs_error': ['count', 'mean'],
    'price_DE': 'mean',
    'chosen_SRMC': 'mean'
}).round(2)
print(import_stats)
print()

# === SPEZIAL-ANALYSEN ===
print("5. EXTREME OFFENDER (>35 EUR/MWh Fehler)")
extreme = df[df['abs_error'] > 35]
print(f"Anzahl: {len(extreme)}")
print(f"Durchschnittlicher Fehler: {extreme['abs_error'].mean():.2f}")
print(f"Hauptsächliche Fuels: {extreme['marginal_fuel'].value_counts().head(3).to_dict()}")
print()

print("6. HYDRO RESERVOIR ANOMALIEN")
hydro = df[df['marginal_fuel'] == 'Hydro Water Reservoir']
if len(hydro) > 0:
    print(f"Anzahl Hydro-Stunden: {len(hydro)}")
    print(f"Durchschnittlicher Fehler: {hydro['abs_error'].mean():.2f}")
    print(f"Typische Preise: {hydro['price_DE'].mean():.2f} EUR/MWh")
    print(f"SRMC meist: {hydro['chosen_SRMC'].mean():.2f} EUR/MWh")
print()

print("7. PEAK-HOUR CHARAKTERISTIKA")
peak_hours = df[df['hour'].isin([7, 8, 18, 19, 20])]  # Typische Peak-Zeiten
print(f"Peak-Hour Offender: {len(peak_hours)}/{len(df)} ({len(peak_hours)/len(df)*100:.1f}%)")
print(f"Durchschnittlicher Peak-Fehler: {peak_hours['abs_error'].mean():.2f}")
print()

# === KORRELATIONS-ANALYSE ===
print("8. KORRELATIONS-MATRIX")
numeric_cols = ['price_DE', 'chosen_SRMC', 'abs_error', 'net_import_MW', 'hour']
corr_matrix = df[numeric_cols].corr().round(3)
print(corr_matrix['abs_error'].sort_values(ascending=False))
print()

# === TOP-10 WORST OFFENDER DETAIL ===
print("9. TOP-10 WORST OFFENDER DETAILS")
top_10 = df.head(10)[['timestamp', 'price_DE', 'chosen_SRMC', 'abs_error', 
                      'marginal_fuel', 'net_import_MW', 'cluster_zones']]
for idx, row in top_10.iterrows():
    print(f"{idx+1:2d}. {row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
          f"Price: {row['price_DE']:6.2f} | SRMC: {row['chosen_SRMC']:6.2f} | "
          f"Error: {row['abs_error']:5.2f} | {row['marginal_fuel']:12s} | "
          f"Import: {row['net_import_MW']:6.0f} MW")
print()

# === IMPROVEMENT POTENTIALS ===
print("=== VERBESSERUNGS-POTENTIALE ===")
print()
print("A) HIGH-PRICE HOURS (>100 EUR/MWh):")
high_price = df[df['price_DE'] > 100]
if len(high_price) > 0:
    potential_A = high_price['abs_error'].sum()
    print(f"   - {len(high_price)} Stunden mit Ø {high_price['abs_error'].mean():.1f} EUR Fehler")
    print(f"   - Gesamtpotential: {potential_A:.1f} EUR/MWh kumuliert")
    print(f"   - Hauptfuels: {high_price['marginal_fuel'].value_counts().head(3).to_dict()}")

print("\nB) IMPORT-HEAVY HOURS (>8000 MW):")
high_import = df[df['net_import_MW'] > 8000]
if len(high_import) > 0:
    potential_B = high_import['abs_error'].sum()
    print(f"   - {len(high_import)} Stunden mit Ø {high_import['abs_error'].mean():.1f} EUR Fehler")
    print(f"   - Gesamtpotential: {potential_B:.1f} EUR/MWh kumuliert")

print("\nC) PEAK TRANSITION HOURS (6-8h, 17-20h):")
transition = df[df['hour'].isin([6,7,8,17,18,19,20])]
if len(transition) > 0:
    potential_C = transition['abs_error'].sum()
    print(f"   - {len(transition)} Stunden mit Ø {transition['abs_error'].mean():.1f} EUR Fehler")
    print(f"   - Gesamtpotential: {potential_C:.1f} EUR/MWh kumuliert")

print(f"\nGESAMT VERBESSERUNGS-POTENTIAL: {df['abs_error'].sum():.1f} EUR/MWh")
print(f"Ø Fehler aktuell: {df['abs_error'].mean():.2f} EUR/MWh")
print("\n=== ENDE ANALYSE ===")