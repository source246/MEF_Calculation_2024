#!/usr/bin/env python3
"""
Erstellt Stacked Area Charts für Lastdeckung - ähnlich dem Global Income Distribution Chart
Zeigt wie verschiedene Technologien zur Lastdeckung in Deutschland beitragen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection
import os

def create_load_coverage_chart(df_res, outdir, args=None):
    """
    Erstellt ein Stacked Area Chart für Lastdeckung nach Technologien
    """
    print("🔥 Erstelle Lastdeckungs-Visualisierung (Stacked Area)...")
    
    # Definiere verfügbare Technologie-Gruppen basierend auf den MEF-Daten
    tech_groups = {
        'Erneuerbare Energien': {
            'cols': ['FEE_MW'],  # Fixed EE
            'color': '#2E8B57',  # Sea Green
            'label': 'Renewable Energy (EE)'
        },
        'Biomasse': {
            'cols': ['BIO_MW'],
            'color': '#90EE90',  # Light Green
            'label': 'Biomass'
        },
        'Abfall': {
            'cols': ['WASTE_MW'],
            'color': '#FFD700',  # Gold
            'label': 'Waste'
        },
        'Fossil Mustrun': {
            'cols': ['DE_fossil_mustrun_required_MW'],
            'color': '#8B4513',  # Saddle Brown
            'label': 'Fossil Mustrun'
        },
        'Öl Mustrun': {
            'cols': ['OIL_MU_used_MW'],
            'color': '#FF4500',  # Orange Red
            'label': 'Oil Mustrun'
        },
        'Import': {
            'cols': ['net_import_total_MW'],
            'color': '#9370DB',  # Medium Purple
            'label': 'Net Import'
        },
        'Sonstige': {
            'cols': ['ND_EXTRA_MW'],
            'color': '#696969',  # Dim Gray
            'label': 'Other Non-Dispatchable'
        }
    }
    
    # Bereite Daten vor
    df_plot = df_res.copy()
    
    # Berechne Gesamtlast aus Residuallast + Erneuerbaren
    if 'residual_domestic_fossil_MW' in df_plot.columns and 'FEE_MW' in df_plot.columns:
        df_plot['total_load_estimated'] = df_plot['residual_domestic_fossil_MW'] + df_plot['FEE_MW'].fillna(0)
        load_col = 'total_load_estimated'
    elif 'residual_after_trade_MW' in df_plot.columns:
        # Verwende Residuallast nach Handel als Proxy
        df_plot['total_load_estimated'] = df_plot['residual_after_trade_MW'] + df_plot['FEE_MW'].fillna(0) + df_plot['net_import_total_MW'].fillna(0).clip(lower=0)
        load_col = 'total_load_estimated'
    else:
        # Fallback: verwende residual_domestic_fossil_MW
        load_col = 'residual_domestic_fossil_MW'
    
    # Sortiere nach geschätzter Gesamtlast für stufenweise Darstellung
    df_plot = df_plot.sort_values(load_col)
    
    # Erstelle kumulierte Werte für Stacking
    cumulative_data = {}
    cumulative_sum = pd.Series(0, index=df_plot.index)
    
    for tech_name, tech_info in tech_groups.items():
        # Summiere verfügbare Spalten für diese Technologie
        tech_sum = pd.Series(0, index=df_plot.index)
        for col in tech_info['cols']:
            if col in df_plot.columns:
                # Nur positive Werte für Erzeugung
                if col == 'net_import_total_MW':
                    tech_sum += df_plot[col].clip(lower=0)  # Nur positive Importe
                else:
                    tech_sum += df_plot[col].fillna(0).clip(lower=0)
        
        cumulative_data[tech_name] = {
            'bottom': cumulative_sum.copy(),
            'height': tech_sum,
            'color': tech_info['color'],
            'label': tech_info['label']
        }
        cumulative_sum += tech_sum
    
    # Erstelle die Visualisierung
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # X-Achse: Stunden sortiert nach Last
    hours = np.arange(len(df_plot))
    
    # Zeichne Stacked Areas
    for tech_name, data in cumulative_data.items():
        if data['height'].sum() > 0:  # Nur wenn Daten vorhanden
            ax.fill_between(
                hours,
                data['bottom'],
                data['bottom'] + data['height'],
                color=data['color'],
                alpha=0.8,
                label=data['label'],
                edgecolor='white',
                linewidth=0.1
            )
    
    # Zeichne geschätzte Gesamtlast als schwarze Linie oben
    total_load = df_plot[load_col].values
    ax.plot(hours, total_load, color='black', linewidth=2, 
            label='Estimated Total Load', alpha=0.9)
    
    # Styling ähnlich dem Income Chart
    ax.set_facecolor('#f8f9fa')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Achsenbeschriftung
    ax.set_xlabel('Hours (sorted by load)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Power Generation [MW]', fontsize=14, fontweight='bold')
    ax.set_title('German Power Generation Mix 2024\nLoad Coverage by Technology', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Formatiere Y-Achse
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f} GW'))
    
    # Formatiere X-Achse 
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}k'))
    
    # Legende
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              frameon=True, fancybox=True, shadow=True)
    
    # Zusätzliche Statistiken als Text
    stats_text = f"""
    Total Hours: {len(df_plot):,}
    Peak Load: {df_plot[load_col].max()/1000:.1f} GW
    Min Load: {df_plot[load_col].min()/1000:.1f} GW
    Avg Load: {df_plot[load_col].mean()/1000:.1f} GW
    """
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Speichere Plot
    plot_dir = os.path.join(outdir, 'analysis', 'plots_load_coverage')
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plot_dir, 'load_coverage_stacked_area.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(plot_dir, 'load_coverage_stacked_area.pdf'), 
                bbox_inches='tight', facecolor='white')
    
    print(f"✅ Load Coverage Chart gespeichert: {plot_dir}")
    plt.close()


def create_duration_curve_chart(df_res, outdir, args=None):
    """
    Erstellt eine Jahresdauerlinie mit Technologie-Stacking
    """
    print("📊 Erstelle Jahresdauerlinie mit Technologie-Mix...")
    
    # Vereinfachtes Technologie-Mapping basierend auf verfügbaren Spalten
    tech_mapping = {
        'Erneuerbare': ['FEE_MW'],
        'Biomasse': ['BIO_MW'],
        'Abfall': ['WASTE_MW'],
        'Fossil Mustrun': ['DE_fossil_mustrun_required_MW'],
        'Öl Mustrun': ['OIL_MU_used_MW'],
        'Import': ['net_import_total_MW'],
        'Sonstige': ['ND_EXTRA_MW']
    }
    
    colors = ['#2E8B57', '#90EE90', '#FFD700', '#8B4513', '#FF4500', '#9370DB', '#696969']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Berechne geschätzte Gesamtlast
    if 'residual_domestic_fossil_MW' in df_res.columns and 'FEE_MW' in df_res.columns:
        df_res['total_load_estimated'] = df_res['residual_domestic_fossil_MW'] + df_res['FEE_MW'].fillna(0)
        load_col = 'total_load_estimated'
    else:
        load_col = 'residual_domestic_fossil_MW'
    
    # Sortiert nach Last
    df_sorted = df_res.sort_values(load_col, ascending=False).reset_index(drop=True)
    
    # Erstelle kumulative Technologie-Beiträge
    bottom = np.zeros(len(df_sorted))
    
    for i, (tech_name, cols) in enumerate(tech_mapping.items()):
        tech_gen = pd.Series(0, index=df_sorted.index)
        for col in cols:
            if col in df_sorted.columns:
                if col == 'net_import_total_MW':
                    tech_gen += df_sorted[col].clip(lower=0)
                else:
                    tech_gen += df_sorted[col].fillna(0).clip(lower=0)
        
        if tech_gen.sum() > 0:
            ax.fill_between(range(len(df_sorted)), bottom, bottom + tech_gen,
                           color=colors[i % len(colors)], alpha=0.8, label=tech_name)
            bottom += tech_gen
    
    # Gesamtlast als schwarze Linie
    ax.plot(range(len(df_sorted)), df_sorted[load_col], 
            color='black', linewidth=2, label='Geschätzte Gesamtlast')
    
    ax.set_xlabel('Stunden im Jahr (sortiert nach Last)')
    ax.set_ylabel('Leistung [MW]')
    ax.set_title('Jahresdauerlinie Deutschland 2024 - Technologie-Mix')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_dir = os.path.join(outdir, 'analysis', 'plots_load_coverage')
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plot_dir, 'duration_curve_tech_mix.png'), 
                dpi=300, bbox_inches='tight')
    
    print(f"✅ Jahresdauerlinie gespeichert: {plot_dir}")
    plt.close()


def create_monthly_load_coverage(df_res, outdir, args=None):
    """
    Erstellt monatliche Load Coverage Charts
    """
    print("📅 Erstelle monatliche Load Coverage Charts...")
    
    df_res.index = pd.to_datetime(df_res.index)
    
    plot_dir = os.path.join(outdir, 'analysis', 'plots_load_coverage', 'monthly')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Berechne geschätzte Gesamtlast
    if 'residual_domestic_fossil_MW' in df_res.columns and 'FEE_MW' in df_res.columns:
        df_res['total_load_estimated'] = df_res['residual_domestic_fossil_MW'] + df_res['FEE_MW'].fillna(0)
        load_col = 'total_load_estimated'
    else:
        load_col = 'residual_domestic_fossil_MW'
    
    for month in range(1, 13):
        df_month = df_res[df_res.index.month == month].copy()
        if len(df_month) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Vereinfachtes Stacking für monatliche Ansicht
        tech_cols = {
            'EE': ['FEE_MW'],
            'Biomasse': ['BIO_MW'],
            'Fossil MR': ['DE_fossil_mustrun_required_MW'],
            'Öl MR': ['OIL_MU_used_MW'],
            'Import': ['net_import_total_MW'],
            'Sonstige': ['ND_EXTRA_MW']
        }
        
        colors = ['#2E8B57', '#90EE90', '#8B4513', '#FF4500', '#9370DB', '#696969']
        
        bottom = np.zeros(len(df_month))
        for i, (tech, cols) in enumerate(tech_cols.items()):
            tech_sum = pd.Series(0, index=df_month.index)
            for col in cols:
                if col in df_month.columns:
                    if col == 'net_import_total_MW':
                        tech_sum += df_month[col].clip(lower=0)
                    else:
                        tech_sum += df_month[col].fillna(0).clip(lower=0)
            
            if tech_sum.sum() > 0:
                ax.fill_between(range(len(df_month)), bottom, bottom + tech_sum,
                               color=colors[i], alpha=0.8, label=tech)
                bottom += tech_sum
        
        # Last als Linie
        ax.plot(range(len(df_month)), df_month[load_col], 
                color='black', linewidth=1.5, label='Geschätzte Last')
        
        ax.set_title(f'Load Coverage - {pd.to_datetime(f"2024-{month:02d}-01").strftime("%B")} 2024')
        ax.set_xlabel('Hours in Month')
        ax.set_ylabel('Power [MW]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'load_coverage_month_{month:02d}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Monatliche Load Coverage Charts gespeichert: {plot_dir}")


if __name__ == "__main__":
    # Test mit existierenden Daten
    import os
    os.chdir('C:/Users/schoenmeiery/Lastgangmanagement/MEF_Berechnung_2024')
    
    outdir = 'out/enhanced_validation_test'
    df_res = pd.read_csv(f'{outdir}/mef_track_c_2024.csv', index_col=0, parse_dates=True)
    
    print("🎨 Erstelle Load Coverage Visualisierungen...")
    
    # Hauptvisualisierung
    create_load_coverage_chart(df_res, outdir)
    
    # Jahresdauerlinie
    create_duration_curve_chart(df_res, outdir)
    
    # Monatliche Charts
    create_monthly_load_coverage(df_res, outdir)
    
    print("✅ Alle Load Coverage Visualisierungen erstellt!")