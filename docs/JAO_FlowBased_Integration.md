# JAO FlowBased Boundary Integration

## Überblick

Das JAO FlowBased Boundary System erweitert die MEF-Analyse um wichtige Informationen über die Position von Deutschland im FlowBased Market Coupling. Dies ermöglicht eine bessere Interpretation von Preisabweichungen und Marginal-Technology-Auswahl.

## Konzept

**Was sind FlowBased Boundaries?**
- JAO publiziert Min/Max Net Positions für jeden Hub (Bidding Zone)
- Diese sind die Union aus finaler FB-Domäne und finalen BEX
- Kein NTC-Ersatz, sondern exakte Referenz für FB Market Coupling
- Nicht simultan realisierbar - nur als Rand-Flag verwendbar

**Warum ist das wichtig?**
- Stunden am Domänenrand haben andere Preissetzungsmechanismen
- Preisabweichungen sind bei Randstunden "by design"
- Marginal Technologies können sich am Rand unterscheiden
- Besserer Kontext für Offender-Analyse

## Verwendung

### 1. JAO Daten herunterladen

```bash
python tools/jao_fetch/jao_fb_downloader.py \
  --region core \
  --hub DE \
  --start 2024-01-01 \
  --end 2025-01-01 \
  --maxnp_endpoint "https://publicationtool.jao.eu/core/api/data/maxNetPositions" \
  --netpos_endpoint "https://publicationtool.jao.eu/core/api/data/netPosition" \
  --out inputs/fb_core_DE_2024.csv
```

**Endpunkt-URLs finden:**
1. Öffne [JAO Publication Tool](https://publicationtool.jao.eu/)
2. Wähle Core/Nordic Region
3. Gehe zu API Tab
4. Nutze den Test-Tab für URLs (z.B. Min/Max Net Positions, Net Position)
5. Kopiere die exakte URL aus dem Test-Tab

### 2. MEF-Analyse mit FB Integration

```bash
python scripts/track_c/mef_dispatch_2024_Final_Version.py \
  --fleet input/de/fleet/Kraftwerke_eff_binned.csv \
  --eta_col Imputed_Effizienz_binned \
  --fuel_prices input/de/fuels/prices_2024.csv \
  --flows flows/flows_scheduled_DE_LU_2024_net.csv \
  --neighbor_gen_dir input/neighbors/gen_2024 \
  --neighbor_load_dir input/neighbors/out_load/2024 \
  --neighbor_prices input/neighbors/prices/neighbor_prices_2024.csv \
  --fb_np_csv inputs/fb_core_DE_2024.csv \
  --year 2024 \
  --start "2024-01-01 00:00" \
  --end "2025-01-01 00:00" \
  --outdir out/mef_with_fb_2024 \
  [weitere MEF Parameter...]
```

### 3. Complete Example

```bash
python example_jao_fb_integration.py
```

## Output-Felder

Das MEF-Ergebnis CSV enthält zusätzliche FlowBased-Spalten:

| Feld | Beschreibung |
|------|-------------|
| `fb_boundary` | Boolean: Ist Deutschland am FB-Domänenrand? |
| `fb_slack_to_min_MW` | Slack zur Min Net Position in MW |
| `fb_slack_to_max_MW` | Slack zur Max Net Position in MW |
| `fb_minNP_MW` | JAO Min Net Position in MW |
| `fb_maxNP_MW` | JAO Max Net Position in MW |
| `fb_NetPosition_MW` | JAO published Net Position in MW |

## Analyse-Ansätze

### 1. Boundary Impact auf MEF

```python
import pandas as pd

df = pd.read_csv("out/mef_with_fb_2024/mef_track_c_2024.csv")

# Boundary Hours Analysis
boundary_hours = df[df['fb_boundary'] == True]
normal_hours = df[df['fb_boundary'] == False]

print(f"Boundary hours: {len(boundary_hours)} ({100*len(boundary_hours)/len(df):.1f}%)")
print(f"Avg MEF at boundary: {boundary_hours['mef_g_per_kwh'].mean():.1f} g/kWh")
print(f"Avg MEF normal: {normal_hours['mef_g_per_kwh'].mean():.1f} g/kWh")
```

### 2. Marginal Technology Distribution

```python
# Marginal technologies at boundary vs normal
boundary_marginals = boundary_hours['marginal_label'].value_counts()
normal_marginals = normal_hours['marginal_label'].value_counts()

print("Marginal technologies at FB boundary:")
print(boundary_marginals.head())
```

### 3. Offender Context

```python
# Load offender analysis results
offenders = pd.read_csv("out/mef_with_fb_2024/corrected_and_offenders.csv")

# Check if price deviations correlate with boundary hours
offenders_boundary = offenders.merge(df[['timestamp', 'fb_boundary']], on='timestamp')
boundary_offenders = offenders_boundary[offenders_boundary['fb_boundary'] == True]

print(f"Offenders at boundary: {len(boundary_offenders)}/{len(offenders)} ({100*len(boundary_offenders)/len(offenders):.1f}%)")
```

## Technische Details

### API Parameter

JAO APIs verwenden verschiedene Parameter-Namen:
- `FromDateTime`, `ToDateTime` (ISO format mit Z suffix)
- `Format=CSV` für CSV Download
- Verschiedene Spaltennamen je View

### Zeitbehandlung

- JAO nutzt UTC timestamps
- MEF Dispatch nutzt Europe/Berlin
- Automatische Konvertierung im Script
- 30min Toleranz für Merge

### Boundary Flag Berechnung

```python
# Toleranz: max(100 MW, 2% der NP-Range)
rng = (maxNP - minNP).clip(lower=1.0)
tau = (0.02 * rng).clip(lower=100.0)

slack_min = (NetPosition - minNP).clip(lower=0.0)
slack_max = (maxNP - NetPosition).clip(lower=0.0)

fb_boundary = (slack_min <= tau) | (slack_max <= tau)
```

## Limitationen

1. **Nicht simultan realisierbar**: Min/Max NPs sind nicht gleichzeitig erreichbar
2. **Keine Flow-Ableitung**: Niemals für Fluss-Berechnung verwenden
3. **Rand-Flag nur**: Nur als binärer Indikator "am Domänenrand"
4. **Publikationsverzögerung**: JAO publiziert i.d.R. um 10:30 D-1
5. **API-Stabilität**: Endpunkt-URLs können sich ändern

## Troubleshooting

### Häufige Probleme

1. **"No data downloaded"**
   - Prüfe Endpunkt-URLs in JAO API Test Tab
   - Prüfe Zeitraum (JAO hat begrenzte Historie)
   - Prüfe Hub-Code (DE, FR, etc.)

2. **"Cannot find column"**
   - Spaltennamen variieren zwischen JAO Views
   - Script hat robuste Column-Detection
   - Prüfe CSV-Format manuell

3. **"No FB boundary integration"**
   - Zeitzone-Mismatch zwischen MEF und JAO
   - Prüfe UTC-Konvertierung
   - Prüfe Merge-Toleranz

### Debug Commands

```bash
# Test JAO download mit verbose
python tools/jao_fetch/jao_fb_downloader.py \
  --region core --hub DE \
  --start 2024-01-01 --end 2024-01-02 \
  --maxnp_endpoint "URL" --netpos_endpoint "URL" \
  --out test_fb.csv --verbose

# Check FB data format
head -20 test_fb.csv
```

## Best Practices

1. **Downloader vor MEF Run**: Immer aktuelle FB-Daten laden
2. **Kurze Tests**: Erst 1-7 Tage testen, dann ganzes Jahr
3. **Endpunkt-Validierung**: URLs regelmäßig prüfen
4. **Backup-Strategy**: FB-Daten archivieren (JAO löscht alte Daten)
5. **Kontext bei Interpretation**: FB-Flag bei Offender-Analyse beachten