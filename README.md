 Track C – MEF Dispatch‑Backcast (DE/LU 2024)

Dieses Repository rekonstruiert stündlich die **marginale Technologie**, deren **Grenzkosten (SRMC)** und den daraus abgeleiteten **marginalen Emissionsfaktor (MEF)** für die Bidding‑Zone **DE/LU** im Jahr **2024**. Es kombiniert eine einheitliche Residual‑Load‑Leiter, Must‑Run‑Logik (domestisch und Nachbarn), preisbasierte Kopplung/Clusterbildung und eine Validierungspipeline inkl. Plot‑Outputs.

> Kernskript: `mef_dispatch_2024_Final_Version.py` • Hilfsmodule: `modules/io_utils.py`, `modules/mustrun.py`, `modules/plots.py`, `modules/validation.py`.

---

## Funktionsumfang (Überblick)

1. **Datenimport & Zeitachsenharmonisierung** (UTC→Europe/Berlin, Schaltjahr 8784 h) für Last, Brennstoffpreise, Flüsse und Nachbarpreise/-erzeugung. Fehlende Stunden werden streng reindiziert und kurze Lücken ≤ 3 h interpoliert. fileciteturn28file1 fileciteturn28file4 fileciteturn28file5  
2. **Flow‑Modi**: `scheduled`, `physical` oder `hybrid` mit Ambivalenz‑Maske (Schwellenwert standardmäßig 0,40). Im Rahmen der Masterarbeit wird die Verwendung von scheduled benutzt. fileciteturn28file14  
3. **Zonale Preiskopplung**: Clusterbildung nach Preisähnlichkeit (ε) und Ankerlogik (`off|closest|threshold`) mit Toleranzband. fileciteturn28file10  
4. **Must‑Run‑Behandlung**: EE‑Abzug + stoffstrom‑spezifische Must‑Run‑Profile (Waste, Nuclear, Bio, Öl, fossile MU). Optional **kostenbasierte** MU‑Profile auf Basis Preis × Wirkungsgradverteilungen (NEI). fileciteturn28file12 fileciteturn28file7  
5. **Einheitliche Residual‑Load‑Leiter (RL0–RL8)**: konsistent pro Stunde angewandt; optionale Reihenfolgevariante „domestic EE nach Must‑Run“. fileciteturn28file10  
6. **Import/Export‑Stacks** (Nachbarn) und **Marginalerkennung**: Bestimmung von Seite (DE vs. IMPORT), Label, Brennstoff, η, SRMC und MEF. fileciteturn28file15  
7. **Flow‑Based‑Boundary (JAO) Integration**: optionaler Import von Net‑Position‑/Slack‑Informationen für Gate‑Flags und Diagnose. fileciteturn28file18 fileciteturn28file13  
8. **Outputs**: Haupt‑Zeitreihe `mef_track_c_2024.csv` plus Debug‑Zeitreihe `_debug_hourly.csv`. fileciteturn28file16  
9. **Validierung**: Korrelation Preis↔SRMC, Import‑Bias‑Gate (konfigurierbar), Negativpreis‑Analyse, Zeitachsen‑Gate (8784 h) und Transformations‑Checks. Reports als CSV/JSON. fileciteturn28file3 fileciteturn28file9 fileciteturn28file2  
10. **Visualisierung** im **BBH‑Stil** (Petrol/Dunkelblau/Umbra/Dunkelrot/Grau; Grid #5F5E5E): Validierungsplots, monatliche Load‑Coverage etc. fileciteturn28file17

---

## Verzeichnis & Module

- `mef_dispatch_2024_Final_Version.py` — Orchestriert Einlesen, RL‑Leiter, Must‑Run, Marginalerkennung, Schreiben der Outputs sowie Validierung/Plots. Enthält die zentrale RL‑Leiter `compute_residual_load_ladder(…)`. fileciteturn28file12 fileciteturn28file10  
- `modules/io_utils.py` — robuste CSV‑Reader, Zeitkonvertierungen (UTC→Europe/Berlin), Schema‑/Index‑Validierung (8784 h), Fluss‑Vorverarbeitung und Mapping von Brennstoffen & Anlagentypen. fileciteturn28file1 fileciteturn28file6 fileciteturn28file14  
- `modules/mustrun.py` — Default‑Verteilungen (NEI), Emissionsfaktoren, klassische **und** kostenbasierte Fossil‑Must‑Run‑Profile, preisinduzierte Lignite/Öl‑Profile. fileciteturn28file7  
- `modules/plots.py` — Validierungsplots, Residual‑/Marginal‑Shares und **Load‑Coverage** im BBH‑Stil (PNG/PDF). fileciteturn28file17  
- `modules/validation.py` — Validierungsgates, Korrelationstabellen, Import‑Bias‑Gate, Negativpreis‑Auswertung sowie Enhanced‑Reports (CSV/JSON). fileciteturn28file3 fileciteturn28file2

---

## Eingabedaten (Formate)

- **Fleet (DE)**: CSV mit Leistung, Brennstoff & (imputierten) Wirkungsgraden; automatische Spaltenerkennung (Leistung, Brennstoff, ID/Name). fileciteturn28file6  
- **Fuel Prices**: Zeitreihe (stündlich) mit numerischen Preis‑Spalten; Zeitspalte wird automatisch erkannt (`timestamp`/`time`/`datetime`/`MTU`). fileciteturn28file14  
- **Flows**: Stündliche Grenzkuppelflüsse; `imp_*`‑Spalten werden zu `net_import_total` aggregiert (falls nicht vorhanden). fileciteturn28file14  
- **Neighbor Prices**: Wide‑Format mit Spalten `price_<ZONE>`; wird auf den Vollindex 2024 reindiziert. fileciteturn28file4  
- **Neighbor Generation/Load**: Pro Zone stündliche Zeitreihen (Dict `zone→DataFrame`), die für MU‑Profile und Stacks genutzt werden. fileciteturn28file7  
- **(Optional) Flow‑Based/JAO**: CSV mit `timestamp_utc`, `fb_boundary`, `slack_to_min`, `slack_to_max`, `minNP`, `maxNP`, `NetPosition`. fileciteturn28file18

Alle Zeitachsen werden auf **Europe/Berlin** konvertiert; der Code erzwingt **8784 Stunden** (Leap‑Year) und bricht bei Abweichungen ab. fileciteturn28file1 fileciteturn28file4

---

## CLI – wichtigste Argumente

> Die Argumente werden in `build_parser()` definiert; hier sind die zentralen Gruppen (nur belegte Beispiele). fileciteturn28file8

**I/O & Zeitraum**  
- `--fleet`, `--fuel_prices`, `--flows`, `--neighbor_gen_dir`, `--neighbor_load_dir`, `--neighbor_prices`, `--outdir`, optional `--start`, `--end`. fileciteturn28file8 fileciteturn28file4

**Flows**  
- `--flow_mode {scheduled,physical,hybrid}`, optional `--flows_physical PATH`, `--flow_hybrid_ambiv_threshold FLOAT`. fileciteturn28file14

**Preis‑Kopplung/Matching**  
- `--epsilon FLOAT` (Preis‑Ähnlichkeit), `--price_anchor {off,closest,threshold}`, `--price_tol FLOAT`. fileciteturn28file10

**Must‑Run (Nachbarn & DE)**  
- klassische oder **kostenbasierte** Fossil‑MU: `--mu_cost_mode q_vs_cost`, `--mu_cost_q`, `--mu_cost_alpha`, optional monatliche/Peak‑Splits. fileciteturn28file7

**Validierung/Gates**  
- z. B. `--corr_drop_neg_prices`, `--corr_cap_mode`, `--corr_cap_tol` (siehe Validation‑Report‑Logik). fileciteturn28file3

> Hinweis: Einige Experten‑Flags werden nur genutzt, wenn die dazugehörigen Daten vorhanden sind (z. B. JAO‑Datei).

---

## Quickstart (Beispiel)

```bash
python mef_dispatch_2024_Final_Version.py \
  --fleet input/de/fleet/Kraftwerke_eff_binned.csv \
  --fuel_prices input/de/fuels/prices_2024.csv \
  --flows flows/flows_scheduled_DE_LU_2024_net.csv \
  --neighbor_gen_dir input/neighbors/gen_2024 \
  --neighbor_load_dir input/neighbors/out_load/2024 \
  --neighbor_prices input/neighbors/prices/neighbor_prices_2024.csv \
  --outdir out/prod \
  --flow_mode hybrid --flow_hybrid_ambiv_threshold 0.40 \
  --epsilon 0.05 --price_anchor threshold --price_tol 1.0
```

Die oben genannten Parameter/Dateien entsprechen den vom Code erwarteten Formaten und werden beim Einlesen strikt geprüft (Zeitachsen/Datentypen). fileciteturn28file14 fileciteturn28file1
