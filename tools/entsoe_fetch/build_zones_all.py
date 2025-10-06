#!/usr/bin/env python3
# build_zones_all.py — erzeugt zones_all.csv mit allen ENTSO-E Bidding-Zonen (Y-EIC)
# Quelle: ENTSO-E EIC XML "allocated-eic-codes.xml" (Y-Codes, Funktion=Bidding Zone)

import sys, csv, xml.etree.ElementTree as ET
import urllib.request

URL = "https://eepublicdownloads.blob.core.windows.net/cio-lio/xml/allocated-eic-codes.xml"
out_csv = "zones_all.csv"

print("Lade EIC XML…")
with urllib.request.urlopen(URL, timeout=120) as r:
    xml_bytes = r.read()

root = ET.fromstring(xml_bytes)

rows = []
for rec in root.findall(".//{*}EicCode"):
    code = rec.findtext("{*}Code") or ""
    # Type 'Y' = Area/Domain
    typecode = rec.findtext("{*}Type") or ""
    func = rec.findtext("{*}Function") or ""
    name = rec.findtext("{*}LongLabel") or rec.findtext("{*}Name") or ""
    # Wir wollen nur Y-Codes mit Funktion = 'BiddingZone' (oder Varianten)
    if typecode.strip().upper() != "Y":
        continue
    if "bidding" not in func.lower():
        continue
    rows.append((code.strip(), name.strip()))

# Duplikate entfernen, sortieren
rows = sorted(set(rows), key=lambda x: x[0])

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["zone_code", "zone_name"])
    w.writerows(rows)

print(f"OK: {len(rows)} Bidding-Zonen → {out_csv}")
