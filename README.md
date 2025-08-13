# Informationsintegration – Bibliojobs

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

Ein Projekt im Rahmen der Hausarbeit im 6. Fachsemester des Studiengangs **Informations- und Datenmanagement** an der **FH Potsdam**. Kursleitung: Prof. Dr. Günther Neher. Entwickelt von Holger Ehrmann.

> **Status:** Dieses Repository befindet sich in aktiver Entwicklung. Feedback und Beiträge sind willkommen.

## Funktionsumfang

- **CSV-Import & Normalisierung**
  - Lädt den Bibliojobs-Datensatz mit dem benutzerdefinierten `_§_`-Delimiter.
  - Normalisiert Spaltennamen, konvertiert Datentypen und unterstützt einen Fortschrittsrückruf.
- **Datenbereinigung**
  - Dekodiert HTML-Entitäten, entfernt Tags und extrahiert Postleitzahlen aus Firmennamen in eine eigene Spalte.
  - Ersetzt Kfz-Kennzeichen durch vollständige Ortsnamen (Wikidata API + Cache).
  - Standardisiert Firmennamen und bereinigt kryptische Werte.
  - Extrahiert Angaben zu Befristung, Arbeitszeit und Vergütung aus dem Attribut `jobdescription` in separate Spalten.
- **Data Profiling & Reporting**
  - Grafische Oberfläche (PyQt6) zur Anzeige von Profiling-Kennzahlen und zur Klassifikation von Datenfehlern nach Naumann/Leser.
  - Export von Bereinigungen und Fehlerberichten als Excel-Datei.
- **Dublettenerkennung**
  - Fuzzy-Matching über "company", "jobtype" und "jobdescription"; der "location"-Wert muss exakt übereinstimmen.
  - Effiziente Kandidatensuche mittels TF-IDF-Vektorisierung und Nearest-Neighbor-Suche.
  - Gefundene Dubletten werden entfernt und in einem eigenen Fenster angezeigt.

## Schnellstart

```bash
# Repository klonen
git clone https://github.com/McNamara84/information-integration.git
cd information-integration

# Virtuelle Umgebung anlegen
python -m venv .venv
source .venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Anwendung starten (optional Pfad zur CSV angeben)
python start.py path/zur/bibliojobs_raw.csv
```

## Tests

```bash
pytest
```

## Projektstruktur

```
├── cleaning.py        # Funktionen zur Bereinigung und Normalisierung
├── load_bibliojobs.py # CSV-Import mit individuellem Delimiter
├── profiling.py       # Data-Profiling und Fehlerklassifikation
├── start.py           # PyQt6-GUI für Profiling, Bereinigung und Export
└── tests/             # Unit- und Integrationstests
```

## Lizenz

Dieses Projekt dient ausschließlich zu Lehr- und Forschungszwecken innerhalb der FH Potsdam.
