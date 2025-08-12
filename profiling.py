"""Data profiling utilities for the Bibliojobs dataset."""
from __future__ import annotations

import re
from typing import Any, List, Tuple

import pandas as pd

# Common placeholders that should be treated as data errors/missing markers.
ERROR_VALUES = ["", "??", "na", "n/a", "null", None]


def get_all_error_types(series: pd.Series, column_name: str) -> List[Tuple[str, float]]:
    """Get all error types for a series according to Naumann/Leser taxonomy.
    
    Returns a list of (error_type, error_rate) tuples for all detected error types.
    """
    total_count = len(series)
    if total_count == 0:
        return [("Keine Daten", 0.0)]
    
    error_types = []
    
    # Count different error types
    missing_count = series.isna().sum()
    empty_count = (series == "").sum()
    error_markers_count = sum((series == val).sum() for val in ERROR_VALUES if val not in [None, ""])
    
    # Fehlende Werte (including various representations)
    total_missing = missing_count + empty_count + error_markers_count
    if total_missing > 0:
        missing_rate = (total_missing / total_count) * 100
        error_types.append(("Fehlende Werte", missing_rate))
    
    # Duplikate (only check for columns that should be unique)
    unique_columns = ["jobid", "url"]
    if column_name.lower() in unique_columns:
        non_null_series = series.dropna()
        if len(non_null_series) > 0:
            duplicate_count = len(non_null_series) - non_null_series.nunique()
            if duplicate_count > 0:
                duplicate_rate = (duplicate_count / total_count) * 100
                error_types.append(("Eindeutigkeitsverletzungen", duplicate_rate))
    
    # Unzulässige Werte (for specific columns)
    if column_name.lower() in ["geo_lat", "geo_lon"]:
        numeric_series = pd.to_numeric(series, errors="coerce")
        invalid_count = 0
        if column_name.lower() == "geo_lat":
            invalid_count = ((numeric_series < -90) | (numeric_series > 90)).sum()
        else:  # geo_lon
            invalid_count = ((numeric_series < -180) | (numeric_series > 180)).sum()
        if invalid_count > 0:
            invalid_values_rate = (invalid_count / total_count) * 100
            error_types.append(("Unzulässige Werte", invalid_values_rate))
    elif column_name.lower() == "country":
        # Check for obviously invalid country values
        non_null = series.dropna()
        if len(non_null) > 0:
            invalid_count = sum(1 for val in non_null if len(str(val)) < 2 or len(str(val)) > 50)
            if invalid_count > 0:
                invalid_values_rate = (invalid_count / total_count) * 100
                error_types.append(("Unzulässige Werte", invalid_values_rate))
    
    # Kryptische Werte (very short codes without clear meaning)
    if column_name.lower() in ["location"]:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Count very short values that might be cryptic codes
            cryptic_count = sum(1 for val in non_null if len(str(val).strip()) <= 3 and str(val).strip().isalpha())
            if cryptic_count > 0:
                cryptic_rate = (cryptic_count / total_count) * 100
                error_types.append(("Kryptische Werte", cryptic_rate))
    
    # Eingebettete Werte (multiple information in one field)
    if column_name.lower() in ["company"]:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Look for addresses embedded in company names (contains numbers and commas)
            embedded_count = sum(1 for val in non_null if re.search(r'\d+.*,.*\d+', str(val)))
            if embedded_count > 0:
                embedded_rate = (embedded_count / total_count) * 100
                error_types.append(("Eingebettete Werte", embedded_rate))
    
    # Schreibfehler (suspicious patterns)
    if column_name.lower() in ["company", "location", "jobdescription"]:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Look for HTML entities, unusual characters, repeated characters
            error_patterns = [
                r'&#\d+;',  # HTML entities
                r'&[a-zA-Z]+;',  # HTML entities
                r'(.)\1{3,}',  # Repeated characters (4+ times)
            ]
            spelling_count = 0
            for val in non_null:
                val_str = str(val)
                if any(re.search(pattern, val_str) for pattern in error_patterns):
                    spelling_count += 1
            if spelling_count > 0:
                spelling_error_rate = (spelling_count / total_count) * 100
                error_types.append(("Schreibfehler", spelling_error_rate))
    
    # Widersprüchliche Werte (logical inconsistencies)
    if column_name.lower() == "date":
        non_null = series.dropna()
        if len(non_null) > 0:
            # Check for future dates (assuming data is from 2014-2025)
            from datetime import datetime
            future_count = 0
            for val in non_null:
                try:
                    if pd.isna(val):
                        continue
                    date_val = pd.to_datetime(val) if not isinstance(val, pd.Timestamp) else val
                    if date_val.year > 2025 or date_val.year < 2000:
                        future_count += 1
                except:
                    pass
            if future_count > 0:
                contradiction_rate = (future_count / total_count) * 100
                error_types.append(("Widersprüchliche Werte", contradiction_rate))
    
    # Falsche Zuordnungen (values in wrong columns)
    if column_name.lower() == "jobtype":
        non_null = series.dropna()
        if len(non_null) > 0:
            # Check if values look like they belong to other columns
            misplaced_count = 0
            for val in non_null:
                val_str = str(val).lower()
                # Check if value looks like a company name, location, or URL
                if any(indicator in val_str for indicator in ['gmbh', 'ag', 'bibliothek', 'universität', 'http', 'www']):
                    misplaced_count += 1
            if misplaced_count > 0:
                misplacement_rate = (misplaced_count / total_count) * 100
                error_types.append(("Falsche Zuordnungen", misplacement_rate))
    
    # Falsche Werte (obviously incorrect information)
    if column_name.lower() in ["country"]:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Check for obviously wrong country values
            wrong_count = 0
            for val in non_null:
                val_str = str(val).lower()
                if val_str in ['test', 'xxx', '123', 'unknown']:
                    wrong_count += 1
            if wrong_count > 0:
                wrong_rate = (wrong_count / total_count) * 100
                error_types.append(("Falsche Werte", wrong_rate))
    
    # Datenkonflikte (conflicting versions of same information)
    # This would typically require cross-record analysis, but we can check for inconsistent formatting
    if column_name.lower() in ["location", "company"]:
        non_null = series.dropna()
        if len(non_null) > 0:
            # Check for different formatting of same values (very basic check)
            value_counts = series.value_counts()
            similar_values = 0
            for val in value_counts.index[:10]:  # Check top 10 values
                val_normalized = re.sub(r'[^\w]', '', str(val).lower())
                for other_val in value_counts.index:
                    if val != other_val:
                        other_normalized = re.sub(r'[^\w]', '', str(other_val).lower())
                        if val_normalized == other_normalized and len(val_normalized) > 3:
                            similar_values += value_counts[other_val]
                            break
            if similar_values > 0:
                conflict_rate = (similar_values / total_count) * 100
                error_types.append(("Datenkonflikte", conflict_rate))
    
    # If no errors found, return empty list
    if not error_types:
        return []
    
    # Sort by error rate (highest first)
    error_types.sort(key=lambda x: x[1], reverse=True)
    return error_types


def classify_error_type(series: pd.Series, column_name: str) -> tuple[str, float]:
    """Classify the primary error type for a series according to Naumann/Leser taxonomy.
    
    Returns the error type and the error rate (0-100).
    """
    all_errors = get_all_error_types(series, column_name)
    if not all_errors:
        return "Keine signifikanten Fehler", 0.0
    return all_errors[0]  # Return the most significant error


def top_error(series: pd.Series) -> tuple[Any, int]:
    """Return the most frequent error marker and its count for *series*.

    Parameters
    ----------
    series:
        The pandas ``Series`` to analyse.
    """
    top: Any = None
    top_count = 0
    for value in ERROR_VALUES:
        # ``==`` works for all markers except ``None`` which we handle via ``isna``.
        if value is None:
            count = series.isna().sum()
        else:
            count = (series == value).sum()
        if count > top_count:
            top = value
            top_count = int(count)
    return top, top_count


def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple profiling statistics for *df*.

    The returned ``DataFrame`` contains one row per column of *df* with the
    following metrics:

    ``Zeilen``
        Gesamtzahl der Zeilen des Datensatzes.
    ``Fehlende Werte`` / ``Fehlende Werte %``
        Anzahl und relative Häufigkeit fehlender Werte.
    ``Eindeutige Werte``
        Zahl der unterschiedlichen Werte (ohne ``NaN``).
    ``Häufigste Fehlerart`` / ``Fehler Häufigkeit`` / ``Fehler %``
        Häufigster Eintrag aus ``ERROR_VALUES`` und seine absolute sowie
        relative Häufigkeit.
    ``Häufigster Wert`` / ``Häufigster Wert %``
        Wert mit der höchsten Auftretenshäufigkeit und sein prozentualer Anteil.
    ``Hauptfehlertyp (Naumann/Leser)``
        Klassifizierter Hauptfehlertyp nach Naumann/Leser Taxonomie.
    ``Hauptfehlerrate %``
        Rate des Hauptfehlertyps.
    """
    total = len(df)
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        error_val, error_count = top_error(series)
        counts = series.value_counts(dropna=True)
        if not counts.empty:
            top_value = counts.idxmax()
            top_value_count = int(counts.iloc[0])
        else:
            top_value = ""
            top_value_count = 0
        if error_count == 0:
            error_display = ""
        elif error_val is None:
            error_display = "None"
        else:
            error_display = str(error_val)

        # Classify main error type according to Naumann/Leser
        main_error_type, main_error_rate = classify_error_type(series, column)

        rows.append(
            {
                "Spalte": column,
                "Zeilen": total,
                "Fehlende Werte": missing,
                "Fehlende Werte %": round(missing / total * 100, 2) if total else 0.0,
                "Eindeutige Werte": unique,
                "Häufigste Fehlerart": error_display,
                "Fehler Häufigkeit": error_count,
                "Fehler %": round(error_count / total * 100, 2) if total else 0.0,
                "Häufigster Wert": top_value,
                "Häufigster Wert %": round(top_value_count / total * 100, 2) if total else 0.0,
                "Hauptfehlertyp (Naumann/Leser)": main_error_type,
                "Hauptfehlerrate %": round(main_error_rate, 2),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["profile_dataframe", "top_error", "classify_error_type", "get_all_error_types"]