"""Data profiling utilities for the Bibliojobs dataset."""
from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd

# Common placeholders that should be treated as data errors/missing markers.
ERROR_VALUES = {"", "??", "na", "n/a", "null", None}


def _top_error(series: pd.Series) -> tuple[str, int]:
    """Return the most frequent error marker and its count for *series*.

    Parameters
    ----------
    series:
        The pandas ``Series`` to analyse.
    """
    top = ""
    top_count = 0
    for value in ERROR_VALUES:
        # ``==`` works for all markers except ``None`` which we handle via ``isna``.
        if value is None:
            count = series.isna().sum()
        else:
            count = (series == value).sum()
        if count > top_count:
            top = "" if value is None else str(value)
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
    """
    total = len(df)
    rows: list[Dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        error_val, error_count = _top_error(series)
        counts = series.value_counts(dropna=True)
        if not counts.empty:
            top_value = counts.idxmax()
            top_value_count = int(counts.iloc[0])
        else:
            top_value = ""
            top_value_count = 0
        rows.append(
            {
                "Spalte": column,
                "Zeilen": total,
                "Fehlende Werte": missing,
                "Fehlende Werte %": round(missing / total * 100, 2) if total else 0.0,
                "Eindeutige Werte": unique,
                "Häufigste Fehlerart": error_val,
                "Fehler Häufigkeit": error_count,
                "Fehler %": round(error_count / total * 100, 2) if total else 0.0,
                "Häufigster Wert": top_value,
                "Häufigster Wert %": round(top_value_count / total * 100, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["profile_dataframe"]
