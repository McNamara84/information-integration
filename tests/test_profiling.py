from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from profiling import profile_dataframe


def test_profile_dataframe_basic():
    df = pd.DataFrame({
        "a": [1, 2, None, "??"],
        "b": ["x", "y", "x", "na"],
    })
    profile = profile_dataframe(df)

    col_a = profile[profile["Spalte"] == "a"].iloc[0]
    assert col_a["Zeilen"] == 4
    assert col_a["Fehlende Werte"] == 1
    assert col_a["Fehlende Werte %"] == 25.0
    assert col_a["Eindeutige Werte"] == 3

    col_b = profile[profile["Spalte"] == "b"].iloc[0]
    assert col_b["Häufigste Fehlerart"] == "na"
    assert col_b["Fehler Häufigkeit"] == 1
    assert col_b["Fehler %"] == 25.0
