import pandas as pd
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


def test_none_as_top_error():
    df = pd.DataFrame({"a": [None, None, "x"]})
    profile = profile_dataframe(df)
    col = profile[profile["Spalte"] == "a"].iloc[0]
    assert col["Häufigste Fehlerart"] == "None"
    assert col["Fehler Häufigkeit"] == 2
