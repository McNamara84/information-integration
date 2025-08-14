import os
import sys
from pathlib import Path
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.append(str(Path(__file__).resolve().parents[1]))
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from start import DuplicatesWindow
from cleaning import DEDUPLICATE_COLUMNS, find_fuzzy_duplicates

CSV_PATH = "bibliojobs_raw.csv"
SEP = "_§_"

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=SEP, engine="python")
    df.columns = [c.strip("_").lower() for c in df.columns]
    for col in ["plz", "fixedterm", "workinghours", "salary"]:
        df[col] = pd.NA
    return df


@pytest.fixture
def duplicates_df() -> pd.DataFrame:
    df = load_raw().head(700)
    _, duplicates = find_fuzzy_duplicates(df, DEDUPLICATE_COLUMNS, threshold=80)
    return duplicates


def test_duplicates_window_filters_probability(duplicates_df: pd.DataFrame) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DuplicatesWindow(duplicates_df)
    expected = (
        duplicates_df[duplicates_df["probability"] == 100]
        .drop(columns=["probability"])
        .reset_index(drop=True)
    )
    assert window._dataframe.equals(expected)
    window.close()
    app.quit()
