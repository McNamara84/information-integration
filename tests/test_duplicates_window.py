import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.append(str(Path(__file__).resolve().parents[1]))
QtWidgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from start import DuplicatesWindow
from cleaning import DEDUPLICATE_COLUMNS, find_fuzzy_duplicates, clean_dataframe
from load_bibliojobs import load_bibliojobs

CSV_PATH = "bibliojobs_raw.csv"


def load_clean_subset(jobids: list[int]) -> pd.DataFrame:
    df = load_bibliojobs(CSV_PATH)
    subset = df[df["jobid"].isin(jobids)].reset_index(drop=True)
    return clean_dataframe(subset)


@pytest.fixture
def duplicates_df() -> pd.DataFrame:
    jobids = [12687, 13048, 13235, 13958]
    df = load_clean_subset(jobids)
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
