import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from start import DuplicatesWindow


def test_duplicates_window_filters_probability():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    df = pd.DataFrame(
        {
            "pair_id": [1, 1, 2, 2],
            "keep": [True, False, True, False],
            "orig_index": [10, 11, 12, 13],
            "probability": [100, 100, 80, 80],
            "company": ["A", "B", "C", "D"],
        }
    )
    window = DuplicatesWindow(df)
    assert "probability" not in window._dataframe.columns
    assert list(window._dataframe["company"]) == ["A", "B"]
    assert len(window._dataframe) == 2
    window.close()
    app.quit()