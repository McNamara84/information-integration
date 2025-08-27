import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt6.QtWidgets", exc_type=ImportError)
from start import dataframe_to_custom_csv


def test_dataframe_to_custom_csv_uppercase_headers_and_no_blank_lines(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    content = dataframe_to_custom_csv(df)
    path = tmp_path / "out.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == ["A_§_B", "1_§_3", "2_§_4"]
