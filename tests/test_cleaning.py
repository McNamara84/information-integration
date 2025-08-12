import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from cleaning import clean_dataframe


def test_clean_dataframe_html_unescape_and_strip():
    df = pd.DataFrame({
        "a": ["AT&amp;T", "<b>Bold</b>", None]
    })
    cleaned = clean_dataframe(df)
    assert cleaned["a"].iloc[0] == "AT&T"
    assert cleaned["a"].iloc[1] == "Bold"
    assert pd.isna(cleaned["a"].iloc[2])
