import html
import re
from typing import Callable, Optional

import pandas as pd


def clean_dataframe(df: pd.DataFrame, progress_callback: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
    """Return a cleaned copy of *df* with HTML entities decoded and tags removed.

    Parameters
    ----------
    df:
        Input DataFrame to clean. Only ``object`` columns are processed.
    progress_callback:
        Optional function receiving the percentage of processed columns as a
        ``float`` between 0 and 100.
    """
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    total = len(object_cols)
    for idx, col in enumerate(object_cols, start=1):
        series = cleaned[col].dropna().astype(str)
        series = series.apply(html.unescape)
        series = series.apply(lambda x: re.sub(r"<.*?>", "", x))
        cleaned.loc[series.index, col] = series
        if progress_callback and total:
            progress_callback(idx / total * 100)
    if progress_callback:
        progress_callback(100.0)
    return cleaned
