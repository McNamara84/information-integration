import html
import re
from functools import lru_cache
from typing import Callable, Optional

import pandas as pd
import requests


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
        cleaned[col] = cleaned[col].apply(
            lambda x: html.unescape(re.sub(r"<.*?>", "", str(x)))
            if pd.notna(x)
            else x
        )
        if progress_callback and total:
            progress_callback(idx / total * 100)

    if "location" in cleaned.columns:
        cleaned["location"] = cleaned["location"].apply(_expand_location)

    if progress_callback:
        progress_callback(100.0)
    return cleaned


def _expand_location(value: str):
    if pd.isna(value):
        return value
    text = str(value).strip()
    if re.fullmatch(r"[A-ZÄÖÜ]{1,3}", text):
        resolved = _get_location_from_wikidata(text)
        return resolved if resolved else value
    return value


@lru_cache(maxsize=None)
def _get_location_from_wikidata(code: str) -> Optional[str]:
    url = "https://query.wikidata.org/sparql"
    query = (
        f'SELECT ?itemLabel WHERE {{ ?item wdt:P395 "{code}".'
        ' SERVICE wikibase:label { bd:serviceParam wikibase:language "de" }. }} LIMIT 1'
    )
    params = {"query": query, "format": "json"}
    headers = {"User-Agent": "information-integration/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        bindings = data.get("results", {}).get("bindings")
        if bindings:
            return bindings[0]["itemLabel"]["value"]
    except Exception:
        return None
    return None
