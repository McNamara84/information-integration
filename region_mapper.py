import pandas as pd
from rapidfuzz import process
from typing import Optional

def match_region_fuzzy(location: str, mapping: pd.DataFrame, threshold: int = 90) -> Optional[str]:
    """Return region name for *location* using fuzzy matching.

    Parameters
    ----------
    location : str
        Location string to match.
    mapping : pd.DataFrame
        DataFrame with at least columns ``location`` and ``region``.
    threshold : int, optional
        Minimum similarity score (0-100) required to accept a match.
        Defaults to 90.

    Returns
    -------
    Optional[str]
        Region name if a match above ``threshold`` is found, ``None`` otherwise.
    """
    if not isinstance(location, str) or not location.strip():
        return None

    choices = mapping["location"].dropna().astype(str).tolist()
    if not choices:
        return None

    result = process.extractOne(location, choices, score_cutoff=threshold)
    if result is None:
        return None

    matched_name, score, idx = result
    try:
        return mapping.iloc[idx]["region"]
    except Exception:
        return None

