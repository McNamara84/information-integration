from pathlib import Path
import re
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
from rapidfuzz import process


@lru_cache()
def region_from_coordinates(lat: float, lon: float) -> Optional[str]:
    """Return German federal state for given ``lat`` and ``lon`` using an API.

    Queries the public Nominatim API which returns address details for the
    provided coordinates. The ``state`` field is used as Bundesland. Results are
    cached to avoid repeated lookups for identical coordinate pairs. Network
    errors or missing data result in ``None``.
    """

    if pd.isna(lat) or pd.isna(lon):
        return None

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": float(lat), "lon": float(lon), "format": "json", "zoom": 3},
            headers={"User-Agent": "information-integration/1.0"},
            timeout=10,
        )
        if response.ok:
            data = response.json()
            return data.get("address", {}).get("state")
    except Exception:
        return None

    return None

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


@lru_cache()
def load_region_mapping(path: str | Path = Path(__file__).with_name("ort_bundesland.sql")) -> pd.DataFrame:
    """Load location to region mapping from the SQL dump.

    Parameters
    ----------
    path:
        Path to the SQL file containing ``INSERT`` statements with ``Name``
        and ``Bundesland`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``location`` and ``region``.
    """

    pattern = re.compile(
        r"\('([^']*)',\s*'([^']*)',\s*'([^']*)',\s*'([^']*)'"
    )
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            for name, region, lat, lon in pattern.findall(line):
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except ValueError:
                    continue
                records.append(
                    {
                        "location": name,
                        "region": region,
                        "geo_lat": lat_f,
                        "geo_lon": lon_f,
                    }
                )

    return pd.DataFrame(records)

