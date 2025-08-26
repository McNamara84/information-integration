from pathlib import Path
import re
import time
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
from rapidfuzz import process


# Some state names in the SQL dump or API responses use English
# descriptions such as ``"State of Berlin"``.  Normalize those to the
# short German names used elsewhere in the project.
REGION_ALIASES = {
    "State of Berlin": "Berlin",
}


_last_nominatim_call = 0.0


def _throttle_nominatim(min_interval: float = 1.0) -> None:
    """Ensure at most one Nominatim request per ``min_interval`` seconds."""

    global _last_nominatim_call
    elapsed = time.monotonic() - _last_nominatim_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_nominatim_call = time.monotonic()


def _normalize_region(region: Optional[str]) -> Optional[str]:
    """Return a standardized region name or ``None``.

    Parameters
    ----------
    region:
        Original region name which might contain English descriptors.

    Returns
    -------
    Optional[str]
        Normalized region name.
    """

    if region is None:
        return None
    return REGION_ALIASES.get(region, region)


@lru_cache()
def region_from_coordinates(lat: float, lon: float) -> Optional[str]:
    """Return German federal state for given ``lat`` and ``lon`` using an API.

    Queries the public Nominatim API which returns address details for the
    provided coordinates. The ``state`` field is used as Bundesland. Results are
    cached to avoid repeated lookups for identical coordinate pairs. Network
    errors or missing data result in ``None``. The function adheres to the
    service's usage policy by issuing at most one request per second and
    retrying once if the service signals rate limiting.
    """

    if pd.isna(lat) or pd.isna(lon):
        return None

    params = {"lat": float(lat), "lon": float(lon), "format": "json", "zoom": 3}
    headers = {
        "User-Agent": "information-integration/1.0",
        "Accept-Language": "de",
    }

    region = None
    for _ in range(2):  # allow a single retry on HTTP 429/503
        try:
            _throttle_nominatim()
            response = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params=params,
                headers=headers,
                timeout=10,
            )
            if response.status_code in {429, 503}:
                time.sleep(1)
                continue
            if response.ok:
                data = response.json()
                region = _normalize_region(data.get("address", {}).get("state"))
            break
        except Exception:
            break

    if region:
        return region

    # Fallback service if Nominatim fails or returns no region
    try:
        response = requests.get(
            "https://geocode.maps.co/reverse",
            params={"lat": float(lat), "lon": float(lon)},
            headers=headers,
            timeout=10,
        )
        if response.ok:
            data = response.json()
            region = _normalize_region(
                data.get("address", {}).get("state")
                or data.get("state")
                or data.get("region")
            )
    except Exception:
        pass
    return region

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
                        "region": _normalize_region(region),
                        "geo_lat": lat_f,
                        "geo_lon": lon_f,
                    }
                )

    return pd.DataFrame(records)

