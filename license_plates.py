"""Helper functions for working with German license plate codes."""
import json
import os
import re
import time
from typing import Callable, Dict, Optional

import pandas as pd
import requests

from utils import make_status_printer


def get_cache_file_path() -> str:
    """Return the path of the licence plate cache file.

    Returns
    -------
    str
        Absolute path to the JSON cache file.
    """
    return os.path.join(
        os.path.dirname(__file__),
        "cache",
        "license_plate_cache.json",
    )


def load_license_plate_cache(
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Load licence plate mapping from the local cache.

    Parameters
    ----------
    status_callback : callable, optional
        Callback receiving status messages.

    Returns
    -------
    dict
        Mapping of licence plate codes to place names.
    """
    _status = make_status_printer(status_callback)
    cache_file = get_cache_file_path()
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as handle:
                return json.load(handle)
    except (json.JSONDecodeError, IOError) as exc:  # pragma: no cover - log path
        _status(f"Warnung: Kennzeichen-Cache konnte nicht geladen werden: {exc}")
    return {}


def save_license_plate_cache(
    license_plate_map: Dict[str, str],
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Persist licence plate mapping to the local cache file.

    Parameters
    ----------
    license_plate_map : dict
        Mapping of licence plate codes to place names.
    status_callback : callable, optional
        Callback receiving status messages.
    """
    _status = make_status_printer(status_callback)
    cache_file = get_cache_file_path()
    try:
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(license_plate_map, handle, ensure_ascii=False, indent=2)
    except IOError as exc:  # pragma: no cover - log path
        _status(f"Warnung: Kennzeichen-Cache konnte nicht gespeichert werden: {exc}")


def fetch_german_license_plates_from_api(
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Query the Wikidata API for German licence plate codes.

    Parameters
    ----------
    status_callback : callable, optional
        Callback receiving status messages.

    Returns
    -------
    dict
        Mapping of licence plate codes to place names. Returns an empty dict on
        failure.
    """
    _status = make_status_printer(status_callback)
    sparql_query = """
    SELECT ?item ?itemLabel ?licencePlate WHERE {
      ?item wdt:P395 ?licencePlate .
      ?item wdt:P17 wd:Q183 .  # Located in Germany
      ?item wdt:P31/wdt:P279* wd:Q56061 .  # Instance of administrative territorial entity
      SERVICE wikibase:label { bd:serviceParam wikibase:language "de,en" . }
    }
    """

    endpoint = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": (
            "Bibliojobs-Data-Cleaning-Tool/1.0 "
            "(Educational Project; Contact: ehrmann@gfz.de)"
        ),
    }

    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                _status(
                    "Wikidata-API wird in "
                    f"{delay} Sekunden erneut aufgerufen... (Versuch {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

            response = requests.get(
                endpoint,
                params={"query": sparql_query},
                headers=headers,
                timeout=45,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    wait_time = min(int(retry_after), 60)
                    _status(
                        f"Von Wikidata ausgebremst. Warte {wait_time} Sekunden..."
                    )
                    time.sleep(wait_time)
                    continue
                continue

            response.raise_for_status()
            data = response.json()
            license_plate_map: Dict[str, str] = {}
            for binding in data.get("results", {}).get("bindings", []):
                if "licencePlate" in binding and "itemLabel" in binding:
                    plate_code = binding["licencePlate"]["value"]
                    place_name = binding["itemLabel"]["value"]
                    # Allow German umlauts in plate codes
                    if re.match(r"^[A-ZÄÖÜ]{1,3}$", plate_code):
                        license_plate_map[plate_code] = place_name
                        # Also store ASCII fallback (e.g. WUE for WÜ)
                        ascii_plate = (
                            plate_code
                            .replace("Ä", "AE")
                            .replace("Ö", "OE")
                            .replace("Ü", "UE")
                        )
                        if ascii_plate != plate_code:
                            license_plate_map[ascii_plate] = place_name

            _status(
                f"Erfolgreich {len(license_plate_map)} Kennzeichen-Zuordnungen von Wikidata geladen"
            )
            return license_plate_map

        except requests.exceptions.Timeout:
            _status(f"Zeitüberschreitung bei Versuch {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                _status("Alle API-Versuche führten zu einer Zeitüberschreitung")
        except requests.exceptions.RequestException as exc:
            _status(f"API-Fehler bei Versuch {attempt + 1}/{max_retries}: {exc}")
            if attempt == max_retries - 1:
                _status("Alle API-Versuche sind fehlgeschlagen")
        except (KeyError, ValueError) as exc:
            _status(f"Fehler beim Verarbeiten der Daten: {exc}")
            break

    return {}


def fetch_german_license_plates(
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Obtain German licence plate codes using cache and API fallback.

    Parameters
    ----------
    status_callback : callable, optional
        Callback receiving status messages.

    Returns
    -------
    dict
        Mapping of licence plate codes to place names.
    """
    _status = make_status_printer(status_callback)
    license_plate_map = load_license_plate_cache(status_callback=status_callback)

    if len(license_plate_map) < 10:
        _status("Kennzeichen-Cache leer oder unvollständig, lade Daten von Wikidata...")
        api_result = fetch_german_license_plates_from_api(status_callback=status_callback)
        if api_result:
            save_license_plate_cache(api_result, status_callback=status_callback)
            return api_result
        _status(
            "Abruf der Kennzeichendaten fehlgeschlagen, verwende vorhandene Cache-Daten"
        )
        return license_plate_map

    _status(
        f"Verwende zwischengespeicherte Kfz-Kennzeichen-Daten ({len(license_plate_map)} Einträge)"
    )
    return license_plate_map


def resolve_license_plates_in_series(
    series: pd.Series, license_plate_map: Dict[str, str]
) -> pd.Series:
    """Replace licence plate codes in a series with full place names.

    Parameters
    ----------
    series : pd.Series
        Series possibly containing licence plate codes.
    license_plate_map : dict
        Mapping of licence plate codes to place names.

    Returns
    -------
    pd.Series
        Series with codes replaced by place names where possible.
    """
    if not license_plate_map:
        return series

    def replace_license_plates(value):
        if pd.isna(value):
            return value
        value_str = str(value).strip()
        if value_str.upper() in license_plate_map:
            return license_plate_map[value_str.upper()]
        return value

    return series.apply(replace_license_plates)
