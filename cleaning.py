import html
import json
import os
import re
import time
from typing import Callable, Optional, Dict
import requests

import pandas as pd


def get_cache_file_path() -> str:
    """Get the path for the license plate cache file."""
    return os.path.join(os.path.dirname(__file__), 'license_plate_cache.json')


def load_license_plate_cache() -> Dict[str, str]:
    """Load license plate mapping from local cache file."""
    cache_file = get_cache_file_path()
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load license plate cache: {e}")
    return {}


def save_license_plate_cache(license_plate_map: Dict[str, str]) -> None:
    """Save license plate mapping to local cache file."""
    cache_file = get_cache_file_path()
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(license_plate_map, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"Warning: Could not save license plate cache: {e}")


def fetch_german_license_plates_from_api() -> Dict[str, str]:
    """Fetch German license plate codes from Wikidata API with retry mechanism.
    
    Returns a dictionary mapping license plate codes to place names.
    """
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
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'Bibliojobs-Data-Cleaning-Tool/1.0 (Educational Project; Contact: bibliojobs@example.com)'
    }
    
    # Retry mechanism with exponential backoff
    max_retries = 3
    base_delay = 2.0  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay before each attempt (except the first one)
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                print(f"Retrying Wikidata API call in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            response = requests.get(
                endpoint, 
                params={'query': sparql_query}, 
                headers=headers,
                timeout=45  # Increased timeout
            )
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = min(int(retry_after), 60)  # Max 1 minute wait
                    print(f"Rate limited by Wikidata. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    # If no Retry-After header, use exponential backoff
                    continue
            
            response.raise_for_status()
            
            data = response.json()
            license_plate_map = {}
            
            for binding in data.get('results', {}).get('bindings', []):
                if 'licencePlate' in binding and 'itemLabel' in binding:
                    plate_code = binding['licencePlate']['value']
                    place_name = binding['itemLabel']['value']
                    
                    # Only add if it looks like a German license plate code (1-3 letters, uppercase)
                    if re.match(r'^[A-Z]{1,3}$', plate_code):
                        license_plate_map[plate_code] = place_name
            
            print(f"Successfully fetched {len(license_plate_map)} license plate mappings from Wikidata")
            return license_plate_map
            
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                print("All API attempts timed out")
        except requests.exceptions.RequestException as e:
            print(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                print("All API attempts failed")
        except (KeyError, ValueError) as e:
            print(f"Data parsing error: {e}")
            break  # Don't retry on parsing errors
    
    return {}


def fetch_german_license_plates() -> Dict[str, str]:
    """Get German license plate codes, using cache first, then API if needed.
    
    Returns a dictionary mapping license plate codes to place names.
    """
    # First, try to load from cache
    license_plate_map = load_license_plate_cache()
    
    # If cache is empty or very small, try to fetch from API
    if len(license_plate_map) < 10:  # Germany has way more than 10 license plates
        print("License plate cache empty or incomplete, fetching from Wikidata...")
        api_result = fetch_german_license_plates_from_api()
        
        if api_result:
            # Save to cache for future use
            save_license_plate_cache(api_result)
            return api_result
        else:
            print("API fetch failed, using cached data (if any)")
            return license_plate_map
    else:
        print(f"Using cached license plate data ({len(license_plate_map)} entries)")
        return license_plate_map


def resolve_license_plates_in_series(series: pd.Series, license_plate_map: Dict[str, str]) -> pd.Series:
    """Replace license plate codes in a series with full place names.
    
    Only replaces values that consist entirely of a license plate code (with optional whitespace).
    Does not replace license plate codes that are part of longer place names.
    
    Parameters
    ----------
    series:
        The series to process
    license_plate_map:
        Dictionary mapping license plate codes to place names
        
    Returns
    -------
    pd.Series
        Series with standalone license plate codes replaced by place names
    """
    if not license_plate_map:
        return series
    
    def replace_license_plates(value):
        if pd.isna(value):
            return value
        
        value_str = str(value).strip()
        
        # Bugfix: Only replace if the entire value (after stripping whitespace) is a license plate code
        # This prevents replacing "AM" in "Frankfurt am Main" while still catching "AM" by itself
        if value_str.upper() in license_plate_map:
            return license_plate_map[value_str.upper()]
        
        # Don't do partial replacements - return original value unchanged
        return value
    
    return series.apply(replace_license_plates)


def clean_dataframe(df: pd.DataFrame, progress_callback: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
    """Return a cleaned copy of *df* with HTML entities decoded, tags removed, and license plates resolved.

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
    
    # Fetch license plate mapping once at the beginning
    license_plate_map = {}
    if 'location' in cleaned.columns:
        if progress_callback:
            progress_callback(5.0)  # 5% for fetching license plates
        license_plate_map = fetch_german_license_plates()
        # Small delay to be respectful
        time.sleep(0.1)
    
    for idx, col in enumerate(object_cols, start=1):
        # HTML cleaning (original functionality)
        cleaned[col] = cleaned[col].apply(
            lambda x: html.unescape(re.sub(r"<.*?>", "", str(x)))
            if pd.notna(x)
            else x
        )
        
        # License plate resolution for location column
        if col == 'location' and license_plate_map:
            cleaned[col] = resolve_license_plates_in_series(cleaned[col], license_plate_map)
        
        if progress_callback and total:
            # Reserve 5% for license plate fetching, use remaining 95% for processing
            progress = 5.0 + (idx / total * 95.0)
            progress_callback(progress)
    
    if progress_callback:
        progress_callback(100.0)
    return cleaned