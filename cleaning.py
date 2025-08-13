import html
import json
import os
import re
import time
from typing import Callable, Optional, Dict

from collections import defaultdict

import pandas as pd
import requests
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


DEDUPLICATE_COLUMNS = ["company", "location", "jobtype", "jobdescription"]

def extract_plz_from_company(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Extract postal codes from company names and return cleaned company names and PLZ.
    
    Parameters
    ----------
    series : pd.Series
        Series containing company names with potential postal codes
        
    Returns
    -------
    tuple[pd.Series, pd.Series]
        Tuple of (cleaned_company_names, extracted_plz)
    """
    
    def extract_plz_and_clean(value):
        if pd.isna(value):
            return value, None
        
        value_str = str(value).strip()
        extracted_plz = None
        
        # Extract German postal codes (5 digits) with optional city
        # Patterns: ", 12345 Stadt" or ", 12345" at end of string
        plz_pattern = r',\s*(\d{5})(?:\s+[A-ZÄÖÜ][a-zäöüß\s-]+)?$'
        match = re.search(plz_pattern, value_str, flags=re.IGNORECASE)
        
        if match:
            extracted_plz = match.group(1)
            # Remove the entire PLZ+city part from company name
            value_str = re.sub(plz_pattern, '', value_str, flags=re.IGNORECASE)
        else:
            # Try to find PLZ without comma (less common but happens)
            # Pattern: " 12345 Stadt" or " 12345" at end, but be careful not to match job IDs etc.
            plz_pattern_no_comma = r'\s+(\d{5})\s+[A-ZÄÖÜ][a-zäöüß\s-]+$'
            match = re.search(plz_pattern_no_comma, value_str, flags=re.IGNORECASE)
            if match:
                extracted_plz = match.group(1)
                value_str = re.sub(plz_pattern_no_comma, '', value_str, flags=re.IGNORECASE)
        
        return value_str.strip(), extracted_plz
    
    # Apply extraction to all values
    results = series.apply(extract_plz_and_clean)
    
    # Separate company names and PLZ
    company_names = pd.Series([result[0] for result in results], index=series.index)
    plz_codes = pd.Series([result[1] for result in results], index=series.index)
    
    return company_names, plz_codes


def clean_company_field(series: pd.Series) -> pd.Series:
    """Clean and standardize company names by removing cities, 
    normalizing formatting, and consolidating similar entries.
    
    Note: PLZ extraction should be done separately before this function.
    
    Parameters
    ----------
    series : pd.Series
        Series containing company names to clean
        
    Returns
    -------
    pd.Series
        Cleaned series with standardized company names
    """
    
    def clean_single_company(value):
        if pd.isna(value):
            return value
        
        value_str = str(value).strip()
        
        # 1. Remove standalone cities at end (Hamburg, Berlin, etc.)
        # Only remove if they appear after a comma
        city_only_pattern = r',\s+(?:Hamburg|Berlin|München|Köln|Frankfurt|Dresden|Leipzig|Hannover|Düsseldorf|Stuttgart|Dortmund|Essen|Bremen|Duisburg|Nürnberg|Bochum|Wuppertal|Bielefeld|Bonn|Münster|Karlsruhe|Mannheim|Augsburg|Wiesbaden|Gelsenkirchen|Mönchengladbach|Braunschweig|Chemnitz|Kiel|Aachen|Halle|Magdeburg|Freiburg|Krefeld|Lübeck|Mainz|Erfurt|Oberhausen|Rostock|Kassel|Hagen|Potsdam|Saarbrücken|Hamm|Mülheim|Ludwigshafen|Leverkusen|Oldenburg|Osnabrück|Solingen|Heidelberg|Herne|Neuss|Darmstadt|Paderborn|Regensburg|Ingolstadt|Würzburg|Fürth|Wolfsburg|Offenbach|Ulm|Heilbronn|Pforzheim|Göttingen|Bottrop|Trier|Recklinghausen|Reutlingen|Bremerhaven|Koblenz|Bergisch|Gladbach|Jena|Remscheid|Erlangen|Moers|Siegen|Hildesheim|Salzgitter|Leimen|Marburg|Kleve|Wildau|Minden|Oberhaching|Böhl-Iggelheim|Groß-Umstadt|Mainburg|Stralsund|Zella-Mehlis)$'
        value_str = re.sub(city_only_pattern, '', value_str, flags=re.IGNORECASE)
        
        # 2. Clean formatting and punctuation
        value_str = re.sub(r'\s*\.\s*$', '', value_str)  # Remove trailing dots
        value_str = re.sub(r',\s*,+', ',', value_str)     # Remove multiple commas
        value_str = re.sub(r',\s*$', '', value_str)       # Remove trailing commas
        value_str = re.sub(r'\s+', ' ', value_str)        # Normalize whitespace
        value_str = re.sub(r'-\s+', '- ', value_str)      # Normalize hyphens
        
        # 3. Standardize common abbreviations and legal forms
        abbreviations = {
            r'\bgGmbH\b': 'gGmbH',
            r'\bGmbH\b': 'GmbH', 
            r'\bAG\b': 'AG',
            r'\be\.V\.\b': 'e.V.',
            r'\beV\b': 'e.V.',
            r'\bLLP\b': 'LLP',
            r'\bBibliothek\b': 'Bibliothek',
            r'\bUniversität\b': 'Universität',
            r'\bHochschule\b': 'Hochschule',
            r'\bInstitut\b': 'Institut',
            r'\bZentrum\b': 'Zentrum',
            r'\bStadt\b': 'Stadt'
        }
        
        for pattern, replacement in abbreviations.items():
            value_str = re.sub(pattern, replacement, value_str, flags=re.IGNORECASE)
        
        # 4. Remove redundant descriptive text that often appears at the end
        redundant_patterns = [
            r',\s*Softwarehersteller für Bibliotheken',
            r',\s*Bibliothek$',
            r',\s*Stadtbibliothek$',
            r',\s*Universitätsbibliothek$',
            r',\s*Referat Benutzung$',
            r',\s*Dienstort\s+\w+$',
            r',\s*Standort\s+\w+$',
            r',\s*Ärztliche Zentralbibliothek$',
            r',\s*Hochschulbibliothek$'
        ]
        
        for pattern in redundant_patterns:
            value_str = re.sub(pattern, '', value_str, flags=re.IGNORECASE)
        
        return value_str.strip()
    
    # First pass: clean individual entries
    cleaned_series = series.apply(clean_single_company)
    
    # Second pass: consolidate similar entries using fuzzy matching
    cleaned_series = consolidate_similar_companies(cleaned_series)
    
    return cleaned_series


def consolidate_similar_companies(series: pd.Series, threshold: int = 85) -> pd.Series:
    """Consolidate similar company names using fuzzy string matching.
    
    Parameters
    ----------
    series : pd.Series
        Series with cleaned company names
    threshold : int
        Similarity threshold (0-100) for considering names as duplicates
        
    Returns
    -------
    pd.Series
        Series with consolidated company names
    """
    
    # Get unique values and their counts
    value_counts = series.value_counts()
    unique_values = value_counts.index.tolist()
    
    # Group similar values
    groups = []
    used = set()
    
    for i, value1 in enumerate(unique_values):
        if value1 in used or pd.isna(value1):
            continue
            
        group = [value1]
        used.add(value1)
        
        for j, value2 in enumerate(unique_values[i+1:], i+1):
            if value2 in used or pd.isna(value2):
                continue
                
            # Use fuzzy matching to compare names
            similarity = fuzz.ratio(str(value1).lower(), str(value2).lower())
            
            if similarity >= threshold:
                group.append(value2)
                used.add(value2)
        
        if len(group) > 1:
            groups.append(group)
    
    # Create mapping from similar names to the most frequent one
    name_mapping = {}
    for group in groups:
        # Choose the most frequent name as canonical
        group_counts = [(name, value_counts[name]) for name in group]
        group_counts.sort(key=lambda x: (-x[1], len(x[0])))  # Most frequent, then shortest
        canonical = group_counts[0][0]
        
        for name in group:
            name_mapping[name] = canonical
    
    # Apply mapping
    def apply_mapping(value):
        if pd.isna(value):
            return value
        return name_mapping.get(value, value)
    
    return series.apply(apply_mapping)

def get_cache_file_path() -> str:
    """Get the path for the license plate cache file."""
    return os.path.join(os.path.dirname(__file__), 'license_plate_cache.json')


def load_license_plate_cache(status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """Load license plate mapping from local cache file."""
    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    cache_file = get_cache_file_path()
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        _status(f"Warnung: Kennzeichen-Cache konnte nicht geladen werden: {e}")
    return {}


def save_license_plate_cache(
    license_plate_map: Dict[str, str],
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Save license plate mapping to local cache file."""
    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    cache_file = get_cache_file_path()
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(license_plate_map, f, ensure_ascii=False, indent=2)
    except IOError as e:
        _status(f"Warnung: Kennzeichen-Cache konnte nicht gespeichert werden: {e}")


def fetch_german_license_plates_from_api(
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, str]:
    """Fetch German license plate codes from Wikidata API with retry mechanism.

    Returns a dictionary mapping license plate codes to place names.
    """
    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)
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
        'User-Agent': 'Bibliojobs-Data-Cleaning-Tool/1.0 (Educational Project; Contact: ehrmann@gfz.de)'
    }
    
    # Retry mechanism with exponential backoff
    max_retries = 3
    base_delay = 2.0  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            # Add delay before each attempt (except the first one)
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))  # Exponential backoff
                _status(
                    f"Wikidata-API wird in {delay} Sekunden erneut aufgerufen... (Versuch {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            
            response = requests.get(
                endpoint, 
                params={'query': sparql_query}, 
                headers=headers,
                timeout=45
            )
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    wait_time = min(int(retry_after), 60)  # Max 1 minute wait
                    _status(f"Von Wikidata ausgebremst. Warte {wait_time} Sekunden...")
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
            
            _status(
                f"Erfolgreich {len(license_plate_map)} Kennzeichen-Zuordnungen von Wikidata geladen"
            )
            return license_plate_map
            
        except requests.exceptions.Timeout:
            _status(f"Zeitüberschreitung bei Versuch {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                _status("Alle API-Versuche führten zu einer Zeitüberschreitung")
        except requests.exceptions.RequestException as e:
            _status(f"API-Fehler bei Versuch {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                _status("Alle API-Versuche sind fehlgeschlagen")
        except (KeyError, ValueError) as e:
            _status(f"Fehler beim Verarbeiten der Daten: {e}")
            break  # Don't retry on parsing errors
    
    return {}


def fetch_german_license_plates(status_callback: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """Get German license plate codes, using cache first, then API if needed.

    Returns a dictionary mapping license plate codes to place names.
    """

    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    # First, try to load from cache
    license_plate_map = load_license_plate_cache(status_callback=_status)

    # If cache is empty or very small, try to fetch from API
    if len(license_plate_map) < 10:  # Germany has way more than 10 license plates
        _status("Kennzeichen-Cache leer oder unvollständig, lade Daten von Wikidata...")
        api_result = fetch_german_license_plates_from_api(status_callback=_status)

        if api_result:
            # Save to cache for future use
            save_license_plate_cache(api_result, status_callback=_status)
            return api_result
        else:
            _status("Abruf der Kennzeichendaten fehlgeschlagen, verwende vorhandene Cache-Daten")
            return license_plate_map
    else:
        _status(f"Verwende zwischengespeicherte Kfz-Kennzeichen-Daten ({len(license_plate_map)} Einträge)")
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


def clean_dataframe(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Return a cleaned copy of *df* with HTML entities decoded, tags removed,
    license plates resolved, company names standardized, and PLZ extracted to separate column.

    Parameters
    ----------
    df:
        Input DataFrame to clean. Only ``object`` columns are processed.
    progress_callback:
        Optional function receiving the percentage of processed columns as a
        ``float`` between 0 and 100.
    status_callback:
        Optional function receiving status messages as ``str``.
    """

    def _status(msg: str) -> None:
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    total = len(object_cols)

    # Fetch license plate mapping once at the beginning
    license_plate_map = {}
    if 'location' in cleaned.columns:
        if progress_callback:
            progress_callback(5.0)  # 5% for fetching license plates
        license_plate_map = fetch_german_license_plates(status_callback)
        time.sleep(0.1)

    # Extract PLZ from company field before other processing
    if 'company' in cleaned.columns:
        if progress_callback:
            progress_callback(10.0)  # Additional 5% for PLZ extraction

        _status("Extrahiere Postleitzahlen aus Firmennamen...")
        company_cleaned, plz_extracted = extract_plz_from_company(cleaned['company'])
        cleaned['company'] = company_cleaned

        # Add PLZ column (or update if it already exists)
        if 'plz' not in cleaned.columns:
            cleaned['plz'] = plz_extracted
        else:
            # If PLZ column exists, only fill empty values
            cleaned['plz'] = cleaned['plz'].fillna(plz_extracted)

        _status(f"PLZ extrahiert: {plz_extracted.notna().sum()} Einträge gefunden")
    
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
        
        # Company name cleaning and standardization (after PLZ extraction)
        if col == 'company':
            if progress_callback:
                progress = 15.0 + (idx / total * 75.0)  # Show progress for company cleaning
                progress_callback(progress)
            _status("Bereinige und standardisiere Firmennamen...")
            cleaned[col] = clean_company_field(cleaned[col])
        
        if progress_callback and total:
            # Reserve 10% for license plates + PLZ, use remaining 90% for processing
            progress = 10.0 + (idx / total * 90.0)
            progress_callback(progress)
    
    if progress_callback:
        progress_callback(100.0)
    return cleaned


def find_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find and remove duplicate rows using fuzzy matching on selected columns.

    Uses a TF-IDF vectorization with nearest-neighbor search to reduce the
    number of pairwise comparisons, avoiding the quadratic complexity of a
    naive nested loop.  Each candidate pair is then compared column-wise and
    considered a duplicate only if *all* selected columns reach the given
    similarity threshold.  This reduces false positives where, for example,
    generic job descriptions are identical but other attributes differ.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to search for duplicates.
    columns : list[str] | None
        Columns to compare for duplicates. If ``None``,
        :data:`DEDUPLICATE_COLUMNS` is used.
    threshold : int
        Similarity threshold (0-100). Higher values mean stricter matching.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (cleaned_dataframe, duplicates_dataframe). The
        duplicates dataframe lists potential duplicate pairs with a
        ``keep`` column indicating the recommended record to retain and a
        ``pair_id`` column grouping corresponding rows.
    """

    columns = columns or DEDUPLICATE_COLUMNS

    # Create comparison keys
    keys = (
        df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    vectorizer = TfidfVectorizer().fit(keys)
    matrix = vectorizer.transform(keys)
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(matrix)

    n_neighbors = min(10, len(df))
    distances, indices = nn.kneighbors(matrix, n_neighbors=n_neighbors)

    drop_indices: set[int] = set()
    pairs: list[tuple[int, int]] = []

    for i, neighbors in enumerate(indices):
        if i in drop_indices:
            continue
        for j in neighbors:
            if j == i or j in drop_indices:
                continue

            row_i = df.iloc[i]
            row_j = df.iloc[j]

            score = fuzz.token_set_ratio(keys.iloc[i], keys.iloc[j])
            company_sim = fuzz.token_set_ratio(
                str(row_i.get("company", "")), str(row_j.get("company", ""))
            )
            jobdesc_sim = fuzz.token_set_ratio(
                str(row_i.get("jobdescription", "")),
                str(row_j.get("jobdescription", "")),
            )

            company_threshold = max(80, threshold - 10)

            if not (
                score >= threshold
                and company_sim >= company_threshold
                and jobdesc_sim >= threshold
            ):
                continue

            nonnull_i = row_i.count()
            nonnull_j = row_j.count()
            if nonnull_i >= nonnull_j:
                keep_idx, drop_idx = i, j
            else:
                keep_idx, drop_idx = j, i
            drop_indices.add(drop_idx)
            pairs.append((keep_idx, drop_idx))
            if drop_idx == i:
                break

    duplicate_rows = []
    for pair_id, (keep_idx, drop_idx) in enumerate(pairs):
        keep_row = df.iloc[keep_idx].copy()
        keep_row["keep"] = True
        keep_row["pair_id"] = pair_id
        drop_row = df.iloc[drop_idx].copy()
        drop_row["keep"] = False
        drop_row["pair_id"] = pair_id
        duplicate_rows.extend([keep_row, drop_row])

    duplicates = pd.DataFrame(duplicate_rows)
    if not duplicates.empty:
        duplicates = duplicates.sort_values(["pair_id", "keep"], ascending=[True, False])
        duplicates = duplicates.reset_index(drop=True)

    cleaned = df.drop(index=drop_indices).reset_index(drop=True)
    return cleaned, duplicates

