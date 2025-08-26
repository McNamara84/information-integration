import html
import re
import time
from typing import Any, Callable, Optional

import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from license_plates import fetch_german_license_plates, resolve_license_plates_in_series
from region_mapper import load_region_mapping, region_from_coordinates
from utils import make_status_printer


DEDUPLICATE_COLUMNS = [
    "jobdescription",
    "jobtype",
    "company",
    "insttype",
    "location",
    "country",
    "geo_lat",
    "geo_lon",
    "plz",
    "fixedterm",
    "workinghours",
    "salary",
]

def extract_plz_from_company(
    series: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Extract postal codes and cities from company names.
    
    Parameters
    ----------
    series : pd.Series
        Series containing company names with potential postal codes
        
    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        Tuple of ``(cleaned_company_names, extracted_plz, extracted_city)``
    """

    def extract_plz_city_and_clean(value):
        if pd.isna(value):
            return value, None, None

        value_str = str(value).strip()
        extracted_plz = None
        extracted_city = None

        # Pattern: ", 12345 Stadt" at end of string
        plz_city_comma = r',\s*(\d{5})\s+([A-ZÄÖÜ][a-zäöüß\s-]+)$'
        match = re.search(plz_city_comma, value_str, flags=re.IGNORECASE)
        if match:
            extracted_plz = match.group(1)
            extracted_city = match.group(2).strip()
            value_str = re.sub(plz_city_comma, '', value_str, flags=re.IGNORECASE)
        else:
            # Pattern: " 12345 Stadt" without comma
            plz_city_no_comma = r'\s+(\d{5})\s+([A-ZÄÖÜ][a-zäöüß\s-]+)$'
            match = re.search(plz_city_no_comma, value_str, flags=re.IGNORECASE)
            if match:
                extracted_plz = match.group(1)
                extracted_city = match.group(2).strip()
                value_str = re.sub(plz_city_no_comma, '', value_str, flags=re.IGNORECASE)
            else:
                # Pattern: ", 12345" or " 12345" (PLZ only)
                plz_only_comma = r',\s*(\d{5})$'
                match = re.search(plz_only_comma, value_str)
                if match:
                    extracted_plz = match.group(1)
                    value_str = re.sub(plz_only_comma, '', value_str)
                else:
                    plz_only_no_comma = r'\s+(\d{5})$'
                    match = re.search(plz_only_no_comma, value_str)
                    if match:
                        extracted_plz = match.group(1)
                        value_str = re.sub(plz_only_no_comma, '', value_str)

        return value_str.strip(), extracted_plz, extracted_city

    # Apply extraction to all values
    results = series.apply(extract_plz_city_and_clean)

    # Separate company names, PLZ, and city
    company_names = pd.Series([result[0] for result in results], index=series.index)
    plz_codes = pd.Series([result[1] for result in results], index=series.index)
    city_names = pd.Series([result[2] for result in results], index=series.index)

    return company_names, plz_codes, city_names


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
    
    # Second pass: consolidate similar entries using vector similarity
    cleaned_series = consolidate_similar_companies(cleaned_series)
    
    return cleaned_series


def consolidate_similar_companies(
    series: pd.Series,
    threshold: int = 85,
    min_occurrence: int = 2,
) -> pd.Series:
    """Consolidate similar company names using a vectorised similarity search.

    This replaces the previous pairwise nested loop with a TF-IDF based approach
    that finds near-duplicate names via a radius search in vector space. A
    frequency threshold can be supplied to skip very rare names up front and
    thereby reduce the number of comparisons.

    Parameters
    ----------
    series : pd.Series
        Series with cleaned company names
    threshold : int
        Similarity threshold (0-100) for considering names as duplicates
    min_occurrence : int, optional
        Minimum number of occurrences of a name for it to participate in the
        similarity check. Higher values reduce the number of comparisons.
        Default is ``2`` so that names occurring only once are skipped.

    Returns
    -------
    pd.Series
        Series with consolidated company names
    """

    value_counts = series.value_counts()

    # Restrict to names that occur at least ``min_occurrence`` times to reduce
    # the number of vectors and comparisons.
    candidates = [str(name) for name in value_counts[value_counts >= min_occurrence].index]
    if len(candidates) < 2:
        return series

    # Vectorise company names as character 3-grams to capture structural
    # similarity.  A radius neighbour search then finds only those names with a
    # cosine similarity above the requested threshold.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    tfidf_matrix = vectorizer.fit_transform(candidates)

    radius = 1 - threshold / 100  # cosine distance equivalent of the threshold
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(tfidf_matrix)
    _, indices = nn.radius_neighbors(tfidf_matrix, radius=radius)

    groups: list[list[str]] = []
    used: set[str] = set()

    for i, neighbors in enumerate(indices):
        name_i = candidates[i]
        if name_i in used or pd.isna(name_i):
            continue

        group = {name_i}
        used.add(name_i)

        for j in neighbors:
            if i == j:
                continue
            name_j = candidates[j]
            if name_j in used or pd.isna(name_j):
                continue
            group.add(name_j)
            used.add(name_j)

        if len(group) > 1:
            groups.append(list(group))

    # Map grouped names to the most frequent canonical name
    name_mapping: dict[str, str] = {}
    for group in groups:
        group_counts = [(name, value_counts[name]) for name in group]
        group_counts.sort(key=lambda x: (-x[1], len(x[0])))  # Most frequent, then shortest
        canonical = group_counts[0][0]
        for name in group:
            name_mapping[name] = canonical

    def apply_mapping(value: Any) -> Any:
        if pd.isna(value):
            return value
        return name_mapping.get(value, value)

    return series.apply(apply_mapping)



def extract_jobdescription_info(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Extract fixed-term status, working hours, and salary from a jobdescription column.

    Parameters
    ----------
    series:
        Series containing textual job descriptions.

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        Tuple of (fixedterm, workinghours, salary) series.
    """

    def parse_terms(value: Any) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if pd.isna(value):
            return None, None, None

        text = str(value)
        lower = text.lower()

        fixedterm: Optional[str] = None
        workinghours: Optional[str] = None
        salary: Optional[str] = None

        # Fixed-term detection
        match = re.search(r"\bunbefristet\b", lower)
        if match:
            fixedterm = text[match.start():match.end()]
        else:
            if not re.search(r"\bbefristete\s+erhöhung\b", lower):
                pattern = (
                    r"\bbefristet\b"
                    r"(?:"
                        r"(?:\s+(?:bis|für|auf|als|zum|zur|in|mit|voraussichtlich|zunächst))?"
                        r"(?:\s+(?:zum|den|die|der|das|ein|eine|einen|zwei|drei))?"
                        r"(?:"
                            r"(?:\s+\d{1,2}\.?\s*(?:januar|februar|märz|april|mai|juni|juli|august|september|oktober|november|dezember|\d{1,2})\.?\s*\d{4})|"
                            r"(?:\s+\d+\s+(?:jahr|jahre|monat|monate|woche|wochen))|"
                            r"(?:\s+\w*vertretung)|"
                            r"(?:\s+elternzeit)|"
                            r"(?:\s+mutterschutz)|"
                            r"(?:\s+der\s+option\s+einer\s+unbefristeten\s+weiterbeschäftigung)"
                        r")*"
                    r")?"
                )
                
                full_pattern = pattern + r"(?![\s\w]*(?:tv-?l|e\s?\d+|vollzeit|teilzeit|stunden|arbeitszeit|wochenarbeitszeit|stellenausschreibung))"
                
                match = re.search(full_pattern, lower, re.IGNORECASE)
                if match:
                    extracted = text[match.start():match.end()]
                    extracted = re.sub(r'\s+(?:zu|in|mit|und|oder)\s*$', '', extracted, flags=re.IGNORECASE)
                    extracted = re.sub(r'\s+bis\s*$', '', extracted, flags=re.IGNORECASE)
                    extracted = re.sub(r'\s+tv-?l.*$', '', extracted, flags=re.IGNORECASE)
                    extracted = re.sub(r'\s+e\s?\d+.*$', '', extracted, flags=re.IGNORECASE)
                    extracted = re.sub(r'\s+(?:vollzeit|teilzeit|in\s+vollzeit|in\s+teilzeit).*$', '', extracted, flags=re.IGNORECASE)
                    extracted = re.sub(r'\s+\d+\s*(?:%|prozent|stunden|std\.).*$', '', extracted, flags=re.IGNORECASE)
                    
                    fixedterm = extracted.strip()

        # Working hours detection
        if re.search(r"\bvollzeit\b", lower):
            workinghours = "Vollzeit"
        elif re.search(r"\bteilzeit\b", lower):
            workinghours = "Teilzeit"
        else:
            hours_week_match = re.search(
                r"(\d+(?:[,\.]\d+)?)\s*(?:stunden|std\.?)\s*(?:/|pro|je|die)?\s*(?:woche|wo\.)",
                lower
            )
            if hours_week_match:
                try:
                    hours = float(hours_week_match.group(1).replace(',', '.'))
                    workinghours = "Vollzeit" if hours >= 36 else "Teilzeit"
                except ValueError:
                    pass
            else:
                hours_month_match = re.search(
                    r"(\d+(?:[,\.]\d+)?)\s*(?:stunden|std\.?)\s*(?:/|pro|je|im)?\s*monat",
                    lower
                )
                if hours_month_match:
                    try:
                        hours_month = float(hours_month_match.group(1).replace(',', '.'))
                        hours_week = hours_month / 4.33
                        workinghours = "Vollzeit" if hours_week >= 36 else "Teilzeit"
                    except ValueError:
                        pass
                else:
                    percent_match = re.search(r"(\d+(?:[,\.]\d+)?)\s*%", lower)
                    if percent_match:
                        try:
                            percent = float(percent_match.group(1).replace(',', '.'))
                            workinghours = "Vollzeit" if percent >= 90 else "Teilzeit"
                        except ValueError:
                            pass
                    else:
                        bare_hours_match = re.search(
                            r"(?:mit|von|umfang|arbeitszeit|wochenarbeitszeit)\s*(?:von)?\s*(\d+(?:[,\.]\d+)?)\s*(?:stunden|std\.?)",
                            lower
                        )
                        if bare_hours_match:
                            try:
                                hours = float(bare_hours_match.group(1).replace(',', '.'))
                                if hours <= 60:
                                    workinghours = "Vollzeit" if hours >= 36 else "Teilzeit"
                            except ValueError:
                                pass

        # Salary detection
        # Priority 1: Look for TV-L, TVöD, TV-öD patterns with valid pay grades
        tv_match = re.search(
            r"(?:tv-?[löd]|tv[öo]d|tv-?h)\s*(?:e|eg)\s*(\d{1,2}[üÜ]?(?:\s*[ab])?)",
            lower
        )
        if tv_match:
            grade = tv_match.group(1)
            grade_num = re.match(r"(\d+)", grade)
            if grade_num and 1 <= int(grade_num.group(1)) <= 15:
                full_match = text[tv_match.start():tv_match.end()]
                salary = re.sub(r'\s+', ' ', full_match.upper())
                salary = re.sub(r'TV-?[LÖD]+', 'TV-L', salary)
                salary = re.sub(r'TV[ÖO]D', 'TVöD', salary)
        
        if not salary:
            # Priority 2: Look for E/EG groups (without TV-L prefix)
            e_match = re.search(
                r"\b(?:e|eg)\s*(\d{1,2}[üÜ]?(?:\s*[ab])?)\b",
                lower
            )
            if e_match:
                grade = e_match.group(1)
                grade_num = re.match(r"(\d+)", grade)
                if grade_num and 1 <= int(grade_num.group(1)) <= 15:
                    after_pos = e_match.end()
                    if after_pos < len(lower):
                        following_text = lower[after_pos:min(after_pos + 10, len(lower))]
                        if not re.match(r'\d', following_text):
                            full_match = text[e_match.start():e_match.end()]
                            if 'eg' in lower[e_match.start():e_match.end()]:
                                salary = f"EG {grade.upper()}"
                            else:
                                salary = f"E {grade.upper()}"
        
        if not salary:
            # Priority 3: Look for A groups (Beamtenbesoldung)
            a_match = re.search(
                r"\ba\s*(\d{1,2})\b",
                lower
            )
            if a_match:
                grade = a_match.group(1)
                if 1 <= int(grade) <= 16:
                    salary = f"A {grade}"
        
        if not salary:
            # Priority 4: IMPROVED Euro amount detection
            
            # Special case: 450 Euro-Job / 450-Euro-Job (Minijob)
            minijob_match = re.search(r"450\s*(?:€|eur|euro)(?:-job)?", lower)
            if minijob_match:
                salary = "450 Euro-Job"
            else:
                euro_patterns = [
                    # Pattern for amounts with space instead of decimal point (e.g., "8 50 Euro")
                    r"(\d{1,2})\s+(\d{2})\s*(?:€|eur|euro)\s*(?:/|pro|je)?\s*(?:stunde|std\.?|monat|jahr|woche|tag)",
                    # X.XXX €/Euro per time period
                    r"(\d{1,5}(?:[.,]\d{1,3})?)\s*(?:€|eur|euro)\s*(?:/|pro|je)?\s*(?:stunde|std\.?|monat|jahr|woche|tag)",
                    # X €/h, X €/Std
                    r"(\d{1,5}(?:[.,]\d{1,3})?)\s*€\s*/\s*(?:h|std\.?)",
                    # Stundenlohn/Monatslohn X €
                    r"(?:stunden|monats|jahres|wochen)lohn\s*(?:von)?\s*(\d{1,5}(?:[.,]\d{1,3})?)\s*(?:€|eur|euro)",
                    # Simple amount without time period (but be careful)
                    r"(\d{3,5})\s*(?:€|eur|euro)(?![/-]job)",
                ]
                
                for pattern in euro_patterns:
                    euro_match = re.search(pattern, lower)
                    if euro_match:
                        # Extract the original text
                        start = euro_match.start()
                        end = euro_match.end()
                        
                        # For the special case with space instead of decimal
                        if pattern == euro_patterns[0] and euro_match.group(1) and euro_match.group(2):
                            # Reconstruct with decimal point
                            amount = f"{euro_match.group(1)},{euro_match.group(2)}"
                            remaining = lower[euro_match.start() + len(euro_match.group(1)) + 1 + len(euro_match.group(2)):euro_match.end()]
                            salary = f"{amount}{text[euro_match.start() + len(euro_match.group(1)) + 1 + len(euro_match.group(2)):euro_match.end()]}"
                        else:
                            salary = text[start:end]
                        
                        # Clean up the extracted salary
                        salary = re.sub(r'\s+', ' ', salary).strip()
                        
                        # Remove trailing punctuation (comma, semicolon, dash) but keep dash in "Euro-Job"
                        salary = re.sub(r'[,;-]+$', '', salary)
                        
                        # Standardize Euro notation
                        salary = salary.replace('eur ', 'Euro ')
                        salary = salary.replace('€', 'Euro')
                        
                        break

        return fixedterm, workinghours, salary

    results = series.apply(parse_terms)
    fixedterm_series = pd.Series([r[0] for r in results], index=series.index)
    workinghours_series = pd.Series([r[1] for r in results], index=series.index)
    salary_series = pd.Series([r[2] for r in results], index=series.index)
    return fixedterm_series, workinghours_series, salary_series


def clean_dataframe(
    df: pd.DataFrame,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    region_mapping: Optional[pd.DataFrame] = None,
    debug: bool = False,
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

    _status = make_status_printer(status_callback)

    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    total = len(object_cols)
    current_progress = 0.0

    def _progress(value: float) -> None:
        nonlocal current_progress
        current_progress = value
        if progress_callback:
            progress_callback(current_progress)

    # Fetch license plate mapping once at the beginning
    license_plate_map: dict[str, str] = {}
    if "location" in cleaned.columns:
        _status("Lade Kfz-Kennzeichen...")
        license_plate_map = fetch_german_license_plates(status_callback)
    _progress(10.0)

    # Extract PLZ from company field before other processing
    if "company" in cleaned.columns:
        _status("Extrahiere Postleitzahlen aus Firmennamen...")
        company_cleaned, plz_extracted, city_extracted = extract_plz_from_company(
            cleaned["company"]
        )
        cleaned["company"] = company_cleaned

        # Add PLZ column (or update if it already exists)
        if "plz" not in cleaned.columns:
            cleaned["plz"] = plz_extracted
        else:
            # If PLZ column exists, only fill empty values
            cleaned["plz"] = cleaned["plz"].fillna(plz_extracted)

        # Use extracted city to correct location if needed
        if "location" in cleaned.columns:
            company_like = r"\b(?:gmbh|mbh|ag|kg|gbr|e\.v\.|generalvikariat)\b"
            mask = (
                cleaned["location"].isna()
                | cleaned["location"].str.contains(company_like, case=False, na=True)
            ) & city_extracted.notna()
            cleaned.loc[mask, "location"] = city_extracted[mask]
        elif city_extracted.notna().any():
            cleaned["location"] = city_extracted

        _status(f"PLZ extrahiert: {plz_extracted.notna().sum()} Einträge gefunden")
    _progress(20.0)

    for idx, col in enumerate(object_cols, start=1):
        _status(f"Bereinige Spalte '{col}'...")

        # HTML cleaning (original functionality)
        cleaned[col] = cleaned[col].apply(
            lambda x: html.unescape(re.sub(r"<.*?>", "", str(x))) if pd.notna(x) else x
        )

        # License plate resolution for location column
        if col == "location" and license_plate_map:
            _status("Ersetze Kfz-Kennzeichen durch Städtenamen...")
            cleaned[col] = resolve_license_plates_in_series(cleaned[col], license_plate_map)

        # Company name cleaning and standardization (after PLZ extraction)
        if col == "company":
            _status("Bereinige und standardisiere Firmennamen...")
            cleaned[col] = clean_company_field(cleaned[col])

        if total:
            # Reserve 60% for column processing
            _progress(20.0 + (idx / total * 60.0))

    if "jobdescription" in cleaned.columns:
        _status("Extrahiere Angaben aus Stellenbeschreibungen...")
        fixedterm, workinghours, salary = extract_jobdescription_info(cleaned["jobdescription"])
        cleaned["fixedterm"] = fixedterm
        cleaned["workinghours"] = workinghours
        cleaned["salary"] = salary
    _progress(90.0)

    # Enrich with region information
    if region_mapping is None:
        if _status:
            _status("Lade Regionszuordnung...")
        try:
            region_mapping = load_region_mapping()
        except FileNotFoundError:
            region_mapping = None

    if "region" not in cleaned.columns:
        cleaned["region"] = None

    if region_mapping is not None:
        _status("Bestimme Regionen...")

        # 1) coordinate-based lookup using nearest neighbours
        if {
            "geo_lat",
            "geo_lon",
        }.issubset(cleaned.columns) and {
            "geo_lat",
            "geo_lon",
        }.issubset(region_mapping.columns):
            mapping_coords = region_mapping.dropna(subset=["geo_lat", "geo_lon"])
            if not mapping_coords.empty:
                nbrs = NearestNeighbors(n_neighbors=1)
                nbrs.fit(mapping_coords[["geo_lat", "geo_lon"]])
                coord_mask = cleaned["geo_lat"].notna() & cleaned["geo_lon"].notna()
                if coord_mask.any():
                    distances, indices = nbrs.kneighbors(
                        cleaned.loc[coord_mask, ["geo_lat", "geo_lon"]]
                    )
                    regions = mapping_coords.reset_index(drop=True)
                    cleaned.loc[coord_mask, "region"] = [
                        regions.iloc[idx]["region"] if dist <= 0.005 else None
                        for dist, idx in zip(distances.ravel(), indices.ravel())
                    ]

        # 2) match by location if still missing
        if "location" in cleaned.columns and "location" in region_mapping.columns:
            location_groups = {
                loc: grp for loc, grp in region_mapping.groupby("location")
            }
            all_locations = list(location_groups.keys())

            def _resolve_location(row):
                loc = row["location"]
                group = location_groups.get(loc)
                if group is None:
                    match = process.extractOne(str(loc), all_locations, processor=None, score_cutoff=90)
                    if match:
                        group = location_groups.get(match[0])
                if group is None:
                    return None
                if len(group) == 1:
                    return group.iloc[0]["region"]
                if {
                    "geo_lat",
                    "geo_lon",
                }.issubset(group.columns) and pd.notna(row.get("geo_lat")) and pd.notna(
                    row.get("geo_lon")
                ):
                    distances = (
                        (group["geo_lat"] - row["geo_lat"]) ** 2
                        + (group["geo_lon"] - row["geo_lon"]) ** 2
                    )
                    return group.loc[distances.idxmin(), "region"]
                return None

            missing_mask = cleaned["region"].isna() & cleaned["location"].notna()
            if missing_mask.any():
                cleaned.loc[missing_mask, "region"] = cleaned.loc[missing_mask].apply(
                    _resolve_location, axis=1
                )

    # 3) API fallback using coordinates
    if {"geo_lat", "geo_lon"}.issubset(cleaned.columns):
        coord_mask = (
            cleaned["region"].isna()
            & cleaned["geo_lat"].notna()
            & cleaned["geo_lon"].notna()
        )
        if coord_mask.any():
            cleaned.loc[coord_mask, "region"] = cleaned.loc[
                coord_mask, ["geo_lat", "geo_lon"]
            ].apply(
                lambda row: region_from_coordinates(
                    row["geo_lat"], row["geo_lon"]
                ),
                axis=1,
            )

    if debug:
        missing_region = cleaned[cleaned["region"].isna()]
        for _, row in missing_region.iterrows():
            reasons: list[str] = []
            loc_val = row.get("location")
            if pd.isna(loc_val) or str(loc_val).strip() == "":
                reasons.append("missing location")
            if pd.isna(row.get("geo_lat")) or pd.isna(row.get("geo_lon")):
                reasons.append("missing coordinates")
            else:
                reasons.append("no region match or API failure")
            jobid = row.get("jobid", _)
            print(
                "[region debug] jobid=%s location=%s lat=%s lon=%s reason=%s"
                % (
                    jobid,
                    loc_val,
                    row.get("geo_lat"),
                    row.get("geo_lon"),
                    ", ".join(reasons) if reasons else "unknown",
                )
            )

    _progress(100.0)
    return cleaned


def generate_candidate_pairs(
    df: pd.DataFrame,
    fuzzy_fields: set[str],
    n_neighbors: int = 5,
) -> set[tuple[int, int]]:
    """Generate potential duplicate candidate pairs using TF-IDF and nearest neighbors.

    This reduces the number of comparisons from O(n^2) to roughly O(n * k),
    where ``k`` is the number of neighbors considered per record.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the records of a group that should be compared.
    fuzzy_fields : set[str]
        Columns used to build the textual representation for similarity search.
    n_neighbors : int, optional
        Number of nearest neighbours to retrieve per record. Includes the record
        itself, so the number of candidates per row is ``n_neighbors - 1``.

    Returns
    -------
    set[tuple[int, int]]
        Set of index pairs (i, j) with ``i < j`` representing candidate
        comparisons. If TF-IDF cannot be computed (e.g. empty vocabulary), the
        function falls back to a full pairwise comparison.
    """

    size = len(df)
    if size < 2:
        return set()

    if not fuzzy_fields:
        return {(i, j) for i in range(size) for j in range(i + 1, size)}

    # Build corpus from fuzzy fields
    corpus = df[list(fuzzy_fields)].fillna("").agg(" ".join, axis=1)

    try:
        vectorizer = TfidfVectorizer(
            min_df=1, max_df=0.95, ngram_range=(1, 2)
        )
        matrix = vectorizer.fit_transform(corpus)

        k = min(n_neighbors, size)
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix, n_neighbors=k)

        pairs: set[tuple[int, int]] = set()
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                if i < j:
                    pairs.add((i, j))
                elif j < i:
                    pairs.add((j, i))
        return pairs
    except ValueError as e:
        if "empty vocabulary" in str(e) or "After pruning, no terms remain" in str(e):
            # Fallback: if TF-IDF fails (e.g., empty vocabulary), compare all pairs
            return {(i, j) for i in range(size) for j in range(i + 1, size)}
        raise
    except Exception:
        # On unexpected errors (e.g., MemoryError), fall back to full comparison
        return {(i, j) for i in range(size) for j in range(i + 1, size)}


def find_fuzzy_duplicates(
    dataframe: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: int = 90,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find and remove duplicate rows using very strict fuzzy matching.

    This version implements much stricter criteria to minimize false positives:
    - Higher similarity thresholds
    - Strict salary grade matching  
    - Company name validation
    - Location validation
    - Additional semantic checks
    """
    columns = columns or DEDUPLICATE_COLUMNS
    
    exact_fields = {"jobtype", "insttype", "country", "fixedterm", "workinghours"}
    fuzzy_fields = {"jobdescription", "company", "location", "plz", "salary"}
    numeric_fields = {"geo_lat", "geo_lon"}
    
    # Filter to only include columns that exist in the dataframe
    exact_fields = {f for f in exact_fields if f in dataframe.columns}
    fuzzy_fields = {f for f in fuzzy_fields if f in dataframe.columns}
    numeric_fields = {f for f in numeric_fields if f in dataframe.columns}
    
    drop_indices: set[int] = set()
    pairs: dict[int, list[tuple[int, int]]] = {}
    
    # Stage 1: Group by exact match fields
    if exact_fields:
        exact_key_cols = list(exact_fields)
        grouping_keys = dataframe[exact_key_cols].fillna("").astype(str).agg("_".join, axis=1)
        groups = dataframe.groupby(grouping_keys).indices
    else:
        groups = {"all": dataframe.index.tolist()}
    
    total_comparisons = sum(len(group_indices) for group_indices in groups.values())
    processed = 0
    
    # Helper function to check salary compatibility
    def are_salaries_compatible(sal1, sal2):
        if pd.isna(sal1) or pd.isna(sal2):
            return True
        
        s1, s2 = str(sal1).upper(), str(sal2).upper()
        
        # Extract numeric grade from salary strings
        def extract_grade(salary_str: str) -> tuple[Optional[int], Optional[str]]:
            # Look for patterns like E9, EG9, E 9, EG 9B, etc.
            match = re.search(r'E\s*G?\s*(\d+)\s*([A-Z]*)', salary_str)
            if match:
                return int(match.group(1)), match.group(2)
            return None, None
        
        grade1, suffix1 = extract_grade(s1)
        grade2, suffix2 = extract_grade(s2)
        
        if grade1 is not None and grade2 is not None:
            # Grades must be identical or differ by max 1 level
            if abs(grade1 - grade2) > 1:
                return False
            # If same grade, suffixes should be similar
            if grade1 == grade2 and suffix1 != suffix2:
                # Allow some flexibility (e.g., E9 vs E9B)
                if suffix1 and suffix2:
                    if suffix1 not in suffix2 and suffix2 not in suffix1:
                        return False
                # At least one suffix is empty, allow compatibility
        
        return True
    
    # Helper function to check company compatibility
    def are_companies_compatible(comp1, comp2):
        if pd.isna(comp1) or pd.isna(comp2):
            return True
        
        c1, c2 = str(comp1).lower(), str(comp2).lower()
        
        # Remove common words that might vary
        stop_words = ['gmbh', 'ag', 'ev', 'e.v.', 'bibliothek', 'stadtbibliothek', 
                     'universitätsbibliothek', 'landesbibliothek', 'stadt', 'universität']
        
        for word in stop_words:
            c1 = c1.replace(word, '')
            c2 = c2.replace(word, '')
        
        # Clean whitespace
        c1 = ' '.join(c1.split())
        c2 = ' '.join(c2.split())
        
        # Must have significant overlap
        if len(c1) > 10 and len(c2) > 10:
            score = int(fuzz.token_sort_ratio(c1, c2))
            return score >= 85
        
        return True
    
    # Helper function to check location compatibility  
    def are_locations_compatible(loc1, loc2):
        if pd.isna(loc1) or pd.isna(loc2):
            return True
        
        l1, l2 = str(loc1).lower().strip(), str(loc2).lower().strip()
        
        # Exact match or very high similarity required
        if l1 == l2:
            return True
        
        score = int(fuzz.ratio(l1, l2))
        return score >= 90
    
    # Stage 2: Within each group, generate candidate pairs and apply very strict fuzzy matching
    for group_indices in groups.values():
        group_size = len(group_indices)
        if group_size < 2:
            processed += group_size
            if progress_callback:
                progress_callback((processed / total_comparisons) * 100)
            continue

        group_df = dataframe.iloc[group_indices]
        candidate_pairs = generate_candidate_pairs(group_df, fuzzy_fields)

        for i, j in candidate_pairs:
            global_i = group_indices[i]
            global_j = group_indices[j]

            if global_i in drop_indices or global_j in drop_indices:
                continue

            row_i = dataframe.iloc[global_i]
            row_j = dataframe.iloc[global_j]

            # Pre-checks: These must pass for any potential duplicate

            # 1. Salary compatibility check
            if 'salary' in dataframe.columns:
                if not are_salaries_compatible(row_i.get('salary'), row_j.get('salary')):
                    continue

            # 2. Company compatibility check
            if 'company' in dataframe.columns:
                if not are_companies_compatible(row_i.get('company'), row_j.get('company')):
                    continue

            # 3. Location compatibility check
            if 'location' in dataframe.columns:
                if not are_locations_compatible(row_i.get('location'), row_j.get('location')):
                    continue

            # 4. Job description must be VERY similar (95%+)
            if 'jobdescription' in dataframe.columns:
                desc1 = str(row_i.get('jobdescription', '')).lower()
                desc2 = str(row_j.get('jobdescription', '')).lower()
                if len(desc1) > 10 and len(desc2) > 10:
                    desc_score = int(fuzz.token_sort_ratio(desc1, desc2))
                    if desc_score < 95:
                        continue

            # Now apply fuzzy matching with very high standards
            scores: list[float] = []
            match = True

            # Check fuzzy fields with very high standards
            for col in fuzzy_fields:
                if col not in dataframe.columns:
                    continue
                val_i = row_i.get(col)
                val_j = row_j.get(col)

                if pd.isna(val_i) and pd.isna(val_j):
                    continue
                if pd.isna(val_i) or pd.isna(val_j):
                    match = False
                    break

                score = int(fuzz.token_sort_ratio(str(val_i), str(val_j)))

                # Very high threshold for each field
                min_score = 95 if col == 'jobdescription' else 90
                if score < min_score:
                    match = False
                    break
                scores.append(score)

            if not match:
                continue

            # Check numeric fields (geo coordinates) with tighter tolerance
            for col in numeric_fields:
                if col not in dataframe.columns:
                    continue
                val_i = row_i.get(col)
                val_j = row_j.get(col)

                if pd.isna(val_i) and pd.isna(val_j):
                    continue
                if pd.isna(val_i) or pd.isna(val_j):
                    match = False
                    break

                try:
                    diff = abs(float(val_i) - float(val_j))
                except (ValueError, TypeError):
                    match = False
                    break

                # Very strict geographic tolerance: 0.01 degrees (~1km)
                max_diff = 0.01
                score = max(0.0, 100 * (1 - min(diff / max_diff, 1)))
                if score < 95:
                    match = False
                    break
                scores.append(score)

            if not match:
                continue

            # Additional final checks
            # Check if job descriptions have substantially different key terms
            if 'jobdescription' in dataframe.columns:
                desc1 = str(row_i.get('jobdescription', '')).lower()
                desc2 = str(row_j.get('jobdescription', '')).lower()

                # Look for contradictory terms
                contradictory_pairs = [
                    ('vollzeit', 'teilzeit'),
                    ('befristet', 'unbefristet'),
                    ('ausbildung', 'arbeitsstelle'),
                    ('leitung', 'mitarbeiter'),
                    ('e13', 'e9'), ('e12', 'e8'), ('e11', 'e7'),
                    ('e10', 'e6'), ('e9', 'e5'),  # Different pay grades
                ]

                for term1, term2 in contradictory_pairs:
                    if (term1 in desc1 and term2 in desc2) or (term2 in desc1 and term1 in desc2):
                        match = False
                        break

            if not match:
                continue

            # Calculate final probability - must be very high
            if scores:
                avg_score = sum(scores) / len(scores)
                probability = int(avg_score)
            else:
                probability = 95

            # Only accept near-perfect matches
            if probability < 95:
                continue

            # Determine which record to keep
            nonnull_i = row_i.count()
            nonnull_j = row_j.count()

            if nonnull_i >= nonnull_j:
                keep_idx, drop_idx = global_i, global_j
            else:
                keep_idx, drop_idx = global_j, global_i

            drop_indices.add(drop_idx)
            pairs.setdefault(keep_idx, []).append((drop_idx, probability))
        
        processed += len(group_indices)
        if progress_callback:
            progress_callback((processed / total_comparisons) * 100)
    
    if progress_callback:
        progress_callback(100)
    
    # Build result DataFrames
    duplicate_rows = []
    for pair_id, (keep_idx, drop_pairs) in enumerate(pairs.items()):
        keep_row = dataframe.iloc[keep_idx].copy()
        keep_row["keep"] = True
        keep_row["pair_id"] = pair_id
        keep_row["orig_index"] = keep_idx
        keep_scores = [score for _, score in drop_pairs]
        keep_row["probability"] = max(keep_scores) if keep_scores else 100
        duplicate_rows.append(keep_row)
        
        for drop_idx, score in drop_pairs:
            drop_row = dataframe.iloc[drop_idx].copy()
            drop_row["keep"] = False
            drop_row["pair_id"] = pair_id
            drop_row["orig_index"] = drop_idx
            drop_row["probability"] = score
            duplicate_rows.append(drop_row)
    
    duplicates = pd.DataFrame(duplicate_rows)
    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            ["probability", "pair_id", "keep"], ascending=[False, True, False]
        )
        duplicates = duplicates.reset_index(drop=True)
    
    cleaned = dataframe.drop(index=list(drop_indices)).reset_index(drop=True)
    return cleaned, duplicates


def prepare_duplicates_export(duplicates: pd.DataFrame) -> pd.DataFrame:
    """Prepare duplicates dataframe for CSV export.

    Adds a ``duplicate_of`` column referencing the ``orig_index`` of the
    record that should be kept for each pair. Rows marked with ``keep=True``
    will have ``duplicate_of`` set to :data:`pandas.NA`.

    Parameters
    ----------
    duplicates : pd.DataFrame
        Dataframe as returned by :func:`find_fuzzy_duplicates`.

    Returns
    -------
    pd.DataFrame
        A copy of ``duplicates`` with the additional ``duplicate_of`` column.
    """

    if duplicates.empty:
        return duplicates.copy()

    pair_to_original = (
        duplicates[duplicates["keep"]].set_index("pair_id")["orig_index"]
    )
    export_df = duplicates.copy()
    export_df["duplicate_of"] = export_df["pair_id"].map(pair_to_original)
    export_df.loc[export_df["keep"], "duplicate_of"] = pd.NA
    return export_df


def format_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with column names formatted for export.

    Column labels are converted to upper case and wrapped with leading and
    trailing underscores as required by the export specification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns should be formatted.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with renamed columns.
    """

    renamed = {col: f"_{col.upper()}_" for col in df.columns}
    return df.rename(columns=renamed)

