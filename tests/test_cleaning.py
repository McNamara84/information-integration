import pathlib
import sys
import pytest
from unittest.mock import patch, Mock
import requests

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from cleaning import clean_dataframe, fetch_german_license_plates, resolve_license_plates_in_series

def test_clean_dataframe_html_unescape_and_strip():
    df = pd.DataFrame({
        "a": ["AT&amp;T", "<b>Bold</b>", None]
    })
    cleaned = clean_dataframe(df)
    assert cleaned["a"].iloc[0] == "AT&T"
    assert cleaned["a"].iloc[1] == "Bold"
    assert pd.isna(cleaned["a"].iloc[2])


def test_fetch_german_license_plates_real_api():
    """Test the real API call to Wikidata."""
    license_plates = fetch_german_license_plates()
    
    # Should return a dictionary
    assert isinstance(license_plates, dict)
    
    # If successful, should contain some well-known German license plates
    if license_plates:  # Only check if API call was successful
        # Check for some common German license plates
        common_plates = ['B', 'M', 'HH', 'K', 'F', 'S', 'D']
        found_common = any(plate in license_plates for plate in common_plates)
        assert found_common, f"Expected at least one common plate, got: {list(license_plates.keys())[:10]}"
        
        # Verify format: keys should be uppercase letters and represent 1-3 characters
        # after normalizing ASCII umlaut sequences (AE/OE/UE -> Ä/Ö/Ü)
        for plate_code in license_plates.keys():
            assert isinstance(plate_code, str)
            assert plate_code.isupper()
            assert plate_code.isalpha()
            normalized_code = (
                plate_code
                .replace("AE", "Ä")
                .replace("OE", "Ö")
                .replace("UE", "Ü")
            )
            assert 1 <= len(normalized_code) <= 3
        
        # Verify values are non-empty strings
        for place_name in license_plates.values():
            assert isinstance(place_name, str)
            assert len(place_name) > 0


def test_fetch_german_license_plates_contains_umlauts():
    """Ensure umlaut license plates are resolved from cache/API."""
    license_plates = fetch_german_license_plates()
    assert license_plates.get("WÜ") == "Würzburg"
    assert license_plates.get("FÜ") == "Fürth"


def test_clean_dataframe_resolves_umlaut_license_plates():
    """Location cleaning should replace license plates with umlauts."""
    df = pd.DataFrame({"location": ["WÜ", "FÜ", None]})
    cleaned = clean_dataframe(df)
    assert cleaned["location"].tolist() == ["Würzburg", "Fürth", None]

def test_resolve_license_plates_in_series():
    """Test license plate resolution in pandas series."""
    license_plate_map = {
        "B": "Berlin",
        "MZ": "Mainz", 
        "HH": "Hamburg",
        "AM": "Amberg"  # Add AM to test the Frankfurt am Main case
    }
    
    series = pd.Series([
        "B",                    # Exact match -> should be replaced
        "MZ",                   # Exact match -> should be replaced
        "b",                    # Case insensitive -> should be replaced
        "AM",                   # Standalone AM -> should be replaced
        "Frankfurt",            # No change
        "Frankfurt am Main",    # Should NOT be replaced (am should stay as is)
        "Berlin Mitte",         # Should NOT be replaced (not a standalone license plate)
        "  B  ",               # With whitespace -> should be replaced
        "HH-City",             # Should NOT be replaced (contains additional text)
        None,                   # No change
        ""                      # No change
    ])
    
    result = resolve_license_plates_in_series(series, license_plate_map)
    
    expected = pd.Series([
        "Berlin",               # B replaced
        "Mainz",                # MZ replaced
        "Berlin",               # b replaced
        "Amberg",               # AM replaced
        "Frankfurt",            # unchanged
        "Frankfurt am Main",    # unchanged (am NOT replaced)
        "Berlin Mitte",         # unchanged
        "Berlin",               # B replaced (whitespace stripped)
        "HH-City",              # unchanged
        None,                   # unchanged
        ""                      # unchanged
    ])
    
    pd.testing.assert_series_equal(result, expected)


def test_resolve_license_plates_no_partial_replacement():
    """Test that license plates in longer texts are not replaced."""
    license_plate_map = {
        "AM": "Amberg",
        "IN": "Ingolstadt",
        "AN": "Ansbach"
    }
    
    series = pd.Series([
        "Frankfurt am Main",
        "Bad Ischl in Austria", 
        "Rothenburg ob der Tauber an der Romantischen Straße",
        "AM",  # This should be replaced
        "IN",  # This should be replaced
        "AN"   # This should be replaced
    ])
    
    result = resolve_license_plates_in_series(series, license_plate_map)
    
    expected = pd.Series([
        "Frankfurt am Main",     # am should NOT be replaced
        "Bad Ischl in Austria",  # in should NOT be replaced
        "Rothenburg ob der Tauber an der Romantischen Straße",  # an should NOT be replaced
        "Amberg",                # AM should be replaced
        "Ingolstadt",            # IN should be replaced  
        "Ansbach"                # AN should be replaced
    ])
    
    pd.testing.assert_series_equal(result, expected)


def test_resolve_license_plates_with_whitespace():
    """Test license plate resolution with various whitespace scenarios."""
    license_plate_map = {
        "B": "Berlin",
        "HH": "Hamburg"
    }
    
    series = pd.Series([
        "B",         # No whitespace
        " B ",       # Spaces around
        "\tB\t",     # Tabs around
        "\nB\n",     # Newlines around
        "  HH  ",    # Multiple spaces
        "B ",        # Trailing space
        " B",        # Leading space
    ])
    
    result = resolve_license_plates_in_series(series, license_plate_map)
    
    expected = pd.Series([
        "Berlin",
        "Berlin", 
        "Berlin",
        "Berlin",
        "Hamburg",
        "Berlin",
        "Berlin"
    ])
    
    pd.testing.assert_series_equal(result, expected)


def test_resolve_license_plates_empty_map():
    """Test license plate resolution with empty mapping."""
    series = pd.Series(["B", "MZ", "Frankfurt"])
    result = resolve_license_plates_in_series(series, {})
    
    # Should return unchanged series
    pd.testing.assert_series_equal(result, series)


def test_clean_dataframe_with_location_column():
    """Test cleaning dataframe with license plate resolution."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        mock_fetch.return_value = {"B": "Berlin", "MZ": "Mainz"}
        
        df = pd.DataFrame({
            "location": ["B", "MZ", "Frankfurt"],
            "other": ["<b>Test</b>", "Normal", "AT&amp;T"]
        })
        
        cleaned = clean_dataframe(df)
        
        # License plates should be resolved
        assert cleaned["location"].iloc[0] == "Berlin"
        assert cleaned["location"].iloc[1] == "Mainz" 
        assert cleaned["location"].iloc[2] == "Frankfurt"
        
        # HTML should still be cleaned
        assert cleaned["other"].iloc[0] == "Test"
        assert cleaned["other"].iloc[2] == "AT&T"


def test_clean_dataframe_without_location_column():
    """Test cleaning dataframe without location column - should not call API."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        df = pd.DataFrame({
            "other": ["<b>Test</b>", "Normal", "AT&amp;T"]
        })
        
        cleaned = clean_dataframe(df)
        
        # Should not have called the API
        mock_fetch.assert_not_called()
        
        # HTML should still be cleaned
        assert cleaned["other"].iloc[0] == "Test"
        assert cleaned["other"].iloc[2] == "AT&T"


def test_clean_dataframe_no_location_column_no_city():
    """No extracted city should avoid creating a new location column."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        df = pd.DataFrame({"company": ["Example Corp", "Another Inc"]})

        cleaned = clean_dataframe(df)

        mock_fetch.assert_not_called()
        assert "location" not in cleaned.columns


def test_clean_dataframe_create_location_from_company():
    """City extracted from company should create location column."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        mock_fetch.return_value = {}

        df = pd.DataFrame({
            "company": ["Stadt Münster, 48127 Münster", "Example Corp"]
        })

        cleaned = clean_dataframe(df)

        assert cleaned["location"].tolist() == ["Münster", None]


def test_clean_dataframe_location_from_company():
    """Locations resembling company names should be replaced by cities from company field."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        mock_fetch.return_value = {}

        df = pd.DataFrame(
            {
                "company": [
                    "Stadt Münster, 48127 Münster",
                    "Stadtbücherei Frankfurt, 60311 Frankfurt am Main",
                ],
                "location": ["Bischöfliches Generalvikariat", "MVB GmbH"],
            }
        )

        cleaned = clean_dataframe(df)

        assert cleaned["location"].tolist() == ["Münster", "Frankfurt am Main"]


def test_extract_jobdescription_info():
    df = pd.DataFrame(
        {
            "jobdescription": [
                "befristet bis 31.12.2025, Vollzeit, Vergütung nach TV-L E13",
                "unbefristet, Teilzeit (50%), E9 TV-L",
                None,
                "befristet, 20 Stunden/Woche, 3000 € monatlich",
            ]
        }
    )

    cleaned = clean_dataframe(df)

    assert cleaned["fixedterm"].tolist() == [
        "befristet bis 31.12.2025",
        "unbefristet",
        None,
        "befristet",
    ]
    assert cleaned["workinghours"].tolist() == [
        "Vollzeit",
        "Teilzeit",
        None,
        "Teilzeit",
    ]
    assert cleaned["salary"].tolist() == [
        "TV-L E13",
        "E 9",
        None,
        "3000 Euro monat",
    ]


def test_clean_dataframe_progress_callback():
    """Test that progress callback works with license plate resolution."""
    progress_calls = []
    
    def progress_callback(value):
        progress_calls.append(value)
    
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        mock_fetch.return_value = {"B": "Berlin"}
        
        df = pd.DataFrame({
            "location": ["B", "Frankfurt"],
            "other": ["Test1", "Test2"]
        })
        
        cleaned = clean_dataframe(df, progress_callback=progress_callback)
        
        # Should have progress calls
        assert len(progress_calls) > 0
        # First call should be 10% (after initial setup)
        assert progress_calls[0] == 10.0
        # Last call should be 100%
        assert progress_calls[-1] == 100.0
        # All calls should be between 0 and 100
        assert all(0 <= call <= 100 for call in progress_calls)


@pytest.mark.integration
def test_integration_clean_dataframe_real_api():
    """Integration test with real Wikidata API call."""
    df = pd.DataFrame({
        "location": ["B", "HH", "MZ", "Frankfurt", "Unknown"],
        "company": ["<b>Test &amp; Co</b>", "Normal Company", "AT&amp;T Corp", "Test", "Another"]
    })
    
    cleaned = clean_dataframe(df)
    
    # HTML should be cleaned
    assert cleaned["company"].iloc[0] == "Test & Co"
    assert cleaned["company"].iloc[2] == "AT&T Corp"
    
    # License plates should be resolved (if API call was successful)
    # We can't guarantee specific results since the API might change or be unavailable
    # But we can check that the function completed without errors
    assert len(cleaned) == len(df)
    # New 'plz' and 'region' columns should be added even if no postal codes were found
    assert list(cleaned.columns) == list(df.columns) + ["plz", "region"]
    assert cleaned["plz"].isna().all()
    
    # Check if any license plates were resolved
    original_location = df["location"].tolist()
    cleaned_location = cleaned["location"].tolist()
    
    # At least "Frankfurt" should remain unchanged
    assert "Frankfurt" in cleaned_location


def test_region_enrichment_fuzzy():
    """Ensure fuzzy region matching enriches locations with minor variations."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch:
        mock_fetch.return_value = {}

        df = pd.DataFrame({"location": ["Berlin", "Münchenn", "Hamburg"]})
        mapping = pd.DataFrame(
            {"location": ["Berlin", "München"], "region": ["Berlin", "Bayern"]}
        )

        cleaned = clean_dataframe(df, region_mapping=mapping)

        assert cleaned.loc[0, "region"] == "Berlin"  # exact match
        assert cleaned.loc[1, "region"] == "Bayern"  # fuzzy match
        assert pd.isna(cleaned.loc[2, "region"])  # no match


def test_region_enrichment_coordinates():
    """Coordinates should resolve to German federal states when names fail."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_coords.return_value = "Mecklenburg-Vorpommern"

        df = pd.DataFrame({
            "jobid": [19065],
            "location": ["Uckerland"],
            "geo_lat": [53.5264244],
            "geo_lon": [13.8020306],
        })

        cleaned = clean_dataframe(df)

        assert cleaned.loc[0, "region"] == "Mecklenburg-Vorpommern"
        mock_coords.assert_called_once()


def test_region_coordinates_match_sql_first():
    """Coordinate mapping via SQL should take precedence over API calls."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_coords.return_value = "ShouldNotBeCalled"

        df = pd.DataFrame({
            "location": ["Irgendwo"],
            "geo_lat": [52.54734],
            "geo_lon": [13.35594],
        })

        mapping = pd.DataFrame({
            "location": ["Wedding"],
            "region": ["Berlin"],
            "geo_lat": [52.54734],
            "geo_lon": [13.35594],
        })

        cleaned = clean_dataframe(df, region_mapping=mapping)

        assert cleaned.loc[0, "region"] == "Berlin"
        mock_coords.assert_not_called()


def test_region_ambiguous_locations_choose_nearest():
    """Ambiguous place names should pick the closest coordinate match."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_coords.return_value = "ShouldNotBeCalled"

        df = pd.DataFrame({
            "jobid": [1],
            "location": ["Berlin"],
            "geo_lat": [52.52],
            "geo_lon": [13.405],
        })

        mapping = pd.DataFrame(
            {
                "location": ["Berlin", "Berlin"],
                "region": ["Schleswig-Holstein", "Berlin"],
                "geo_lat": [54.0, 52.52],
                "geo_lon": [9.0, 13.405],
            }
        )

        cleaned = clean_dataframe(df, region_mapping=mapping)

        assert cleaned.loc[0, "region"] == "Berlin"
        mock_coords.assert_not_called()


def test_region_api_fallback_without_location_column():
    """When only coordinates are available, API fallback should determine the region."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.load_region_mapping') as mock_mapping, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_mapping.return_value = pd.DataFrame()
        mock_coords.return_value = "Berlin"

        df = pd.DataFrame({"geo_lat": [52.52], "geo_lon": [13.405]})

        cleaned = clean_dataframe(df)

        assert cleaned.loc[0, "region"] == "Berlin"
        mock_fetch.assert_not_called()
        mock_coords.assert_called_once()


def test_region_api_fallback_with_unmapped_location():
    """Fallback API should run when location is unmapped but coordinates exist."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.load_region_mapping') as mock_mapping, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_mapping.return_value = pd.DataFrame({
            'location': ['Hamburg'],
            'region': ['Hamburg'],
            'geo_lat': [53.55],
            'geo_lon': [10.0],
        })
        mock_coords.return_value = 'Berlin'

        df = pd.DataFrame({
            'location': ['Unbekannt'],
            'geo_lat': [52.52],
            'geo_lon': [13.405],
        })

        cleaned = clean_dataframe(df)

        assert cleaned.loc[0, 'region'] == 'Berlin'
        mock_coords.assert_called_once()


def test_region_location_with_qualifiers_resolves_via_mapping():
    """Locations with qualifiers should map to the correct region without API calls."""
    with patch('cleaning.fetch_german_license_plates') as mock_fetch, \
         patch('cleaning.region_from_coordinates') as mock_coords:
        mock_fetch.return_value = {}
        mock_coords.return_value = "ShouldNotBeCalled"

        df = pd.DataFrame({
            "location": ["Stolberg (Rheinland)", "Frankenthal (Pfalz)"],
            "geo_lat": [50.7891808, 49.5340652],
            "geo_lon": [6.2235961, 8.3520517],
        })

        mapping = pd.DataFrame(
            {
                "location": [
                    "Stolberg",
                    "Stolberg",
                    "Frankenthal",
                    "Frankenthal",
                ],
                "region": [
                    "Nordrhein-Westfalen",
                    "Sachsen-Anhalt",
                    "Rheinland-Pfalz",
                    "Thüringen",
                ],
                "geo_lat": [50.77368, 51.57426, 49.53414, 50.8778],
                "geo_lon": [6.22595, 10.95582, 8.35357, 12.01292],
            }
        )

        cleaned = clean_dataframe(df, region_mapping=mapping)

    assert cleaned["region"].tolist() == ["Nordrhein-Westfalen", "Rheinland-Pfalz"]
    mock_coords.assert_not_called()

