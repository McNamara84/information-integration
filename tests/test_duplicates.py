import pandas as pd

from cleaning import DEDUPLICATE_COLUMNS, find_fuzzy_duplicates


def test_find_fuzzy_duplicates_new_rules():
    df = pd.DataFrame(
        {
            "jobdescription": [
                "Manage books and media",
                "Manage books and archives",
                "Archive documents",
            ],
            "jobtype": ["Librarian", "Librarian", "Librarian"],
            "company": ["ABC GmbH", "ABCD GmbH", "XYZ AG"],
            "insttype": ["Public", "Public", "Public"],
            "location": ["Berlin", "Berlin", "Hamburg"],
            "country": ["DE", "DE", "DE"],
            "geo_lat": [52.52, 52.5205, 53.55],
            "geo_lon": [13.405, 13.4055, 9.993],
            "plz": ["10115", "10116", "20095"],
            "fixedterm": ["No", "No", "No"],
            "workinghours": ["Vollzeit", "Vollzeit", "Vollzeit"],
            "salary": ["E9", "E9", "E7"],
        }
    )

    cleaned, duplicates = find_fuzzy_duplicates(
        df, DEDUPLICATE_COLUMNS, threshold=80
    )

    assert len(duplicates) == 2
    assert duplicates.iloc[0]["keep"]
    assert not duplicates.iloc[1]["keep"]
    assert duplicates["probability"].between(50, 100).all()
    assert "XYZ AG" in cleaned["company"].values


def test_no_duplicates_when_jobtype_differs():
    df = pd.DataFrame(
        {
            "jobdescription": ["Manage books", "Manage books"],
            "jobtype": ["Librarian", "Archivist"],
            "company": ["ABC GmbH", "ABC GmbH"],
            "insttype": ["Public", "Public"],
            "location": ["Berlin", "Berlin"],
            "country": ["DE", "DE"],
            "geo_lat": [52.52, 52.5205],
            "geo_lon": [13.405, 13.4055],
            "plz": ["10115", "10115"],
            "fixedterm": ["No", "No"],
            "workinghours": ["Vollzeit", "Vollzeit"],
            "salary": ["E9", "E9"],
        }
    )

    cleaned, duplicates = find_fuzzy_duplicates(
        df, DEDUPLICATE_COLUMNS, threshold=80
    )

    assert duplicates.empty
    assert len(cleaned) == 2
