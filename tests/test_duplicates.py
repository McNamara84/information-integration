import pandas as pd

from cleaning import (
    DEDUPLICATE_COLUMNS,
    find_fuzzy_duplicates,
    prepare_duplicates_export,
)


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


def test_prepare_duplicates_export_adds_reference():
    df = pd.DataFrame(
        {
            "jobdescription": [
                "Manage books and media",
                "Manage books and archives",
            ],
            "jobtype": ["Librarian", "Librarian"],
            "company": ["ABC GmbH", "ABCD GmbH"],
            "insttype": ["Public", "Public"],
            "location": ["Berlin", "Berlin"],
            "country": ["DE", "DE"],
            "geo_lat": [52.52, 52.5205],
            "geo_lon": [13.405, 13.4055],
            "plz": ["10115", "10116"],
            "fixedterm": ["No", "No"],
            "workinghours": ["Vollzeit", "Vollzeit"],
            "salary": ["E9", "E9"],
        }
    )

    _, duplicates = find_fuzzy_duplicates(
        df, DEDUPLICATE_COLUMNS, threshold=80
    )

    export_df = prepare_duplicates_export(duplicates)
    assert "duplicate_of" in export_df.columns
    keep_row = export_df[export_df["keep"]].iloc[0]
    dup_row = export_df[~export_df["keep"]].iloc[0]
    assert pd.isna(keep_row["duplicate_of"])
    assert dup_row["duplicate_of"] == keep_row["orig_index"]
