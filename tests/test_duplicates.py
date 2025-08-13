import pandas as pd

from cleaning import DEDUPLICATE_COLUMNS, find_fuzzy_duplicates


def test_find_fuzzy_duplicates():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "A.B.C. GmbH", "XYZ AG"],
        "location": ["Berlin", "Berlin", "Hamburg"],
        "jobtype": ["Librarian", "Librarian", "Archivist"],
        "jobdescription": ["Manage books", "manage books", "Archive documents"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert len(duplicates) == 2
    assert duplicates.iloc[0]["company"] == "ABC GmbH"
    assert duplicates.iloc[0]["keep"]
    assert duplicates.iloc[1]["company"] == "A.B.C. GmbH"
    assert not duplicates.iloc[1]["keep"]
    assert len(cleaned) == 2
    assert "XYZ AG" in cleaned["company"].values


def test_no_false_duplicates_with_different_company():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "XYZ GmbH"],
        "location": ["Berlin", "Berlin"],
        "jobtype": ["Librarian", "Librarian"],
        "jobdescription": ["Manage books", "Manage books"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert duplicates.empty
    assert len(cleaned) == 2
