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

    assert len(duplicates) == 1
    assert duplicates.iloc[0]["company"] == "A.B.C. GmbH"
    assert len(cleaned) == 2
    assert "XYZ AG" in cleaned["company"].values
