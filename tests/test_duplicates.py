import pathlib
import sys
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from cleaning import find_fuzzy_duplicates


def test_find_fuzzy_duplicates():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "A.B.C. GmbH", "XYZ AG"],
        "location": ["Berlin", "Berlin", "Hamburg"],
        "jobtype": ["Librarian", "Librarian", "Archivist"],
        "jobdescription": ["Manage books", "manage books", "Archive documents"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        ["company", "location", "jobtype", "jobdescription"],
        threshold=90,
    )

    assert len(duplicates) == 1
    assert duplicates.iloc[0]["company"] == "A.B.C. GmbH"
    assert len(cleaned) == 2
    assert "XYZ AG" in cleaned["company"].values
