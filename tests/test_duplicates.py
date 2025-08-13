import pandas as pd
from fuzzywuzzy import fuzz

from cleaning import DEDUPLICATE_COLUMNS, find_fuzzy_duplicates


def test_find_fuzzy_duplicates():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "A.B.C. GmbH", "XYZ AG"],
        "location": ["Berlin", "Berlin", "Hamburg"],
        "jobtype": ["Librarian", "Librarian", "Archivist"],
        "jobdescription": ["Manage books", "manage books", "Archive documents"],
        "fixedterm": [None, None, None],
        "workinghours": ["Vollzeit", "Vollzeit", "Teilzeit"],
        "salary": ["E 9", "E 9", "E 7"],
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
    assert "probability" in duplicates.columns
    assert duplicates["probability"].between(0, 100).all()


def test_no_false_duplicates_with_different_company():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "XYZ GmbH"],
        "location": ["Berlin", "Berlin"],
        "jobtype": ["Librarian", "Librarian"],
        "jobdescription": ["Manage books", "Manage books"],
        "fixedterm": [None, None],
        "workinghours": ["Vollzeit", "Vollzeit"],
        "salary": ["E 9", "E 9"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert duplicates.empty
    assert len(cleaned) == 2


def test_no_duplicates_when_workinghours_differs():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "A.B.C. GmbH"],
        "location": ["Berlin", "Berlin"],
        "jobtype": ["Librarian", "Librarian"],
        "jobdescription": ["Manage books", "manage books"],
        "fixedterm": [None, None],
        "workinghours": ["Vollzeit", "Teilzeit"],
        "salary": ["E 9", "E 9"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert duplicates.empty
    assert len(cleaned) == 2


def test_multiple_duplicates_are_grouped():
    df = pd.DataFrame(
        {
            "company": ["ABC GmbH", "A.B.C. GmbH", "A B C GmbH", "XYZ AG"],
            "location": ["Berlin", "Berlin", "Berlin", "Hamburg"],
            "jobtype": ["Librarian", "Librarian", "Librarian", "Archivist"],
            "jobdescription": [
                "Manage books",
                "manage books",
                "Manage books",
                "Archive documents",
            ],
            "fixedterm": [None, None, None, None],
            "workinghours": ["Vollzeit", "Vollzeit", "Vollzeit", "Teilzeit"],
            "salary": ["E 9", "E 9", "E 9", "E 7"],
        }
    )

    _, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert len(duplicates) == 3
    assert duplicates["pair_id"].nunique() == 1
    assert duplicates.iloc[0]["keep"]
    assert (~duplicates.iloc[1:]["keep"]).all()


def test_no_false_duplicates_with_missing_vs_string_value():
    df = pd.DataFrame({
        "company": ["ABC GmbH", "ABC GmbH"],
        "location": ["Berlin", "Berlin"],
        "jobtype": ["Librarian", "Librarian"],
        "jobdescription": ["Manage books", "Manage books"],
        "fixedterm": [None, None],
        "workinghours": ["Vollzeit", "Vollzeit"],
        "salary": [None, "None provided"],
    })

    cleaned, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=90,
    )

    assert duplicates.empty
    assert len(cleaned) == 2


def test_duplicates_sorted_by_probability_desc():
    df = pd.DataFrame(
        {
            "company": ["ABC GmbH", "A.B.C. GmbH", "XYZ GmbH", "XZ GmbH"],
            "location": ["Berlin", "Berlin", "Hamburg", "Hamburg"],
            "jobtype": ["Librarian", "Librarian", "Archivist", "Archivist"],
            "jobdescription": [
                "Manage books",
                "manage books",
                "Archive documents",
                "Archive docs",
            ],
            "fixedterm": [None, None, None, None],
            "workinghours": ["Vollzeit", "Vollzeit", "Teilzeit", "Teilzeit"],
            "salary": ["E 9", "E 9", "E 7", "E 7"],
        }
    )

    _, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=80,
    )

    probs = duplicates["probability"].to_list()
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
    assert len(set(probs)) > 1


def test_probability_matches_each_drop_row():
    df = pd.DataFrame(
        {
            "company": ["ABC GmbH", "ABCD GmbH", "ABC GMBH"],
            "location": ["Berlin", "Berlin", "Berlin"],
            "jobtype": ["Librarian", "Librarian", "Librarian"],
            "jobdescription": ["Manage books", "Manage books", "Manage books"],
            "fixedterm": [None, None, None],
            "workinghours": ["Vollzeit", "Vollzeit", "Vollzeit"],
            "salary": ["E 9", "E 9", "E 9"],
        }
    )

    _, duplicates = find_fuzzy_duplicates(
        df,
        DEDUPLICATE_COLUMNS,
        threshold=80,
    )

    drop_rows = duplicates[~duplicates["keep"]]
    prob_map = dict(zip(drop_rows["company"], drop_rows["probability"]))
    expected_abcd = [
        fuzz.token_set_ratio("ABC GmbH", "ABCD GmbH"),
        fuzz.token_set_ratio("Manage books", "Manage books"),
        fuzz.token_set_ratio("Berlin", "Berlin"),
        fuzz.token_set_ratio("Librarian", "Librarian"),
        100,
        fuzz.token_set_ratio("Vollzeit", "Vollzeit"),
        fuzz.token_set_ratio("E 9", "E 9"),
    ]
    expected_gmbh = [
        fuzz.token_set_ratio("ABC GmbH", "ABC GMBH"),
        fuzz.token_set_ratio("Manage books", "Manage books"),
        fuzz.token_set_ratio("Berlin", "Berlin"),
        fuzz.token_set_ratio("Librarian", "Librarian"),
        100,
        fuzz.token_set_ratio("Vollzeit", "Vollzeit"),
        fuzz.token_set_ratio("E 9", "E 9"),
    ]
    expected1 = int(sum(expected_abcd) / len(expected_abcd))
    expected2 = int(sum(expected_gmbh) / len(expected_gmbh))
    assert prob_map["ABCD GmbH"] == expected1
    assert prob_map["ABC GMBH"] == expected2
