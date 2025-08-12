from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from deduplication import find_content_duplicates


def test_find_content_duplicates_basic():
    df = pd.DataFrame({
        "jobid": [1, 2, 3],
        "title": ["Bibliothekar", "Bibliothekarin", "Archiv"],
        "jobdescription": [
            "Leitung der Stadtbibliothek",
            "Leitung Stadtbibliothek",
            "Arbeiten im Archiv",
        ],
        "company": ["Stadtbibliothek Berlin", "Stadt-Bibliothek Berlin", "Archiv Berlin"],
        "location": ["Berlin", "Berlin", "Berlin"],
    })

    dup_df = find_content_duplicates(df, threshold=80)
    assert set(dup_df.index) == {0, 1}
    assert len(dup_df) == 2


def test_find_content_duplicates_none():
    df = pd.DataFrame({
        "jobid": [1, 2],
        "title": ["Alpha", "Beta"],
        "jobdescription": ["Erste Beschreibung", "Zweite Info"],
        "company": ["Firma Eins", "Unternehmen Zwei"],
        "location": ["Ort A", "Ort B"],
    })

    dup_df = find_content_duplicates(df, threshold=80)
    assert dup_df.empty


def test_find_content_duplicates_missing_columns():
    """Missing optional columns are ignored during duplicate detection."""
    df = pd.DataFrame({
        "jobid": [1, 2],
        # The "title" column is intentionally omitted here
        "jobdescription": ["Leitung der Stadtbibliothek", "Leitung Stadtbibliothek"],
        "company": ["Stadtbibliothek Berlin", "Stadt-Bibliothek Berlin"],
        "location": ["Berlin", "Berlin"],
    })

    dup_df = find_content_duplicates(df, threshold=80)
    assert set(dup_df.index) == {0, 1}
    assert "__dup_group" in dup_df.columns
