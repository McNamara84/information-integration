import pathlib
import sys
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from duplicates import find_and_remove_duplicates


def test_find_and_remove_duplicates():
    df = pd.DataFrame({
        'company': ['M\u00fcller GmbH', 'Mueller GmbH', 'Schmidt AG'],
        'location': ['Berlin', 'Berlin', 'Hamburg'],
        'jobtype': ['Bibliothekar', 'Bibliothekar', 'Archiv'],
        'jobdescription': ['Verwaltung der Bibliothek', 'Verwaltung Bibliothek', 'Archivierung']
    })
    cleaned, duplicates = find_and_remove_duplicates(df, threshold=85)
    assert len(cleaned) == 2
    assert len(duplicates) == 1
    assert duplicates.iloc[0]['company'] == 'Mueller GmbH'

