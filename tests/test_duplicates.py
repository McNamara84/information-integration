import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from cleaning import (
    DEDUPLICATE_COLUMNS,
    find_fuzzy_duplicates,
    prepare_duplicates_export,
    clean_dataframe,
)
from load_bibliojobs import load_bibliojobs

CSV_PATH = "bibliojobs_raw.csv"


def load_clean_subset(jobids: list[int]) -> pd.DataFrame:
    df = load_bibliojobs(CSV_PATH)
    subset = df[df["jobid"].isin(jobids)].reset_index(drop=True)
    return clean_dataframe(subset)


@pytest.fixture
def duplicates_df() -> pd.DataFrame:
    jobids = [12687, 13048, 13235, 13958]
    df = load_clean_subset(jobids)
    _, duplicates = find_fuzzy_duplicates(df, DEDUPLICATE_COLUMNS, threshold=80)
    return duplicates


def test_find_fuzzy_duplicates_new_rules(duplicates_df: pd.DataFrame) -> None:
    pair = duplicates_df[duplicates_df["jobid"].isin([12687, 13048])]
    assert len(pair) == 2
    assert pair["probability"].eq(100).all()
    assert {True, False} == set(pair["keep"])


def test_no_duplicates_when_jobtype_differs() -> None:
    df = load_clean_subset([102390, 106191])
    cleaned, duplicates = find_fuzzy_duplicates(
        df, DEDUPLICATE_COLUMNS, threshold=80
    )
    assert duplicates.empty
    assert len(cleaned) == 2


def test_prepare_duplicates_export_adds_reference(duplicates_df: pd.DataFrame) -> None:
    export_df = prepare_duplicates_export(duplicates_df)
    assert "duplicate_of" in export_df.columns
    keep_row = export_df[export_df["keep"]].iloc[0]
    dup_row = export_df[~export_df["keep"]].iloc[0]
    assert pd.isna(keep_row["duplicate_of"])
    assert dup_row["duplicate_of"] == keep_row["orig_index"]
