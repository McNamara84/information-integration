import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from cleaning import (
    DEDUPLICATE_COLUMNS,
    find_fuzzy_duplicates,
    prepare_duplicates_export,
)

CSV_PATH = "bibliojobs_raw.csv"
SEP = "_§_"

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=SEP, engine="python")
    df.columns = [c.strip("_").lower() for c in df.columns]
    for col in ["plz", "fixedterm", "workinghours", "salary"]:
        df[col] = pd.NA
    return df


@pytest.fixture
def duplicates_df() -> pd.DataFrame:
    df = load_raw().head(700)
    _, duplicates = find_fuzzy_duplicates(df, DEDUPLICATE_COLUMNS, threshold=80)
    return duplicates


def test_find_fuzzy_duplicates_new_rules(duplicates_df: pd.DataFrame) -> None:
    pair = duplicates_df[duplicates_df["pair_id"] == 0]
    assert len(pair) == 2
    assert pair["probability"].eq(100).all()
    assert {True, False} == set(pair["keep"])


def test_no_duplicates_when_jobtype_differs() -> None:
    df = load_raw()
    subset = df[df["jobid"].isin([102390, 106191])]
    cleaned, duplicates = find_fuzzy_duplicates(
        subset, DEDUPLICATE_COLUMNS, threshold=80
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
