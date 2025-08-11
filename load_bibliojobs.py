"""Utilities to load and clean the Bibliojobs dataset."""
import pandas as pd
from typing import Union

def load_bibliojobs(path: Union[str, bytes] = "bibliojobs_raw.csv") -> pd.DataFrame:
    """Read the bibliojobs CSV using the custom `_§_` delimiter and clean it.

    Parameters
    ----------
    path:
        Path to the raw CSV file. Defaults to ``"bibliojobs_raw.csv"`` in the
        repository root.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalised column names and corrected dtypes.
    """
    # Read with explicit UTF-8 encoding and the `_§_` delimiter.
    df = pd.read_csv(path, sep="_§_", engine="python", encoding="utf-8")

    # Remove leading/trailing underscores and standardise column names.
    df.columns = df.columns.str.strip("_").str.lower()

    # Convert numeric columns.
    df["jobid"] = pd.to_numeric(df["jobid"], errors="coerce").astype("Int64")
    df["geo_lat"] = pd.to_numeric(df["geo_lat"], errors="coerce")
    df["geo_lon"] = pd.to_numeric(df["geo_lon"], errors="coerce")

    # Parse the date column.
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")

    return df


if __name__ == "__main__":
    dataframe = load_bibliojobs()
    print(dataframe.dtypes)
