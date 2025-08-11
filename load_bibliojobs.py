"""Utilities to load and clean the Bibliojobs dataset."""
import logging
import os
from typing import Callable, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

def load_bibliojobs(
    path: Union[str, bytes] = "bibliojobs_raw.csv",
    *,
    date_format: str = "%d-%m-%Y",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Read the bibliojobs CSV using the custom `_§_` delimiter and clean it.

    Parameters
    ----------
    path:
        Path to the raw CSV file. Defaults to ``"bibliojobs_raw.csv"`` in the
        repository root.
    date_format:
        Expected ``strftime`` format of the ``date`` column. Defaults to
        ``"%d-%m-%Y"``.
    progress_callback:
        Optional function that receives the percentage of bytes read from the
        file as a ``float`` between 0 and 100.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalised column names and corrected dtypes.
    """
    # Read with explicit UTF-8 encoding and the `_§_` delimiter. If a
    # ``progress_callback`` is supplied the file is read in chunks so that the
    # caller can be informed about the progress of the operation.
    if progress_callback:
        chunks = []
        total_size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8") as handle:
            reader = pd.read_csv(
                handle,
                sep="_§_",
                engine="python",
                chunksize=1000,
            )
            for chunk in reader:
                chunks.append(chunk)
                progress_callback(handle.buffer.tell() / total_size * 100)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep="_§_", engine="python", encoding="utf-8")

    # Remove leading/trailing underscores and standardise column names.
    normalised = df.columns.str.strip("_").str.lower()
    duplicates = normalised[normalised.duplicated()]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate column names after normalisation: "
            f"{duplicates.tolist()}"
        )
    df.columns = normalised

    # Convert numeric columns.
    jobid_numeric = pd.to_numeric(df["jobid"], errors="coerce")
    invalid_jobids = df.loc[jobid_numeric.isna(), "jobid"].dropna().unique()
    if invalid_jobids.size:
        logger.warning(
            "Non-numeric jobid values encountered: %s",
            invalid_jobids.tolist(),
        )
    df["jobid"] = jobid_numeric.astype("Int64")
    df["geo_lat"] = pd.to_numeric(df["geo_lat"], errors="coerce")
    df["geo_lon"] = pd.to_numeric(df["geo_lon"], errors="coerce")

    # Parse the date column.
    raw_dates = df["date"].copy()
    df["date"] = pd.to_datetime(raw_dates, format=date_format, errors="coerce")
    invalid_dates = raw_dates[df["date"].isna() & raw_dates.notna()].unique()
    if invalid_dates.size:
        logger.warning(
            "Date values not matching format %s encountered: %s",
            date_format,
            invalid_dates.tolist(),
        )

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataframe = load_bibliojobs()
    print(dataframe.dtypes)
