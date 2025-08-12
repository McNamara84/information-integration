import pandas as pd
from fuzzywuzzy import fuzz


def find_and_remove_duplicates(df: pd.DataFrame, threshold: int = 90, progress_callback=None):
    """Find duplicates based on fuzzy matching of selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : int, optional
        Similarity threshold for considering rows as duplicates (0-100).
    progress_callback : callable, optional
        Function called with progress percentage (0-100).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Tuple of (deduplicated dataframe, dataframe of duplicates).
    """
    columns = ["company", "location", "jobtype", "jobdescription"]
    if not set(columns).issubset(df.columns):
        raise ValueError("Dataframe must contain company, location, jobtype and jobdescription columns")

    combined = (
        df[columns]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    duplicate_indices = set()
    duplicates_rows = []
    total = len(df)
    for i in range(total):
        if progress_callback:
            progress_callback(int((i / max(total - 1, 1)) * 100))
        if i in duplicate_indices:
            continue
        for j in range(i + 1, total):
            if j in duplicate_indices:
                continue
            score = fuzz.token_set_ratio(combined.iloc[i], combined.iloc[j])
            if score >= threshold:
                duplicate_indices.add(j)
                duplicates_rows.append(df.iloc[j])
    if progress_callback:
        progress_callback(100)
    duplicates_df = pd.DataFrame(duplicates_rows, columns=df.columns)
    deduped_df = df.drop(index=list(duplicate_indices)).reset_index(drop=True)
    return deduped_df, duplicates_df.reset_index(drop=True)

