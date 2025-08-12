import re
from collections import defaultdict
from typing import Iterable, List

import pandas as pd
from fuzzywuzzy import fuzz


# Regular expression for non-word characters
_non_word_re = re.compile(r"\W+")


def _normalise(text: str) -> str:
    """Return a normalised representation of *text*.

    The normalisation removes non-word characters and collapses whitespace.
    ``None`` values are treated as empty strings.
    """
    if pd.isna(text):
        return ""
    return _non_word_re.sub(" ", str(text).lower()).strip()


def find_content_duplicates(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    *,
    threshold: int = 80,
) -> pd.DataFrame:
    """Return rows of *df* that are content duplicates.

    Two rows are considered duplicates if the concatenation of their selected
    ``columns`` is similar according to ``fuzz.token_set_ratio`` with a score of
    at least ``threshold`` (0-100).

    Parameters
    ----------
    df:
        The input DataFrame.
    columns:
        Iterable of column names to use for the similarity comparison. If
        ``None`` the default columns ``("title", "jobdescription", "company",
        "location")`` are used.
    threshold:
         Minimum similarity score (0-100) required to treat two rows as
         duplicates. Defaults to ``80``.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the duplicate rows. An additional column
        ``__dup_group`` is added which indicates duplicate group membership.
    """
    if columns is None:
        columns = ("title", "jobdescription", "company", "location")

    # Prepare normalised combined text for each row
    relevant = df.loc[:, columns].fillna("").map(_normalise)
    combined = relevant.agg(" ".join, axis=1)

    # Pairwise comparison to find similar rows
    n = len(df)
    groups: dict[int, int] = {}

    def find(x: int) -> int:
        # Disjoint-set find with path compression
        parent = groups.setdefault(x, x)
        if parent != x:
            groups[x] = find(parent)
        return groups[x]

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            groups[root_y] = root_x

    for i in range(n):
        text_i = combined.iloc[i]
        for j in range(i + 1, n):
            score = fuzz.token_set_ratio(text_i, combined.iloc[j])
            if score >= threshold:
                union(i, j)

    # Collect indices belonging to duplicate groups
    inverse: defaultdict[int, List[int]] = defaultdict(list)
    for idx in groups:
        root = find(idx)
        inverse[root].append(idx)

    dup_indices = [i for lst in inverse.values() if len(lst) > 1 for i in lst]
    if not dup_indices:
        return pd.DataFrame(columns=list(df.columns) + ["__dup_group"])

    result = df.loc[dup_indices].copy()
    result["__dup_group"] = [find(i) for i in dup_indices]
    return result
