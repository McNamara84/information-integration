import pathlib
import sys

import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from cleaning import clean_dataframe


def test_clean_dataframe_html_unescape_and_strip():
    df = pd.DataFrame({
        "a": ["AT&amp;T", "<b>Bold</b>", None]
    })
    cleaned = clean_dataframe(df)
    assert cleaned["a"].iloc[0] == "AT&T"
    assert cleaned["a"].iloc[1] == "Bold"
    assert pd.isna(cleaned["a"].iloc[2])


def test_clean_dataframe_replace_location(monkeypatch):
    df = pd.DataFrame({"location": ["MZ", "Darmstadt", "TR"]})

    def fake_get(url, params=None, headers=None, timeout=10):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                code = params["query"].split('P395 "')[1].split('"')[0]
                mapping = {"MZ": "Mainz", "TR": "Trier"}
                name = mapping.get(code)
                bindings = (
                    [{"itemLabel": {"value": name}}]
                    if name
                    else []
                )
                return {"results": {"bindings": bindings}}

        return FakeResponse()

    monkeypatch.setattr("cleaning.requests.get", fake_get)
    cleaned = clean_dataframe(df)
    assert list(cleaned["location"]) == ["Mainz", "Darmstadt", "Trier"]
