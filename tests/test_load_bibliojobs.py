from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from load_bibliojobs import load_bibliojobs


def write_csv(tmp_path, df):
    path = tmp_path / "bibliojobs.csv"
    csv_str = df.to_csv(sep="\t", index=False)
    path.write_text(csv_str.replace("\t", "_§_"), encoding="utf-8")
    return path


def test_load_bibliojobs_parses_and_casts(tmp_path):
    df = pd.DataFrame({
        "_JobID_": ["1"],
        "date": ["01-02-2020"],
        "geo_lat": ["52.5"],
        "geo_lon": ["13.4"],
    })
    path = write_csv(tmp_path, df)

    loaded = load_bibliojobs(path)

    assert list(loaded.columns) == ["jobid", "date", "geo_lat", "geo_lon"]
    assert loaded["jobid"].dtype == "Int64"
    assert pd.api.types.is_float_dtype(loaded["geo_lat"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def test_load_bibliojobs_warns_non_numeric_jobid(tmp_path, caplog):
    df = pd.DataFrame({
        "_JobID_": ["abc"],
        "date": ["01-02-2020"],
        "geo_lat": ["52"],
        "geo_lon": ["13"],
    })
    path = write_csv(tmp_path, df)

    with caplog.at_level("WARNING"):
        loaded = load_bibliojobs(path)

    assert "Non-numeric jobid values encountered" in caplog.text
    assert pd.isna(loaded.loc[0, "jobid"])


def test_load_bibliojobs_warns_invalid_date(tmp_path, caplog):
    df = pd.DataFrame({
        "_JobID_": ["1"],
        "date": ["2020/01/01"],
        "geo_lat": ["52"],
        "geo_lon": ["13"],
    })
    path = write_csv(tmp_path, df)

    with caplog.at_level("WARNING"):
        loaded = load_bibliojobs(path, date_format="%d-%m-%Y")

    assert "Date values not matching format" in caplog.text
    assert pd.isna(loaded.loc[0, "date"])


def test_load_bibliojobs_respects_custom_date_format(tmp_path):
    df = pd.DataFrame({
        "_JobID_": ["1"],
        "date": ["2020/01/02"],
        "geo_lat": ["52"],
        "geo_lon": ["13"],
    })
    path = write_csv(tmp_path, df)

    loaded = load_bibliojobs(path, date_format="%Y/%m/%d")

    assert loaded.loc[0, "date"] == pd.Timestamp(2020, 1, 2)


def test_load_bibliojobs_duplicate_columns_error(tmp_path):
    df = pd.DataFrame({
        "_JobID_": ["1"],
        "_Dup_": ["a"],
        "_dup_": ["b"],
        "date": ["01-02-2020"],
        "geo_lat": ["52"],
        "geo_lon": ["13"],
    })
    path = write_csv(tmp_path, df)

    with pytest.raises(ValueError, match="Duplicate column names"):
        load_bibliojobs(path)


def test_load_bibliojobs_reports_progress(tmp_path):
    df = pd.DataFrame({
        "_JobID_": [str(i) for i in range(2500)],
        "date": ["01-02-2020"] * 2500,
        "geo_lat": ["52"] * 2500,
        "geo_lon": ["13"] * 2500,
    })
    path = write_csv(tmp_path, df)

    calls = []
    load_bibliojobs(path, progress_callback=lambda v: calls.append(v))

    assert len(calls) >= 2
    assert calls == sorted(calls)
    assert calls[-1] == pytest.approx(100.0)
    assert all(0 < c <= 100 for c in calls)


def test_load_bibliojobs_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.csv"
    with pytest.raises(FileNotFoundError, match="CSV-Datei nicht gefunden"):
        load_bibliojobs(missing)
