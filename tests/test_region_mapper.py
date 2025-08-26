import pathlib
from unittest.mock import Mock, patch

from region_mapper import load_region_mapping, region_from_coordinates


def test_load_region_mapping_normalizes_berlin():
    mapping = load_region_mapping(pathlib.Path(__file__).resolve().parent.parent / "ort_bundesland.sql")
    assert "State of Berlin" not in mapping["region"].unique()
    berlin_regions = mapping[mapping["location"] == "Berlin"]["region"].unique()
    assert "Berlin" in berlin_regions


def test_region_from_coordinates_normalizes_api_response():
    region_from_coordinates.cache_clear()
    fake_response = Mock()
    fake_response.ok = True
    fake_response.json.return_value = {"address": {"state": "State of Berlin"}}
    with patch("region_mapper.requests.get", return_value=fake_response):
        result = region_from_coordinates(52.5, 13.4)
    assert result == "Berlin"


def test_region_from_coordinates_retries_after_rate_limit():
    region_from_coordinates.cache_clear()
    rate_limited = Mock()
    rate_limited.status_code = 429
    rate_limited.ok = False

    success = Mock()
    success.status_code = 200
    success.ok = True
    success.json.return_value = {"address": {"state": "State of Berlin"}}

    with patch("region_mapper.requests.get", side_effect=[rate_limited, success]) as mock_get, \
         patch("region_mapper.time.sleep") as mock_sleep:
        mock_sleep.return_value = None
        result = region_from_coordinates(52.5, 13.4)

    assert result == "Berlin"
    assert mock_get.call_count == 2
    assert mock_sleep.called


def test_region_from_coordinates_falls_back_to_secondary_service():
    region_from_coordinates.cache_clear()
    failure = Mock()
    failure.ok = False
    failure.status_code = 500

    success = Mock()
    success.ok = True
    success.json.return_value = {"address": {"state": "State of Berlin"}}

    with patch("region_mapper.requests.get", side_effect=[failure, success]) as mock_get:
        result = region_from_coordinates(52.5, 13.4)

    assert result == "Berlin"
    assert mock_get.call_count == 2
