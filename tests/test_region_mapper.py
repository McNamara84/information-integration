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
