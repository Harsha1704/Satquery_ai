"""Regression tests for analysis-context publication guards."""

from backend.app.services.analysis_service import _execution_source_consistency


def test_execution_source_accepts_planned_collection_id_or_label():
    expected = {
        "before": {"year": 2020, "collection_id": "COPERNICUS/S2_SR_HARMONIZED", "collection_label": "Sentinel-2 SR"},
        "after": {"year": 2024, "collection_id": "COPERNICUS/S2_SR_HARMONIZED", "collection_label": "Sentinel-2 SR"},
    }
    actual = [
        {"year": 2020, "collection_id": "Sentinel-2 SR"},
        {"year": 2024, "collection_id": "Sentinel-2 SR"},
    ]

    result = _execution_source_consistency(expected, actual)

    assert result["matched"] is True
    assert result["status"] == "matched"


def test_execution_source_rejects_an_unplanned_date_or_collection():
    expected = {
        "before": {"year": 2020, "collection_id": "COPERNICUS/S2_SR_HARMONIZED"},
        "after": {"year": 2024, "collection_id": "COPERNICUS/S2_SR_HARMONIZED"},
    }
    actual = [
        {"year": 2020, "collection_id": "COPERNICUS/S2_SR_HARMONIZED"},
        {"year": 2025, "collection_id": "LANDSAT/LC08/C02/T1_L2"},
    ]

    result = _execution_source_consistency(expected, actual)

    assert result["matched"] is False
    assert result["status"] == "changed"


def test_map_client_contains_a_stale_run_guard_and_aoi_snapshot():
    source = open("frontend/static/js/map.js", encoding="utf-8").read()

    assert "let activeRunToken = 0" in source
    assert "const submittedAoi = JSON.parse(JSON.stringify(currentAoi))" in source
    assert "if (runToken !== activeRunToken) return" in source
