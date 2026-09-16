"""Real-raster regressions for invalid pixels, AOI topology and evidence grids."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from query_engine.runtime import configure_raster_runtime
from query_engine.schemas import AOI
from query_engine.policy import QueryError
from ai.multispectral.indices import normalized_difference, index_statistics
from ai.multispectral.analyzer import MultispectralAnalyzer
from ai.data.multispectral_loader import MultispectralLoader
from ai.temporal import TemporalPair, TemporalChangeEngine
from backend.app.services.analysis_service import _generate_task_change_layer

configure_raster_runtime()


def raster(path, data, *, transform=None, nodata=-9999, mask=None):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1],
                       count=data.shape[0], dtype="float32", crs="EPSG:4326",
                       transform=transform or from_origin(0, 1, .01, .01), nodata=nodata) as dst:
        dst.write(data.astype("float32"))
        if mask is not None:
            dst.write_mask(mask)
    return path


def test_invalid_spectral_pixels_are_unavailable():
    values = normalized_difference(np.array([.8, 0, np.nan, np.inf]), np.array([.2, 0, .2, .2]))
    assert values[0] == pytest.approx(.6)
    assert np.isnan(values[1:]).all()
    assert index_statistics(values)["valid_pixel_count"] == 1
    with pytest.raises(ValueError, match="no finite"):
        index_statistics(values[1:])


def test_spectral_summary_uses_only_valid_pixels(tmp_path):
    data = np.full((12, 2, 2), .2, dtype="float32")
    data[7] = .8  # B08 in the documented twelve-band order
    data[:, 0, 0] = -9999
    image = MultispectralLoader().load(raster(tmp_path / "scene.tif", data))
    assert np.isnan(image.data[:, 0, 0]).all()
    summary = MultispectralAnalyzer().summary(image.data)
    assert summary["ndvi"]["valid_pixel_count"] == 3
    assert summary["vegetation_coverage_percent"] == 100


@pytest.mark.parametrize("hole", [
    [(2, 2), (3, 2), (3, 3), (2, 2)],
    [(.2, .2), (.8, .8), (.2, .8), (.8, .2), (.2, .2)],
])
def test_invalid_polygon_holes_rejected(hole):
    with pytest.raises(ValueError):
        AOI(coordinates=[[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], hole])


def test_tiny_valid_polygon_is_not_expanded_or_rejected():
    coordinates = [[(77, 12), (77.0000001, 12), (77.0000001, 12.0000001), (77, 12)]]
    assert AOI(coordinates=coordinates).coordinates == coordinates


def test_temporal_internal_mask_is_excluded(tmp_path):
    data = np.full((1, 30, 30), .2, dtype="float32")
    first = raster(tmp_path / "before.tif", data)
    after = data.copy()
    after[:, 8:22, 8:22] = .8
    mask = np.full((30, 30), 255, dtype="uint8")
    mask[8:22, 8:22] = 0
    second = raster(tmp_path / "after.tif", after, mask=mask)
    result = TemporalChangeEngine().analyze(TemporalPair(first, second))
    assert result["statistics"]["changed_percentage"] == 0
    assert result["statistics"]["valid_pixel_percentage"] == pytest.approx((900 - 196) / 9)


def test_temporal_uniform_change_is_not_reported_unchanged(tmp_path):
    data = np.full((1, 10, 10), .2, dtype="float32")
    pair = TemporalPair(raster(tmp_path / "before.tif", data), raster(tmp_path / "after.tif", data * 4))
    with pytest.raises(ValueError, match="UNRESOLVED_UNIFORM_CHANGE"):
        TemporalChangeEngine().analyze(pair)


def evidence_pair(tmp_path, *, shifted=False):
    directory = tmp_path / "imagery"
    directory.mkdir()
    before = np.zeros((3, 10, 10), dtype="float32")
    after = before.copy()
    after[:, :, 5:] = .8  # change entirely outside the selected half
    raster(directory / "gee_indices_2020_test.tif", before)
    raster(directory / "gee_indices_2025_test.tif", after,
           transform=from_origin(.01 if shifted else 0, 1, .01, .01))
    plan = SimpleNamespace(parsed=SimpleNamespace(query="vegetation change", targets=[], years=[2020, 2025]))
    request = SimpleNamespace(aoi=AOI(coordinates=[[(0, .9), (.05, .9), (.05, 1), (0, 1), (0, .9)]]))
    return plan, request


def test_evidence_masks_requested_aoi_and_retains_raster_bounds(tmp_path):
    plan, request = evidence_pair(tmp_path)
    layer = _generate_task_change_layer(tmp_path, plan, request)
    assert layer["changed_percentage"] == 0
    assert layer["valid_pixel_count"] == 50
    assert layer["bounds_wgs84"] == [0, .9, .1, 1]
    assert layer["affected_area_km2"] is None
    assert (tmp_path / layer["artifact"]).is_file()


def test_evidence_rejects_shifted_grids_with_identical_dimensions(tmp_path):
    plan, request = evidence_pair(tmp_path, shifted=True)
    with pytest.raises(QueryError, match="matching CRS"):
        _generate_task_change_layer(tmp_path, plan, request)


def test_spatial_validator_rejects_unrelated_georeferenced_evidence(tmp_path):
    from query_engine.spatial_context import build_analysis_context, evidence_records, validate_spatial_consistency
    plan, request = evidence_pair(tmp_path)
    context = build_analysis_context(analysis_id="ana_test", query="vegetation", aoi=request.aoi,
                                     inputs={"image_path": tmp_path / "imagery/gee_indices_2020_test.tif"})
    record = evidence_records(analysis_id="ana_test", context=context, artifacts=["wrong.png"])[0]
    record.update(georeferenced=True, crs="EPSG:4326", bounds=[30, 30, 31, 31])
    assert validate_spatial_consistency(context, [record])["status"] == "FAIL"
    context["result"]["bounds"] = [30, 30, 31, 31]
    assert "RESULT_ROI_MISMATCH" in validate_spatial_consistency(context, [])["failures"]


def test_earth_engine_connectivity_failure_is_actionable(monkeypatch):
    import gee_temporal
    from query_engine.planner import QueryPlanner
    from query_engine.schemas import AnalysisRequest
    from query_engine.worker import run

    monkeypatch.setattr(gee_temporal, "is_configured", lambda: True)
    monkeypatch.setattr(gee_temporal, "plan_temporal_fetch", lambda *_: (_ for _ in ()).throw(ConnectionError("offline")))
    request = AnalysisRequest(query="Compare 2020 and 2025", aoi={"type": "Polygon", "coordinates": [[[77, 12], [77.01, 12], [77.01, 12.01], [77, 12]]]})
    plan = QueryPlanner().plan(request)
    with pytest.raises(QueryError) as error:
        run({"plan": plan.model_dump(mode="json"), "request": request.model_dump(mode="json"), "inputs": {}})
    assert error.value.code == "imagery_service_unavailable"
    assert error.value.status_code == 503
