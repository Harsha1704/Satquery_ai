"""QA regressions for legacy UI bi-temporal evidence contracts."""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from query_engine.runtime import configure_raster_runtime
from ai.router.orchestrator import SatQueryOrchestrator
from ai.validation import InputValidator

configure_raster_runtime()


def write_rgb(path: Path, data: np.ndarray, *, transform=None, mask=None):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=3,
                       dtype="uint8", crs="EPSG:4326", transform=transform or from_origin(77, 13, .001, .001), nodata=0) as dst:
        dst.write(data)
        if mask is not None:
            dst.write_mask(mask)
    return path


def test_visual_metrics_exactly_measure_a_known_25_percent_change(tmp_path):
    before = np.full((3, 10, 10), 10, dtype=np.uint8)
    after = before.copy()
    after[:, :5, :5] = 110
    first = write_rgb(tmp_path / "before_2017.tif", before)
    second = write_rgb(tmp_path / "after_2025.tif", after)
    result = SatQueryOrchestrator._assess_bitemporal_visual_support(str(first), str(second))
    assert result["available"] is True
    assert result["valid_pixel_count"] == 100
    assert result["pixels_difference_gt_20_pct"] == pytest.approx(25)
    assert result["mean_absolute_rgb_difference"] == pytest.approx(25)


def test_visual_metrics_exclude_nodata_from_denominators(tmp_path):
    before = np.full((3, 10, 10), 10, dtype=np.uint8)
    after = before.copy()
    after[:, :5, :5] = 110
    mask = np.full((10, 10), 255, dtype=np.uint8)
    mask[:5, :5] = 0
    first = write_rgb(tmp_path / "before.tif", before)
    second = write_rgb(tmp_path / "after.tif", after, mask=mask)
    result = SatQueryOrchestrator._assess_bitemporal_visual_support(str(first), str(second))
    assert result["valid_pixel_count"] == 75
    assert result["pixels_difference_gt_20_pct"] == 0
    assert result["mean_absolute_rgb_difference"] == 0


def test_identical_pair_has_zero_visual_difference(tmp_path):
    image = np.full((3, 8, 8), 40, dtype=np.uint8)
    first = write_rgb(tmp_path / "before.tif", image)
    second = write_rgb(tmp_path / "after.tif", image)
    result = SatQueryOrchestrator._assess_bitemporal_visual_support(str(first), str(second))
    assert result["mean_absolute_rgb_difference"] == 0
    assert result["pixels_difference_gt_20_pct"] == 0


def test_equal_sized_shifted_geotiffs_are_rejected_before_pixel_comparison(tmp_path):
    image = np.full((3, 10, 10), 40, dtype=np.uint8)
    first = write_rgb(tmp_path / "before_2017.tif", image)
    second = write_rgb(tmp_path / "after_2025.tif", image, transform=from_origin(77.001, 13, .001, .001))
    validation = InputValidator().validate_change_pair(str(first), str(second))
    assert validation["valid"] is False
    assert any(item["code"] == "GRID_MISALIGNMENT" for item in validation["errors"])
    result = SatQueryOrchestrator()._execute_change_detection("What changed?", str(first), str(second))
    assert result["success"] is False
    assert result["error"] == "SpatialInputValidationFailed"


def test_reversed_dated_inputs_are_rejected_before_inference(tmp_path):
    image = np.full((3, 8, 8), 40, dtype=np.uint8)
    first = write_rgb(tmp_path / "before_2025.tif", image)
    second = write_rgb(tmp_path / "after_2017.tif", image)
    result = SatQueryOrchestrator()._execute_change_detection("What changed?", str(first), str(second))
    assert result["success"] is False
    assert result["error"] == "InvalidTemporalOrder"


@pytest.mark.parametrize("unknown,passed", [(34.9, True), (35.0, True), (35.1, False)])
def test_semantic_unknown_threshold_is_explicit(unknown, passed):
    before = {"water": 100 - unknown, "vegetation": 0, "built_up": 0, "bare_land": 0, "unknown": unknown, "known_total": 100 - unknown}
    gate = SatQueryOrchestrator._evaluate_semantic_quality(before, before, {"water": 0, "vegetation": 0, "built_up": 0, "bare_land": 0})
    assert gate["passed"] is passed
    assert "above 35.0%" in gate["thresholds"]["semantics"]


def test_noida_visual_values_match_the_stored_scene_pair():
    root = Path("outputs/temporal_high_change/noida_airport_jewar")
    result = SatQueryOrchestrator._assess_bitemporal_visual_support(str(root / "noida_airport_jewar_2017.tif"), str(root / "noida_airport_jewar_2025.tif"))
    assert result["mean_absolute_rgb_difference"] == pytest.approx(31.0969, abs=0.0001)
    assert result["pixels_difference_gt_20_pct"] == pytest.approx(48.5345, abs=0.0001)
    assert result["valid_pixel_count"] > 0
