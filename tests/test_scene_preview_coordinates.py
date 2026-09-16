from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.app.services.analysis_service import _generate_scene_comparison


def test_scene_previews_preserve_source_bounds_but_comparison_is_not_a_map(tmp_path):
    imagery = tmp_path / "imagery"
    imagery.mkdir()
    for year in (2024, 2026):
        with rasterio.open(imagery / f"gee_{year}_test.tif", "w", driver="GTiff",
                           width=8, height=6, count=3, dtype="uint8", crs="EPSG:4326",
                           transform=from_origin(74, 30, .001, .001)) as dst:
            dst.write(np.full((3, 6, 8), 100, dtype=np.uint8))
    result = _generate_scene_comparison(tmp_path, SimpleNamespace(parsed=SimpleNamespace(years=[2024, 2026])), None)
    assert result is not None
    refs = result["scene_references"]
    assert len(refs) == 2
    assert refs[result["before_artifact"]]["bounds"] == [74, 29.994, 74.008, 30]
    assert refs[result["after_artifact"]]["source_artifact"] == "imagery/gee_2026_test.tif"
    assert result["comparison_artifact"] not in refs
