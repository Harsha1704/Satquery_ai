import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

import frontend.app as frontend
from ai.report import SatQueryReportGenerator


class MapROITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="roi-test-", dir=frontend.ROOT / "outputs")
        self.directory = Path(self.temp.name).resolve()
        self.assertTrue(self.directory.is_relative_to(frontend.ROOT / "outputs"))
        self.addCleanup(self.temp.cleanup)
        for name in ("ROI_DIR", "REPORT_DIR", "UPLOAD_DIR", "MAP_PREVIEW_DIR"):
            directory = self.directory / name.lower()
            directory.mkdir()
            patcher = patch.object(frontend, name, directory)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(frontend, "report_generator", SatQueryReportGenerator(output_dir=str(frontend.REPORT_DIR)))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(frontend, "ANALYSIS_REGISTRY", {})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = frontend.app.test_client()

    def upload(self, filename="sentinel2_test.tif", query="Calculate NDVI", field="optical_image"):
        path = frontend.ROOT / "data/test/geotiff" / filename
        with path.open("rb") as file:
            response = self.client.post("/api/analyze", data={"query": query, field: (file, filename)})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"], data)
        self.assertIn(data["analysis_id"], frontend.ANALYSIS_REGISTRY)
        return data

    @staticmethod
    def inner_bounds(data):
        west, south, east, north = data["geospatial"]["layers"][0]["bounds_wgs84"]
        return [west + (east-west)*.25, south + (north-south)*.25,
                west + (east-west)*.75, south + (north-south)*.75]

    def test_ndvi_roi_and_repeat_roi_generate_distinct_reports(self):
        original = self.upload()
        result = original
        for expected_size in (60, 30):
            response = self.client.post("/api/map-analyze", json={
                "analysis_id": result["analysis_id"],
                "query": "Calculate NDVI for this selected area",
                "bounds": self.inner_bounds(result),
                # Paths from the browser are ignored; only the registry source is used.
                "image_path": "C:/not-a-registered-image.tif",
            })
            self.assertEqual(response.status_code, 200, response.get_json())
            result = response.get_json()
            self.assertTrue(result["success"], result)
            layer = result["geospatial"]["layers"][0]
            self.assertEqual((layer["width"], layer["height"]), (expected_size, expected_size))
            self.assertTrue(result["geospatial"]["has_result_overlay"])
            self.assertIn("NDVI", result["answer"])
            self.assertNotEqual(original["reports"]["html"], result["reports"]["html"])
            with self.client.get(result["reports"]["html"]) as report:
                self.assertEqual(report.status_code, 200)
                self.assertIn(b"selected area", report.data)
            record = frontend.ANALYSIS_REGISTRY[result["analysis_id"]]
            self.assertTrue(all(Path(path).suffix == ".tif" for path in record["inputs"].values()))

    def test_bad_requests_and_outside_roi_do_not_register_an_analysis(self):
        for body, status in (([], 400), ({}, 400),
                             ({"analysis_id": "missing", "query": "Calculate NDVI", "bounds": [0, 0, 1, 1]}, 404)):
            self.assertEqual(self.client.post("/api/map-analyze", json=body).status_code, status)
        original = self.upload()
        count = len(frontend.ANALYSIS_REGISTRY)
        for bounds in ([0, 0, 1], [0, 0, 200, 100], [float("nan"), 0, 1, 1], [0, 0, 1, 1]):
            with self.subTest(bounds=bounds):
                response = self.client.post("/api/map-analyze", json={
                    "analysis_id": original["analysis_id"], "query": "Calculate NDVI", "bounds": bounds,
                })
                self.assertEqual(response.status_code, 400, response.get_json())
                self.assertFalse(response.get_json()["success"])
        self.assertEqual(len(frontend.ANALYSIS_REGISTRY), count)

    def test_crop_preserves_pixels_crs_band_names_mask_and_tags(self):
        source = self.directory / "source.tif"
        destination = self.directory / "crop.tif"
        pixels = np.arange(100, dtype=np.uint16).reshape(1, 10, 10)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        mask[3, 3] = 0
        with rasterio.open(source, "w", driver="GTiff", width=10, height=10, count=1,
                           dtype="uint16", crs="EPSG:4326", transform=from_origin(0, 10, 1, 1)) as raster:
            raster.write(pixels)
            raster.write_mask(mask)
            raster.set_band_description(1, "B04")
            raster.update_tags(sensor="test")
        frontend.crop_geotiff_to_roi(str(source), [2, 2, 8, 8], destination)
        with rasterio.open(destination) as crop:
            np.testing.assert_array_equal(crop.read(), pixels[:, 2:8, 2:8])
            np.testing.assert_array_equal(crop.dataset_mask(), mask[2:8, 2:8])
            self.assertEqual(crop.transform, from_origin(2, 8, 1, 1))
            self.assertEqual(crop.crs.to_epsg(), 4326)
            self.assertEqual(crop.descriptions, ("B04",))
            self.assertEqual(crop.tags()["sensor"], "test")

    def test_visual_roi_registry_sources_remain_georeferenced(self):
        source = frontend.ROOT / "data/test/geotiff/sentinel2_test.tif"
        metadata = frontend.validator.validate_single_image(str(source))
        geo = frontend.build_geospatial_payload(metadata, {}, "single_image_vqa")
        result = {"geospatial": geo}
        roi = frontend.prepare_roi_execution("test", "single_image_vqa", {"image_path": str(source)}, self.inner_bounds(result))
        self.assertEqual(Path(roi["image_path"]).suffix, ".png")
        self.assertEqual(Path(roi["source_inputs"]["image_path"]).suffix, ".tif")

    def test_registry_evicts_oldest_records(self):
        with patch.object(frontend, "MAX_REGISTERED_ANALYSES", 2):
            ids = [frontend.register_analysis(str(index), "sar_analysis", {}, None, {}, {"success": True}) for index in range(3)]
        self.assertNotIn(ids[0], frontend.ANALYSIS_REGISTRY)
        self.assertEqual(list(frontend.ANALYSIS_REGISTRY), ids[1:])


if __name__ == "__main__":
    unittest.main()
