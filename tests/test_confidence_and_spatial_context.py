from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin

from query_engine.runtime import configure_raster_runtime

configure_raster_runtime()

from query_engine.confidence_engine import build_confidence
from query_engine.schemas import AOI, InputConfiguration, InputKind
from query_engine.spatial_context import build_analysis_context, evidence_records, validate_spatial_consistency


class ConfidenceAndSpatialContextTests(unittest.TestCase):
    def test_confidence_keeps_model_probability_unavailable_and_caps_spatial_weakness(self):
        confidence = build_confidence(
            routing=0.95,
            input_configuration=InputConfiguration(kind=InputKind.AOI_TEMPORAL, image_count=0),
            validation={"status": "good", "valid_coverage_pct": 55, "alignment": {"status": "passed"}},
        )
        self.assertIsNone(confidence.model_confidence)
        self.assertEqual(confidence.confidence_provenance["model_confidence"]["source"], "UNAVAILABLE")
        self.assertEqual(confidence.overall_type, "SYSTEM_RELIABILITY_SCORE")
        self.assertLessEqual(confidence.overall_confidence, 0.55)

    def test_context_uses_content_hashes_and_evidence_links(self):
        aoi = AOI(coordinates=[[(77.0, 12.0), (77.1, 12.0), (77.1, 12.1), (77.0, 12.0)]])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "same-name.png"
            path.write_bytes(b"source-a")
            context = build_analysis_context(analysis_id="ana_one", query="Inspect", aoi=aoi, inputs={"image_path": path})
            records = evidence_records(analysis_id="ana_one", context=context, artifacts=["artifacts/result.png"], aoi=aoi)
        self.assertTrue(context["sources"][0]["source_id"].startswith("src_"))
        self.assertEqual(len(context["sources"][0]["input_hash"]), 64)
        self.assertEqual(records[0]["analysis_id"], "ana_one")
        self.assertEqual(validate_spatial_consistency(context, records)["status"], "WARNING")

    def test_self_intersecting_aoi_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "self-intersects"):
            AOI(coordinates=[[(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]])

    def _geotiff(self, directory, name="source.tif"):
        path = Path(directory) / name
        with rasterio.open(path, "w", driver="GTiff", width=50, height=50, count=1, dtype="uint8",
                           crs="EPSG:4326", transform=from_origin(77, 13, .01, .01), nodata=0) as dst:
            dst.write(np.ones((1, 50, 50), dtype="uint8"))
        return path

    def test_outside_aoi_fails_real_geotiff_intersection(self):
        with TemporaryDirectory() as directory:
            context = build_analysis_context(analysis_id="ana_outside", query="Inspect",
                aoi=AOI(coordinates=[[(78, 14), (78.1, 14), (78.1, 14.1), (78, 14),]]),
                inputs={"image_path": self._geotiff(directory)})
        result = validate_spatial_consistency(context, [])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("AOI_OUTSIDE_SOURCE", result["failures"])

    def test_partial_aoi_preserves_effective_roi_geometry(self):
        with TemporaryDirectory() as directory:
            context = build_analysis_context(analysis_id="ana_partial", query="Inspect",
                aoi=AOI(coordinates=[[(76.8, 12.8), (77.2, 12.8), (77.2, 13.2), (76.8, 12.8)]]),
                inputs={"image_path": self._geotiff(directory)})
        check = context["aoi_source_checks"][0]
        self.assertEqual(validate_spatial_consistency(context, [])["status"], "WARNING")
        self.assertTrue(check["intersects"])
        self.assertGreater(check["requested_aoi_coverage_pct"], 0)
        self.assertLess(check["requested_aoi_coverage_pct"], 100)
        self.assertEqual(context["roi"]["parent_source_id"], context["sources"][0]["source_id"])
        self.assertEqual(context["roi"]["crs"], "EPSG:4326")
        self.assertGreater(context["roi"]["width"], 0)

    def test_wrong_evidence_result_link_fails(self):
        with TemporaryDirectory() as directory:
            source = self._geotiff(directory)
            aoi = AOI(coordinates=[[(77, 12.5), (77.1, 12.5), (77.1, 13), (77, 12.5)]])
            context = build_analysis_context(analysis_id="ana_a", query="Inspect", aoi=aoi, inputs={"image_path": source})
        record = evidence_records(analysis_id="ana_a", context=context, artifacts=["result.png"])[0]
        record["result_id"] = "res_other"
        result = validate_spatial_consistency(context, [record])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("EVIDENCE_RESULT_MISMATCH", result["failures"])

    def test_wgs84_aoi_is_transformed_before_projected_source_intersection(self):
        from pyproj import Transformer
        with TemporaryDirectory() as directory:
            path = Path(directory) / "utm.tif"
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
            x, y = transformer.transform(77.0, 12.5)
            with rasterio.open(path, "w", driver="GTiff", width=50, height=50, count=1, dtype="uint8",
                               crs="EPSG:32643", transform=from_origin(x, y + 1000, 20, 20)) as dst:
                dst.write(np.ones((1, 50, 50), dtype="uint8"))
            aoi = AOI(coordinates=[[(77.0, 12.5), (77.005, 12.5), (77.005, 12.505), (77.0, 12.5)]])
            context = build_analysis_context(analysis_id="ana_utm", query="Inspect", aoi=aoi, inputs={"image_path": path})
        check = context["aoi_source_checks"][0]
        self.assertTrue(check["transformation_performed"])
        self.assertEqual(context["roi"]["crs"], "EPSG:32643")
        self.assertIn(check["status"], {"PASS", "WARNING"})
