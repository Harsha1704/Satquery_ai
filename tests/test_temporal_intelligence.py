import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import rasterio
from rasterio.transform import from_origin
from query_engine.runtime import configure_raster_runtime
from ai.temporal import TemporalPair, TemporalChangeEngine, TemporalTrendAnalyzer, TemporalChangeVQA
from query_engine.schemas import AnalysisPlan, AnalysisRequest, Inputs, Intent, ParsedQuery
from query_engine.worker import run

configure_raster_runtime()

class TemporalIntelligenceTests(unittest.TestCase):
 def _pair(self, directory, *, shifted=False, after_resolution=.001):
  before=np.full((1,40,40),.2,dtype="float32"); after=before.copy();after[:,12:28,20:36]=.8
  transform=from_origin(77,13,.001,.001); paths=[]
  for name,data,trans in (("before.tif",before,transform),("after.tif",after,from_origin(77.01 if shifted else 77,13,after_resolution,after_resolution))):
   path=Path(directory)/name
   with rasterio.open(path,"w",driver="GTiff",width=40,height=40,count=1,dtype="float32",crs="EPSG:4326",transform=trans,nodata=-9999) as dst:dst.write(data)
   paths.append(path)
  return paths
 def test_known_square_has_change_area_and_georeferenced_polygon(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory); result=TemporalChangeEngine().analyze(TemporalPair(before,after,"2023-01-01","2024-01-01"),minimum_component_pixels=4)
  self.assertGreater(result["statistics"]["changed_area_m2"],0); self.assertGreater(len(result["change_polygons"]["features"]),0)
  self.assertGreaterEqual(result["change_polygons"]["features"][0]["properties"]["pixel_count"],240)
  self.assertEqual(result["effective_roi"]["crs"],"EPSG:4326");self.assertEqual(result["change_polygons"]["features"][0]["properties"]["geometry_crs"],"EPSG:4326");self.assertIsNone(result["model_confidence"])
 def test_semantic_transition_matrix_excludes_unchanged_classes(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory); a=np.ones((40,40),dtype=np.uint8);b=a.copy();b[12:28,20:36]=2
   result=TemporalChangeEngine().analyze(TemporalPair(before,after),before_semantic=a,after_semantic=b,minimum_component_pixels=4)
  self.assertEqual(result["transition_matrix"],{"1_to_2":256});self.assertEqual(result["dominant_transition"],"1_to_2")
 def test_common_grid_handles_shifted_source(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory,shifted=True);result=TemporalChangeEngine().analyze(TemporalPair(before,after),minimum_component_pixels=4)
  self.assertEqual(result["registration"]["method"],"metadata_common_grid_reprojection")
  self.assertLess(result["effective_roi"]["width"],40)
 def test_resolution_mismatch_is_recorded_as_a_quality_warning(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory,after_resolution=.004);result=TemporalChangeEngine().analyze(TemporalPair(before,after),minimum_component_pixels=1)
  self.assertTrue(any("resolution differs" in warning for warning in result["registration"]["quality_warnings"]))
 def test_temporal_pair_rejects_reverse_dates(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory)
   with self.assertRaisesRegex(ValueError,"INVALID_TEMPORAL_PAIR"): TemporalPair(before,after,"2025-01-01","2024-01-01").validate()
   with self.assertRaisesRegex(ValueError,"INVALID_TEMPORAL_PAIR"): TemporalPair(before,after,"not-a-date","2024-01-01").validate()
 def test_identical_pair_is_not_false_change(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory); import shutil;shutil.copy2(before,after);result=TemporalChangeEngine().analyze(TemporalPair(before,after),minimum_component_pixels=4)
  self.assertEqual(result["statistics"]["changed_percentage"],0);self.assertEqual(result["change_polygons"]["features"],[])
 def test_change_vqa_answers_only_from_result(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory);result=TemporalChangeEngine().analyze(TemporalPair(before,after),minimum_component_pixels=4)
  answer=TemporalChangeVQA().answer("How much area changed?",result)
  self.assertIn("hectares",answer["answer"]);self.assertEqual(answer["evidence_result_id"],result["result_id"])
 def test_trend_requires_three_observations(self):
  self.assertEqual(TemporalTrendAnalyzer().analyze([2023,2024],[.4,.3])["status"],"insufficient_observations")
  self.assertEqual(TemporalTrendAnalyzer().analyze([2022,2023,2024],[.6,.5,.4])["status"],"decreasing")
 def test_local_worker_uses_temporal_engine_and_publishes_geojson(self):
  with TemporaryDirectory() as directory:
   before,after=self._pair(directory); plan=AnalysisPlan(parsed=ParsedQuery(query="How much area changed?",intent=Intent.CHANGE),tools=["change_reasoning"],source="local")
   request=AnalysisRequest(query="How much area changed?",inputs=Inputs(before_path="before.tif",after_path="after.tif")); original=Path.cwd(); os.chdir(directory)
   try: output=run({"plan":plan.model_dump(mode="json"),"request":request.model_dump(mode="json"),"inputs":{"before_path":str(before),"after_path":str(after)}})
   finally: os.chdir(original)
   self.assertTrue(output["success"]);self.assertIn("temporal",output["statistics"]);self.assertTrue((Path(directory)/"artifacts"/"temporal_change_polygons.geojson").is_file())
if __name__=="__main__":unittest.main()
