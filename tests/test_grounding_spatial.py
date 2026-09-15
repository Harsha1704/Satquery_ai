import math
import unittest

from ai.grounding.spatial import pixel_box_to_geo, validate_box
from query_engine.capability_validator import CapabilityValidator
from query_engine.schemas import InputConfiguration, InputKind, InspectedImage, Intent, Modality, ParsedQuery
from query_engine.tool_registry import descriptor

ROI={"analysis_id":"ana_x","roi_id":"roi_x","parent_source_id":"src_x","crs":"EPSG:4326","transform":[.01,0,77,0,-.01,13,0,0,1],"width":100,"height":100}
class GroundingSpatialTests(unittest.TestCase):
 def test_pixel_box_uses_roi_affine_not_requested_aoi_bounds(self):
  result=pixel_box_to_geo([10,20,30,40],ROI)
  self.assertEqual(result["analysis_id"],"ana_x"); self.assertEqual(result["roi_id"],"roi_x")
  self.assertEqual(result["geometry"]["coordinates"][0][0],(77.1,12.8))
 def test_invalid_and_outside_boxes_are_rejected_or_clipped(self):
  self.assertEqual(validate_box([-2,-2,110,110],100,100),[0.0,0.0,100.0,100.0])
  with self.assertRaisesRegex(ValueError,"INVALID_GROUNDING_RESULT"): validate_box([1,2,1,4],100,100)
  with self.assertRaisesRegex(ValueError,"INVALID_GROUNDING_RESULT"): validate_box([math.nan,2,3,4],100,100)
 def test_registry_distinguishes_region_retrieval_from_grounding(self):
  self.assertEqual(descriptor("text_grounding").capability_type,"remote_sensing_region_retrieval")
  self.assertEqual(descriptor("text_grounding").status,"experimental")
 def test_precise_grounding_is_not_silently_substituted(self):
  config=InputConfiguration(kind=InputKind.SINGLE_OPTICAL,image_count=1,images=[InspectedImage(role="image_path",modality=Modality.OPTICAL)])
  decision=CapabilityValidator().validate(ParsedQuery(query="Locate aircraft",intent=Intent.GROUNDING),config,"local")
  self.assertFalse(decision.executable); self.assertEqual(decision.code,"grounding_experimental")
 def test_vqa_rejects_ndvi_question(self):
  config=InputConfiguration(kind=InputKind.SINGLE_OPTICAL,image_count=1,images=[InspectedImage(role="image_path",modality=Modality.OPTICAL)])
  decision=CapabilityValidator().validate(ParsedQuery(query="Calculate NDVI",intent=Intent.VQA),config,"local")
  self.assertFalse(decision.executable); self.assertEqual(decision.code,"unsupported_vqa_query")
if __name__=="__main__": unittest.main()
