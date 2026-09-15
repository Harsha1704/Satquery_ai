"""Deterministic controller-contract tests; no model downloads or inference."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from query_engine.capability_validator import CapabilityValidator
from query_engine.execution_engine import validate_step_graph
from query_engine.planner import QueryPlanner
from query_engine.policy import QueryError
from query_engine.schemas import (
    AnalysisRequest, InputConfiguration, InputKind, InspectedImage,
    Intent, Modality, ParsedQuery, PlanStep,
)
from query_engine.tool_registry import REGISTRY, descriptor
from query_engine.input_inspector import InputInspector


def config(kind, *modalities, overlap=True, registered=True):
    return InputConfiguration(
        kind=kind, image_count=len(modalities), geographic_overlap=overlap,
        co_registered=registered,
        images=[InspectedImage(role=f"image_{index}", modality=modality, modality_confidence=.9)
                for index, modality in enumerate(modalities)],
    )


class AgenticControllerTests(unittest.TestCase):
    def test_registry_declares_real_tool_metadata(self):
        tool = descriptor("multispectral_analysis")
        self.assertEqual(tool.required_inputs, ("red", "nir"))
        self.assertIn("statistics", tool.produced_outputs)
        self.assertEqual(REGISTRY[tool.name], tool)

    def test_ndvi_requires_verified_multispectral_input(self):
        parsed = ParsedQuery(query="Calculate NDVI", intent=Intent.MULTISPECTRAL)
        decision = CapabilityValidator().validate(
            parsed, config(InputKind.SINGLE_OPTICAL, Modality.OPTICAL), "local"
        )
        self.assertFalse(decision.executable)
        self.assertEqual(decision.code, "missing_nir_band")

    def test_change_requires_bitemporal_pair(self):
        parsed = ParsedQuery(query="What changed?", intent=Intent.CHANGE)
        decision = CapabilityValidator().validate(
            parsed, config(InputKind.SINGLE_OPTICAL, Modality.OPTICAL), "local"
        )
        self.assertFalse(decision.executable)
        self.assertEqual(decision.code, "insufficient_images")

    def test_optical_sar_pair_is_recognized(self):
        parsed = ParsedQuery(query="Fuse optical and SAR", intent=Intent.FUSION)
        decision = CapabilityValidator().validate(
            parsed, config(InputKind.OPTICAL_SAR_PAIR, Modality.OPTICAL, Modality.SAR), "local"
        )
        self.assertTrue(decision.executable)

    def test_metadata_inspection_classifies_rgb_png_as_optical(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            Image.new("RGB", (8, 8), (20, 30, 40)).save(path)
            inspected = InputInspector().inspect({"image_path": path})
        self.assertEqual(inspected.kind, InputKind.SINGLE_OPTICAL)
        self.assertEqual(inspected.images[0].modality, Modality.OPTICAL)
        self.assertEqual(inspected.images[0].bands, 3)

    def test_change_plan_has_ordered_dependencies(self):
        request = AnalysisRequest(query="What changed?", inputs={"before_path": "before.tif", "after_path": "after.tif"})
        plan = QueryPlanner().plan(
            request,
            config(InputKind.BI_TEMPORAL_OPTICAL, Modality.OPTICAL, Modality.OPTICAL),
        )
        self.assertEqual(plan.steps[0].tool, "input.validator")
        self.assertEqual(plan.steps[1].tool, "geospatial.alignment")
        self.assertEqual(plan.steps[-1].tool, "evidence.publisher")
        validate_step_graph(plan)

    def test_dependency_graph_rejects_out_of_order_step(self):
        request = AnalysisRequest(query="What changed?", inputs={"before_path": "before.tif", "after_path": "after.tif"})
        plan = QueryPlanner().plan(
            request, config(InputKind.BI_TEMPORAL_OPTICAL, Modality.OPTICAL, Modality.OPTICAL)
        )
        broken = plan.model_copy(update={"steps": [PlanStep(step_id="later", tool="x", dependencies=["missing"], expected_output="x")]})
        with self.assertRaises(QueryError):
            validate_step_graph(broken)


if __name__ == "__main__":
    unittest.main()
