"""Closed tool registry. Client-supplied function names are never executed."""
from types import MappingProxyType
from dataclasses import dataclass

from query_engine.policy import QueryError
from query_engine.schemas import AnalysisPlan, Intent
from query_engine.execution_engine import validate_step_graph


TOOLS = MappingProxyType({
    Intent.SEMANTIC: "semantic_segmentation",
    Intent.VQA: "single_image_vqa",
    Intent.DETECTION: "object_detection",
    Intent.MULTISPECTRAL: "multispectral_analysis",
    Intent.SAR: "sar_analysis",
    Intent.FUSION: "optical_sar_fusion",
    Intent.CHANGE: "change_reasoning",
    Intent.GROUNDING: "text_grounding",
})


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    version: str
    supported_intents: tuple[Intent, ...]
    supported_modalities: tuple[str, ...]
    required_inputs: tuple[str, ...]
    produced_outputs: tuple[str, ...]
    status: str = "available"
    fallback: str | None = None
    capability_type: str = "specialist"


REGISTRY = MappingProxyType({
    "single_image_vqa": ToolDescriptor("single_image_vqa", "blip-vqa-base", (Intent.VQA,), ("optical", "multispectral"), ("image",), ("answer",), "experimental", capability_type="generic_vqa"),
    "object_detection": ToolDescriptor("object_detection", "legacy-v1", (Intent.DETECTION,), ("optical",), ("image",), ("detections", "answer"), "experimental"),
    "semantic_segmentation": ToolDescriptor("semantic_segmentation", "legacy-v1", (Intent.SEMANTIC,), ("optical", "multispectral"), ("image",), ("mask", "statistics"), "experimental"),
    "multispectral_analysis": ToolDescriptor("multispectral_analysis", "legacy-v1", (Intent.MULTISPECTRAL,), ("multispectral",), ("red", "nir"), ("indices", "statistics")),
    "sar_analysis": ToolDescriptor("sar_analysis", "legacy-v1", (Intent.SAR,), ("sar",), ("sar",), ("statistics", "answer"), "experimental"),
    "optical_sar_fusion": ToolDescriptor("optical_sar_fusion", "legacy-v1", (Intent.FUSION,), ("optical", "sar"), ("optical", "sar"), ("fusion", "answer"), "experimental"),
    "change_reasoning": ToolDescriptor("change_reasoning", "legacy-v1", (Intent.CHANGE,), ("optical", "multispectral", "sar"), ("before", "after"), ("change", "statistics", "answer")),
    "text_grounding": ToolDescriptor("text_grounding", "RemoteCLIP-ViT-B-32", (Intent.GROUNDING,), ("optical", "multispectral"), ("image", "text"), ("regions", "answer"), "experimental", capability_type="remote_sensing_region_retrieval"),
})


def descriptor(name: str) -> ToolDescriptor:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise QueryError("tool_unavailable", f"No registered tool named {name!r}.", 503) from exc


def allowed_tools(intent: Intent, source: str) -> list[str]:
    if intent not in TOOLS:
        raise QueryError(
            "unknown_intent",
            "SatQuery could not map this question to a supported geospatial workflow. "
            "Try a vegetation, built-up/urban, water/flood, or general temporal-change question.",
            422,
        )
    if source == "earth_engine":
        # The current worldwide AOI executor is a validated temporal-change
        # pipeline. Do not pretend that unsupported object-detection/VQA/SAR
        # routes are executable merely because the language router recognized
        # the words.
        if intent != Intent.CHANGE:
            raise QueryError(
                "unsupported_aoi_analysis",
                "This worldwide AOI executor currently supports temporal change analysis. "
                "Ask about vegetation change, urban/built-up change, water/flood extent change, "
                "or general change between two dates. Other recognized tasks require a separate "
                "validated executor and will not be simulated.",
                422,
            )
        return ["satellite_retrieval", TOOLS[intent], "temporal_evidence"]
    return [TOOLS[intent]]


def validate_plan(plan: AnalysisPlan) -> None:
    if plan.tools != allowed_tools(plan.parsed.intent, plan.source):
        raise QueryError("invalid_plan", "Plan contains tools outside the allowed execution sequence.")
    validate_step_graph(plan)
