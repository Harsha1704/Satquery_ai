"""Versioned contracts shared by planning, execution, and HTTP adapters."""
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Intent(str, Enum):
    SEMANTIC = "semantic_analysis"
    VQA = "single_image_vqa"
    DETECTION = "object_detection"
    MULTISPECTRAL = "multispectral_analysis"
    SAR = "sar_analysis"
    FUSION = "optical_sar_fusion"
    CHANGE = "change_detection"
    GROUNDING = "text_guided_grounding"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    UNKNOWN = "unknown"


class InputKind(str, Enum):
    SINGLE_OPTICAL = "single_optical"
    SINGLE_MULTISPECTRAL = "single_multispectral"
    SINGLE_SAR = "single_sar"
    BI_TEMPORAL_OPTICAL = "bi_temporal_optical"
    BI_TEMPORAL_MULTISPECTRAL = "bi_temporal_multispectral"
    BI_TEMPORAL_SAR = "bi_temporal_sar"
    OPTICAL_SAR_PAIR = "optical_sar_pair"
    UNSUPPORTED_PAIR = "unsupported_pair"
    INVALID_INPUT = "invalid_input"
    AOI_TEMPORAL = "aoi_temporal"


class InspectedImage(Contract):
    role: str
    modality: Modality = Modality.UNKNOWN
    modality_confidence: float | None = Field(default=None, ge=0, le=1)
    modality_evidence: list[str] = Field(default_factory=list)
    format: str | None = None
    bands: int | None = Field(default=None, ge=1)
    width: int | None = Field(default=None, ge=1)
    height: int | None = Field(default=None, ge=1)
    crs: str | None = None
    bounds: list[float] | None = None
    resolution: list[float] | None = None
    georeferenced: bool = False


class InputConfiguration(Contract):
    kind: InputKind
    image_count: int = Field(ge=0, le=2)
    images: list[InspectedImage] = Field(default_factory=list)
    geographic_overlap: bool | None = None
    co_registered: bool | None = None
    validation_issues: list[dict[str, str]] = Field(default_factory=list)


class PlanStep(Contract):
    step_id: str
    tool: str
    dependencies: list[str] = Field(default_factory=list)
    required: bool = True
    fallback_tool: str | None = None
    expected_output: str


class CapabilityDecision(Contract):
    executable: bool
    code: str = "ok"
    message: str = ""
    missing: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AOI(Contract):
    """WGS84 polygon used as the analysis/statistics footprint.

    Retrieval may use :pyattr:`bounds` as a rectangular envelope, but the
    polygon itself remains authoritative for masking, statistics, provenance,
    and report language.  This prevents a drawn polygon from silently becoming
    a rectangle downstream.
    """

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[tuple[float, float]]]

    @model_validator(mode="after")
    def validate_polygon(self):
        if not self.coordinates:
            raise ValueError("AOI polygon requires at least one ring.")

        for ring_index, ring in enumerate(self.coordinates):
            if len(ring) < 4:
                raise ValueError(f"AOI ring {ring_index} requires at least four positions.")
            if ring[0] != ring[-1]:
                raise ValueError(f"AOI ring {ring_index} must be closed.")
            for x, y in ring:
                if not (-180 <= x <= 180 and -90 <= y <= 90):
                    raise ValueError("AOI coordinates must be WGS84 longitude, latitude.")

        outer = self.coordinates[0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        if max(xs) == min(xs) or max(ys) == min(ys):
            raise ValueError("AOI polygon has zero spatial extent.")
        if max(xs) - min(xs) >= 180:
            raise ValueError("Antimeridian-spanning AOIs are not supported by this interactive workflow.")

        # Lightweight nonzero-area test in lon/lat space. Earth Engine performs
        # the geodesic area calculation later for final statistics.
        area2 = 0.0
        for a, b in zip(outer, outer[1:]):
            area2 += a[0] * b[1] - b[0] * a[1]
        # Reject bow-tie polygons early.  The test is deliberately small and
        # dependency-free; adjacent edges share a vertex and are excluded.
        def orientation(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        def intersects(a, b, c, d):
            o1, o2 = orientation(a, b, c), orientation(a, b, d)
            o3, o4 = orientation(c, d, a), orientation(c, d, b)
            return o1 * o2 < 0 and o3 * o4 < 0
        edges = list(zip(outer, outer[1:]))
        for index, (a, b) in enumerate(edges):
            for other, (c, d) in enumerate(edges[index + 1:], start=index + 1):
                if other in {index + 1, len(edges) - 1 if index == 0 else -1}:
                    continue
                if intersects(a, b, c, d):
                    raise ValueError("AOI polygon self-intersects.")
        if abs(area2) < 1e-12:
            raise ValueError("AOI polygon area is effectively zero.")
        return self

    @property
    def bounds(self) -> list[float]:
        points = [point for ring in self.coordinates for point in ring]
        xs, ys = zip(*points)
        return [min(xs), min(ys), max(xs), max(ys)]


class Inputs(Contract):
    # Paths are relative to a server-controlled input root, never arbitrary paths.
    image_path: str | None = None
    before_path: str | None = None
    after_path: str | None = None
    optical_path: str | None = None
    sar_path: str | None = None


class AnalysisRequest(Contract):
    query: str = Field(min_length=1, max_length=4000)
    aoi: AOI | None = None
    inputs: Inputs = Field(default_factory=Inputs)
    project_id: str = Field(default="local", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    before_year: int | None = Field(default=None, ge=1985)
    after_year: int | None = Field(default=None, ge=1985)


class ParsedQuery(Contract):
    query: str
    intent: Intent
    targets: list[str] = Field(default_factory=list)
    operation: str = "analyze"
    years: list[int] = Field(default_factory=list)
    routing_confidence: float | None = Field(default=None, ge=0, le=1)
    change_direction: str = "none"
    transition_from: str = "none"
    transition_to: str = "none"
    parser: str = "legacy_rules_v1"


class AnalysisPlan(Contract):
    version: Literal["1"] = "1"
    parsed: ParsedQuery
    tools: list[str]
    source: Literal["local", "earth_engine"]
    warnings: list[str] = Field(default_factory=list)
    input_configuration: InputConfiguration | None = None
    capability: CapabilityDecision | None = None
    steps: list[PlanStep] = Field(default_factory=list)


class Confidence(Contract):
    # Legacy scalar fields remain for existing API clients.  The explicit
    # component fields below carry the public provenance needed to interpret a
    # score safely.
    routing: float | None = Field(default=None, ge=0, le=1)
    model: float | None = Field(default=None, ge=0, le=1)
    grounding_relevance: float | None = Field(default=None, ge=0, le=1)
    semantic_quality: float | None = Field(default=None, ge=0, le=1)
    data_quality: float | None = Field(default=None, ge=0, le=1)
    alignment_quality: float | None = Field(default=None, ge=0, le=1)
    evidence_strength: float | None = Field(default=None, ge=0, le=1)
    overall: float | None = Field(default=None, ge=0, le=1)
    method: str | None = None
    routing_confidence: float | None = Field(default=None, ge=0, le=1)
    input_confidence: float | None = Field(default=None, ge=0, le=1)
    modality_confidence: float | None = Field(default=None, ge=0, le=1)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_confidence: float | None = Field(default=None, ge=0, le=1)
    spatial_confidence: float | None = Field(default=None, ge=0, le=1)
    data_quality_confidence: float | None = Field(default=None, ge=0, le=1)
    overall_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_available: bool = False
    confidence_method: str | None = None
    confidence_level: str | None = None
    overall_type: Literal["SYSTEM_RELIABILITY_SCORE", "MODEL_PROBABILITY", "UNAVAILABLE"] = "UNAVAILABLE"
    calibrated: bool = False
    confidence_provenance: dict[str, dict[str, Any]] = Field(default_factory=dict)
    confidence_warnings: list[str] = Field(default_factory=list)


class TraceStep(Contract):
    step_id: str
    tool: str
    status: Literal["pending", "running", "success", "failed", "skipped", "fallback"]
    duration_ms: int | None = Field(default=None, ge=0)
    output: str | None = None
    warning: str | None = None


class ExecutionTrace(Contract):
    analysis_id: str
    task: str
    input_configuration: InputConfiguration | None = None
    selected_tools: list[str] = Field(default_factory=list)
    steps: list[TraceStep] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int | None = Field(default=None, ge=0)


class AnalysisResult(Contract):
    answer: str
    statistics: dict[str, Any] = Field(default_factory=dict)
    layers: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=Confidence)
    limitations: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    execution_trace: ExecutionTrace | None = None


class JobResponse(Contract):
    job_id: str
    project_id: str
    user_id: Literal["local"] = "local"
    status: Literal["completed", "failed"]
    started_at: str
    completed_at: str
    plan: AnalysisPlan
    result: AnalysisResult | None = None
    error_code: str | None = None
    error: str | None = None
