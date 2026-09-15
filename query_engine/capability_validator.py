"""Truthful task capability checks before specialist execution."""
from __future__ import annotations

from query_engine.policy import QueryError
from query_engine.schemas import CapabilityDecision, InputConfiguration, InputKind, Intent, Modality, ParsedQuery


class CapabilityValidator:
    def validate(self, parsed: ParsedQuery, config: InputConfiguration | None, source: str) -> CapabilityDecision:
        if source == "earth_engine":
            return CapabilityDecision(executable=True)
        if config is None:
            return CapabilityDecision(executable=False, code="invalid_input", message="The supplied imagery could not be classified from metadata.")
        if parsed.intent == Intent.CHANGE and not config.kind.value.startswith("bi_temporal_"):
            return CapabilityDecision(executable=False, code="insufficient_images", message="Bi-temporal analysis requires two corresponding observations.", missing=["before_path", "after_path"])
        if parsed.intent == Intent.FUSION and config.kind != InputKind.OPTICAL_SAR_PAIR:
            return CapabilityDecision(executable=False, code="modality_mismatch", message="Optical-SAR analysis requires one optical image and one SAR image.", missing=["optical_path", "sar_path"])
        if parsed.intent == Intent.MULTISPECTRAL:
            image = config.images[0] if config.images else None
            if not image or image.modality != Modality.MULTISPECTRAL:
                return CapabilityDecision(executable=False, code="missing_nir_band", message="NDVI and multispectral indices require a verified multispectral input with RED and NIR bands.", missing=["NIR band"])
        if parsed.intent == Intent.VQA and any(term in parsed.query.lower() for term in ("ndvi", "soil nitrogen", "population", "changed since", "change between")):
            return CapabilityDecision(executable=False, code="unsupported_vqa_query", message="This question requires a spectral, temporal, or external-data workflow and cannot be answered from a single RGB VQA image.")
        if parsed.intent == Intent.GROUNDING:
            return CapabilityDecision(executable=False, code="grounding_experimental", message="Precise text-guided grounding is not available yet. RemoteCLIP tile relevance remains an explicitly experimental region-retrieval capability.")
        if config.kind in {InputKind.UNSUPPORTED_PAIR, InputKind.INVALID_INPUT}:
            return CapabilityDecision(executable=False, code="invalid_input", message="The supplied imagery is not a supported input configuration.")
        if config.image_count > 1 and config.validation_issues and any(issue.get("level") == "error" for issue in config.validation_issues):
            return CapabilityDecision(executable=False, code="input_validation_failed", message="Input validation found incompatible spatial or raster metadata.")
        return CapabilityDecision(executable=True)

    def enforce(self, parsed: ParsedQuery, config: InputConfiguration | None, source: str) -> CapabilityDecision:
        decision = self.validate(parsed, config, source)
        if not decision.executable:
            raise QueryError(decision.code, decision.message, 422)
        return decision
