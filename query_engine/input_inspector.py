"""Metadata-first local-input inspection for the agentic planning layer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from query_engine.schemas import InputConfiguration, InputKind, InspectedImage, Modality


def _modality(role: str, metadata: dict) -> tuple[Modality, float, list[str]]:
    """Classify only from declared role and inspected metadata, never pixels."""
    bands = metadata.get("bands")
    suffix = Path(str(metadata.get("path") or "")).suffix.lower()
    if role == "sar_path":
        return Modality.SAR, 0.98, ["Input role is SAR."]
    if role == "optical_path":
        return Modality.OPTICAL, 0.98, ["Input role is optical."]
    if isinstance(bands, int) and bands >= 4:
        return Modality.MULTISPECTRAL, 0.90, [f"{bands} raster bands available."]
    if isinstance(bands, int) and bands == 3:
        return Modality.OPTICAL, 0.85, ["Three image bands available (RGB/optical)."]
    return Modality.UNKNOWN, 0.0, [f"{suffix or 'Input'} metadata does not establish modality."]


class InputInspector:
    """Build a typed configuration from server-resolved inputs.

    This adapter reuses the mature project validator for raster metadata and
    spatial comparisons rather than implementing a second GeoTIFF reader.
    """

    def __init__(self, validator: Any | None = None):
        # Importing ``ai`` at API boot would load optional specialist-model
        # modules. Keep health/planning imports lightweight; this validator is
        # instantiated only for a real local-input request.
        self.validator = validator

    def inspect(self, paths: Mapping[str, Path], *, aoi_present: bool = False) -> InputConfiguration:
        if aoi_present:
            return InputConfiguration(kind=InputKind.AOI_TEMPORAL, image_count=0)

        if self.validator is None:
            from ai.validation.input_validator import InputValidator
            self.validator = InputValidator()

        images: list[InspectedImage] = []
        issues: list[dict[str, str]] = []
        for role, path in paths.items():
            raw = self.validator.validate_single_image(str(path))
            issues.extend(raw.get("issues") or [])
            metadata = raw.get("metadata") or {}
            modality, confidence, evidence = _modality(role, metadata)
            bounds = metadata.get("bounds")
            resolution = [metadata.get("resolution_x"), metadata.get("resolution_y")]
            images.append(InspectedImage(
                role=role, modality=modality, modality_confidence=confidence,
                modality_evidence=evidence, format=metadata.get("format"), bands=metadata.get("bands"),
                width=metadata.get("width"), height=metadata.get("height"), crs=metadata.get("crs"),
                bounds=list(bounds) if bounds else None,
                resolution=resolution if all(x is not None for x in resolution) else None,
                georeferenced=bool(metadata.get("georeferenced")),
            ))

        kind = InputKind.INVALID_INPUT
        overlap = registered = None
        if set(paths) == {"image_path"}:
            kind = {
                Modality.OPTICAL: InputKind.SINGLE_OPTICAL,
                Modality.MULTISPECTRAL: InputKind.SINGLE_MULTISPECTRAL,
                Modality.SAR: InputKind.SINGLE_SAR,
            }.get(images[0].modality, InputKind.SINGLE_OPTICAL)
        elif set(paths) == {"optical_path", "sar_path"}:
            kind = InputKind.OPTICAL_SAR_PAIR
            pair = self.validator.validate_optical_sar_pair(str(paths["optical_path"]), str(paths["sar_path"]))
            issues.extend(pair.get("issues") or [])
        elif set(paths) == {"before_path", "after_path"}:
            pair = self.validator.validate_change_pair(str(paths["before_path"]), str(paths["after_path"]))
            issues.extend(pair.get("issues") or [])
            modalities = {image.modality for image in images}
            kind = {
                frozenset({Modality.OPTICAL}): InputKind.BI_TEMPORAL_OPTICAL,
                frozenset({Modality.MULTISPECTRAL}): InputKind.BI_TEMPORAL_MULTISPECTRAL,
                frozenset({Modality.SAR}): InputKind.BI_TEMPORAL_SAR,
            }.get(frozenset(modalities), InputKind.UNSUPPORTED_PAIR)
            error_codes = {str(issue.get("code")) for issue in pair.get("issues") or [] if issue.get("level") == "error"}
            overlap = "NO_GEOGRAPHIC_OVERLAP" not in error_codes
            registered = not bool(error_codes & {"CRS_MISMATCH", "TRANSFORM_MISMATCH", "DIMENSION_MISMATCH", "RESOLUTION_MISMATCH"})
        return InputConfiguration(
            kind=kind, image_count=len(images), images=images,
            geographic_overlap=overlap, co_registered=registered,
            validation_issues=issues,
        )
