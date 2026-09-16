# ai/validation/input_validator.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None


SUPPORTED_IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".npy",
}

RASTER_EXTENSIONS = {
    ".tif",
    ".tiff",
}


@dataclass
class ValidationIssue:
    level: str
    code: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageMetadata:
    path: str
    extension: str
    format: str

    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    dtype: Optional[str] = None

    crs: Optional[str] = None
    transform: Optional[Tuple[float, ...]] = None

    resolution_x: Optional[float] = None
    resolution_y: Optional[float] = None

    bounds: Optional[Tuple[float, float, float, float]] = None

    nodata: Optional[float] = None

    georeferenced: bool = False

    array_shape: Optional[Tuple[int, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InputValidator:
    """
    SatQuery AI input/modality validator.

    Validates:
        - path existence
        - supported file type
        - image dimensions
        - band count
        - GeoTIFF metadata
        - CRS
        - transform
        - spatial resolution
        - optical/SAR compatibility
        - bi-temporal compatibility
        - expected modality
        - co-registration indicators

    Important:
        This module validates compatibility.
        It does not automatically reproject or co-register imagery.
    """

    def __init__(
        self,
        resolution_tolerance: float = 0.05,
        shape_tolerance_pixels: int = 2,
    ):
        self.resolution_tolerance = resolution_tolerance
        self.shape_tolerance_pixels = shape_tolerance_pixels

    # ================================================================
    # PUBLIC API
    # ================================================================

    def validate_single_image(
        self,
        path: str,
        expected_modality: Optional[str] = None,
    ) -> Dict[str, Any]:

        issues: List[ValidationIssue] = []

        metadata = self.inspect_image(
            path=path,
            issues=issues,
        )

        if metadata is None:
            return self._build_result(
                valid=False,
                issues=issues,
                metadata=None,
            )

        if expected_modality:
            self._validate_modality(
                metadata=metadata,
                expected_modality=expected_modality,
                issues=issues,
            )

        valid = not self._has_errors(issues)

        return self._build_result(
            valid=valid,
            issues=issues,
            metadata=metadata.to_dict(),
        )

    def validate_change_pair(
        self,
        before_path: str,
        after_path: str,
    ) -> Dict[str, Any]:

        issues: List[ValidationIssue] = []

        before = self.inspect_image(
            before_path,
            issues,
        )

        after = self.inspect_image(
            after_path,
            issues,
        )

        if before is None or after is None:
            return self._build_result(
                valid=False,
                issues=issues,
                metadata={
                    "before": before.to_dict() if before else None,
                    "after": after.to_dict() if after else None,
                },
            )

        self._compare_dimensions(
            before,
            after,
            issues,
            context="bi_temporal",
        )

        self._compare_band_count(
            before,
            after,
            issues,
            strict=False,
        )

        self._compare_geospatial_metadata(
            first=before,
            second=after,
            issues=issues,
            context="bi_temporal",
        )

        valid = not self._has_errors(issues)

        return self._build_result(
            valid=valid,
            issues=issues,
            metadata={
                "before": before.to_dict(),
                "after": after.to_dict(),
            },
        )

    def validate_optical_sar_pair(
        self,
        optical_path: str,
        sar_path: str,
    ) -> Dict[str, Any]:

        issues: List[ValidationIssue] = []

        optical = self.inspect_image(
            optical_path,
            issues,
        )

        sar = self.inspect_image(
            sar_path,
            issues,
        )

        if optical is None or sar is None:
            return self._build_result(
                valid=False,
                issues=issues,
                metadata={
                    "optical": optical.to_dict() if optical else None,
                    "sar": sar.to_dict() if sar else None,
                },
            )

        self._validate_modality(
            metadata=optical,
            expected_modality="optical",
            issues=issues,
        )

        self._validate_modality(
            metadata=sar,
            expected_modality="sar",
            issues=issues,
        )

        self._compare_geospatial_metadata(
            first=optical,
            second=sar,
            issues=issues,
            context="optical_sar",
        )

        self._compare_dimensions(
            optical,
            sar,
            issues,
            context="optical_sar",
            warning_only=True,
        )

        valid = not self._has_errors(issues)

        return self._build_result(
            valid=valid,
            issues=issues,
            metadata={
                "optical": optical.to_dict(),
                "sar": sar.to_dict(),
            },
        )

    def validate_multispectral(
        self,
        path: str,
        minimum_bands: int = 4,
    ) -> Dict[str, Any]:

        result = self.validate_single_image(
            path=path,
            expected_modality="multispectral",
        )

        if result["metadata"] is None:
            return result

        bands = result["metadata"].get("bands")

        if bands is not None and bands < minimum_bands:
            result["issues"].append(
                ValidationIssue(
                    level="error",
                    code="INSUFFICIENT_MULTISPECTRAL_BANDS",
                    message=(
                        f"Multispectral analysis requires at least "
                        f"{minimum_bands} bands, but {bands} were found."
                    ),
                ).to_dict()
            )

            result["valid"] = False

        return result

    # ================================================================
    # IMAGE INSPECTION
    # ================================================================

    def inspect_image(
        self,
        path: str,
        issues: Optional[List[ValidationIssue]] = None,
    ) -> Optional[ImageMetadata]:

        if issues is None:
            issues = []

        image_path = Path(path)

        if not image_path.exists():
            issues.append(
                ValidationIssue(
                    level="error",
                    code="FILE_NOT_FOUND",
                    message=f"Input file does not exist: {path}",
                )
            )
            return None

        if not image_path.is_file():
            issues.append(
                ValidationIssue(
                    level="error",
                    code="NOT_A_FILE",
                    message=f"Input path is not a file: {path}",
                )
            )
            return None

        extension = image_path.suffix.lower()

        if extension not in SUPPORTED_IMAGE_EXTENSIONS:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="UNSUPPORTED_FORMAT",
                    message=(
                        f"Unsupported image format '{extension}'. "
                        f"Supported formats: "
                        f"{sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
                    ),
                )
            )
            return None

        if extension == ".npy":
            return self._inspect_numpy(
                image_path,
                issues,
            )

        if extension in RASTER_EXTENSIONS:
            return self._inspect_geotiff(
                image_path,
                issues,
            )

        return self._inspect_standard_image(
            image_path,
            issues,
        )

    # ================================================================
    # NPY
    # ================================================================

    def _inspect_numpy(
        self,
        path: Path,
        issues: List[ValidationIssue],
    ) -> Optional[ImageMetadata]:

        try:
            array = np.load(
                path,
                mmap_mode="r",
            )
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="NPY_LOAD_FAILED",
                    message=f"Unable to load NPY image: {exc}",
                )
            )
            return None

        shape = array.shape

        if array.ndim == 2:
            height, width = shape
            bands = 1

        elif array.ndim == 3:

            if shape[0] <= 32:
                bands, height, width = shape

            elif shape[-1] <= 32:
                height, width, bands = shape

            else:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="AMBIGUOUS_ARRAY_LAYOUT",
                        message=(
                            f"Unable to determine channel dimension "
                            f"for array shape {shape}."
                        ),
                    )
                )
                return None

        else:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="INVALID_ARRAY_DIMENSION",
                    message=(
                        f"Expected 2D or 3D raster array, "
                        f"received shape {shape}."
                    ),
                )
            )
            return None

        issues.append(
            ValidationIssue(
                level="warning",
                code="NO_GEOSPATIAL_METADATA",
                message=(
                    "NPY arrays do not contain CRS, transform, "
                    "bounds, or spatial resolution metadata."
                ),
            )
        )

        return ImageMetadata(
            path=str(path),
            extension=path.suffix.lower(),
            format="NPY",
            width=int(width),
            height=int(height),
            bands=int(bands),
            dtype=str(array.dtype),
            georeferenced=False,
            array_shape=tuple(int(v) for v in shape),
        )

    # ================================================================
    # GEOTIFF
    # ================================================================

    def _inspect_geotiff(
        self,
        path: Path,
        issues: List[ValidationIssue],
    ) -> Optional[ImageMetadata]:

        if rasterio is None:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="RASTERIO_NOT_AVAILABLE",
                    message=(
                        "Rasterio is required for GeoTIFF validation."
                    ),
                )
            )
            return None

        try:
            with rasterio.open(path) as dataset:

                transform = dataset.transform

                crs = (
                    dataset.crs.to_string()
                    if dataset.crs
                    else None
                )

                bounds = (
                    float(dataset.bounds.left),
                    float(dataset.bounds.bottom),
                    float(dataset.bounds.right),
                    float(dataset.bounds.top),
                )

                resolution_x = abs(float(transform.a))
                resolution_y = abs(float(transform.e))

                georeferenced = (
                    dataset.crs is not None
                    and not transform.is_identity
                )

                if dataset.crs is None:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            code="MISSING_CRS",
                            message=(
                                f"GeoTIFF has no CRS: {path.name}"
                            ),
                        )
                    )

                if transform.is_identity:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            code="IDENTITY_TRANSFORM",
                            message=(
                                f"GeoTIFF has no meaningful "
                                f"geospatial transform: {path.name}"
                            ),
                        )
                    )

                if dataset.count <= 0:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="NO_RASTER_BANDS",
                            message="Raster contains no image bands.",
                        )
                    )

                if dataset.width <= 0 or dataset.height <= 0:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="INVALID_RASTER_SIZE",
                            message="Raster dimensions are invalid.",
                        )
                    )

                dtype = (
                    dataset.dtypes[0]
                    if dataset.dtypes
                    else None
                )

                return ImageMetadata(
                    path=str(path),
                    extension=path.suffix.lower(),
                    format="GeoTIFF",
                    width=int(dataset.width),
                    height=int(dataset.height),
                    bands=int(dataset.count),
                    dtype=dtype,
                    crs=crs,
                    transform=tuple(transform)[:6],
                    resolution_x=resolution_x,
                    resolution_y=resolution_y,
                    bounds=bounds,
                    nodata=dataset.nodata,
                    georeferenced=georeferenced,
                    array_shape=(
                        int(dataset.count),
                        int(dataset.height),
                        int(dataset.width),
                    ),
                )

        except Exception as exc:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="GEOTIFF_READ_FAILED",
                    message=f"Unable to read GeoTIFF: {exc}",
                )
            )
            return None

    # ================================================================
    # PNG / JPEG
    # ================================================================

    def _inspect_standard_image(
        self,
        path: Path,
        issues: List[ValidationIssue],
    ) -> Optional[ImageMetadata]:

        try:
            from PIL import Image

            with Image.open(path) as image:

                width, height = image.size

                bands = len(
                    image.getbands()
                )

                issues.append(
                    ValidationIssue(
                        level="warning",
                        code="NON_GEOSPATIAL_IMAGE",
                        message=(
                            f"{path.suffix.upper()} input does not "
                            f"contain geospatial CRS/transform metadata."
                        ),
                    )
                )

                return ImageMetadata(
                    path=str(path),
                    extension=path.suffix.lower(),
                    format=image.format or path.suffix.upper(),
                    width=int(width),
                    height=int(height),
                    bands=int(bands),
                    dtype="uint8",
                    georeferenced=False,
                    array_shape=(
                        int(height),
                        int(width),
                        int(bands),
                    ),
                )

        except Exception as exc:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="IMAGE_READ_FAILED",
                    message=f"Unable to read image: {exc}",
                )
            )

            return None

    # ================================================================
    # MODALITY
    # ================================================================

    def _validate_modality(
        self,
        metadata: ImageMetadata,
        expected_modality: str,
        issues: List[ValidationIssue],
    ):

        modality = expected_modality.lower().strip()

        bands = metadata.bands or 0

        if modality == "sar":

            if bands not in {1, 2}:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        code="UNEXPECTED_SAR_BANDS",
                        message=(
                            f"SAR input usually contains 1 or 2 bands "
                            f"(e.g. VV/VH), but {bands} bands were found."
                        ),
                    )
                )

        elif modality == "multispectral":

            if bands < 4:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="NOT_MULTISPECTRAL",
                        message=(
                            f"Multispectral analysis requires at least "
                            f"4 bands, but {bands} were found."
                        ),
                    )
                )

        elif modality == "optical":

            if bands < 3:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="INVALID_OPTICAL_BANDS",
                        message=(
                            f"Optical imagery requires at least "
                            f"3 bands, but {bands} were found."
                        ),
                    )
                )

        elif modality == "rgb":

            if bands < 3:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="INVALID_RGB_IMAGE",
                        message=(
                            f"RGB input requires at least 3 bands, "
                            f"but {bands} were found."
                        ),
                    )
                )

    # ================================================================
    # PAIR COMPARISON
    # ================================================================

    def _compare_dimensions(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
        context: str,
        warning_only: bool = False,
    ):

        if (
            first.width is None
            or first.height is None
            or second.width is None
            or second.height is None
        ):
            return

        width_difference = abs(
            first.width - second.width
        )

        height_difference = abs(
            first.height - second.height
        )

        if (
            width_difference > self.shape_tolerance_pixels
            or height_difference > self.shape_tolerance_pixels
        ):

            level = (
                "warning"
                if warning_only
                else "error"
            )

            issues.append(
                ValidationIssue(
                    level=level,
                    code="SPATIAL_DIMENSION_MISMATCH",
                    message=(
                        f"{context} inputs have different dimensions: "
                        f"{first.width}x{first.height} versus "
                        f"{second.width}x{second.height}."
                    ),
                )
            )

    def _compare_band_count(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
        strict: bool = False,
    ):

        if (
            first.bands is None
            or second.bands is None
        ):
            return

        if first.bands != second.bands:

            level = (
                "error"
                if strict
                else "warning"
            )

            issues.append(
                ValidationIssue(
                    level=level,
                    code="BAND_COUNT_MISMATCH",
                    message=(
                        f"Input band counts differ: "
                        f"{first.bands} versus {second.bands}."
                    ),
                )
            )

    def _compare_geospatial_metadata(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
        context: str,
    ):

        if not first.georeferenced or not second.georeferenced:

            issues.append(
                ValidationIssue(
                    level="warning",
                    code="PAIR_GEOSPATIAL_VALIDATION_LIMITED",
                    message=(
                        f"{context} compatibility cannot be fully "
                        f"verified because one or both inputs lack "
                        f"geospatial metadata."
                    ),
                )
            )

            return

        if first.crs != second.crs:

            issues.append(
                ValidationIssue(
                    level="error",
                    code="CRS_MISMATCH",
                    message=(
                        f"CRS mismatch: "
                        f"{first.crs} versus {second.crs}. "
                        f"Reprojection is required."
                    ),
                )
            )

        self._compare_resolution(
            first,
            second,
            issues,
        )

        self._compare_bounds(
            first,
            second,
            issues,
        )

        self._compare_transform(
            first,
            second,
            issues,
            context,
        )

    def _compare_resolution(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
    ):

        values = [
            first.resolution_x,
            first.resolution_y,
            second.resolution_x,
            second.resolution_y,
        ]

        if any(value is None for value in values):
            return

        x_reference = max(
            first.resolution_x,
            second.resolution_x,
            1e-12,
        )

        y_reference = max(
            first.resolution_y,
            second.resolution_y,
            1e-12,
        )

        x_difference = abs(
            first.resolution_x - second.resolution_x
        ) / x_reference

        y_difference = abs(
            first.resolution_y - second.resolution_y
        ) / y_reference

        if (
            x_difference > self.resolution_tolerance
            or y_difference > self.resolution_tolerance
        ):

            issues.append(
                ValidationIssue(
                    level="error",
                    code="RESOLUTION_MISMATCH",
                    message=(
                        f"Spatial resolution mismatch: "
                        f"({first.resolution_x}, {first.resolution_y}) "
                        f"versus "
                        f"({second.resolution_x}, {second.resolution_y})."
                    ),
                )
            )

    def _compare_bounds(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
    ):

        if (
            first.bounds is None
            or second.bounds is None
        ):
            return

        overlap = self._bounds_overlap(
            first.bounds,
            second.bounds,
        )

        if not overlap:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="NO_SPATIAL_OVERLAP",
                    message=(
                        "The two georeferenced rasters do not "
                        "spatially overlap."
                    ),
                )
            )

    def _compare_transform(
        self,
        first: ImageMetadata,
        second: ImageMetadata,
        issues: List[ValidationIssue],
        context: str,
    ):

        if (
            first.transform is None
            or second.transform is None
        ):
            return

        if not np.allclose(
            np.asarray(first.transform),
            np.asarray(second.transform),
            rtol=1e-5,
            atol=1e-5,
        ):
            level = "error" if context == "bi_temporal" else "warning"
            issues.append(
                ValidationIssue(
                    level=level,
                    code="GRID_MISALIGNMENT" if context == "bi_temporal" else "TRANSFORM_MISMATCH",
                    message=(
                        "Raster transforms differ. "
                        "Images are not on the same comparison grid."
                    ),
                )
            )

    # ================================================================
    # UTILS
    # ================================================================

    @staticmethod
    def _bounds_overlap(
        first: Tuple[float, float, float, float],
        second: Tuple[float, float, float, float],
    ) -> bool:

        left1, bottom1, right1, top1 = first
        left2, bottom2, right2, top2 = second

        return not (
            right1 <= left2
            or right2 <= left1
            or top1 <= bottom2
            or top2 <= bottom1
        )

    @staticmethod
    def _has_errors(
        issues: List[ValidationIssue],
    ) -> bool:

        return any(
            issue.level == "error"
            for issue in issues
        )

    @staticmethod
    def _build_result(
        valid: bool,
        issues: List[ValidationIssue],
        metadata: Any,
    ) -> Dict[str, Any]:

        errors = [
            issue.to_dict()
            for issue in issues
            if issue.level == "error"
        ]

        warnings = [
            issue.to_dict()
            for issue in issues
            if issue.level == "warning"
        ]

        return {
            "valid": valid,
            "metadata": metadata,
            "errors": errors,
            "warnings": warnings,
            "issues": [
                issue.to_dict()
                for issue in issues
            ],
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
