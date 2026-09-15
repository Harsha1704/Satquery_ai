from pathlib import Path

from .input_types import (
    InputMode,
    InputRequest
)


class InputValidator:

    def validate(self, request: InputRequest):

        if request.mode == InputMode.SINGLE:
            return self._validate_single(request)

        if request.mode == InputMode.BI_TEMPORAL:
            return self._validate_pair(
                request,
                "Bi-temporal analysis"
            )

        if request.mode == InputMode.OPTICAL_SAR:
            return self._validate_pair(
                request,
                "Optical-SAR analysis"
            )

        raise ValueError(
            f"Unsupported input mode: {request.mode}"
        )

    def _validate_single(
        self,
        request: InputRequest
    ):

        if not request.primary_image:
            raise ValueError(
                "Primary image is required."
            )

        self._check_file(request.primary_image)

        return True

    def _validate_pair(
        self,
        request: InputRequest,
        analysis_name: str
    ):

        if not request.primary_image:
            raise ValueError(
                f"{analysis_name}: primary image is required."
            )

        if not request.secondary_image:
            raise ValueError(
                f"{analysis_name}: secondary image is required."
            )

        self._check_file(request.primary_image)
        self._check_file(request.secondary_image)

        if request.primary_image == request.secondary_image:
            raise ValueError(
                "Primary and secondary images cannot be identical."
            )

        return True

    def _check_file(self, image_path: str):

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Input file does not exist: {image_path}"
            )