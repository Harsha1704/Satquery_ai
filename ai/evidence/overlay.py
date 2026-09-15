# ai/evidence/overlay.py

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


class EvidenceOverlay:
    """
    Visual evidence generation utilities for SatQuery-AI.

    Supports:
        - Binary change-mask overlays
        - Generic mask overlays
        - Before/after/change comparison images
        - ChangeFormer visual evidence
    """

    @staticmethod
    def _prepare_output(
        output_path: str,
    ) -> Path:

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output

    @staticmethod
    def _normalize_mask(
        mask: np.ndarray,
    ) -> np.ndarray:

        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                "Evidence mask must be a 2D array."
            )

        if mask.size == 0:
            raise ValueError(
                "Evidence mask cannot be empty."
            )

        if np.issubdtype(
            mask.dtype,
            np.bool_,
        ):
            return mask.astype(
                np.uint8
            )

        mask_float = mask.astype(
            np.float32,
            copy=False,
        )

        mask_float = np.nan_to_num(
            mask_float,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        unique_values = np.unique(
            mask_float
        )

        if (
            unique_values.size <= 2
            and np.all(
                np.isin(
                    unique_values,
                    [0, 1, 255],
                )
            )
        ):
            return (
                mask_float > 0
            ).astype(
                np.uint8
            )

        minimum = float(
            np.min(
                mask_float
            )
        )

        maximum = float(
            np.max(
                mask_float
            )
        )

        if maximum <= minimum:
            return np.zeros(
                mask_float.shape,
                dtype=np.uint8,
            )

        normalized = (
            mask_float
            - minimum
        ) / (
            maximum
            - minimum
        )

        return (
            normalized >= 0.5
        ).astype(
            np.uint8
        )

    @staticmethod
    def _resize_mask(
        mask: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:

        mask_image = Image.fromarray(
            (
                mask.astype(
                    np.uint8
                )
                * 255
            ),
            mode="L",
        )

        mask_image = mask_image.resize(
            size,
            resample=Image.Resampling.NEAREST,
        )

        return (
            np.asarray(
                mask_image
            )
            > 0
        ).astype(
            np.uint8
        )

    def overlay_mask(
        self,
        base_image_path: str,
        mask: np.ndarray,
        output_path: str,
        alpha: float = 0.55,
        color: Tuple[int, int, int] = (
            255,
            0,
            0,
        ),
    ) -> str:
        """
        Overlay changed/masked pixels on top of the original image.

        Unchanged pixels remain exactly as they appear in the
        original image.

        Default overlay:
            Red = changed area.
        """

        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(
                "alpha must be between 0 and 1."
            )

        base_path = Path(
            base_image_path
        )

        if not base_path.exists():
            raise FileNotFoundError(
                f"Base image not found: "
                f"{base_image_path}"
            )

        base = Image.open(
            base_path
        ).convert(
            "RGB"
        )

        base_array = np.asarray(
            base
        ).astype(
            np.float32
        )

        binary_mask = (
            self._normalize_mask(
                mask
            )
        )

        if (
            binary_mask.shape[1]
            != base.width
            or binary_mask.shape[0]
            != base.height
        ):
            binary_mask = (
                self._resize_mask(
                    binary_mask,
                    base.size,
                )
            )

        mask_bool = (
            binary_mask.astype(
                bool
            )
        )

        result = (
            base_array.copy()
        )

        overlay_color = np.array(
            color,
            dtype=np.float32,
        )

        result[
            mask_bool
        ] = (
            base_array[
                mask_bool
            ]
            * (
                1.0
                - alpha
            )
            + overlay_color
            * alpha
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(
            np.uint8
        )

        output = (
            self._prepare_output(
                output_path
            )
        )

        Image.fromarray(
            result,
            mode="RGB",
        ).save(
            output
        )

        return str(
            output
        )

    def create_binary_mask(
        self,
        mask: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Save a binary mask.

        255 = changed
        0   = stable
        """

        binary_mask = (
            self._normalize_mask(
                mask
            )
        )

        image = Image.fromarray(
            (
                binary_mask
                * 255
            ).astype(
                np.uint8
            ),
            mode="L",
        )

        output = (
            self._prepare_output(
                output_path
            )
        )

        image.save(
            output
        )

        return str(
            output
        )

    def create_change_comparison(
        self,
        before_image_path: str,
        after_image_path: str,
        mask: np.ndarray,
        output_path: str,
        alpha: float = 0.55,
    ) -> str:
        """
        Create a single evidence image containing:

            BEFORE | AFTER | CHANGE OVERLAY
        """

        before_path = Path(
            before_image_path
        )

        after_path = Path(
            after_image_path
        )

        if not before_path.exists():
            raise FileNotFoundError(
                f"Before image not found: "
                f"{before_image_path}"
            )

        if not after_path.exists():
            raise FileNotFoundError(
                f"After image not found: "
                f"{after_image_path}"
            )

        before = Image.open(
            before_path
        ).convert(
            "RGB"
        )

        after = Image.open(
            after_path
        ).convert(
            "RGB"
        )

        if before.size != after.size:
            after = after.resize(
                before.size,
                resample=(
                    Image.Resampling.BILINEAR
                ),
            )

        binary_mask = (
            self._normalize_mask(
                mask
            )
        )

        if (
            binary_mask.shape[1]
            != after.width
            or binary_mask.shape[0]
            != after.height
        ):
            binary_mask = (
                self._resize_mask(
                    binary_mask,
                    after.size,
                )
            )

        after_array = np.asarray(
            after
        ).astype(
            np.float32
        )

        mask_bool = (
            binary_mask.astype(
                bool
            )
        )

        overlay_array = (
            after_array.copy()
        )

        red = np.array(
            [
                255,
                0,
                0,
            ],
            dtype=np.float32,
        )

        overlay_array[
            mask_bool
        ] = (
            after_array[
                mask_bool
            ]
            * (
                1.0
                - alpha
            )
            + red
            * alpha
        )

        overlay = Image.fromarray(
            np.clip(
                overlay_array,
                0,
                255,
            ).astype(
                np.uint8
            ),
            mode="RGB",
        )

        width = before.width
        height = before.height

        header_height = 40

        comparison = Image.new(
            "RGB",
            (
                width * 3,
                height
                + header_height,
            ),
            (
                255,
                255,
                255,
            ),
        )

        comparison.paste(
            before,
            (
                0,
                header_height,
            ),
        )

        comparison.paste(
            after,
            (
                width,
                header_height,
            ),
        )

        comparison.paste(
            overlay,
            (
                width * 2,
                header_height,
            ),
        )

        draw = ImageDraw.Draw(
            comparison
        )

        draw.text(
            (
                10,
                12,
            ),
            "BEFORE",
            fill=(
                0,
                0,
                0,
            ),
        )

        draw.text(
            (
                width + 10,
                12,
            ),
            "AFTER",
            fill=(
                0,
                0,
                0,
            ),
        )

        draw.text(
            (
                width * 2 + 10,
                12,
            ),
            "DETECTED CHANGE",
            fill=(
                255,
                0,
                0,
            ),
        )

        output = (
            self._prepare_output(
                output_path
            )
        )

        comparison.save(
            output
        )

        return str(
            output
        )

    def create_change_evidence(
        self,
        before_image_path: str,
        after_image_path: str,
        mask: np.ndarray,
        output_dir: str,
        prefix: str = "changeformer",
        alpha: float = 0.55,
    ) -> Dict[str, str]:
        """
        Generate the complete visual-evidence package.

        Outputs:
            binary mask
            before-image overlay
            after-image overlay
            before/after/comparison panel
        """

        output_directory = Path(
            output_dir
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        mask_path = (
            output_directory
            / f"{prefix}_change_mask.png"
        )

        before_overlay_path = (
            output_directory
            / f"{prefix}_before_overlay.png"
        )

        after_overlay_path = (
            output_directory
            / f"{prefix}_after_overlay.png"
        )

        comparison_path = (
            output_directory
            / f"{prefix}_comparison.png"
        )

        self.create_binary_mask(
            mask=mask,
            output_path=str(
                mask_path
            ),
        )

        self.overlay_mask(
            base_image_path=(
                before_image_path
            ),
            mask=mask,
            output_path=str(
                before_overlay_path
            ),
            alpha=alpha,
        )

        self.overlay_mask(
            base_image_path=(
                after_image_path
            ),
            mask=mask,
            output_path=str(
                after_overlay_path
            ),
            alpha=alpha,
        )

        self.create_change_comparison(
            before_image_path=(
                before_image_path
            ),
            after_image_path=(
                after_image_path
            ),
            mask=mask,
            output_path=str(
                comparison_path
            ),
            alpha=alpha,
        )

        return {
            "change_mask": str(
                mask_path
            ),
            "before_overlay": str(
                before_overlay_path
            ),
            "after_overlay": str(
                after_overlay_path
            ),
            "comparison": str(
                comparison_path
            ),
        }


__all__ = [
    "EvidenceOverlay",
]