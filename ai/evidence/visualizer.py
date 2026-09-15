from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class EvidenceVisualizer:
    """
    CPU-friendly visualization utilities for SatQuery-AI.

    Supported visual evidence:
        - NDVI
        - NDWI
        - NDBI
        - Semantic masks
        - SAR VV
        - SAR VH
        - SAR VV/VH ratio
        - Generic grayscale arrays
        - Before/after change maps
        - RGB previews from multispectral data
    """

    # ------------------------------------------------------------------
    # BASIC NORMALIZATION
    # ------------------------------------------------------------------

    def normalize(
        self,
        array: np.ndarray,
        lower_percentile: float = 2.0,
        upper_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Normalize an array to uint8 [0, 255].

        Percentile clipping makes the visualization more robust
        to extreme values.
        """

        array = np.asarray(array)

        if array.size == 0:
            raise ValueError(
                "Cannot visualize an empty array."
            )

        array = array.astype(
            np.float32,
            copy=False,
        )

        valid = np.isfinite(array)

        if not np.any(valid):
            return np.zeros(
                array.shape,
                dtype=np.uint8,
            )

        values = array[valid]

        low = np.percentile(
            values,
            lower_percentile,
        )

        high = np.percentile(
            values,
            upper_percentile,
        )

        if high <= low:
            low = float(np.min(values))
            high = float(np.max(values))

        if high <= low:
            return np.zeros(
                array.shape,
                dtype=np.uint8,
            )

        normalized = (
            (array - low)
            / (high - low)
            * 255.0
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        return np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # SAVE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_output(
        output_path: str,
    ) -> Path:
        """
        Create output directory and return output path.
        """

        output = Path(output_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output

    @staticmethod
    def _save_grayscale(
        array: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Save a uint8 2D array as grayscale PNG/image.
        """

        output = EvidenceVisualizer._prepare_output(
            output_path
        )

        image = Image.fromarray(
            array,
            mode="L",
        )

        image.save(output)

        return str(output)

    # ------------------------------------------------------------------
    # GENERIC MAP
    # ------------------------------------------------------------------

    def create_grayscale_map(
        self,
        array: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create a generic grayscale evidence image.
        """

        array = np.asarray(array)

        if array.ndim != 2:
            raise ValueError(
                "Grayscale map requires a 2D array."
            )

        normalized = self.normalize(
            array
        )

        return self._save_grayscale(
            normalized,
            output_path,
        )

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    def create_ndvi_map(
        self,
        ndvi: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create an NDVI visualization.

        NDVI normally ranges approximately from -1 to +1.
        """

        ndvi = np.asarray(ndvi)

        if ndvi.ndim != 2:
            raise ValueError(
                "NDVI map requires a 2D array."
            )

        # Fixed NDVI range provides consistent interpretation
        # between different images.
        clipped = np.clip(
            ndvi.astype(np.float32),
            -1.0,
            1.0,
        )

        normalized = (
            (clipped + 1.0)
            / 2.0
            * 255.0
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        return self._save_grayscale(
            normalized,
            output_path,
        )

    # ------------------------------------------------------------------
    # NDWI
    # ------------------------------------------------------------------

    def create_ndwi_map(
        self,
        ndwi: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create an NDWI visualization.

        NDWI is normalized using the common -1 to +1 range.
        """

        ndwi = np.asarray(ndwi)

        if ndwi.ndim != 2:
            raise ValueError(
                "NDWI map requires a 2D array."
            )

        clipped = np.clip(
            ndwi.astype(np.float32),
            -1.0,
            1.0,
        )

        normalized = (
            (clipped + 1.0)
            / 2.0
            * 255.0
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        return self._save_grayscale(
            normalized,
            output_path,
        )

    # ------------------------------------------------------------------
    # NDBI
    # ------------------------------------------------------------------

    def create_ndbi_map(
        self,
        ndbi: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create an NDBI visualization.

        NDBI is normalized using the common -1 to +1 range.
        """

        ndbi = np.asarray(ndbi)

        if ndbi.ndim != 2:
            raise ValueError(
                "NDBI map requires a 2D array."
            )

        clipped = np.clip(
            ndbi.astype(np.float32),
            -1.0,
            1.0,
        )

        normalized = (
            (clipped + 1.0)
            / 2.0
            * 255.0
        )

        normalized = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        normalized = np.clip(
            normalized,
            0,
            255,
        ).astype(np.uint8)

        return self._save_grayscale(
            normalized,
            output_path,
        )

    # ------------------------------------------------------------------
    # SEMANTIC MASK
    # ------------------------------------------------------------------

    def create_semantic_mask(
        self,
        mask: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Save a semantic segmentation mask.

        Each class ID is preserved as a grayscale intensity.
        """

        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                "Semantic mask requires a 2D array."
            )

        if not np.issubdtype(
            mask.dtype,
            np.integer,
        ):
            mask = np.rint(mask)

        mask = mask.astype(
            np.int32,
            copy=False,
        )

        maximum = int(
            np.max(mask)
        )

        if maximum <= 0:
            normalized = np.zeros(
                mask.shape,
                dtype=np.uint8,
            )
        else:
            normalized = (
                mask.astype(np.float32)
                / maximum
                * 255.0
            ).astype(np.uint8)

        return self._save_grayscale(
            normalized,
            output_path,
        )

    # ------------------------------------------------------------------
    # SAR VV
    # ------------------------------------------------------------------

    def create_sar_vv_map(
        self,
        vv: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create SAR VV grayscale visualization.
        """

        vv = np.asarray(vv)

        if vv.ndim != 2:
            raise ValueError(
                "SAR VV map requires a 2D array."
            )

        return self.create_grayscale_map(
            vv,
            output_path,
        )

    # ------------------------------------------------------------------
    # SAR VH
    # ------------------------------------------------------------------

    def create_sar_vh_map(
        self,
        vh: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create SAR VH grayscale visualization.
        """

        vh = np.asarray(vh)

        if vh.ndim != 2:
            raise ValueError(
                "SAR VH map requires a 2D array."
            )

        return self.create_grayscale_map(
            vh,
            output_path,
        )

    # ------------------------------------------------------------------
    # SAR VV/VH RATIO
    # ------------------------------------------------------------------

    def create_sar_ratio_map(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create a VV/VH ratio visualization.
        """

        vv = np.asarray(vv)
        vh = np.asarray(vh)

        if vv.ndim != 2 or vh.ndim != 2:
            raise ValueError(
                "VV and VH must both be 2D arrays."
            )

        if vv.shape != vh.shape:
            raise ValueError(
                "VV and VH must have identical spatial dimensions."
            )

        denominator = np.where(
            np.abs(vh) < 1e-6,
            1e-6,
            vh,
        )

        ratio = (
            vv.astype(np.float32)
            / denominator.astype(np.float32)
        )

        return self.create_grayscale_map(
            ratio,
            output_path,
        )

    # ------------------------------------------------------------------
    # MULTISPECTRAL RGB PREVIEW
    # ------------------------------------------------------------------

    def create_rgb_preview(
        self,
        data: np.ndarray,
        output_path: str,
        red_band: int = 3,
        green_band: int = 2,
        blue_band: int = 1,
    ) -> str:
        """
        Create an RGB preview from multispectral data.

        Expected input:
            (bands, H, W)

        Band indices are zero-based.
        """

        data = np.asarray(data)

        if data.ndim != 3:
            raise ValueError(
                "Multispectral data must have shape "
                "(bands, H, W)."
            )

        bands = data.shape[0]

        for index in (
            red_band,
            green_band,
            blue_band,
        ):
            if index < 0 or index >= bands:
                raise ValueError(
                    f"Band index {index} is invalid "
                    f"for {bands} bands."
                )

        red = self.normalize(
            data[red_band]
        )

        green = self.normalize(
            data[green_band]
        )

        blue = self.normalize(
            data[blue_band]
        )

        rgb = np.stack(
            [
                red,
                green,
                blue,
            ],
            axis=-1,
        )

        output = self._prepare_output(
            output_path
        )

        image = Image.fromarray(
            rgb,
            mode="RGB",
        )

        image.save(output)

        return str(output)

    # ------------------------------------------------------------------
    # CHANGE MAP
    # ------------------------------------------------------------------

    def create_change_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
        output_path: str,
        threshold: float = 0.0,
    ) -> str:
        """
        Create a binary change map from two 2D arrays.

        Pixels whose absolute difference is greater than
        threshold are marked as changed.
        """

        before = np.asarray(before)
        after = np.asarray(after)

        if before.ndim != 2:
            raise ValueError(
                "Before image must be 2D."
            )

        if after.ndim != 2:
            raise ValueError(
                "After image must be 2D."
            )

        if before.shape != after.shape:
            raise ValueError(
                "Before and after images must have "
                "the same spatial dimensions."
            )

        difference = np.abs(
            before.astype(np.float32)
            - after.astype(np.float32)
        )

        changed = (
            difference > threshold
        ).astype(np.uint8) * 255

        return self._save_grayscale(
            changed,
            output_path,
        )

    # ------------------------------------------------------------------
    # CHANGE DIFFERENCE MAP
    # ------------------------------------------------------------------

    def create_difference_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
        output_path: str,
    ) -> str:
        """
        Create a normalized absolute difference map.
        """

        before = np.asarray(before)
        after = np.asarray(after)

        if before.ndim != 2:
            raise ValueError(
                "Before image must be 2D."
            )

        if after.ndim != 2:
            raise ValueError(
                "After image must be 2D."
            )

        if before.shape != after.shape:
            raise ValueError(
                "Before and after images must have "
                "the same spatial dimensions."
            )

        difference = np.abs(
            before.astype(np.float32)
            - after.astype(np.float32)
        )

        return self.create_grayscale_map(
            difference,
            output_path,
        )

    # ------------------------------------------------------------------
    # MULTISPECTRAL BAND MAP
    # ------------------------------------------------------------------

    def create_band_map(
        self,
        data: np.ndarray,
        band_index: int,
        output_path: str,
    ) -> str:
        """
        Create a visualization for one multispectral band.
        """

        data = np.asarray(data)

        if data.ndim != 3:
            raise ValueError(
                "Expected multispectral data with "
                "shape (bands, H, W)."
            )

        if (
            band_index < 0
            or band_index >= data.shape[0]
        ):
            raise ValueError(
                f"Invalid band index {band_index}. "
                f"Available bands: {data.shape[0]}."
            )

        return self.create_grayscale_map(
            data[band_index],
            output_path,
        )