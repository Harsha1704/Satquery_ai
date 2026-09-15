"""
Sentinel-2 multispectral analyzer for SatQuery AI.

This module converts the 12-band Sentinel-2 data into
interpretable spectral information.
"""

from typing import Dict

import numpy as np

from .band_config import get_band_index
from .indices import (
    ndvi,
    ndwi,
    ndbi,
    index_statistics,
)


class MultispectralAnalyzer:
    """
    Analyze Sentinel-2 multispectral imagery.
    """

    def __init__(self):
        self.required_bands = {
            "red": get_band_index("B04"),
            "green": get_band_index("B03"),
            "nir": get_band_index("B08"),
            "swir1": get_band_index("B11"),
        }

    def _validate_data(
        self,
        data: np.ndarray
    ) -> np.ndarray:

        data = np.asarray(
            data,
            dtype=np.float32
        )

        if data.ndim != 3:
            raise ValueError(
                "Sentinel-2 data must have shape "
                "(bands, height, width)."
            )

        if data.shape[0] < 12:
            raise ValueError(
                "Expected at least 12 Sentinel-2 bands."
            )

        return data

    def calculate_indices(
        self,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate NDVI, NDWI and NDBI.
        """

        data = self._validate_data(data)

        red = data[
            self.required_bands["red"]
        ]

        green = data[
            self.required_bands["green"]
        ]

        nir = data[
            self.required_bands["nir"]
        ]

        swir1 = data[
            self.required_bands["swir1"]
        ]

        return {
            "ndvi": ndvi(red, nir),
            "ndwi": ndwi(green, nir),
            "ndbi": ndbi(nir, swir1),
        }

    def analyze(
        self,
        data: np.ndarray
    ) -> Dict:

        data = self._validate_data(data)

        indices = self.calculate_indices(data)

        statistics = {
            name: index_statistics(value)
            for name, value in indices.items()
        }

        return {
            "bands": int(data.shape[0]),
            "height": int(data.shape[1]),
            "width": int(data.shape[2]),
            "indices": statistics,
        }

    def vegetation_mask(
        self,
        data: np.ndarray,
        threshold: float = 0.30
    ) -> np.ndarray:

        indices = self.calculate_indices(data)

        return (
            indices["ndvi"] >= threshold
        ).astype(np.uint8)

    def water_mask(
        self,
        data: np.ndarray,
        threshold: float = 0.20
    ) -> np.ndarray:

        indices = self.calculate_indices(data)

        return (
            indices["ndwi"] >= threshold
        ).astype(np.uint8)

    def built_up_mask(
        self,
        data: np.ndarray,
        threshold: float = 0.00
    ) -> np.ndarray:

        indices = self.calculate_indices(data)

        return (
            indices["ndbi"] >= threshold
        ).astype(np.uint8)

    @staticmethod
    def coverage_percentage(
        mask: np.ndarray
    ) -> float:

        mask = np.asarray(mask)

        if mask.size == 0:
            return 0.0

        return float(
            np.mean(mask > 0) * 100.0
        )

    def summary(
        self,
        data: np.ndarray
    ) -> Dict:

        data = self._validate_data(data)

        indices = self.calculate_indices(data)

        vegetation = (
            indices["ndvi"] >= 0.30
        )

        water = (
            indices["ndwi"] >= 0.20
        )

        built_up = (
            indices["ndbi"] >= 0.00
        )

        return {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "ndvi": index_statistics(
                indices["ndvi"]
            ),
            "ndwi": index_statistics(
                indices["ndwi"]
            ),
            "ndbi": index_statistics(
                indices["ndbi"]
            ),
            "vegetation_coverage_percent":
                float(np.mean(vegetation) * 100),
            "water_coverage_percent":
                float(np.mean(water) * 100),
            "built_up_coverage_percent":
                float(np.mean(built_up) * 100),
        }