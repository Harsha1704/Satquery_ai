"""
SatQuery AI - Sentinel-1 SAR analysis.
"""

from typing import Dict

import numpy as np

from .config import EPSILON


class SARAnalyzer:
    """
    Analyze Sentinel-1 VV/VH SAR imagery.
    """

    @staticmethod
    def _validate(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)

        if data.ndim != 3:
            raise ValueError(
                "SAR data must have shape (bands, height, width)."
            )

        if data.shape[0] < 1:
            raise ValueError("SAR data contains no bands.")

        return data.astype(np.float32)

    @staticmethod
    def _stats(array: np.ndarray) -> Dict:
        array = np.asarray(array, dtype=np.float32)

        return {
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "std": float(np.std(array)),
        }

    def vv_statistics(self, data: np.ndarray) -> Dict:
        data = self._validate(data)

        return self._stats(data[0])

    def vh_statistics(self, data: np.ndarray) -> Dict:
        data = self._validate(data)

        if data.shape[0] < 2:
            raise ValueError(
                "VH band is required for VH statistics."
            )

        return self._stats(data[1])

    def vv_vh_ratio(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate VV/VH ratio.
        """

        data = self._validate(data)

        if data.shape[0] < 2:
            raise ValueError(
                "Both VV and VH bands are required."
            )

        vv = data[0]
        vh = data[1]

        return vv / (vh + EPSILON)

    def vv_vh_ratio_statistics(self, data: np.ndarray) -> Dict:
        ratio = self.vv_vh_ratio(data)

        return self._stats(ratio)

    @staticmethod
    def to_db(data: np.ndarray) -> np.ndarray:
        """
        Convert linear SAR intensity/power to dB.

        Formula:

            dB = 10 * log10(power)
        """

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        return 10.0 * np.log10(
            np.maximum(data, EPSILON)
        )

    def summary(self, data: np.ndarray) -> Dict:
        """
        Complete SAR statistical analysis.
        """

        data = self._validate(data)

        result = {
            "bands": int(data.shape[0]),
            "height": int(data.shape[1]),
            "width": int(data.shape[2]),
            "vv": self.vv_statistics(data),
        }

        if data.shape[0] >= 2:
            result["vh"] = self.vh_statistics(data)
            result["vv_vh_ratio"] = (
                self.vv_vh_ratio_statistics(data)
            )

        return result