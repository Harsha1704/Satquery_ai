from typing import Dict

import numpy as np


class FusionAnalyzer:
    """
    Analyze fused optical-SAR features.
    """

    def analyze(
        self,
        fused: np.ndarray
    ) -> Dict:

        fused = np.asarray(fused)

        if fused.ndim != 3:
            raise ValueError(
                "Fused data must have shape (bands, H, W)."
            )

        bands, height, width = fused.shape

        band_statistics = {}

        for index in range(bands):

            band = fused[index]

            band_statistics[index] = {
                "min": float(np.min(band)),
                "max": float(np.max(band)),
                "mean": float(np.mean(band)),
                "median": float(np.median(band)),
                "std": float(np.std(band)),
            }

        return {
            "bands": int(bands),
            "height": int(height),
            "width": int(width),
            "shape": tuple(fused.shape),
            "statistics": band_statistics,
        }

    def describe(
        self,
        result: Dict
    ) -> str:

        return (
            f"Optical-SAR fusion produced "
            f"{result['bands']} feature bands over a "
            f"{result['height']}x{result['width']} spatial grid."
        )