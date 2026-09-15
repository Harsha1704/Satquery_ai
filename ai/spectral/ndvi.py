import numpy as np


class NDVIEngine:
    """
    Calculate and interpret NDVI from Red and NIR bands.
    """

    def calculate(
        self,
        red_band: np.ndarray,
        nir_band: np.ndarray
    ) -> np.ndarray:

        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)

        denominator = nir + red

        ndvi = np.divide(
            nir - red,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float32),
            where=denominator != 0
        )

        return np.clip(ndvi, -1.0, 1.0)

    def statistics(self, ndvi: np.ndarray) -> dict:

        valid = ndvi[np.isfinite(ndvi)]

        if valid.size == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "vegetation_percentage": 0.0
            }

        vegetation_pixels = np.sum(valid > 0.2)

        return {
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "vegetation_percentage": float(
                vegetation_pixels / valid.size * 100
            )
        }

    def classify(self, value: float) -> str:

        if value < -0.1:
            return "Water / very low vegetation"

        if value < 0.2:
            return "Bare soil / built-up / low vegetation"

        if value < 0.4:
            return "Sparse or moderate vegetation"

        if value < 0.6:
            return "Healthy vegetation"

        return "Dense healthy vegetation"