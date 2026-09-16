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

        from ai.multispectral.indices import normalized_difference
        ndvi = normalized_difference(nir_band, red_band)

        return np.clip(ndvi, -1.0, 1.0)

    def statistics(self, ndvi: np.ndarray) -> dict:

        valid = ndvi[np.isfinite(ndvi)]

        if valid.size == 0:
            raise ValueError("Spectral index contains no finite values.")

        vegetation_pixels = np.sum(valid > 0.2)

        return {
            "valid_pixel_count": int(valid.size),
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
