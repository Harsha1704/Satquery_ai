import numpy as np


class SpectralIndices:

    @staticmethod
    def normalized_difference(
        band_a: np.ndarray,
        band_b: np.ndarray
    ) -> np.ndarray:

        from ai.multispectral.indices import normalized_difference
        return normalized_difference(band_a, band_b)

    def ndvi(
        self,
        red: np.ndarray,
        nir: np.ndarray
    ) -> np.ndarray:

        return self.normalized_difference(nir, red)

    def ndwi(
        self,
        green: np.ndarray,
        nir: np.ndarray
    ) -> np.ndarray:

        return self.normalized_difference(green, nir)

    def ndbi(
        self,
        nir: np.ndarray,
        swir: np.ndarray
    ) -> np.ndarray:

        return self.normalized_difference(swir, nir)
