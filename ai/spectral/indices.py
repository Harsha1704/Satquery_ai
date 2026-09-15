import numpy as np


class SpectralIndices:

    @staticmethod
    def normalized_difference(
        band_a: np.ndarray,
        band_b: np.ndarray
    ) -> np.ndarray:

        a = band_a.astype(np.float32)
        b = band_b.astype(np.float32)

        denominator = a + b

        return np.divide(
            a - b,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0
        )

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