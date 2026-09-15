"""
Spectral indices for SatQuery AI.

Supported indices:

NDVI  - Normalized Difference Vegetation Index
NDWI  - Normalized Difference Water Index
NDBI  - Normalized Difference Built-up Index
"""

import numpy as np


def _validate_bands(band_a, band_b):
    """
    Validate two spectral bands.
    """

    band_a = np.asarray(band_a, dtype=np.float32)
    band_b = np.asarray(band_b, dtype=np.float32)

    if band_a.shape != band_b.shape:
        raise ValueError(
            "Spectral bands must have identical shapes."
        )

    return band_a, band_b


def normalized_difference(
    band_a,
    band_b,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Generic normalized difference:

        (A - B) / (A + B)
    """

    band_a, band_b = _validate_bands(
        band_a,
        band_b
    )

    denominator = band_a + band_b

    result = np.divide(
        band_a - band_b,
        denominator,
        out=np.zeros_like(band_a),
        where=np.abs(denominator) > epsilon
    )

    return result.astype(np.float32)


def ndvi(
    red,
    nir,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate NDVI.

    NDVI = (NIR - Red) / (NIR + Red)
    """

    return normalized_difference(
        nir,
        red,
        epsilon
    )


def ndwi(
    green,
    nir,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate NDWI.

    NDWI = (Green - NIR) / (Green + NIR)
    """

    return normalized_difference(
        green,
        nir,
        epsilon
    )


def ndbi(
    nir,
    swir,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Calculate NDBI.

    NDBI = (SWIR - NIR) / (SWIR + NIR)
    """

    return normalized_difference(
        swir,
        nir,
        epsilon
    )


def index_statistics(index: np.ndarray) -> dict:
    """
    Calculate basic statistics for a spectral index.
    """

    index = np.asarray(
        index,
        dtype=np.float32
    )

    if index.size == 0:
        raise ValueError(
            "Spectral index cannot be empty."
        )

    finite = index[np.isfinite(index)]

    if finite.size == 0:
        raise ValueError(
            "Spectral index contains no finite values."
        )

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
    }