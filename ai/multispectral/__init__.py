"""
SatQuery AI multispectral analysis package.
"""

from .band_config import (
    SENTINEL2_BANDS,
    BAND_NAMES,
    BAND_INDEX,
    get_band_index,
    get_band_name,
)

from .indices import (
    ndvi,
    ndwi,
    ndbi,
    normalized_difference,
    index_statistics,
)

from .analyzer import (
    MultispectralAnalyzer,
)


__all__ = [
    "SENTINEL2_BANDS",
    "BAND_NAMES",
    "BAND_INDEX",
    "get_band_index",
    "get_band_name",
    "ndvi",
    "ndwi",
    "ndbi",
    "normalized_difference",
    "index_statistics",
    "MultispectralAnalyzer",
]