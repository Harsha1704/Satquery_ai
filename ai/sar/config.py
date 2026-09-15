"""
SatQuery AI - Sentinel-1 SAR configuration.
"""

SAR_BANDS = {
    0: "VV",
    1: "VH",
}

SAR_BAND_INDEX = {
    "VV": 0,
    "VH": 1,
}

SUPPORTED_SAR_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".npy",
}

EPSILON = 1e-8


def get_band_index(band_name: str) -> int:
    """
    Return the zero-based channel index for a SAR band.
    """
    band_name = band_name.upper()

    if band_name not in SAR_BAND_INDEX:
        raise ValueError(
            f"Unsupported SAR band: {band_name}. "
            f"Available bands: {list(SAR_BAND_INDEX)}"
        )

    return SAR_BAND_INDEX[band_name]