"""
Sentinel-2 band configuration for SatQuery AI.

The current Sentinel-2 dataset contains 12 bands:
B01, B02, B03, B04, B05, B06, B07, B08,
B8A, B09, B11, B12

Band indices below are zero-based.
"""

SENTINEL2_BANDS = {
    0: "B01",
    1: "B02",
    2: "B03",
    3: "B04",
    4: "B05",
    5: "B06",
    6: "B07",
    7: "B08",
    8: "B8A",
    9: "B09",
    10: "B11",
    11: "B12",
}


BAND_NAMES = {
    "B01": "Coastal Aerosol",
    "B02": "Blue",
    "B03": "Green",
    "B04": "Red",
    "B05": "Vegetation Red Edge 1",
    "B06": "Vegetation Red Edge 2",
    "B07": "Vegetation Red Edge 3",
    "B08": "Near Infrared",
    "B8A": "Narrow Near Infrared",
    "B09": "Water Vapour",
    "B11": "Short Wave Infrared 1",
    "B12": "Short Wave Infrared 2",
}


BAND_INDEX = {
    name: index
    for index, name in SENTINEL2_BANDS.items()
}


def get_band_index(band_name: str) -> int:
    """
    Return the zero-based array index for a Sentinel-2 band.
    """

    band_name = band_name.upper()

    if band_name not in BAND_INDEX:
        raise ValueError(
            f"Unknown Sentinel-2 band: {band_name}"
        )

    return BAND_INDEX[band_name]


def get_band_name(index: int) -> str:
    """
    Return Sentinel-2 band name from zero-based index.
    """

    if index not in SENTINEL2_BANDS:
        raise ValueError(
            f"Invalid Sentinel-2 band index: {index}"
        )

    return SENTINEL2_BANDS[index]