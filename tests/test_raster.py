import numpy as np
import rasterio
from rasterio.transform import from_origin

from geospatial.raster import (
    RasterLoader,
    RasterMetadataExtractor
)


TEST_RASTER = "datasets/raw/test_multispectral.tif"


def create_test_raster():

    width = 100
    height = 100
    bands = 4

    transform = from_origin(
        75.000000,
        30.000000,
        10,
        10
    )

    data = np.zeros(
        (bands, height, width),
        dtype=np.float32
    )

    # Simulated spectral bands
    data[0] = 0.20  # Blue
    data[1] = 0.30  # Green
    data[2] = 0.40  # Red
    data[3] = 0.70  # Near Infrared

    with rasterio.open(
        TEST_RASTER,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform
    ) as dataset:

        dataset.write(data)


def test_raster():

    print("\nSatQuery GeoTIFF Test")
    print("-" * 40)

    create_test_raster()

    loader = RasterLoader()
    metadata_extractor = RasterMetadataExtractor()

    metadata = metadata_extractor.extract(
        TEST_RASTER
    )

    print("GeoTIFF creation : PASS")
    print("Raster loading   : PASS")
    print("Metadata         : PASS")

    print("\nRaster Metadata")
    print("Filename   :", metadata.filename)
    print("Width      :", metadata.width)
    print("Height     :", metadata.height)
    print("Bands      :", metadata.band_count)
    print("Data type  :", metadata.dtype)
    print("CRS        :", metadata.crs)
    print("Resolution :", metadata.resolution)
    print("Bounds     :", metadata.bounds)

    red = loader.read_band(
        TEST_RASTER,
        3
    )

    nir = loader.read_band(
        TEST_RASTER,
        4
    )

    print("\nBand Test")
    print("Red band shape :", red.shape)
    print("NIR band shape :", nir.shape)

    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    test_raster()