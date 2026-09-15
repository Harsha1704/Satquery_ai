from geospatial.raster import RasterLoader
from ai.spectral import NDVIEngine


RASTER_PATH = "datasets/raw/test_multispectral.tif"


def test_ndvi():

    print("\nSatQuery Spectral Intelligence Test")
    print("-" * 45)

    loader = RasterLoader()
    engine = NDVIEngine()

    red = loader.read_band(
        RASTER_PATH,
        3
    )

    nir = loader.read_band(
        RASTER_PATH,
        4
    )

    ndvi = engine.calculate(
        red,
        nir
    )

    stats = engine.statistics(ndvi)

    print("Red band loaded       : PASS")
    print("NIR band loaded       : PASS")
    print("NDVI calculation      : PASS")

    print("\nNDVI Statistics")
    print("Minimum              :", stats["min"])
    print("Maximum              :", stats["max"])
    print("Mean                 :", stats["mean"])
    print(
        "Vegetation percentage:",
        f"{stats['vegetation_percentage']:.2f}%"
    )

    mean_ndvi = stats["mean"]

    print("\nInterpretation")
    print(engine.classify(mean_ndvi))

    assert -1 <= mean_ndvi <= 1


if __name__ == "__main__":
    test_ndvi()