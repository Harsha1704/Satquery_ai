# tests/test_geotiff_support.py

from pathlib import Path
from pprint import pprint

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ai.validation import InputValidator


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "test" / "geotiff"
TEST_DIR.mkdir(parents=True, exist_ok=True)

OPTICAL_PATH = TEST_DIR / "sentinel2_test.tif"
SAR_PATH = TEST_DIR / "sar_test.tif"
BEFORE_PATH = TEST_DIR / "before_test.tif"
AFTER_PATH = TEST_DIR / "after_test.tif"


def create_optical_geotiff():

    height = 120
    width = 120
    bands = 12

    rng = np.random.default_rng(42)

    data = rng.integers(
        low=0,
        high=10000,
        size=(bands, height, width),
        dtype=np.uint16,
    )

    transform = from_origin(
        77.0000,
        31.0000,
        10.0,
        10.0,
    )

    with rasterio.open(
        OPTICAL_PATH,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=data.dtype,
        crs="EPSG:32643",
        transform=transform,
        nodata=0,
    ) as dataset:

        dataset.write(data)

        band_names = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]

        for index, name in enumerate(
            band_names,
            start=1,
        ):
            dataset.set_band_description(
                index,
                name,
            )


def create_sar_geotiff():

    height = 120
    width = 120

    rng = np.random.default_rng(43)

    vv = rng.random(
        (height, width),
        dtype=np.float32,
    )

    vh = rng.random(
        (height, width),
        dtype=np.float32,
    )

    data = np.stack(
        [vv, vh],
        axis=0,
    )

    transform = from_origin(
        77.0000,
        31.0000,
        10.0,
        10.0,
    )

    with rasterio.open(
        SAR_PATH,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
        nodata=0.0,
    ) as dataset:

        dataset.write(data)

        dataset.set_band_description(
            1,
            "VV",
        )

        dataset.set_band_description(
            2,
            "VH",
        )


def create_change_pair():

    height = 256
    width = 256

    before = np.zeros(
        (3, height, width),
        dtype=np.uint8,
    )

    after = before.copy()

    before[:, 40:100, 40:100] = 100

    after[:, 40:100, 40:100] = 100
    after[:, 140:200, 140:200] = 220

    transform = from_origin(
        500000.0,
        3500000.0,
        0.5,
        0.5,
    )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": transform,
        "nodata": 0,
    }

    with rasterio.open(
        BEFORE_PATH,
        "w",
        **profile,
    ) as dataset:
        dataset.write(before)

    with rasterio.open(
        AFTER_PATH,
        "w",
        **profile,
    ) as dataset:
        dataset.write(after)


def inspect_with_rasterio(path):

    with rasterio.open(path) as dataset:

        return {
            "driver": dataset.driver,
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": dataset.dtypes,
            "crs": (
                dataset.crs.to_string()
                if dataset.crs
                else None
            ),
            "transform": tuple(
                dataset.transform
            )[:6],
            "resolution": dataset.res,
            "bounds": tuple(
                dataset.bounds
            ),
            "nodata": dataset.nodata,
            "band_descriptions": (
                dataset.descriptions
            ),
        }


def main():

    print("\n" + "=" * 70)
    print("CREATING GEOTIFF TEST DATA")
    print("=" * 70)

    create_optical_geotiff()
    create_sar_geotiff()
    create_change_pair()

    print("OPTICAL:", OPTICAL_PATH)
    print("SAR:", SAR_PATH)
    print("BEFORE:", BEFORE_PATH)
    print("AFTER:", AFTER_PATH)

    validator = InputValidator()

    print("\n" + "=" * 70)
    print("1. OPTICAL GEOTIFF METADATA")
    print("=" * 70)

    pprint(
        inspect_with_rasterio(
            OPTICAL_PATH
        )
    )

    print("\nVALIDATOR RESULT:")

    optical_result = (
        validator.validate_multispectral(
            str(OPTICAL_PATH)
        )
    )

    pprint(optical_result)

    assert optical_result["valid"] is True

    assert (
        optical_result["metadata"]["bands"]
        == 12
    )

    assert (
        optical_result["metadata"]["crs"]
        == "EPSG:32643"
    )

    assert (
        optical_result["metadata"][
            "georeferenced"
        ]
        is True
    )

    print("\n" + "=" * 70)
    print("2. SAR GEOTIFF METADATA")
    print("=" * 70)

    pprint(
        inspect_with_rasterio(
            SAR_PATH
        )
    )

    print("\nVALIDATOR RESULT:")

    sar_result = (
        validator.validate_single_image(
            str(SAR_PATH),
            expected_modality="sar",
        )
    )

    pprint(sar_result)

    assert sar_result["valid"] is True
    assert sar_result["metadata"]["bands"] == 2
    assert sar_result["metadata"]["crs"] == "EPSG:32643"

    print("\n" + "=" * 70)
    print("3. OPTICAL + SAR GEOSPATIAL COMPATIBILITY")
    print("=" * 70)

    fusion_result = (
        validator.validate_optical_sar_pair(
            optical_path=str(
                OPTICAL_PATH
            ),
            sar_path=str(
                SAR_PATH
            ),
        )
    )

    pprint(fusion_result)

    assert fusion_result["valid"] is True

    assert fusion_result[
        "error_count"
    ] == 0

    print("\n" + "=" * 70)
    print("4. BI-TEMPORAL GEOTIFF COMPATIBILITY")
    print("=" * 70)

    change_result = (
        validator.validate_change_pair(
            before_path=str(
                BEFORE_PATH
            ),
            after_path=str(
                AFTER_PATH
            ),
        )
    )

    pprint(change_result)

    assert change_result["valid"] is True

    assert change_result[
        "error_count"
    ] == 0

    print("\n" + "=" * 70)
    print("5. DIRECT RASTER READ TEST")
    print("=" * 70)

    with rasterio.open(
        OPTICAL_PATH
    ) as dataset:

        optical_array = dataset.read()

    with rasterio.open(
        SAR_PATH
    ) as dataset:

        sar_array = dataset.read()

    print(
        "OPTICAL ARRAY:",
        optical_array.shape,
        optical_array.dtype,
    )

    print(
        "SAR ARRAY:",
        sar_array.shape,
        sar_array.dtype,
    )

    assert optical_array.shape == (
        12,
        120,
        120,
    )

    assert sar_array.shape == (
        2,
        120,
        120,
    )

    print("\n" + "=" * 70)
    print("PRIORITY 12 GEOTIFF/TIFF SUPPORT: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()