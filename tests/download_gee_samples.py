from pathlib import Path
import shutil

import rasterio

import gee_temporal


OUTPUT_DIR = Path("outputs") / "temporal_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SAMPLES = [
    {
        "name": "delhi",
        "bounds": [77.20, 28.55, 77.24, 28.59],
        "before_year": 2010,
        "after_year": 2025,
    },
    {
        "name": "mumbai",
        "bounds": [72.84, 18.95, 72.88, 18.99],
        "before_year": 2018,
        "after_year": 2025,
    },
    {
        "name": "bengaluru",
        "bounds": [77.56, 12.95, 77.60, 12.99],
        "before_year": 2020,
        "after_year": 2025,
    },
]


def move_result(source_path: Path, target_path: Path) -> Path:
    source_path = Path(source_path)
    target_path = Path(target_path)

    if target_path.exists():
        target_path.unlink()

    shutil.move(
        str(source_path),
        str(target_path),
    )

    return target_path


def print_raster_info(path: Path):
    with rasterio.open(path) as src:
        print(f"File       : {path}")
        print(f"CRS        : {src.crs}")
        print(f"Bands      : {src.count}")
        print(f"Width      : {src.width}")
        print(f"Height     : {src.height}")
        print(f"Bounds     : {src.bounds}")
        print(f"Data types : {src.dtypes}")
        print("-" * 70)


def download_pair(sample):
    name = sample["name"]
    bounds = sample["bounds"]
    before_year = sample["before_year"]
    after_year = sample["after_year"]

    print()
    print("=" * 70)
    print(
        f"DOWNLOADING {name.upper()} "
        f"{before_year} → {after_year}"
    )
    print("=" * 70)

    plan = gee_temporal.plan_temporal_fetch(
        bounds_wgs84=bounds,
        before_year=before_year,
        after_year=after_year,
    )

    print("Before sensor :", plan.before_spec.get("label"))
    print("After sensor  :", plan.after_spec.get("label"))
    print("Before images :", plan.before_count)
    print("After images  :", plan.after_count)
    print("Target scale  :", plan.target_scale_m)
    print(
        "Dimensions    :",
        plan.width_px,
        "x",
        plan.height_px,
    )
    print("Cross sensor  :", plan.cross_sensor)

    dimensions = (
        plan.width_px,
        plan.height_px,
    )

    before = gee_temporal.fetch_year_composite(
        bounds_wgs84=bounds,
        year=before_year,
        workdir=OUTPUT_DIR,
        spec=plan.before_spec,
        dimensions=dimensions,
    )

    after = gee_temporal.fetch_year_composite(
        bounds_wgs84=bounds,
        year=after_year,
        workdir=OUTPUT_DIR,
        spec=plan.after_spec,
        dimensions=dimensions,
    )

    before_target = (
        OUTPUT_DIR
        / f"{name}_{before_year}.tif"
    )

    after_target = (
        OUTPUT_DIR
        / f"{name}_{after_year}.tif"
    )

    before_target = move_result(
        before.path,
        before_target,
    )

    after_target = move_result(
        after.path,
        after_target,
    )

    print()
    print("BEFORE SPECTRAL STATS")
    print(before.stats)

    print()
    print("AFTER SPECTRAL STATS")
    print(after.stats)

    print()
    print("RASTER INFORMATION")
    print_raster_info(before_target)
    print_raster_info(after_target)


def main():
    print("=" * 70)
    print("SATQUERY AI - GEE GEOTIFF SAMPLE DOWNLOADER")
    print("=" * 70)

    gee_temporal.init_earth_engine()

    print("Earth Engine: READY")

    for sample in SAMPLES:
        try:
            download_pair(sample)

        except Exception as exc:
            print()
            print(
                f"FAILED: {sample['name']}"
            )
            print(
                type(exc).__name__,
                ":",
                exc,
            )

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(
        "Files saved in:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()