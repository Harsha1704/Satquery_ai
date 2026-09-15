from pathlib import Path

import rasterio


SUPPORTED_RASTER_EXTENSIONS = {
    ".tif",
    ".tiff"
}


class RasterLoader:
    """Load and inspect geospatial raster datasets."""

    def validate_path(self, raster_path: str) -> Path:
        path = Path(raster_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Raster file not found: {raster_path}"
            )

        if path.suffix.lower() not in SUPPORTED_RASTER_EXTENSIONS:
            raise ValueError(
                f"Unsupported raster format: {path.suffix}"
            )

        return path

    def open(self, raster_path: str):
        path = self.validate_path(raster_path)

        return rasterio.open(path)

    def read_band(
        self,
        raster_path: str,
        band_number: int
    ):
        with self.open(raster_path) as dataset:

            if band_number < 1 or band_number > dataset.count:
                raise ValueError(
                    f"Band {band_number} does not exist. "
                    f"Available bands: 1-{dataset.count}"
                )

            return dataset.read(band_number)

    def read_all_bands(self, raster_path: str):
        with self.open(raster_path) as dataset:
            return dataset.read()