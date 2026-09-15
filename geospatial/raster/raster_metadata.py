from dataclasses import dataclass
from typing import Optional

import rasterio


@dataclass
class RasterMetadata:
    filename: str
    width: int
    height: int
    band_count: int
    dtype: str
    crs: Optional[str]
    bounds: tuple
    resolution: tuple
    transform: str


class RasterMetadataExtractor:
    """Extract important geospatial metadata."""

    def extract(self, raster_path: str) -> RasterMetadata:

        with rasterio.open(raster_path) as dataset:

            crs_value = None

            if dataset.crs is not None:
                crs_value = dataset.crs.to_string()

            return RasterMetadata(
                filename=dataset.name.split("\\")[-1],
                width=dataset.width,
                height=dataset.height,
                band_count=dataset.count,
                dtype=str(dataset.dtypes[0]),
                crs=crs_value,
                bounds=(
                    dataset.bounds.left,
                    dataset.bounds.bottom,
                    dataset.bounds.right,
                    dataset.bounds.top
                ),
                resolution=(
                    dataset.res[0],
                    dataset.res[1]
                ),
                transform=str(dataset.transform)
            )