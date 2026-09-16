from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None


class MultispectralImage:
    """
    Container for a loaded multispectral or RGB image.

    Attributes
    ----------
    data : np.ndarray
        Image data in (bands, height, width) format.
    width : int
        Image width.
    height : int
        Image height.
    bands : int
        Number of spectral bands.
    crs : optional
        Coordinate Reference System for GeoTIFF images.
    transform : optional
        GeoTIFF affine transform.
    metadata : dict
        Raster metadata.
    path : str
        Source file path.
    """

    def __init__(
        self,
        data: np.ndarray,
        width: int,
        height: int,
        bands: int,
        crs=None,
        transform=None,
        metadata: Optional[Dict] = None,
        path: Optional[str] = None,
    ):
        self.data = data
        self.width = width
        self.height = height
        self.bands = bands
        self.crs = crs
        self.transform = transform
        self.metadata = metadata or {}
        self.path = path

    def __repr__(self) -> str:
        return (
            f"MultispectralImage("
            f"bands={self.bands}, "
            f"height={self.height}, "
            f"width={self.width}, "
            f"dtype={self.data.dtype}"
            f")"
        )

    def get_band(self, band_index: int) -> np.ndarray:
        """
        Return one band using zero-based indexing.
        """

        if band_index < 0 or band_index >= self.bands:
            raise IndexError(
                f"Band index {band_index} is out of range. "
                f"Available bands: 0-{self.bands - 1}"
            )

        return self.data[band_index]

    def to_hwc(self) -> np.ndarray:
        """
        Convert data from:

            (bands, height, width)

        to:

            (height, width, bands)
        """

        return np.transpose(self.data, (1, 2, 0))


class MultispectralLoader:
    """
    Unified image loader for SatQuery AI.

    Supported formats
    -----------------
    Standard RGB:
        .jpg
        .jpeg
        .png
        .jfif
        .bmp

    Geospatial:
        .tif
        .tiff
        .geotiff

    RGB images are converted to:

        (3, H, W)

    GeoTIFF images preserve their original number of bands:

        (B, H, W)

    GeoTIFF metadata such as CRS and transform is preserved.
    """

    RGB_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".jfif",
        ".bmp",
    }

    TIFF_EXTENSIONS = {
        ".tif",
        ".tiff",
        ".geotiff",
    }

    SUPPORTED_EXTENSIONS = RGB_EXTENSIONS | TIFF_EXTENSIONS

    def __init__(self):
        if rasterio is None:
            raise ImportError(
                "Rasterio is required for GeoTIFF support. "
                "Install it using: "
                "python -m pip install rasterio"
            )

    def load(
        self,
        image_path: Union[str, Path],
    ) -> MultispectralImage:
        """
        Load an RGB or multispectral image.

        Parameters
        ----------
        image_path:
            Path to image.

        Returns
        -------
        MultispectralImage
        """

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {extension}. "
                f"Supported formats: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if extension in self.TIFF_EXTENSIONS:
            return self._load_geotiff(path)

        return self._load_rgb(path)

    def _load_rgb(
        self,
        path: Path,
    ) -> MultispectralImage:
        """
        Load a normal RGB image.

        Output shape:

            (3, H, W)
        """

        image = Image.open(path).convert("RGB")

        array = np.asarray(image)

        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image, got shape {array.shape}"
            )

        # HWC -> CHW
        data = np.transpose(
            array,
            (2, 0, 1)
        )

        height, width = array.shape[:2]

        metadata = {
            "driver": "PIL",
            "format": path.suffix.lower(),
            "dtype": str(data.dtype),
            "bands": 3,
            "is_geospatial": False,
        }

        return MultispectralImage(
            data=data,
            width=width,
            height=height,
            bands=3,
            crs=None,
            transform=None,
            metadata=metadata,
            path=str(path),
        )

    def _load_geotiff(
        self,
        path: Path,
    ) -> MultispectralImage:
        """
        Load a GeoTIFF while preserving all spectral bands.

        Rasterio returns data as:

            (bands, height, width)

        which is exactly the representation required
        by the multispectral pipeline.
        """

        if rasterio is None:
            raise ImportError(
                "Rasterio is required for GeoTIFF loading."
            )

        with rasterio.open(path) as src:

            data = src.read(masked=True).astype(np.float32).filled(np.nan)

            height = src.height
            width = src.width
            bands = src.count

            crs = src.crs
            transform = src.transform

            metadata = src.meta.copy()

            metadata.update(
                {
                    "driver": src.driver,
                    "dtype": str(data.dtype),
                    "bands": bands,
                    "width": width,
                    "height": height,
                    "crs": str(crs) if crs else None,
                    "is_geospatial": True,
                    "nodata": src.nodata,
                }
            )

        if data.ndim != 3:
            raise ValueError(
                f"Expected GeoTIFF data with 3 dimensions, "
                f"got shape {data.shape}"
            )

        return MultispectralImage(
            data=data,
            width=width,
            height=height,
            bands=bands,
            crs=crs,
            transform=transform,
            metadata=metadata,
            path=str(path),
        )

    @staticmethod
    def normalize(
        data: np.ndarray,
        lower_percentile: float = 2.0,
        upper_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Percentile-based normalization for remote-sensing data.

        Each band is normalized independently to [0, 1].

        Parameters
        ----------
        data:
            Array in (bands, height, width).

        lower_percentile:
            Lower clipping percentile.

        upper_percentile:
            Upper clipping percentile.

        Returns
        -------
        np.ndarray
            Float32 normalized array.
        """

        if data.ndim != 3:
            raise ValueError(
                "Expected data with shape "
                "(bands, height, width)."
            )

        if not (
            0 <= lower_percentile < upper_percentile <= 100
        ):
            raise ValueError(
                "Invalid percentile values."
            )

        data = data.astype(
            np.float32,
            copy=False
        )

        normalized = np.zeros_like(
            data,
            dtype=np.float32
        )

        for band in range(data.shape[0]):

            band_data = data[band]

            valid = np.isfinite(band_data)

            if not np.any(valid):
                continue

            low = np.percentile(
                band_data[valid],
                lower_percentile
            )

            high = np.percentile(
                band_data[valid],
                upper_percentile
            )

            if high <= low:
                normalized[band] = 0.0
                continue

            normalized[band] = np.clip(
                (band_data - low)
                / (high - low),
                0.0,
                1.0,
            )

        return normalized

    @staticmethod
    def get_rgb(
        data: np.ndarray,
        red_band: int = 0,
        green_band: int = 1,
        blue_band: int = 2,
    ) -> np.ndarray:
        """
        Extract three bands as an RGB image.

        Parameters are zero-based band indexes.

        Returns
        -------
        np.ndarray
            RGB image with shape (H, W, 3).
        """

        if data.ndim != 3:
            raise ValueError(
                "Expected data with shape "
                "(bands, height, width)."
            )

        bands = data.shape[0]

        indexes = [
            red_band,
            green_band,
            blue_band,
        ]

        for index in indexes:
            if index < 0 or index >= bands:
                raise IndexError(
                    f"Band {index} is unavailable. "
                    f"Image contains {bands} bands."
                )

        rgb = np.stack(
            [
                data[red_band],
                data[green_band],
                data[blue_band],
            ],
            axis=-1,
        )

        return rgb

    @staticmethod
    def summary(
        image: MultispectralImage,
    ) -> Dict:
        """
        Return useful information about a loaded image.
        """

        return {
            "path": image.path,
            "width": image.width,
            "height": image.height,
            "bands": image.bands,
            "shape": tuple(image.data.shape),
            "dtype": str(image.data.dtype),
            "crs": (
                str(image.crs)
                if image.crs is not None
                else None
            ),
            "is_geospatial": image.metadata.get(
                "is_geospatial",
                False,
            ),
        }


def load_image(
    image_path: Union[str, Path],
) -> MultispectralImage:
    """
    Convenience function.

    Example
    -------
    image = load_image("image.tif")
    """

    loader = MultispectralLoader()

    return loader.load(image_path)
