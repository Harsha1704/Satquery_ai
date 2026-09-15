"""
SatQuery AI - Sentinel-1 SAR data loader.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class SARSample:
    """
    Loaded Sentinel-1 SAR sample.
    """

    path: str
    data: np.ndarray
    bands: tuple[str, ...]
    crs: object = None
    transform: object = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype


class SARLoader:
    """
    Loader for Sentinel-1 SAR data.

    Supported:
        - NumPy .npy files
        - GeoTIFF .tif / .tiff files

    Expected channel layout:

        (2, H, W)

    where:

        channel 0 = VV
        channel 1 = VH
    """

    def load(self, path: Union[str, Path]) -> SARSample:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"SAR file not found: {path}"
            )

        suffix = path.suffix.lower()

        if suffix == ".npy":
            return self._load_numpy(path)

        if suffix in {".tif", ".tiff"}:
            return self._load_geotiff(path)

        raise ValueError(
            f"Unsupported SAR format: {suffix}. "
            "Supported formats: .npy, .tif, .tiff"
        )

    def _load_numpy(self, path: Path) -> SARSample:
        data = np.load(path)

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        if data.ndim != 3:
            raise ValueError(
                "SAR NumPy data must have shape "
                "(bands, height, width)."
            )

        if data.shape[0] == 1:
            bands = ("VV",)

        elif data.shape[0] >= 2:
            data = data[:2]
            bands = ("VV", "VH")

        else:
            raise ValueError("Invalid SAR band count.")

        return SARSample(
            path=str(path),
            data=data,
            bands=bands,
        )

    def _load_geotiff(self, path: Path) -> SARSample:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError(
                "rasterio is required for GeoTIFF SAR files. "
                "Install it with: pip install rasterio"
            ) from exc

        with rasterio.open(path) as src:
            data = src.read()

            if data.ndim != 3:
                raise ValueError(
                    "GeoTIFF SAR data must have shape "
                    "(bands, height, width)."
                )

            if data.shape[0] == 1:
                bands = ("VV",)

            elif data.shape[0] >= 2:
                data = data[:2]
                bands = ("VV", "VH")

            else:
                raise ValueError(
                    "GeoTIFF contains no SAR bands."
                )

            return SARSample(
                path=str(path),
                data=data,
                bands=bands,
                crs=src.crs,
                transform=src.transform,
            )

    @staticmethod
    def normalize(
        data: np.ndarray,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
    ) -> np.ndarray:
        """
        Percentile-based normalization to [0, 1].
        """

        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 3:
            raise ValueError(
                "SAR data must be 3D: (bands, height, width)."
            )

        output = np.zeros_like(data, dtype=np.float32)

        for band in range(data.shape[0]):
            image = data[band]

            low = np.percentile(image, percentile_low)
            high = np.percentile(image, percentile_high)

            if high <= low:
                output[band] = 0.0
                continue

            output[band] = np.clip(
                (image - low) / (high - low),
                0.0,
                1.0,
            )

        return output

    @staticmethod
    def summary(sample: SARSample) -> dict:
        """
        Return metadata about a SAR sample.
        """

        return {
            "path": sample.path,
            "shape": sample.data.shape,
            "dtype": str(sample.data.dtype),
            "bands": list(sample.bands),
            "height": int(sample.data.shape[1]),
            "width": int(sample.data.shape[2]),
            "crs": (
                str(sample.crs)
                if sample.crs is not None
                else None
            ),
            "is_geospatial": sample.crs is not None,
        }