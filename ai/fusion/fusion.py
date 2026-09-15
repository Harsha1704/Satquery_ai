from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FusionResult:
    """
    Result of optical-SAR feature fusion.

    optical:
        Optical Sentinel-2 data, expected as (bands, H, W).

    sar:
        SAR data, expected as (2, H, W):
            0 -> VV
            1 -> VH

    fused:
        Combined feature tensor.
    """

    optical: np.ndarray
    sar: np.ndarray
    fused: np.ndarray
    metadata: Dict


class OpticalSARFusion:
    """
    Feature-level fusion between Sentinel-2 optical
    imagery and Sentinel-1 SAR imagery.

    This implementation performs deterministic feature
    concatenation after spatial alignment.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fuse(
        self,
        optical: np.ndarray,
        sar: np.ndarray,
    ) -> FusionResult:

        optical = self._validate_optical(optical)
        sar = self._validate_sar(sar)

        if optical.shape[1:] != sar.shape[1:]:
            sar = self._resize_to_match(
                sar,
                optical.shape[1],
                optical.shape[2]
            )

        if self.normalize:
            optical = self._normalize(optical)
            sar = self._normalize(sar)

        fused = np.concatenate(
            [optical, sar],
            axis=0
        ).astype(np.float32)

        metadata = {
            "optical_bands": int(optical.shape[0]),
            "sar_bands": int(sar.shape[0]),
            "fused_bands": int(fused.shape[0]),
            "height": int(fused.shape[1]),
            "width": int(fused.shape[2]),
            "normalized": self.normalize,
        }

        return FusionResult(
            optical=optical,
            sar=sar,
            fused=fused,
            metadata=metadata,
        )

    @staticmethod
    def _validate_optical(
        data: np.ndarray
    ) -> np.ndarray:

        data = np.asarray(data)

        if data.ndim != 3:
            raise ValueError(
                "Optical data must have shape (bands, H, W)."
            )

        if data.shape[0] < 3:
            raise ValueError(
                "Optical data must contain at least 3 bands."
            )

        return data

    @staticmethod
    def _validate_sar(
        data: np.ndarray
    ) -> np.ndarray:

        data = np.asarray(data)

        if data.ndim != 3:
            raise ValueError(
                "SAR data must have shape (bands, H, W)."
            )

        if data.shape[0] != 2:
            raise ValueError(
                "SAR data must contain exactly VV and VH bands."
            )

        return data

    @staticmethod
    def _normalize(
        data: np.ndarray
    ) -> np.ndarray:

        data = data.astype(np.float32)

        result = np.empty_like(data)

        for band in range(data.shape[0]):

            layer = data[band]

            minimum = np.nanmin(layer)
            maximum = np.nanmax(layer)

            if maximum > minimum:
                result[band] = (
                    (layer - minimum)
                    / (maximum - minimum)
                )
            else:
                result[band] = 0.0

        return result

    @staticmethod
    def _resize_to_match(
        data: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:

        from PIL import Image

        resized_bands = []

        for band in data:

            image = Image.fromarray(
                band.astype(np.float32),
                mode="F"
            )

            image = image.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR
            )

            resized_bands.append(
                np.asarray(image, dtype=np.float32)
            )

        return np.stack(resized_bands, axis=0)