from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np


class Sentinel2Sample:
    """
    Represents one Sentinel-2 multispectral sample.
    """

    def __init__(
        self,
        path: str,
        data: np.ndarray,
        sample_id: str,
        labels: Optional[List[str]] = None,
    ):
        self.path = path
        self.data = data
        self.sample_id = sample_id
        self.labels = labels or []

    @property
    def bands(self) -> int:
        return int(self.data.shape[0])

    @property
    def height(self) -> int:
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        return int(self.data.shape[2])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(self.data.shape)

    def __repr__(self) -> str:
        return (
            f"Sentinel2Sample("
            f"id='{self.sample_id}', "
            f"shape={self.shape}, "
            f"dtype={self.data.dtype}, "
            f"labels={self.labels}"
            f")"
        )


class Sentinel2Dataset:
    """
    Loader for SatQuery Sentinel-2 multispectral samples.

    Expected input format:

        (bands, height, width)

    Each sample is stored as a .npy file.

    The loader does not assume a fixed number of bands.
    This allows Sentinel-2 datasets with different band
    selections to be used during development.
    """

    def __init__(
        self,
        data_dir: str = "data/multispectral/data",
        labels_file: str = "data/multispectral/coastal_labels.json",
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = Path(labels_file)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Sentinel-2 data directory not found: "
                f"{self.data_dir}"
            )

        self.labels = self._load_labels()

        self.files = sorted(
            self.data_dir.glob("*.npy")
        )

    def _load_labels(self) -> Dict:
        """
        Load optional label metadata.
        """

        if not self.labels_file.exists():
            return {}

        try:
            with open(
                self.labels_file,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if isinstance(data, dict):
                return data

            return {}

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON label file: "
                f"{self.labels_file}"
            ) from exc

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> Sentinel2Sample:

        if index < 0 or index >= len(self.files):
            raise IndexError(
                f"Sample index {index} out of range. "
                f"Dataset contains {len(self.files)} samples."
            )

        path = self.files[index]

        data = np.load(
            path,
            allow_pickle=False,
        )

        data = self._validate_array(
            data,
            path,
        )

        sample_id = path.stem

        labels = self._get_labels(
            sample_id
        )

        return Sentinel2Sample(
            path=str(path),
            data=data,
            sample_id=sample_id,
            labels=labels,
        )

    def _validate_array(
        self,
        data: np.ndarray,
        path: Path,
    ) -> np.ndarray:
        """
        Validate Sentinel-2 array structure.

        Expected:

            (bands, height, width)
        """

        if not isinstance(data, np.ndarray):
            raise TypeError(
                f"{path.name} did not contain a NumPy array."
            )

        if data.ndim != 3:
            raise ValueError(
                f"{path.name}: expected a 3D array "
                f"(bands, height, width), "
                f"got shape {data.shape}"
            )

        if data.shape[0] < 1:
            raise ValueError(
                f"{path.name}: no spectral bands found."
            )

        if data.shape[1] < 1 or data.shape[2] < 1:
            raise ValueError(
                f"{path.name}: invalid spatial dimensions "
                f"{data.shape}"
            )

        return data

    def _get_labels(
        self,
        sample_id: str,
    ) -> List[str]:
        """
        Retrieve labels for a sample.

        Supports several common JSON layouts.
        """

        if not self.labels:
            return []

        entry = self.labels.get(sample_id)

        if entry is None:
            return []

        if isinstance(entry, list):
            return [
                str(label)
                for label in entry
            ]

        if isinstance(entry, dict):

            for key in (
                "labels",
                "classes",
                "categories",
                "land_cover",
            ):
                value = entry.get(key)

                if isinstance(value, list):
                    return [
                        str(label)
                        for label in value
                    ]

            return []

        if isinstance(entry, str):
            return [entry]

        return []

    def get_all_samples(self) -> List[Sentinel2Sample]:
        """
        Load all available samples.
        """

        return [
            self[index]
            for index in range(len(self))
        ]

    @staticmethod
    def normalize(
        data: np.ndarray,
        lower_percentile: float = 2.0,
        upper_percentile: float = 98.0,
    ) -> np.ndarray:
        """
        Normalize every spectral band independently.

        Output range:

            [0, 1]

        Output dtype:

            float32
        """

        if data.ndim != 3:
            raise ValueError(
                "Expected shape "
                "(bands, height, width)."
            )

        if not (
            0 <= lower_percentile < upper_percentile <= 100
        ):
            raise ValueError(
                "Invalid percentile range."
            )

        source = data.astype(
            np.float32,
            copy=False,
        )

        output = np.zeros_like(
            source,
            dtype=np.float32,
        )

        for band_index in range(
            source.shape[0]
        ):

            band = source[band_index]

            valid = np.isfinite(band)

            if not np.any(valid):
                continue

            low = np.percentile(
                band[valid],
                lower_percentile,
            )

            high = np.percentile(
                band[valid],
                upper_percentile,
            )

            if high <= low:
                output[band_index] = 0.0
                continue

            output[band_index] = np.clip(
                (band - low) / (high - low),
                0.0,
                1.0,
            )

        return output

    @staticmethod
    def statistics(
        data: np.ndarray,
    ) -> List[Dict]:
        """
        Calculate per-band statistics.
        """

        if data.ndim != 3:
            raise ValueError(
                "Expected shape "
                "(bands, height, width)."
            )

        results = []

        for band_index in range(
            data.shape[0]
        ):

            band = data[band_index]

            valid = np.isfinite(band)

            if not np.any(valid):
                results.append(
                    {
                        "band": band_index,
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
                )
                continue

            values = band[valid]

            results.append(
                {
                    "band": band_index,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            )

        return results

    def summary(self) -> Dict:
        """
        Return dataset-level information.
        """

        return {
            "data_dir": str(self.data_dir),
            "labels_file": str(self.labels_file),
            "samples": len(self),
            "files": [
                file.name
                for file in self.files
            ],
        }