from pathlib import Path
import cv2

from .input_types import ImageMetadata, ImageType


class MetadataExtractor:

    def extract(self, image_path: str, image) -> ImageMetadata:

        path = Path(image_path)

        height, width = image.shape[:2]

        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]

        image_type = self._detect_image_type(
            channels,
            path.name
        )

        return ImageMetadata(
            filename=path.name,
            width=width,
            height=height,
            channels=channels,
            dtype=str(image.dtype),
            image_type=image_type,
            format=path.suffix.lower()
        )

    def _detect_image_type(
        self,
        channels: int,
        filename: str
    ) -> ImageType:

        filename_lower = filename.lower()

        if any(
            keyword in filename_lower
            for keyword in ["sar", "sentinel1", "risat"]
        ):
            return ImageType.SAR

        if channels > 4:
            return ImageType.MULTISPECTRAL

        if channels in (1, 3, 4):
            return ImageType.OPTICAL

        return ImageType.UNKNOWN