from pathlib import Path
from PIL import Image
import cv2


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}


class ImageLoader:

    def __init__(self):
        pass

    def validate_path(self, image_path: str) -> Path:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}"
            )

        return path

    def load(self, image_path: str):
        path = self.validate_path(image_path)

        image = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        return image

    def get_pillow_image(self, image_path: str):
        path = self.validate_path(image_path)

        return Image.open(path)