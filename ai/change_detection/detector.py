import numpy as np


class ChangeDetectionEngine:
    """
    Lightweight pixel-level change detection engine.

    Compares two aligned raster arrays and identifies
    pixels whose normalized difference exceeds a threshold.
    """

    def normalize(self, image: np.ndarray) -> np.ndarray:

        image = image.astype(np.float32)

        minimum = np.nanmin(image)
        maximum = np.nanmax(image)

        if maximum - minimum == 0:
            return np.zeros_like(image, dtype=np.float32)

        return (
            (image - minimum)
            / (maximum - minimum)
        )

    def calculate_difference(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> np.ndarray:

        if before.shape != after.shape:
            raise ValueError(
                "Before and after images must have "
                "the same dimensions."
            )

        before_normalized = self.normalize(before)
        after_normalized = self.normalize(after)

        difference = np.abs(
            after_normalized - before_normalized
        )

        return difference

    def create_change_mask(
        self,
        difference: np.ndarray,
        threshold: float = 0.20
    ) -> np.ndarray:

        if not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        return (
            difference >= threshold
        ).astype(np.uint8)

    def detect(
        self,
        before: np.ndarray,
        after: np.ndarray,
        threshold: float = 0.20
    ) -> dict:

        difference = self.calculate_difference(
            before,
            after
        )

        change_mask = self.create_change_mask(
            difference,
            threshold
        )

        return {
            "difference": difference,
            "change_mask": change_mask,
            "threshold": threshold
        }