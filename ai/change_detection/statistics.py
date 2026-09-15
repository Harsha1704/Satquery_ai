import numpy as np


class ChangeStatistics:

    def calculate(
        self,
        difference: np.ndarray,
        change_mask: np.ndarray
    ) -> dict:

        valid_pixels = np.isfinite(
            difference
        )

        total_pixels = int(
            np.sum(valid_pixels)
        )

        changed_pixels = int(
            np.sum(
                (change_mask == 1)
                & valid_pixels
            )
        )

        if total_pixels == 0:
            return {
                "total_pixels": 0,
                "changed_pixels": 0,
                "change_percentage": 0.0,
                "mean_difference": 0.0,
                "max_difference": 0.0
            }

        return {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "change_percentage": (
                changed_pixels
                / total_pixels
                * 100
            ),
            "mean_difference": float(
                np.nanmean(difference)
            ),
            "max_difference": float(
                np.nanmax(difference)
            )
        }

    def classify(
        self,
        change_percentage: float
    ) -> str:

        if change_percentage < 5:
            return "Minimal change"

        if change_percentage < 20:
            return "Moderate change"

        if change_percentage < 50:
            return "Significant change"

        return "Extensive change"