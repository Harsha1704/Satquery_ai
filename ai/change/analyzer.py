from typing import Dict

import numpy as np


class ChangeAnalyzer:
    """
    Analyze a 2D binary change mask.

    Class convention:
        0 -> stable
        1 -> changed
    """

    def analyze(
        self,
        change_mask: np.ndarray,
    ) -> Dict:

        if change_mask is None:
            raise ValueError(
                "Change mask cannot be None."
            )

        change_mask = np.asarray(change_mask)

        if change_mask.ndim != 2:
            raise ValueError(
                "Change mask must be a 2D array."
            )

        if change_mask.size == 0:
            raise ValueError(
                "Change mask cannot be empty."
            )

        total_pixels = int(
            change_mask.size
        )

        changed_pixels = int(
            np.count_nonzero(change_mask)
        )

        stable_pixels = (
            total_pixels - changed_pixels
        )

        changed_percentage = (
            changed_pixels
            / total_pixels
            * 100
        )

        stable_percentage = (
            stable_pixels
            / total_pixels
            * 100
        )

        return {
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "stable_pixels": stable_pixels,
            "changed_percentage": float(
                changed_percentage
            ),
            "stable_percentage": float(
                stable_percentage
            ),
            "change_detected": (
                changed_pixels > 0
            ),
        }

    def describe(
        self,
        result: Dict,
    ) -> str:

        if not result.get(
            "change_detected",
            False,
        ):
            return (
                "No significant change was detected "
                "in the analyzed image pair."
            )

        percentage = result[
            "changed_percentage"
        ]

        pixels = result[
            "changed_pixels"
        ]

        return (
            "Change was detected across approximately "
            f"{percentage:.2f}% of the analyzed image "
            f"({pixels:,} pixels)."
        )