from typing import Dict, Optional

import numpy as np

from .classes import (
    LandCoverClass,
    CLASS_DESCRIPTIONS,
)


class SemanticAnalyzer:
    """
    Analyze a SatQuery semantic prediction mask.

    The prediction mask must contain SatQuery class IDs:

        0 -> water
        1 -> vegetation
        2 -> built_up
        3 -> bare_land
        4 -> unknown
    """

    def analyze(
        self,
        prediction_mask: np.ndarray,
        class_map: Optional[Dict[int, str]] = None,
        requested_class: Optional[str] = None,
    ) -> Dict:
        """
        Calculate semantic class statistics.
        """

        if prediction_mask is None:
            raise ValueError(
                "Prediction mask cannot be None."
            )

        if not isinstance(
            prediction_mask,
            np.ndarray,
        ):
            prediction_mask = np.asarray(
                prediction_mask
            )

        if prediction_mask.ndim != 2:
            raise ValueError(
                "Prediction mask must be a 2D array."
            )

        total_pixels = prediction_mask.size

        if total_pixels == 0:
            raise ValueError(
                "Prediction mask cannot be empty."
            )

        if class_map is None:
            class_map = {
                0: "water",
                1: "vegetation",
                2: "built_up",
                3: "bare_land",
                4: "unknown",
            }

        unique_classes, counts = np.unique(
            prediction_mask,
            return_counts=True,
        )

        class_statistics = {}

        for class_id, count in zip(
            unique_classes,
            counts,
        ):

            class_name = class_map.get(
                int(class_id),
                LandCoverClass.UNKNOWN.value,
            )

            percentage = (
                float(count)
                / total_pixels
                * 100.0
            )

            class_statistics[class_name] = {
                "pixels": int(count),
                "percentage": percentage,
            }

        dominant_class = max(
            class_statistics.items(),
            key=lambda item: item[1]["pixels"],
        )

        result = {
            "total_pixels": int(total_pixels),

            "classes": class_statistics,

            "dominant_class": dominant_class[0],

            "dominant_percentage": (
                dominant_class[1]["percentage"]
            ),
        }

        # Optional query-specific analysis
        if requested_class is not None:

            requested_class = (
                requested_class
                .strip()
                .lower()
                .replace(" ", "_")
            )

            requested_data = class_statistics.get(
                requested_class
            )

            if requested_data is None:

                result["requested_class"] = requested_class
                result["requested_pixels"] = 0
                result["requested_percentage"] = 0.0
                result["requested_class_found"] = False

            else:

                result["requested_class"] = requested_class
                result["requested_pixels"] = (
                    requested_data["pixels"]
                )
                result["requested_percentage"] = (
                    requested_data["percentage"]
                )
                result["requested_class_found"] = True

        return result

    def describe(
        self,
        analysis_result: Dict,
    ) -> str:
        """
        Convert analysis result into natural language.
        """

        if "requested_class" in analysis_result:

            requested = analysis_result[
                "requested_class"
            ]

            percentage = analysis_result[
                "requested_percentage"
            ]

            pixels = analysis_result[
                "requested_pixels"
            ]

            found = analysis_result.get(
                "requested_class_found",
                False,
            )

            if not found:

                return (
                    f"No {requested.replace('_', ' ')} "
                    f"areas were detected in the analyzed image."
                )

            try:

                land_class = LandCoverClass(
                    requested
                )

                description = CLASS_DESCRIPTIONS[
                    land_class
                ]

            except ValueError:

                description = CLASS_DESCRIPTIONS[
                    LandCoverClass.UNKNOWN
                ]

            return (
                f"{requested.replace('_', ' ').capitalize()} "
                f"covers approximately "
                f"{percentage:.2f}% of the analyzed image "
                f"({pixels:,} pixels). "
                f"{description}"
            )

        dominant = analysis_result[
            "dominant_class"
        ]

        percentage = analysis_result[
            "dominant_percentage"
        ]

        try:

            land_class = LandCoverClass(
                dominant
            )

            description = CLASS_DESCRIPTIONS[
                land_class
            ]

        except ValueError:

            description = CLASS_DESCRIPTIONS[
                LandCoverClass.UNKNOWN
            ]

        return (
            f"The dominant land-cover class is "
            f"{dominant.replace('_', ' ')} "
            f"({percentage:.2f}% of analyzed pixels). "
            f"{description}"
        )