from typing import Dict, Optional, Any

import numpy as np

from .types import (
    ChangeTransition,
    SemanticChangeResult,
)


class ChangeReasoner:
    """
    Semantic reasoning over two land-cover prediction masks.

    SatQuery-AI classes:

        0 -> water
        1 -> vegetation
        2 -> built_up
        3 -> bare_land
        4 -> unknown

    Responsibilities:

        - Validate prediction masks
        - Compare before/after semantic masks
        - Calculate pixel-level change statistics
        - Calculate class-to-class transitions
        - Support requested-class filtering
        - Separate known semantic changes from unknown changes
        - Generate human-readable explanations
    """

    DEFAULT_CLASS_MAP = {
        0: "water",
        1: "vegetation",
        2: "built_up",
        3: "bare_land",
        4: "unknown",
    }

    UNKNOWN_CLASS_NAME = "unknown"

    def __init__(
        self,
        class_map: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the change reasoner.
        """

        self.class_map = (
            class_map.copy()
            if class_map is not None
            else self.DEFAULT_CLASS_MAP.copy()
        )

        self.class_map = {
            int(key): str(value).strip().lower()
            for key, value in self.class_map.items()
        }

    # ==========================================================
    # MAIN COMPARISON
    # ==========================================================

    def compare(
        self,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
    ) -> SemanticChangeResult:
        """
        Compare two semantic segmentation masks.
        """

        before = self._validate(
            before_mask,
            "before_mask",
        )

        after = self._validate(
            after_mask,
            "after_mask",
        )

        if before.shape != after.shape:
            raise ValueError(
                "Before and after masks must have "
                f"the same shape. Got {before.shape} "
                f"and {after.shape}."
            )

        total_pixels = int(before.size)

        if total_pixels <= 0:
            raise ValueError(
                "Semantic masks cannot contain zero pixels."
            )

        # ------------------------------------------------------
        # Pixel-level change
        # ------------------------------------------------------

        changed_pixels = int(
            np.count_nonzero(
                before != after
            )
        )

        stable_pixels = (
            total_pixels - changed_pixels
        )

        changed_percentage = (
            changed_pixels
            / total_pixels
            * 100.0
        )

        stable_percentage = (
            stable_pixels
            / total_pixels
            * 100.0
        )

        # ------------------------------------------------------
        # Semantic transitions
        # ------------------------------------------------------

        transitions = self._calculate_transitions(
            before=before,
            after=after,
            total_pixels=total_pixels,
        )

        return SemanticChangeResult(
            total_pixels=total_pixels,
            changed_pixels=changed_pixels,
            stable_pixels=stable_pixels,
            changed_percentage=changed_percentage,
            stable_percentage=stable_percentage,
            change_detected=(
                changed_pixels > 0
            ),
            transitions=transitions,
        )

    # ==========================================================
    # TRANSITIONS
    # ==========================================================

    def _calculate_transitions(
        self,
        before: np.ndarray,
        after: np.ndarray,
        total_pixels: int,
    ) -> Dict[str, ChangeTransition]:
        """
        Calculate every non-stable semantic transition.
        """

        transitions: Dict[
            str,
            ChangeTransition
        ] = {}

        before_classes = np.unique(before)
        after_classes = np.unique(after)

        for from_id in before_classes:

            from_id_int = int(from_id)

            for to_id in after_classes:

                to_id_int = int(to_id)

                # Same class = stable.
                if from_id_int == to_id_int:
                    continue

                count = int(
                    np.count_nonzero(
                        (before == from_id_int)
                        & (after == to_id_int)
                    )
                )

                if count <= 0:
                    continue

                from_name = self._class_name(
                    from_id_int
                )

                to_name = self._class_name(
                    to_id_int
                )

                key = (
                    f"{from_name}_to_{to_name}"
                )

                transitions[key] = ChangeTransition(
                    from_class=from_name,
                    to_class=to_name,
                    pixels=count,
                    percentage_of_image=(
                        count
                        / total_pixels
                        * 100.0
                    ),
                )

        return transitions

    # ==========================================================
    # REQUESTED CLASS
    # ==========================================================

    def find_change(
        self,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        requested_class: str,
    ) -> Dict[str, Any]:
        """
        Find changes involving a requested semantic class.
        """

        if not requested_class:
            raise ValueError(
                "requested_class cannot be empty."
            )

        result = self.compare(
            before_mask,
            after_mask,
        )

        requested_class = (
            requested_class
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        matches = []

        for transition in result.transitions.values():

            if (
                transition.from_class
                == requested_class
                or transition.to_class
                == requested_class
            ):
                matches.append(
                    transition
                )

        matches = sorted(
            matches,
            key=lambda item: (
                -item.pixels,
                item.from_class,
                item.to_class,
            ),
        )

        pixels = int(
            sum(
                item.pixels
                for item in matches
            )
        )

        percentage = (
            pixels
            / result.total_pixels
            * 100.0
            if result.total_pixels > 0
            else 0.0
        )

        return {
            "total_pixels": result.total_pixels,

            "changed_pixels": result.changed_pixels,

            "stable_pixels": result.stable_pixels,

            "changed_percentage": (
                result.changed_percentage
            ),

            "stable_percentage": (
                result.stable_percentage
            ),

            "change_detected": (
                result.change_detected
            ),

            "transitions": {
                key: vars(value)
                for key, value
                in result.transitions.items()
            },

            "requested_class": requested_class,

            "requested_change_pixels": pixels,

            "requested_change_percentage": float(
                percentage
            ),

            "requested_changes": [
                vars(item)
                for item in matches
            ],

            "requested_class_found": (
                len(matches) > 0
            ),

            "meaningful_changed_pixels": (
                result.meaningful_changed_pixels
            ),

            "meaningful_changed_percentage": (
                result.meaningful_changed_percentage
            ),

            "unknown_changed_pixels": (
                result.unknown_changed_pixels
            ),

            "unknown_changed_percentage": (
                result.unknown_changed_percentage
            ),

            "meaningful_changes": [
                vars(item)
                for item in result.meaningful_changes
            ],

            "unknown_changes": [
                vars(item)
                for item in result.unknown_changes
            ],
        }

    # ==========================================================
    # DESCRIPTION
    # ==========================================================

    def describe(
        self,
        result,
    ) -> str:
        """
        Generate a natural-language explanation.

        Supports:

            SemanticChangeResult

        and:

            dictionary results from find_change().
        """

        if isinstance(
            result,
            SemanticChangeResult,
        ):
            return self._describe_structured(
                result
            )

        if not isinstance(
            result,
            dict,
        ):
            return (
                "Semantic change analysis returned "
                "an invalid result."
            )

        if not result.get(
            "change_detected",
            False,
        ):
            return (
                "No semantic land-cover changes "
                "were detected."
            )

        requested = result.get(
            "requested_class"
        )

        # ======================================================
        # REQUESTED CLASS
        # ======================================================

        if requested:

            pixels = int(
                result.get(
                    "requested_change_pixels",
                    0,
                )
            )

            percentage = float(
                result.get(
                    "requested_change_percentage",
                    0.0,
                )
            )

            if not result.get(
                "requested_class_found",
                False,
            ):
                return (
                    f"No changes involving "
                    f"{requested.replace('_', ' ')} "
                    "were detected."
                )

            text = (
                f"Changes involving "
                f"{requested.replace('_', ' ')} "
                f"were detected across approximately "
                f"{percentage:.2f}% of the image "
                f"({pixels:,} pixels)."
            )

            requested_changes = result.get(
                "requested_changes",
                [],
            )

            # Prefer known -> known transitions.
            meaningful_requested = [
                item
                for item in requested_changes
                if (
                    isinstance(item, dict)
                    and item.get("from_class") != "unknown"
                    and item.get("to_class") != "unknown"
                )
            ]

            descriptions = (
                self._transition_dict_descriptions(
                    meaningful_requested,
                    limit=3,
                )
            )

            if descriptions:
                text += (
                    " Major semantic transitions: "
                    + "; ".join(descriptions)
                    + "."
                )

            unknown_requested = [
                item
                for item in requested_changes
                if (
                    isinstance(item, dict)
                    and (
                        item.get("from_class")
                        == "unknown"
                        or item.get("to_class")
                        == "unknown"
                    )
                )
            ]

            if unknown_requested:
                unknown_pixels = sum(
                    int(
                        item.get(
                            "pixels",
                            0,
                        )
                    )
                    for item in unknown_requested
                )

                unknown_percentage = (
                    unknown_pixels
                    / result.get(
                        "total_pixels",
                        1,
                    )
                    * 100.0
                )

                text += (
                    f" {unknown_percentage:.2f}% "
                    "involved uncertain transitions "
                    "to or from the unknown class."
                )

            return text

        # ======================================================
        # GENERAL DESCRIPTION
        # ======================================================

        changed_percentage = float(
            result.get(
                "changed_percentage",
                0.0,
            )
        )

        changed_pixels = int(
            result.get(
                "changed_pixels",
                0,
            )
        )

        meaningful_percentage = float(
            result.get(
                "meaningful_changed_percentage",
                0.0,
            )
        )

        meaningful_pixels = int(
            result.get(
                "meaningful_changed_pixels",
                0,
            )
        )

        unknown_percentage = float(
            result.get(
                "unknown_changed_percentage",
                0.0,
            )
        )

        text = (
            "Semantic land-cover changes were "
            f"detected across approximately "
            f"{changed_percentage:.2f}% of the image "
            f"({changed_pixels:,} pixels)."
        )

        # ======================================================
        # VERIFIED SEMANTIC CHANGES
        # ======================================================

        transitions = result.get(
            "transitions",
            {},
        )

        meaningful = []

        if isinstance(
            transitions,
            dict,
        ):
            for transition in transitions.values():

                if not isinstance(
                    transition,
                    dict,
                ):
                    continue

                from_class = str(
                    transition.get(
                        "from_class",
                        "unknown",
                    )
                )

                to_class = str(
                    transition.get(
                        "to_class",
                        "unknown",
                    )
                )

                if (
                    from_class == self.UNKNOWN_CLASS_NAME
                    or to_class == self.UNKNOWN_CLASS_NAME
                ):
                    continue

                meaningful.append(
                    transition
                )

        meaningful.sort(
            key=lambda item: (
                -int(
                    item.get(
                        "pixels",
                        0,
                    )
                ),
                str(
                    item.get(
                        "from_class",
                        "",
                    )
                ),
                str(
                    item.get(
                        "to_class",
                        "",
                    )
                ),
            )
        )

        if meaningful:

            descriptions = (
                self._transition_dict_descriptions(
                    meaningful,
                    limit=3,
                )
            )

            text += (
                " Major semantic transitions: "
                + "; ".join(descriptions)
                + "."
            )

        else:

            text += (
                " No transitions between known "
                "land-cover classes were identified."
            )

        # ======================================================
        # UNKNOWN / UNCERTAIN CHANGES
        # ======================================================

        if unknown_percentage > 0:

            text += (
                f" Approximately "
                f"{unknown_percentage:.2f}% of the image "
                "involved transitions to or from "
                "the unknown class, so these areas "
                "should be treated as uncertain."
            )

        # ======================================================
        # INTERPRETATION
        # ======================================================

        if (
            meaningful_pixels > 0
            and changed_pixels > 0
        ):

            verified_ratio = (
                meaningful_pixels
                / changed_pixels
                * 100.0
            )

            text += (
                f" Verified known-class transitions "
                f"account for approximately "
                f"{verified_ratio:.2f}% of all detected "
                "changed pixels."
            )

        return text

    # ==========================================================
    # STRUCTURED DESCRIPTION
    # ==========================================================

    @staticmethod
    def _describe_structured(
        result: SemanticChangeResult,
    ) -> str:
        """
        Generate a description directly from
        SemanticChangeResult.
        """

        if not result.change_detected:
            return (
                "No semantic land-cover changes "
                "were detected."
            )

        text = (
            "Semantic land-cover changes were "
            f"detected across approximately "
            f"{result.changed_percentage:.2f}% "
            "of the image "
            f"({result.changed_pixels:,} pixels)."
        )

        # ======================================================
        # VERIFIED CHANGES
        # ======================================================

        major = result.meaningful_changes[:3]

        if major:

            transitions = "; ".join(
                item.description
                for item in major
            )

            text += (
                " Major semantic transitions: "
                f"{transitions}."
            )

        else:

            text += (
                " No transitions between known "
                "land-cover classes were identified."
            )

        # ======================================================
        # UNKNOWN CHANGES
        # ======================================================

        if result.unknown_changed_pixels > 0:

            text += (
                f" {result.unknown_changed_percentage:.2f}% "
                "of the image involved transitions "
                "to or from the unknown class, "
                "which should be treated as uncertain."
            )

        # ======================================================
        # VERIFIED RATIO
        # ======================================================

        if result.meaningful_changed_pixels > 0:

            ratio = (
                result.meaningful_changed_pixels
                / result.changed_pixels
                * 100.0
                if result.changed_pixels > 0
                else 0.0
            )

            text += (
                f" Known-class transitions account for "
                f"approximately {ratio:.2f}% of all "
                "detected changed pixels."
            )

        return text

    # ==========================================================
    # CLASS NAME
    # ==========================================================

    def _class_name(
        self,
        class_id: int,
    ) -> str:
        """
        Resolve a numeric class ID to a semantic name.
        """

        return self.class_map.get(
            int(class_id),
            self.UNKNOWN_CLASS_NAME,
        )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @staticmethod
    def _validate(
        mask: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """
        Validate and normalize a semantic mask.
        """

        if mask is None:
            raise ValueError(
                f"{name} cannot be None."
            )

        mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D class mask. "
                f"Got shape {mask.shape}."
            )

        if mask.size == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.issubdtype(
            mask.dtype,
            np.integer,
        ):

            if np.all(
                np.isfinite(mask)
            ):

                mask = mask.astype(
                    np.int64
                )

            else:

                raise ValueError(
                    f"{name} contains non-finite values."
                )

        return mask

    # ==========================================================
    # DICTIONARY TRANSITION DESCRIPTION
    # ==========================================================

    @staticmethod
    def _transition_dict_descriptions(
        transitions,
        limit: int = 3,
    ):
        """
        Convert serialized transition dictionaries
        into human-readable descriptions.
        """

        descriptions = []

        if not isinstance(
            transitions,
            list,
        ):
            return descriptions

        for item in transitions[:limit]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            from_class = str(
                item.get(
                    "from_class",
                    "unknown",
                )
            )

            to_class = str(
                item.get(
                    "to_class",
                    "unknown",
                )
            )

            pixels = int(
                item.get(
                    "pixels",
                    0,
                )
            )

            percentage = float(
                item.get(
                    "percentage_of_image",
                    0.0,
                )
            )

            descriptions.append(
                f"{from_class.replace('_', ' ')} "
                f"to {to_class.replace('_', ' ')}: "
                f"{pixels:,} pixels "
                f"({percentage:.2f}%)"
            )

        return descriptions