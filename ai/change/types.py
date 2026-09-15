from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChangeTransition:
    """
    Represents a semantic land-cover transition.

    Example:
        vegetation -> built_up
        bare_land -> vegetation
    """

    from_class: str
    to_class: str
    pixels: int
    percentage_of_image: float

    @property
    def description(self) -> str:
        """Return a human-readable transition description."""

        return (
            f"{self.from_class.replace('_', ' ')} "
            f"to {self.to_class.replace('_', ' ')}: "
            f"{self.pixels:,} pixels "
            f"({self.percentage_of_image:.2f}%)"
        )

    @property
    def is_unknown_transition(self) -> bool:
        """
        Whether this transition involves the unknown class.
        """

        return (
            self.from_class == "unknown"
            or self.to_class == "unknown"
        )

    @property
    def is_known_transition(self) -> bool:
        """
        Whether both sides of the transition
        are known semantic classes.
        """

        return not self.is_unknown_transition


@dataclass
class SemanticChangeResult:
    """
    Structured semantic change-analysis result.

    Keeps the complete pixel-level change statistics while
    separating:

        1. All detected changes
        2. Known semantic changes
        3. Unknown/uncertain changes
    """

    total_pixels: int
    changed_pixels: int
    stable_pixels: int

    changed_percentage: float
    stable_percentage: float

    change_detected: bool

    transitions: Dict[str, ChangeTransition] = field(
        default_factory=dict
    )

    # ==========================================================
    # ALL CHANGES
    # ==========================================================

    @property
    def major_changes(self) -> List[ChangeTransition]:
        """
        Return the largest transitions.

        Includes unknown transitions.
        """

        return sorted(
            self.transitions.values(),
            key=lambda item: (
                -item.pixels,
                item.from_class,
                item.to_class,
            ),
        )

    # ==========================================================
    # MEANINGFUL CHANGES
    # ==========================================================

    @property
    def meaningful_changes(self) -> List[ChangeTransition]:
        """
        Return only known -> known semantic transitions.

        Example:

            bare_land -> vegetation

        is meaningful.

            bare_land -> unknown

        is not considered a verified semantic transition.
        """

        return sorted(
            (
                transition
                for transition in self.transitions.values()
                if transition.is_known_transition
            ),
            key=lambda item: (
                -item.pixels,
                item.from_class,
                item.to_class,
            ),
        )

    # ==========================================================
    # UNKNOWN CHANGES
    # ==========================================================

    @property
    def unknown_changes(self) -> List[ChangeTransition]:
        """
        Return all transitions involving unknown.
        """

        return sorted(
            (
                transition
                for transition in self.transitions.values()
                if transition.is_unknown_transition
            ),
            key=lambda item: (
                -item.pixels,
                item.from_class,
                item.to_class,
            ),
        )

    # ==========================================================
    # MEANINGFUL PIXELS
    # ==========================================================

    @property
    def meaningful_changed_pixels(self) -> int:
        """
        Total pixels participating in known -> known
        semantic transitions.
        """

        return int(
            sum(
                transition.pixels
                for transition in self.meaningful_changes
            )
        )

    @property
    def meaningful_changed_percentage(self) -> float:
        """
        Percentage of the complete image represented by
        known -> known semantic transitions.
        """

        if self.total_pixels <= 0:
            return 0.0

        return (
            self.meaningful_changed_pixels
            / self.total_pixels
            * 100.0
        )

    # ==========================================================
    # UNKNOWN PIXELS
    # ==========================================================

    @property
    def unknown_changed_pixels(self) -> int:
        """
        Total changed pixels involving unknown.
        """

        return int(
            sum(
                transition.pixels
                for transition in self.unknown_changes
            )
        )

    @property
    def unknown_changed_percentage(self) -> float:
        """
        Percentage of the complete image represented by
        transitions involving unknown.
        """

        if self.total_pixels <= 0:
            return 0.0

        return (
            self.unknown_changed_pixels
            / self.total_pixels
            * 100.0
        )

    # ==========================================================
    # VERIFIED CHANGE RATIO
    # ==========================================================

    @property
    def meaningful_change_ratio(self) -> float:
        """
        Fraction of detected changed pixels that are
        known -> known semantic changes.

        Returns a value from 0.0 to 1.0.
        """

        if self.changed_pixels <= 0:
            return 0.0

        return (
            self.meaningful_changed_pixels
            / self.changed_pixels
        )

    # ==========================================================
    # UNKNOWN CHANGE RATIO
    # ==========================================================

    @property
    def unknown_change_ratio(self) -> float:
        """
        Fraction of detected changed pixels involving unknown.
        """

        if self.changed_pixels <= 0:
            return 0.0

        return (
            self.unknown_changed_pixels
            / self.changed_pixels
        )