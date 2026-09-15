from enum import Enum


class LandCoverClass(str, Enum):
    """
    SatQuery semantic land-cover classes.
    """

    WATER = "water"
    VEGETATION = "vegetation"
    BUILT_UP = "built_up"
    BARE_LAND = "bare_land"
    UNKNOWN = "unknown"


CLASS_DESCRIPTIONS = {
    LandCoverClass.WATER:
        "Water bodies such as lakes, rivers, ponds, and reservoirs.",

    LandCoverClass.VEGETATION:
        "Vegetated areas including crops, forests, and grassland.",

    LandCoverClass.BUILT_UP:
        "Human-made built-up areas such as buildings and urban surfaces.",

    LandCoverClass.BARE_LAND:
        "Bare soil, exposed ground, sand, or sparsely vegetated land.",

    LandCoverClass.UNKNOWN:
        "Areas that could not be confidently assigned to a supported land-cover class.",
}