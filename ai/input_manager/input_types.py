from enum import Enum
from dataclasses import dataclass
from typing import Optional


class InputMode(Enum):
    SINGLE = "single"
    BI_TEMPORAL = "bi_temporal"
    OPTICAL_SAR = "optical_sar"


class ImageType(Enum):
    OPTICAL = "optical"
    SAR = "sar"
    MULTISPECTRAL = "multispectral"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    filename: str
    width: int
    height: int
    channels: int
    dtype: str
    image_type: ImageType
    format: str


@dataclass
class InputRequest:
    mode: InputMode
    primary_image: str
    secondary_image: Optional[str] = None