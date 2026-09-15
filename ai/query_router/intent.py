from enum import Enum


class Intent(Enum):

    SPECTRAL_ANALYSIS = "spectral_analysis"
    CHANGE_DETECTION = "change_detection"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    VQA = "vqa"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    UNKNOWN = "unknown"