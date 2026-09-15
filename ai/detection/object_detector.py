"""Backward-compatible import for the SatQuery-AI detection engine.

The project has one canonical detector implementation:
    ai.detection.detector.ObjectDetector

This wrapper prevents older imports from accidentally using a second,
inconsistent detector implementation.
"""

from .detector import ObjectDetector

__all__ = ["ObjectDetector"]
