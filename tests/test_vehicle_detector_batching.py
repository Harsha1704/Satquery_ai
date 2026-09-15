from __future__ import annotations

from PIL import Image

from ai.detection.detector import ObjectDetector


class _FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [type("Result", (), {"boxes": None})() for _ in kwargs["source"]]


def test_tiles_are_sent_in_a_single_model_batch() -> None:
    detector = ObjectDetector.__new__(ObjectDetector)
    detector.model = _FakeModel()
    detector.device = "cpu"
    detector.inference_size = 640
    detector.tile_size = 512
    detector.tile_overlap = 0.25
    detector.tile_batch_size = 4
    detector.tile_upscale = 1.5

    detections = detector._detect_with_tiles(
        Image.new("RGB", (1024, 512)),
        requested_class="car",
        confidence_threshold=0.25,
    )

    assert detections == []
    assert len(detector.model.calls) == 1
    assert len(detector.model.calls[0]["source"]) == 3
    assert detector.model.calls[0]["imgsz"] == 640
