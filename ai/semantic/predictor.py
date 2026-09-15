from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ai.models import SemanticModel


class SemanticPredictor:
    """
    High-level semantic prediction interface.

    Connects:
        Image
          ↓
        SemanticModel
          ↓
        SatQuery semantic mask
    """

    def __init__(
        self,
        model: SemanticModel = None,
    ):
        self.model = model or SemanticModel()

    def predict(
        self,
        image: Union[
            str,
            Path,
            Image.Image,
            np.ndarray,
        ],
    ) -> np.ndarray:
        """
        Generate a SatQuery semantic mask.
        """

        if not self.model.is_loaded():
            raise RuntimeError(
                "SemanticModel is not loaded."
            )

        mask = self.model.predict(image)

        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)

        if mask.ndim != 2:
            raise ValueError(
                "Semantic prediction must be a 2D mask."
            )

        return mask.astype(np.uint8)

    def predict_with_metadata(
        self,
        image,
    ) -> dict:
        """
        Generate prediction together with model metadata.
        """

        mask = self.predict(image)

        return {
            "mask": mask,
            "shape": mask.shape,
            "dtype": str(mask.dtype),
            "classes": np.unique(mask).tolist(),
            "class_map": self.model.get_class_map(),
        }