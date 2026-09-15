from typing import Optional

from ai.models import SemanticModel


class SemanticSegmentationModel:
    """
    Compatibility wrapper around the real SatQuery SemanticModel.
    """

    def __init__(
        self,
        model: Optional[SemanticModel] = None,
    ):
        self.model = model or SemanticModel()

    def load(self):
        """
        Load the real SegFormer model.
        """

        if not self.model.is_loaded():
            self.model.load()

        return self.model

    def predict(self, image):
        """
        Generate a semantic segmentation mask.
        """

        return self.model.predict(image)

    def is_loaded(self) -> bool:
        """
        Return model loading status.
        """

        return self.model.is_loaded()

    def get_class_map(self):
        """
        Return SatQuery class mapping.
        """

        return self.model.get_class_map()

    def get_original_class_map(self):
        """
        Return LoveDA class mapping.
        """

        return self.model.get_original_class_map()