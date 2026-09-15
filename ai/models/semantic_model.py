from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class SemanticModel:
    """
    Real semantic land-cover segmentation model for SatQuery AI.

    Backend:
        SegFormer B2 fine-tuned on LoveDA.

    LoveDA classes:
        0 -> ignore
        1 -> background
        2 -> building
        3 -> road
        4 -> water
        5 -> barren
        6 -> forest
        7 -> agricultural

    SatQuery classes:
        0 -> water
        1 -> vegetation
        2 -> built_up
        3 -> bare_land
        4 -> unknown
    """

    MODEL_NAME = "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"

    # SatQuery's normalized class IDs
    CLASS_MAP = {
        0: "water",
        1: "vegetation",
        2: "built_up",
        3: "bare_land",
        4: "unknown",
    }

    # Original LoveDA class IDs
    ORIGINAL_CLASS_MAP = {
        0: "ignore",
        1: "background",
        2: "building",
        3: "road",
        4: "water",
        5: "barren",
        6: "forest",
        7: "agricultural",
    }

    # LoveDA -> SatQuery mapping
    #
    # Ignore/background:
    #     -> unknown
    #
    # Building/road:
    #     -> built_up
    #
    # Water:
    #     -> water
    #
    # Barren:
    #     -> bare_land
    #
    # Forest/agricultural:
    #     -> vegetation
    LOVE_DA_TO_SATQUERY = {
        0: 4,  # ignore -> unknown
        1: 4,  # background -> unknown
        2: 2,  # building -> built_up
        3: 2,  # road -> built_up
        4: 0,  # water -> water
        5: 3,  # barren -> bare_land
        6: 1,  # forest -> vegetation
        7: 1,  # agricultural -> vegetation
    }

    def __init__(
        self,
        model=None,
        processor=None,
        device: Optional[str] = None,
    ):
        """
        Initialize the semantic segmentation model.

        Parameters
        ----------
        model:
            Optional already-loaded Hugging Face model.

        processor:
            Optional already-loaded SegFormer processor.

        device:
            "cpu" or "cuda".
            Automatically detected if omitted.
        """

        self.device = self._select_device(device)

        self.model = model
        self.processor = processor

        self.loaded = False

        # If both are supplied, consider the model ready.
        if self.model is not None and self.processor is not None:
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True

        # Otherwise automatically load the real model.
        else:
            self._load_pretrained_model()

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------

    @staticmethod
    def _select_device(device: Optional[str]) -> torch.device:
        """
        Select inference device.
        """

        if device is not None:
            requested = device.lower()

            if requested == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "CUDA was requested but CUDA is not available."
                    )

                return torch.device("cuda")

            if requested == "cpu":
                return torch.device("cpu")

            raise ValueError(
                "Device must be either 'cpu' or 'cuda'."
            )

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def _load_pretrained_model(self) -> None:
        """
        Load the pretrained/fine-tuned LoveDA SegFormer model.
        """

        try:
            self.processor = SegformerImageProcessor.from_pretrained(
                self.MODEL_NAME
            )

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.MODEL_NAME
            )

            self.model.to(self.device)
            self.model.eval()

            self.loaded = True

        except Exception as exc:
            self.model = None
            self.processor = None
            self.loaded = False

            raise RuntimeError(
                "Failed to load the SatQuery semantic segmentation "
                f"model '{self.MODEL_NAME}'. Error: {exc}"
            ) from exc

    def load(self, model, processor=None) -> None:
        """
        Attach a custom segmentation model.

        Parameters
        ----------
        model:
            Hugging Face/PyTorch segmentation model.

        processor:
            Corresponding image processor.
        """

        if model is None:
            raise ValueError("Model cannot be None.")

        self.model = model

        if processor is not None:
            self.processor = processor

        if self.processor is None:
            raise ValueError(
                "A SegFormer image processor is required."
            )

        self.model.to(self.device)
        self.model.eval()

        self.loaded = True

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        original: bool = False,
    ) -> np.ndarray:
        """
        Run real semantic segmentation.

        Parameters
        ----------
        image:
            PIL image, NumPy array, or image file path.

        Returns
        -------
        np.ndarray
            2D uint8 class mask. By default this uses SatQuery IDs.
            If original=True, the original LoveDA class IDs are returned.

        SatQuery IDs:
            0 -> water
            1 -> vegetation
            2 -> built_up
            3 -> bare_land
            4 -> unknown
        """

        if not self.loaded:
            raise RuntimeError(
                "Semantic model is not loaded."
            )

        image_array = self._load_image(image)

        height, width = image_array.shape[:2]

        pil_image = Image.fromarray(
            image_array.astype(np.uint8)
        ).convert("RGB")

        # Prepare image for SegFormer.
        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Inference.
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits

        # SegFormer outputs:
        # [batch, classes, height, width]
        prediction = torch.argmax(
            logits,
            dim=1,
        )[0]

        # Resize prediction back to original image dimensions.
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(0).unsqueeze(0).float(),
            size=(height, width),
            mode="nearest",
        )

        prediction = prediction[
            0,
            0,
        ].cpu().numpy().astype(np.uint8)

        # Building extraction needs the original LoveDA class IDs.
        # Normal semantic analysis continues to receive the normalized
        # SatQuery class IDs.
        if original:
            return prediction

        # Convert LoveDA IDs to SatQuery IDs.
        satquery_mask = self._convert_to_satquery_classes(
            prediction
        )

        return satquery_mask

    # ------------------------------------------------------------------
    # CLASS CONVERSION
    # ------------------------------------------------------------------

    def _convert_to_satquery_classes(
        self,
        loveda_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Convert original LoveDA class IDs into SatQuery IDs.
        """

        if loveda_mask.ndim != 2:
            raise ValueError(
                "LoveDA prediction mask must be 2D."
            )

        result = np.full(
            loveda_mask.shape,
            4,
            dtype=np.uint8,
        )

        for loveda_id, satquery_id in self.LOVE_DA_TO_SATQUERY.items():
            result[loveda_mask == loveda_id] = satquery_id

        return result

    # ------------------------------------------------------------------
    # IMAGE LOADING
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> np.ndarray:
        """
        Load an image into a NumPy RGB array.
        """

        if isinstance(image, (str, Path)):
            path = Path(image)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image file not found: {path}"
                )

            with Image.open(path) as img:
                return np.asarray(
                    img.convert("RGB")
                )

        if isinstance(image, Image.Image):
            return np.asarray(
                image.convert("RGB")
            )

        if isinstance(image, np.ndarray):

            if image.ndim == 2:
                # Grayscale -> RGB
                return np.stack(
                    [image, image, image],
                    axis=-1,
                )

            if image.ndim == 3:
                if image.shape[2] == 1:
                    return np.repeat(
                        image,
                        3,
                        axis=2,
                    )

                if image.shape[2] >= 3:
                    return image[:, :, :3]

            raise ValueError(
                "NumPy image must have shape "
                "(H, W) or (H, W, C)."
            )

        raise TypeError(
            "Image must be a file path, PIL Image, "
            "or NumPy array."
        )

    # ------------------------------------------------------------------
    # INFORMATION METHODS
    # ------------------------------------------------------------------

    def get_class_map(self) -> Dict[int, str]:
        """
        Return SatQuery class mapping.
        """

        return self.CLASS_MAP.copy()

    def get_original_class_map(self) -> Dict[int, str]:
        """
        Return original LoveDA class mapping.
        """

        return self.ORIGINAL_CLASS_MAP.copy()

    def get_mapping(self) -> Dict[int, int]:
        """
        Return LoveDA -> SatQuery class mapping.
        """

        return self.LOVE_DA_TO_SATQUERY.copy()

    def is_loaded(self) -> bool:
        """
        Check whether the real model is loaded.
        """

        return (
            self.loaded
            and self.model is not None
            and self.processor is not None
        )

    def get_device(self) -> str:
        """
        Return the current inference device.
        """

        return str(self.device)

    def get_model_name(self) -> str:
        """
        Return the Hugging Face model name.
        """

        return self.MODEL_NAME