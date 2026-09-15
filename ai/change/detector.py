# ai/change/detector.py

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np


@dataclass
class ChangeResult:
    mask: np.ndarray
    metadata: Dict


class Sentinel2ChangeDetector:
    """
    Existing spectral-difference detector.

    Input:
        before -> (bands, height, width)
        after  -> (bands, height, width)

    Output:
        0 -> stable
        1 -> changed
    """

    def __init__(
        self,
        threshold: float = 0.10,
        normalize: bool = True,
    ):
        if threshold < 0:
            raise ValueError(
                "Threshold must be >= 0."
            )

        self.threshold = float(threshold)
        self.normalize = normalize

    def detect(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:

        before = self._validate(
            before,
            "before",
        )

        after = self._validate(
            after,
            "after",
        )

        if before.shape != after.shape:
            raise ValueError(
                "Before and after images must have "
                "identical shapes. "
                f"Got {before.shape} and {after.shape}."
            )

        before_float = before.astype(
            np.float32
        )

        after_float = after.astype(
            np.float32
        )

        if self.normalize:

            before_float = self._normalize(
                before_float
            )

            after_float = self._normalize(
                after_float
            )

        difference = np.mean(
            np.abs(
                after_float - before_float
            ),
            axis=0,
        )

        mask = (
            difference >= self.threshold
        ).astype(
            np.uint8
        )

        return mask

    def detect_with_result(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> ChangeResult:

        mask = self.detect(
            before,
            after,
        )

        changed_pixels = int(
            np.sum(mask)
        )

        total_pixels = int(
            mask.size
        )

        metadata = {
            "detector": (
                "sentinel2_difference"
            ),
            "before_shape": tuple(
                before.shape
            ),
            "after_shape": tuple(
                after.shape
            ),
            "threshold": (
                self.threshold
            ),
            "changed_pixels": (
                changed_pixels
            ),
            "stable_pixels": (
                total_pixels
                - changed_pixels
            ),
            "changed_percentage": (
                changed_pixels
                / total_pixels
                * 100
                if total_pixels
                else 0.0
            ),
        }

        return ChangeResult(
            mask=mask,
            metadata=metadata,
        )

    @staticmethod
    def _validate(
        image: np.ndarray,
        name: str,
    ) -> np.ndarray:

        if image is None:
            raise ValueError(
                f"{name} image cannot be None."
            )

        image = np.asarray(
            image
        )

        if image.ndim != 3:
            raise ValueError(
                f"{name} image must be a "
                "3D array "
                "(bands, height, width). "
                f"Got {image.ndim}D."
            )

        if image.shape[0] < 1:
            raise ValueError(
                f"{name} image must contain "
                "at least one band."
            )

        return image

    @staticmethod
    def _normalize(
        image: np.ndarray,
    ) -> np.ndarray:

        minimum = image.min(
            axis=(1, 2),
            keepdims=True,
        )

        maximum = image.max(
            axis=(1, 2),
            keepdims=True,
        )

        denominator = (
            maximum - minimum
        )

        denominator[
            denominator == 0
        ] = 1.0

        return (
            image - minimum
        ) / denominator


class ChangeFormerDetector:
    """
    Pretrained ChangeFormerV6 detector.

    Designed for RGB bi-temporal images.

    Model:
        ChangeFormerV6

    Training benchmark:
        LEVIR-CD

    Output:
        0 -> unchanged
        1 -> changed

    Notes:
        LEVIR-CD primarily represents building /
        structural change.

        This detector must not be interpreted as
        a generic water/vegetation/bare-land
        semantic transition model.
    """

    DEFAULT_CHECKPOINT_FOLDER = (
        "CD_ChangeFormerV6_LEVIR_"
        "b16_lr0.0001_adamw_train_test_200_"
        "linear_ce_multi_train_True_"
        "multi_infer_False_shuffle_AB_False_"
        "embed_dim_256"
    )

    def __init__(
        self,
        checkpoint_path: Optional[
            str
        ] = None,
        tile_size: int = 256,
        device: str = "cpu",
    ):

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        self.changeformer_root = (
            self.project_root
            / "external"
            / "ChangeFormer"
        )

        if checkpoint_path:

            self.checkpoint_path = Path(
                checkpoint_path
            )

        else:

            self.checkpoint_path = (
                self.changeformer_root
                / "checkpoints"
                / self.DEFAULT_CHECKPOINT_FOLDER
                / "best_ckpt.pt"
            )

        self.tile_size = int(
            tile_size
        )

        if self.tile_size <= 0:
            raise ValueError(
                "tile_size must be > 0."
            )

        self.device_name = device

        self.device = None
        self.model = None
        self.torch = None

    def is_loaded(
        self,
    ) -> bool:

        return (
            self.model is not None
        )

    def load(
        self,
    ) -> None:

        if self.model is not None:
            return

        if not (
            self.changeformer_root.exists()
        ):
            raise FileNotFoundError(
                "ChangeFormer repository was "
                "not found at: "
                f"{self.changeformer_root}"
            )

        if not (
            self.checkpoint_path.exists()
        ):
            raise FileNotFoundError(
                "ChangeFormer checkpoint was "
                "not found at: "
                f"{self.checkpoint_path}"
            )

        changeformer_path = str(
            self.changeformer_root
        )

        if (
            changeformer_path
            not in sys.path
        ):
            sys.path.insert(
                0,
                changeformer_path,
            )

        import torch

        from models.networks import (
            define_G
        )

        self.torch = torch

        if (
            self.device_name.lower()
            == "cuda"
            and torch.cuda.is_available()
        ):
            self.device = (
                torch.device("cuda")
            )
            gpu_ids = [0]

        else:
            self.device = (
                torch.device("cpu")
            )
            gpu_ids = []

        args = SimpleNamespace(
            net_G="ChangeFormerV6",
            gpu_ids=gpu_ids,
            n_class=2,
            embed_dim=256,
        )

        self.model = define_G(
            args=args,
            gpu_ids=gpu_ids,
        )

        checkpoint = torch.load(
            str(
                self.checkpoint_path
            ),
            map_location=self.device,
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model_G_state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "model_G_state_dict"
            ]

        elif (
            isinstance(checkpoint, dict)
            and "state_dict"
            in checkpoint
        ):
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            state_dict = checkpoint

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _preprocess(
        self,
        image,
    ):

        array = np.asarray(
            image,
            dtype=np.float32,
        ).copy()

        array /= 255.0

        array = np.transpose(
            array,
            (2, 0, 1),
        )

        tensor = (
            self.torch
            .from_numpy(array)
        )

        tensor = (
            tensor - 0.5
        ) / 0.5

        tensor = (
            tensor
            .unsqueeze(0)
            .to(self.device)
        )

        return tensor

    def detect(
        self,
        before_path: str,
        after_path: str,
    ) -> np.ndarray:

        self.load()

        from PIL import Image

        before_file = Path(
            before_path
        )

        after_file = Path(
            after_path
        )

        if not before_file.exists():
            raise FileNotFoundError(
                f"Before image not found: "
                f"{before_file}"
            )

        if not after_file.exists():
            raise FileNotFoundError(
                f"After image not found: "
                f"{after_file}"
            )

        before = Image.open(
            before_file
        ).convert(
            "RGB"
        )

        after = Image.open(
            after_file
        ).convert(
            "RGB"
        )

        if before.size != after.size:
            raise ValueError(
                "Before and after images must "
                "have identical dimensions. "
                f"Got {before.size} and "
                f"{after.size}."
            )

        width, height = (
            before.size
        )

        prediction = np.zeros(
            (
                height,
                width,
            ),
            dtype=np.uint8,
        )

        tile_size = (
            self.tile_size
        )

        with self.torch.no_grad():

            for y in range(
                0,
                height,
                tile_size,
            ):

                for x in range(
                    0,
                    width,
                    tile_size,
                ):

                    x2 = min(
                        x + tile_size,
                        width,
                    )

                    y2 = min(
                        y + tile_size,
                        height,
                    )

                    before_tile = (
                        before.crop(
                            (
                                x,
                                y,
                                x2,
                                y2,
                            )
                        )
                    )

                    after_tile = (
                        after.crop(
                            (
                                x,
                                y,
                                x2,
                                y2,
                            )
                        )
                    )

                    original_width = (
                        before_tile.width
                    )

                    original_height = (
                        before_tile.height
                    )

                    needs_resize = (
                        original_width
                        != tile_size
                        or original_height
                        != tile_size
                    )

                    if needs_resize:

                        before_tile = (
                            before_tile.resize(
                                (
                                    tile_size,
                                    tile_size,
                                ),
                                Image.Resampling.BICUBIC,
                            )
                        )

                        after_tile = (
                            after_tile.resize(
                                (
                                    tile_size,
                                    tile_size,
                                ),
                                Image.Resampling.BICUBIC,
                            )
                        )

                    input_a = (
                        self._preprocess(
                            before_tile
                        )
                    )

                    input_b = (
                        self._preprocess(
                            after_tile
                        )
                    )

                    output = self.model(
                        input_a,
                        input_b,
                    )

                    if isinstance(
                        output,
                        (list, tuple),
                    ):
                        logits = (
                            output[-1]
                        )
                    else:
                        logits = output

                    mask = (
                        self.torch
                        .argmax(
                            logits,
                            dim=1,
                        )
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(
                            np.uint8
                        )
                    )

                    if needs_resize:

                        mask = np.asarray(
                            Image.fromarray(
                                mask
                            ).resize(
                                (
                                    original_width,
                                    original_height,
                                ),
                                Image.Resampling.NEAREST,
                            )
                        ).astype(
                            np.uint8
                        )

                    prediction[
                        y:y2,
                        x:x2,
                    ] = mask[
                        : y2 - y,
                        : x2 - x,
                    ]

        return prediction

    def detect_with_result(
        self,
        before_path: str,
        after_path: str,
        output_path: Optional[
            str
        ] = None,
    ) -> ChangeResult:

        mask = self.detect(
            before_path=before_path,
            after_path=after_path,
        )

        total_pixels = int(
            mask.size
        )

        changed_pixels = int(
            np.count_nonzero(
                mask
            )
        )

        stable_pixels = (
            total_pixels
            - changed_pixels
        )

        changed_percentage = (
            changed_pixels
            / total_pixels
            * 100
            if total_pixels
            else 0.0
        )

        saved_path = None

        if output_path:

            from PIL import Image

            output_file = Path(
                output_path
            )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            Image.fromarray(
                (
                    mask * 255
                ).astype(
                    np.uint8
                )
            ).save(
                output_file
            )

            saved_path = str(
                output_file
            )

        metadata = {
            "detector": (
                "ChangeFormerV6"
            ),
            "training_dataset": (
                "LEVIR-CD"
            ),
            "device": str(
                self.device
            ),
            "tile_size": (
                self.tile_size
            ),
            "checkpoint": str(
                self.checkpoint_path
            ),
            "before_path": str(
                before_path
            ),
            "after_path": str(
                after_path
            ),
            "mask_shape": tuple(
                mask.shape
            ),
            "total_pixels": (
                total_pixels
            ),
            "changed_pixels": (
                changed_pixels
            ),
            "stable_pixels": (
                stable_pixels
            ),
            "changed_percentage": float(
                changed_percentage
            ),
            "evidence_path": (
                saved_path
            ),
            "model_scope": (
                "binary structural/building "
                "change detection"
            ),
        }

        return ChangeResult(
            mask=mask,
            metadata=metadata,
        )