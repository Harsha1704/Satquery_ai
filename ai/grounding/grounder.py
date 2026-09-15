# ai/grounding/remoteclip_grounder.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
import open_clip


class RemoteCLIPGrounder:
    MODEL_NAME = "ViT-B-32"
    REPO_ID = "chendelong/RemoteCLIP"
    CHECKPOINT_NAME = "RemoteCLIP-ViT-B-32.pt"

    def __init__(
        self,
        device: Optional[str] = None,
        tile_size: int = 224,
        stride: int = 112,
        top_k: int = 8,
        minimum_score: float = 0.20,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.top_k = int(top_k)
        self.minimum_score = float(minimum_score)

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        checkpoint_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.CHECKPOINT_NAME,
            cache_dir=r".\checkpoints",
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.MODEL_NAME
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint,
            strict=True,
        )

        model.to(self.device)
        model.eval()

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(
            self.MODEL_NAME
        )

        self._loaded = True

    @staticmethod
    def _validate_image(
        image_path: str,
    ) -> Path:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Not a file: {image_path}"
            )

        return path

    @staticmethod
    def _normalize_prompt(
        prompt: str,
    ) -> str:
        text = prompt.lower().strip()

        replacements = {
            "buildings": "building",
            "built-up": "building",
            "built up": "building",
            "built_up": "building",
            "roads": "road",
            "water bodies": "water",
            "water body": "water",
            "trees": "forest",
            "vegetation": "forest",
            "cars": "vehicle",
        }

        for source, target in replacements.items():
            text = text.replace(
                source,
                target,
            )

        prefixes = [
            "show ",
            "find ",
            "locate ",
            "highlight ",
            "identify ",
            "detect ",
            "where are ",
            "where is ",
        ]

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        suffixes = [
            " in this image",
            " in the image",
            " in this satellite image",
            " on this image",
            " on the image",
        ]

        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[:-len(suffix)]

        return text.strip(" ?.")

    def _build_tiles(
        self,
        image: Image.Image,
    ) -> List[Tuple[int, int, int, int, Image.Image]]:
        width, height = image.size

        tiles = []

        y_positions = list(
            range(
                0,
                max(
                    height - self.tile_size + 1,
                    1,
                ),
                self.stride,
            )
        )

        x_positions = list(
            range(
                0,
                max(
                    width - self.tile_size + 1,
                    1,
                ),
                self.stride,
            )
        )

        if not y_positions:
            y_positions = [0]

        if not x_positions:
            x_positions = [0]

        last_y = max(
            0,
            height - self.tile_size,
        )

        last_x = max(
            0,
            width - self.tile_size,
        )

        if last_y not in y_positions:
            y_positions.append(last_y)

        if last_x not in x_positions:
            x_positions.append(last_x)

        for y in y_positions:
            for x in x_positions:
                x2 = min(
                    width,
                    x + self.tile_size,
                )

                y2 = min(
                    height,
                    y + self.tile_size,
                )

                tile = image.crop(
                    (
                        x,
                        y,
                        x2,
                        y2,
                    )
                )

                tiles.append(
                    (
                        x,
                        y,
                        x2,
                        y2,
                        tile,
                    )
                )

        return tiles

    @torch.inference_mode()
    def _encode_text(
        self,
        target: str,
    ) -> torch.Tensor:
        prompts = [
            target,
            f"a satellite image of {target}",
            f"an aerial image containing {target}",
            f"remote sensing imagery showing {target}",
        ]

        tokens = self.tokenizer(
            prompts
        ).to(
            self.device
        )

        features = self.model.encode_text(
            tokens
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

        features = features.mean(
            dim=0,
            keepdim=True,
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True,
        )

        return features

    @torch.inference_mode()
    def _score_tiles(
        self,
        tiles,
        target: str,
    ) -> List[Dict[str, Any]]:
        text_features = self._encode_text(
            target
        )

        scored = []

        for (
            x1,
            y1,
            x2,
            y2,
            tile,
        ) in tiles:
            image_tensor = self.preprocess(
                tile.convert("RGB")
            ).unsqueeze(0).to(
                self.device
            )

            image_features = self.model.encode_image(
                image_tensor
            )

            image_features = (
                image_features
                / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            similarity = float(
                (
                    image_features
                    @ text_features.T
                )
                .squeeze()
                .cpu()
                .item()
            )

            scored.append(
                {
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],
                    "score": similarity,
                    "target": target,
                }
            )

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored

    @staticmethod
    def _normalize_scores(
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not detections:
            return detections

        values = np.array(
            [
                item["score"]
                for item in detections
            ],
            dtype=np.float32,
        )

        minimum = float(
            values.min()
        )

        maximum = float(
            values.max()
        )

        if maximum - minimum < 1e-8:
            for item in detections:
                item["normalized_score"] = 1.0

            return detections

        for item in detections:
            item["normalized_score"] = float(
                (
                    item["score"] - minimum
                )
                / (
                    maximum - minimum
                )
            )

        return detections

    @staticmethod
    def _iou(
        box_a,
        box_b,
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(
            ax1,
            bx1,
        )

        iy1 = max(
            ay1,
            by1,
        )

        ix2 = min(
            ax2,
            bx2,
        )

        iy2 = min(
            ay2,
            by2,
        )

        iw = max(
            0,
            ix2 - ix1,
        )

        ih = max(
            0,
            iy2 - iy1,
        )

        intersection = iw * ih

        area_a = max(
            0,
            ax2 - ax1,
        ) * max(
            0,
            ay2 - ay1,
        )

        area_b = max(
            0,
            bx2 - bx1,
        ) * max(
            0,
            by2 - by1,
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    def _nms(
        self,
        detections: List[Dict[str, Any]],
        threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        selected = []

        for detection in detections:
            keep = True

            for existing in selected:
                if (
                    self._iou(
                        detection["box"],
                        existing["box"],
                    )
                    >= threshold
                ):
                    keep = False
                    break

            if keep:
                selected.append(
                    detection
                )

        return selected

    @staticmethod
    def _make_heatmap(
        image_width: int,
        image_height: int,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        heatmap = np.zeros(
            (
                image_height,
                image_width,
            ),
            dtype=np.float32,
        )

        weight = np.zeros_like(
            heatmap
        )

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]

            score = float(
                detection.get(
                    "normalized_score",
                    0.0,
                )
            )

            heatmap[
                y1:y2,
                x1:x2,
            ] += score

            weight[
                y1:y2,
                x1:x2,
            ] += 1.0

        valid = weight > 0

        heatmap[
            valid
        ] /= weight[
            valid
        ]

        return heatmap

    @staticmethod
    def _save_evidence(
        image: Image.Image,
        detections: List[Dict[str, Any]],
        heatmap: np.ndarray,
        target: str,
        output_path: str,
    ) -> str:
        rgb = np.asarray(
            image.convert("RGB")
        )

        heat_u8 = np.clip(
            heatmap * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

        colored = cv2.applyColorMap(
            heat_u8,
            cv2.COLORMAP_JET,
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB,
        )

        blended = cv2.addWeighted(
            rgb,
            0.70,
            colored,
            0.30,
            0,
        )

        output = Image.fromarray(
            blended
        )

        draw = ImageDraw.Draw(
            output
        )

        for index, detection in enumerate(
            detections,
            start=1,
        ):
            x1, y1, x2, y2 = detection["box"]

            score = detection.get(
                "normalized_score",
                0.0,
            )

            draw.rectangle(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
                outline=(
                    255,
                    255,
                    0,
                ),
                width=3,
            )

            draw.text(
                (
                    x1 + 4,
                    y1 + 4,
                ),
                f"{target} {score:.2f}",
                fill=(
                    255,
                    255,
                    0,
                ),
            )

        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.save(
            path
        )

        return str(
            path
        )

    def ground(
        self,
        image_path: str,
        text_prompt: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = self._validate_image(
            image_path
        )

        if not text_prompt.strip():
            return {
                "success": False,
                "error": "MissingTextPrompt",
            }

        self.load()

        target = self._normalize_prompt(
            text_prompt
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        tiles = self._build_tiles(
            image
        )

        detections = self._score_tiles(
            tiles,
            target,
        )

        detections = self._normalize_scores(
            detections
        )

        candidates = [
            item
            for item in detections
            if item.get(
                "normalized_score",
                0.0,
            )
            >= self.minimum_score
        ]

        candidates = candidates[
            : max(
                self.top_k * 3,
                self.top_k,
            )
        ]

        candidates = self._nms(
            candidates
        )

        candidates = candidates[
            : self.top_k
        ]

        heatmap = self._make_heatmap(
            image.width,
            image.height,
            detections,
        )

        evidence_path = None

        if output_path:
            evidence_path = self._save_evidence(
                image=image,
                detections=candidates,
                heatmap=heatmap,
                target=target,
                output_path=output_path,
            )

        confidence = max(
            (
                item.get(
                    "normalized_score",
                    0.0,
                )
                for item in candidates
            ),
            default=0.0,
        )

        return {
            "success": True,
            "task": "text_guided_region_grounding",
            "grounding_mode": "remoteclip_tile_grounding",
            "image_path": str(path),
            "requested_prompt": text_prompt,
            "target": target,
            "count": len(candidates),
            "detections": candidates,
            "confidence": float(confidence),
            "visual_evidence": evidence_path,
            "model": "RemoteCLIP-ViT-B-32",
            "device": self.device,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "total_tiles": len(tiles),
            "remote_sensing_adapted": True,
            "limitations": (
                "RemoteCLIP grounding is tile-level visual-text matching. "
                "Boxes represent high-relevance regions rather than exact "
                "instance boundaries."
            ),
        }