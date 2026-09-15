# ============================================================
# FILE: ai/vqa/single_image_vqa.py
# REPLACE THE COMPLETE FILE WITH THIS VERSION
# ============================================================

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image, ImageOps
from transformers import (
    AutoProcessor,
    BlipForQuestionAnswering,
)


class SingleImageVQA:
    """
    Optimized BLIP single-image VQA for SatQuery AI.

    Competition-oriented optimizations:
        - Shared model cache across all instances
        - Thread-safe model loading
        - Automatic CUDA / CPU selection
        - model.eval()
        - torch.inference_mode()
        - CUDA autocast when available
        - Greedy decoding
        - Image pre-resize before processor
        - Answer cache for repeated queries
        - Detailed timing telemetry
    """

    MODEL_NAME = (
        "Salesforce/blip-vqa-base"
    )

    _MODEL_CACHE: Dict[
        str,
        Tuple[
            Any,
            BlipForQuestionAnswering,
        ],
    ] = {}

    _MODEL_CACHE_LOCK = (
        threading.Lock()
    )

    _TORCH_CONFIGURED = False

    _TORCH_CONFIG_LOCK = (
        threading.Lock()
    )

    def __init__(
        self,
        device: str = "auto",
        max_new_tokens: int = 12,
        max_input_side: int = 1024,
        answer_cache_size: int = 64,
    ):

        self.device = (
            self._resolve_device(
                device
            )
        )

        self.max_new_tokens = int(
            max_new_tokens
        )

        self.max_input_side = int(
            max_input_side
        )

        self.answer_cache_size = int(
            answer_cache_size
        )

        self.processor = None
        self.model = None

        self.model_load_ms = 0.0

        self.loaded_from_cache = False

        self._answer_cache = (
            OrderedDict()
        )

        self._configure_torch()

        self._load_model()

    # ========================================================
    # DEVICE
    # ========================================================

    @staticmethod
    def _resolve_device(
        requested: Optional[str],
    ) -> str:

        value = (
            requested
            or "auto"
        ).lower()

        if (
            value
            not in {
                "auto",
                "cpu",
                "cuda",
                "mps",
            }
        ):
            value = "auto"

        if value == "cuda":

            if torch.cuda.is_available():
                return "cuda"

            return "cpu"

        if value == "mps":

            if (
                hasattr(
                    torch.backends,
                    "mps",
                )
                and torch.backends
                .mps
                .is_available()
            ):
                return "mps"

            return "cpu"

        if value == "cpu":
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(
                torch.backends,
                "mps",
            )
            and torch.backends
            .mps
            .is_available()
        ):
            return "mps"

        return "cpu"

    # ========================================================
    # TORCH RUNTIME
    # ========================================================

    def _configure_torch(
        self,
    ) -> None:

        with self._TORCH_CONFIG_LOCK:

            if (
                SingleImageVQA
                ._TORCH_CONFIGURED
            ):
                return

            if self.device == "cpu":

                default_threads = min(
                    8,
                    max(
                        1,
                        os.cpu_count()
                        or 4,
                    ),
                )

                try:

                    threads = int(
                        os.getenv(
                            "SATQUERY_TORCH_THREADS",
                            str(
                                default_threads
                            ),
                        )
                    )

                except Exception:

                    threads = (
                        default_threads
                    )

                threads = max(
                    1,
                    threads,
                )

                try:
                    torch.set_num_threads(
                        threads
                    )
                except Exception:
                    pass

                try:

                    interop_threads = int(
                        os.getenv(
                            "SATQUERY_TORCH_INTEROP_THREADS",
                            "2",
                        )
                    )

                    torch.set_num_interop_threads(
                        max(
                            1,
                            interop_threads,
                        )
                    )

                except Exception:
                    pass

            if (
                self.device == "cuda"
                and torch.cuda.is_available()
            ):

                try:

                    torch.backends.cuda.matmul.allow_tf32 = True

                    torch.backends.cudnn.allow_tf32 = True

                except Exception:
                    pass

            try:

                torch.set_float32_matmul_precision(
                    "high"
                )

            except Exception:
                pass

            SingleImageVQA._TORCH_CONFIGURED = True

    # ========================================================
    # MODEL CACHE
    # ========================================================

    def _cache_key(
        self,
    ) -> str:

        return (
            f"{self.MODEL_NAME}"
            f"::{self.device}"
        )

    def _load_model(
        self,
    ) -> None:

        cache_key = (
            self._cache_key()
        )

        start = time.perf_counter()

        with self._MODEL_CACHE_LOCK:

            cached = (
                self._MODEL_CACHE.get(
                    cache_key
                )
            )

            if cached is not None:

                (
                    self.processor,
                    self.model,
                ) = cached

                self.loaded_from_cache = (
                    True
                )

                self.model_load_ms = (
                    (
                        time.perf_counter()
                        - start
                    )
                    * 1000.0
                )

                return

            print(
                "[SatQuery][BLIP] "
                f"Loading {self.MODEL_NAME} "
                f"on {self.device}..."
            )

            processor = (
                AutoProcessor
                .from_pretrained(
                    self.MODEL_NAME
                )
            )

            model = (
                BlipForQuestionAnswering
                .from_pretrained(
                    self.MODEL_NAME
                )
            )

            model.eval()

            model.to(
                self.device
            )

            self._MODEL_CACHE[
                cache_key
            ] = (
                processor,
                model,
            )

            self.processor = (
                processor
            )

            self.model = (
                model
            )

            self.loaded_from_cache = (
                False
            )

        self.model_load_ms = (
            (
                time.perf_counter()
                - start
            )
            * 1000.0
        )

        print(
            "[SatQuery][BLIP] "
            "Model ready in "
            f"{self.model_load_ms / 1000.0:.2f}s"
        )

    # ========================================================
    # IMAGE
    # ========================================================

    def _prepare_image(
        self,
        image_path: str,
    ) -> Tuple[
        Image.Image,
        Dict[str, Any],
    ]:

        path = Path(
            image_path
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        with Image.open(
            path
        ) as source:

            source = (
                ImageOps
                .exif_transpose(
                    source
                )
            )

            original_width = int(
                source.width
            )

            original_height = int(
                source.height
            )

            original_mode = (
                source.mode
            )

            image = (
                source
                .convert(
                    "RGB"
                )
            )

            resized = False

            if max(
                image.size
            ) > self.max_input_side:

                image.thumbnail(
                    (
                        self.max_input_side,
                        self.max_input_side,
                    ),
                    Image.Resampling.LANCZOS,
                )

                resized = True

            image = image.copy()

        return (
            image,
            {
                "original_width":
                    original_width,

                "original_height":
                    original_height,

                "original_mode":
                    original_mode,

                "inference_width":
                    int(
                        image.width
                    ),

                "inference_height":
                    int(
                        image.height
                    ),

                "pre_resized":
                    resized,
            },
        )

    # ========================================================
    # ANSWER CACHE
    # ========================================================

    def _answer_cache_key(
        self,
        image_path: str,
        question: str,
    ):

        path = Path(
            image_path
        )

        try:

            stat = path.stat()

            identity = (
                str(
                    path.resolve()
                ),
                int(
                    stat.st_size
                ),
                int(
                    stat.st_mtime_ns
                ),
            )

        except Exception:

            identity = (
                str(path),
                0,
                0,
            )

        return (
            identity,
            question
            .strip()
            .lower(),
        )

    def _get_cached_answer(
        self,
        key,
    ):

        if (
            key
            not in self._answer_cache
        ):
            return None

        value = (
            self._answer_cache.pop(
                key
            )
        )

        self._answer_cache[
            key
        ] = value

        return dict(
            value
        )

    def _cache_answer(
        self,
        key,
        value: Dict[str, Any],
    ) -> None:

        self._answer_cache[
            key
        ] = dict(
            value
        )

        self._answer_cache.move_to_end(
            key
        )

        while (
            len(
                self._answer_cache
            )
            >
            self.answer_cache_size
        ):

            self._answer_cache.popitem(
                last=False
            )

    # ========================================================
    # INFERENCE
    # ========================================================

    def answer(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, Any]:

        total_start = (
            time.perf_counter()
        )

        question = (
            question
            or ""
        ).strip()

        if not question:

            return {
                "success":
                    False,

                "message":
                    "Question cannot be empty.",

                "error":
                    "EmptyQuestion",
            }

        cache_key = (
            self._answer_cache_key(
                image_path,
                question,
            )
        )

        cached = (
            self._get_cached_answer(
                cache_key
            )
        )

        if cached is not None:

            cached[
                "cache_hit"
            ] = True

            cached[
                "timing_ms"
            ] = {
                "total":
                    round(
                        (
                            time.perf_counter()
                            - total_start
                        )
                        * 1000.0,
                        2,
                    ),

                "model_load":
                    0.0,

                "image_prepare":
                    0.0,

                "preprocess":
                    0.0,

                "inference":
                    0.0,

                "decode":
                    0.0,
            }

            print(
                "[SatQuery][BLIP] "
                "Answer cache hit."
            )

            return cached

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_start = (
            time.perf_counter()
        )

        (
            image,
            image_metadata,
        ) = self._prepare_image(
            image_path
        )

        image_ms = (
            (
                time.perf_counter()
                - image_start
            )
            * 1000.0
        )

        # ----------------------------------------------------
        # PROCESSOR
        # ----------------------------------------------------

        preprocess_start = (
            time.perf_counter()
        )

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
        )

        inputs = {
            key:
                tensor.to(
                    self.device,
                    non_blocking=(
                        self.device
                        == "cuda"
                    ),
                )
            for key, tensor
            in inputs.items()
        }

        preprocess_ms = (
            (
                time.perf_counter()
                - preprocess_start
            )
            * 1000.0
        )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        inference_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            if (
                self.device == "cuda"
                and torch.cuda.is_available()
            ):

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    generated = (
                        self.model.generate(
                            **inputs,
                            max_new_tokens=(
                                self.max_new_tokens
                            ),
                            num_beams=1,
                            do_sample=False,
                        )
                    )

            else:

                generated = (
                    self.model.generate(
                        **inputs,
                        max_new_tokens=(
                            self.max_new_tokens
                        ),
                        num_beams=1,
                        do_sample=False,
                    )
                )

        inference_ms = (
            (
                time.perf_counter()
                - inference_start
            )
            * 1000.0
        )

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        decode_start = (
            time.perf_counter()
        )

        answer = (
            self.processor.decode(
                generated[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        decode_ms = (
            (
                time.perf_counter()
                - decode_start
            )
            * 1000.0
        )

        total_ms = (
            (
                time.perf_counter()
                - total_start
            )
            * 1000.0
        )

        timing = {
            "model_load":
                round(
                    self.model_load_ms,
                    2,
                ),

            "image_prepare":
                round(
                    image_ms,
                    2,
                ),

            "preprocess":
                round(
                    preprocess_ms,
                    2,
                ),

            "inference":
                round(
                    inference_ms,
                    2,
                ),

            "decode":
                round(
                    decode_ms,
                    2,
                ),

            "total":
                round(
                    total_ms,
                    2,
                ),
        }

        print(
            "[SatQuery][BLIP] "
            f"prepare={image_ms:.1f}ms | "
            f"preprocess={preprocess_ms:.1f}ms | "
            f"inference={inference_ms:.1f}ms | "
            f"decode={decode_ms:.1f}ms | "
            f"total={total_ms:.1f}ms"
        )

        result = {
            "success":
                True,

            "task":
                "single_image_vqa",

            "answer":
                answer,

            "question":
                question,

            "model":
                self.MODEL_NAME,

            "device":
                self.device,

            "cache_hit":
                False,

            "timing_ms":
                timing,

            "metadata": {
                "input_mode":
                    "single_image",

                "image_width":
                    image_metadata[
                        "original_width"
                    ],

                "image_height":
                    image_metadata[
                        "original_height"
                    ],

                "image_mode":
                    image_metadata[
                        "original_mode"
                    ],

                "inference_width":
                    image_metadata[
                        "inference_width"
                    ],

                "inference_height":
                    image_metadata[
                        "inference_height"
                    ],

                "pre_resized":
                    image_metadata[
                        "pre_resized"
                    ],

                "max_new_tokens":
                    self.max_new_tokens,

                "model_cached":
                    self.loaded_from_cache,

                "remote_sensing_adapted":
                    False,

                "model_family": "GENERIC_VQA",

                "model_confidence": None,

                "confidence_provenance": "UNAVAILABLE",

                "baseline":
                    True,

                "timing_ms":
                    timing,
            },

            "limitations": (
                "BLIP is used as the single-image VQA "
                "baseline. Remote-sensing-specific components "
                "are used elsewhere in SatQuery AI."
            ),
        }

        self._cache_answer(
            cache_key,
            result,
        )

        return result

    # ========================================================
    # OPTIONAL STARTUP WARMUP
    # ========================================================

    def warmup(
        self,
    ) -> Dict[str, Any]:

        start = (
            time.perf_counter()
        )

        image = Image.new(
            "RGB",
            (
                384,
                384,
            ),
            color=(
                110,
                130,
                110,
            ),
        )

        inputs = self.processor(
            images=image,
            text="What is visible?",
            return_tensors="pt",
        )

        inputs = {
            key:
                tensor.to(
                    self.device
                )
            for key, tensor
            in inputs.items()
        }

        with torch.inference_mode():

            self.model.generate(
                **inputs,
                max_new_tokens=2,
                num_beams=1,
                do_sample=False,
            )

        duration = (
            (
                time.perf_counter()
                - start
            )
            * 1000.0
        )

        print(
            "[SatQuery][BLIP] "
            f"Warmup completed in "
            f"{duration:.1f}ms"
        )

        return {
            "success":
                True,

            "device":
                self.device,

            "warmup_ms":
                round(
                    duration,
                    2,
                ),
        }
