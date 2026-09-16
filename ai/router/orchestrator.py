from pathlib import Path
import re
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from .intent import Intent
from .planner import QueryPlanner


class SatQueryOrchestrator:
    """
    Central orchestration layer for SatQuery-AI.

    Supported capabilities:

        - Semantic analysis
        - Single-image visual question answering
        - Text-guided region grounding
        - Object detection
        - Building extraction
        - Multispectral analysis
        - SAR analysis
        - Optical + SAR fusion
        - Bi-temporal change detection
    """

    def __init__(
        self,
        planner: Optional[QueryPlanner] = None,
    ):

        self.planner = (
            planner
            or QueryPlanner()
        )

        # Lazy-loaded.
        self.object_detector = None
        self.single_image_vqa = None
        self._vqa_lock = RLock()
        self.remoteclip_grounder = None
    # ==========================================================
    # PLAN
    # ==========================================================

    def plan(self, query: str) -> Dict[str, Any]:
        """Build a structured executable plan from natural language."""
        if not query or not query.strip():
            return {
                "query": query or "", "intent": Intent.UNKNOWN.value,
                "confidence": 0.0, "matched_keywords": [],
                "target": "overall", "targets": [], "operation": "analyze",
                "change_direction": "none", "years": [],
                "transition_from": "none", "transition_to": "none",
                "required_tools": [], "success": False,
            }

        r = self.planner.plan(query)
        return {
            "query": query,
            "intent": r.intent.value,
            "confidence": r.confidence,
            "matched_keywords": list(r.matched_keywords),
            "target": r.target,
            "targets": list(r.targets),
            "operation": r.operation,
            "change_direction": r.change_direction,
            "years": list(r.years),
            "transition_from": r.transition_from,
            "transition_to": r.transition_to,
            "required_tools": self.planner.required_tools(r.intent),
            "success": r.intent != Intent.UNKNOWN,
        }

    def route(self, query: str) -> Dict[str, Any]:
        return self.plan(query)

    @staticmethod
    def _extract_year_from_path(
        path: Optional[str],
    ) -> Optional[int]:
        if not path:
            return None

        name = Path(str(path)).name
        matches = re.findall(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)",
            name,
        )

        if not matches:
            return None

        try:
            return int(matches[0])
        except (TypeError, ValueError):
            return None

    @classmethod
    def _prepare_change_pair_query(
        cls,
        query: str,
        before_path: Optional[str],
        after_path: Optional[str],
    ) -> str:
        original = str(query or "").strip()

        before_year = cls._extract_year_from_path(
            before_path
        )
        after_year = cls._extract_year_from_path(
            after_path
        )

        if before_year and after_year:
            temporal_prefix = (
                f"What changed between {before_year} "
                f"and {after_year}?"
            )
        else:
            temporal_prefix = (
                "What changed between the before "
                "and after images?"
            )

        if not original:
            return temporal_prefix

        return (
            f"{temporal_prefix} "
            f"User focus: {original}"
        )

    def execute(
        self,
        query: str,
        image_path: Optional[str] = None,
        before_path: Optional[str] = None,
        after_path: Optional[str] = None,
        optical_path: Optional[str] = None,
        sar_path: Optional[str] = None,
        requested_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        user_query = str(query or "").strip()
        execution_query = user_query

        # When a before/after pair is explicitly supplied, the calling
        # workflow has already selected bi-temporal analysis. Do not let an
        # ambiguous user phrase such as "find buildings" reroute that pair
        # into single-image object detection.
        explicit_change_pair = bool(
            before_path
            and after_path
        )

        if explicit_change_pair:
            execution_query = (
                self._prepare_change_pair_query(
                    query=user_query,
                    before_path=before_path,
                    after_path=after_path,
                )
            )

        plan = self.plan(
            execution_query
        )

        if explicit_change_pair:
            plan["intent"] = (
                Intent.CHANGE_DETECTION.value
            )
            plan["success"] = True
            plan["operation"] = "compare"
            plan["required_tools"] = (
                self.planner.required_tools(
                    Intent.CHANGE_DETECTION
                )
            )

        if not plan["success"]:
            output = {
                **plan,
                "validation": {
                    "success": False,
                    "errors": [
                        "Unable to determine the requested analysis."
                    ],
                    "warnings": [],
                },
                "execution_summary": {
                    "status": "not_executed",
                    "reason": "unknown_intent",
                },
                "execution": {
                    "success": False,
                    "message": (
                        "Unable to determine the required "
                        "SatQuery analysis capability."
                    ),
                },
            }
            output["query"] = user_query
            return output

        intent = plan["intent"]

        effective_requested_class = (
            requested_class
        )

        # ChangeFormer supports structural/building change. Preserve a
        # building-oriented user focus, but do not force vegetation/water
        # targets into ChangeFormer because those are handled by the spectral
        # historical analysis.
        if (
            explicit_change_pair
            and not effective_requested_class
        ):
            lowered = user_query.lower()

            if re.search(
                r"\b(building|buildings|house|houses)\b",
                lowered,
            ):
                effective_requested_class = "building"

            elif re.search(
                r"\b(built[ _-]?up|urban|construction)\b",
                lowered,
            ):
                effective_requested_class = "built_up"

        resolved_class = (
            self._resolve_requested_class(
                plan,
                effective_requested_class,
            )
        )

        validation = (
            self._validate_execution_inputs(
                intent,
                image_path,
                before_path,
                after_path,
                optical_path,
                sar_path,
            )
        )

        if not validation["success"]:
            output = {
                **plan,
                "validation": validation,
                "execution_summary": {
                    "status": "not_executed",
                    "intent": intent,
                    "selected_tools": plan[
                        "required_tools"
                    ],
                    "configured_target": (
                        resolved_class
                    ),
                    "reason": (
                        "input_validation_failed"
                    ),
                },
                "execution": {
                    "success": False,
                    "intent": intent,
                    "message": (
                        "Input validation failed."
                    ),
                    "errors": validation[
                        "errors"
                    ],
                },
            }
            output["query"] = user_query
            return output

        try:
            if (
                intent
                == Intent.SEMANTIC_ANALYSIS.value
            ):
                result = self._execute_semantic(
                    image_path,
                    resolved_class,
                )

            elif (
                intent
                == Intent.TEXT_GUIDED_GROUNDING.value
            ):
                result = (
                    self._execute_text_guided_grounding(
                        image_path=image_path,
                        query=execution_query,
                        requested_class=resolved_class,
                    )
                )

            elif (
                intent
                == Intent.OBJECT_DETECTION.value
            ):
                result = self._execute_object_detection(
                    image_path,
                    resolved_class,
                    execution_query,
                )

            elif (
                intent
                == Intent.SINGLE_IMAGE_VQA.value
            ):
                result = (
                    self._execute_single_image_vqa(
                        image_path=image_path,
                        query=execution_query,
                    )
                )

            elif (
                intent
                == Intent.MULTISPECTRAL_ANALYSIS.value
            ):
                result = self._execute_multispectral(
                    image_path
                )

            elif (
                intent
                == Intent.SAR_ANALYSIS.value
            ):
                result = self._execute_sar(
                    sar_path
                    or image_path
                )

            elif (
                intent
                == Intent.OPTICAL_SAR_FUSION.value
            ):
                result = self._execute_fusion(
                    optical_path,
                    sar_path,
                )

            elif (
                intent
                == Intent.CHANGE_DETECTION.value
            ):
                result = (
                    self._execute_change_detection(
                        execution_query,
                        before_path,
                        after_path,
                        resolved_class,
                    )
                )

            else:
                result = {
                    "success": False,
                    "message": (
                        f"Execution for intent "
                        f"'{intent}' is not "
                        "implemented."
                    ),
                }

        except Exception as exc:
            output = {
                **plan,
                "validation": validation,
                "execution_summary": {
                    "status": "failed",
                    "intent": intent,
                    "selected_tools": plan[
                        "required_tools"
                    ],
                    "configured_target": (
                        resolved_class
                    ),
                    "error": type(exc).__name__,
                },
                "execution": {
                    "success": False,
                    "intent": intent,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            }
            output["query"] = user_query
            return output

        summary = {
            "status": (
                "completed"
                if result.get("success", False)
                else "failed"
            ),
            "intent": intent,
            "confidence": plan["confidence"],
            "selected_tools": plan[
                "required_tools"
            ],
            "operation": plan["operation"],
            "target": plan["target"],
            "targets": plan["targets"],
            "configured_class": (
                resolved_class
            ),
            "years": plan["years"],
            "change_direction": plan[
                "change_direction"
            ],
            "transition": {
                "from": plan[
                    "transition_from"
                ],
                "to": plan[
                    "transition_to"
                ],
            },
        }

        output = {
            **plan,
            "validation": validation,
            "execution_summary": summary,
            "execution": result,
        }
        output["query"] = user_query
        return output

    def _resolve_requested_class(
        self, plan: Dict[str, Any], requested_class: Optional[str]
    ) -> Optional[str]:
        if requested_class:
            return self._normalize_analysis_class(requested_class)

        intent = plan.get("intent")
        target = plan.get("target", "overall")
        targets = plan.get("targets", [])

        if plan.get("transition_from", "none") != "none" and plan.get("transition_to", "none") != "none":
            return None
        if len(targets) > 1:
            return None

        land = {"water": "water", "vegetation": "vegetation", "built_up": "built_up", "bare_land": "bare_land"}
        if intent in {Intent.SEMANTIC_ANALYSIS.value, Intent.CHANGE_DETECTION.value} and target in land:
            return land[target]

        if intent == Intent.TEXT_GUIDED_GROUNDING.value:
            return {
                "water": "water",
                "vegetation": "vegetation",
                "built_up": "building",
                "bare_land": "bare land",
                "vehicle": "vehicle",
            }.get(target)

        if intent == Intent.OBJECT_DETECTION.value:
            return {"vehicle": "car", "building": "building", "built_up": "building"}.get(target)
        return None

    @staticmethod
    def _normalize_analysis_class(requested_class: Optional[str]) -> Optional[str]:
        if not requested_class:
            return None
        value = requested_class.lower().strip()
        aliases = {
            "water body": "water", "water bodies": "water", "water": "water",
            "vegetation": "vegetation", "forest": "vegetation", "forests": "vegetation",
            "crop": "vegetation", "crops": "vegetation",
            "built up": "built_up", "built-up": "built_up", "built_up": "built_up", "urban": "built_up",
            "bare land": "bare_land", "bare-land": "bare_land", "bare_land": "bare_land", "barren": "bare_land",
            "cars": "car", "car": "car", "vehicles": "car", "vehicle": "car",
            "trucks": "truck", "truck": "truck", "buses": "bus", "bus": "bus",
            "buildings": "building", "building": "building", "houses": "building", "house": "building",
        }
        return aliases.get(value, value)

    def _validate_execution_inputs(
        self, intent: str, image_path: Optional[str] = None,
        before_path: Optional[str] = None, after_path: Optional[str] = None,
        optical_path: Optional[str] = None, sar_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        if intent == Intent.SEMANTIC_ANALYSIS.value and not image_path:
            errors.append("Semantic analysis requires image_path.")
        elif intent == Intent.TEXT_GUIDED_GROUNDING.value and not image_path:
            errors.append("Text-guided grounding requires image_path.")
        elif intent == Intent.OBJECT_DETECTION.value and not image_path:
            errors.append("Object detection requires image_path.")
        elif intent == Intent.SINGLE_IMAGE_VQA.value and not image_path:
            errors.append("Single-image VQA requires image_path.")
        elif intent == Intent.MULTISPECTRAL_ANALYSIS.value and not image_path:
            errors.append("Multispectral analysis requires image_path.")
        elif intent == Intent.SAR_ANALYSIS.value and not (sar_path or image_path):
            errors.append("SAR analysis requires sar_path or image_path.")
        elif intent == Intent.OPTICAL_SAR_FUSION.value:
            if not optical_path:
                errors.append("Optical-SAR analysis requires optical_path.")
            if not sar_path:
                errors.append("Optical-SAR analysis requires sar_path.")
        elif intent == Intent.CHANGE_DETECTION.value:
            if not before_path:
                errors.append("Change detection requires before_path.")
            if not after_path:
                errors.append("Change detection requires after_path.")

        for name, path in {
            "image_path": image_path, "before_path": before_path,
            "after_path": after_path, "optical_path": optical_path, "sar_path": sar_path,
        }.items():
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                errors.append(f"{name} does not exist: {path}")
            elif not p.is_file():
                errors.append(f"{name} is not a file: {path}")

        return {"success": not errors, "errors": errors, "warnings": warnings}

    # ==========================================================
    # TEXT-GUIDED REGION GROUNDING
    # ==========================================================

    def _execute_text_guided_grounding(
        self,
        image_path: Optional[str],
        query: str,
        requested_class: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not image_path:
            return {
                "success": False,
                "intent": Intent.TEXT_GUIDED_GROUNDING.value,
                "task": "text_guided_region_grounding",
                "message": (
                    "A satellite image is required "
                    "for text-guided grounding."
                ),
                "error": "MissingImagePath",
            }

        self._check_file(image_path)

        try:
            from ai.grounding import RemoteCLIPGrounder

            if self.remoteclip_grounder is None:
                self.remoteclip_grounder = RemoteCLIPGrounder(
                    device="cpu",
                    tile_size=224,
                    stride=112,
                    top_k=6,
                    minimum_score=0.45,
                    batch_size=8,
                )

            grounding_prompt = (
                requested_class
                if requested_class
                and requested_class not in {"overall", "none", ""}
                else query
            )

            output_dir = Path("outputs/evidence")
            output_dir.mkdir(parents=True, exist_ok=True)

            safe_target = (
                str(grounding_prompt)
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )

            safe_target = "".join(
                character
                for character in safe_target
                if character.isalnum() or character == "_"
            )

            safe_target = safe_target[:40] or "grounding"

            evidence_path = (
                output_dir
                / f"remoteclip_{safe_target}.png"
            )

            result = self.remoteclip_grounder.ground(
                image_path=image_path,
                text_prompt=grounding_prompt,
                output_path=str(evidence_path),
            )

            if not result.get("success", False):
                return {
                    "success": False,
                    "intent": Intent.TEXT_GUIDED_GROUNDING.value,
                    "task": "text_guided_region_grounding",
                    "message": result.get(
                        "message",
                        "Text-guided grounding failed.",
                    ),
                    "error": result.get("error"),
                    "details": result.get("details"),
                }

            detections = result.get("detections", [])
            target = result.get("target", grounding_prompt)
            count = len(detections)

            if count > 0:
                answer = (
                    f"RemoteCLIP identified {count} high-relevance "
                    f"region{'s' if count != 1 else ''} for '{target}'."
                )
            else:
                answer = (
                    "No sufficiently relevant regions were found "
                    f"for '{target}'."
                )

            return {
                "success": True,
                "intent": Intent.TEXT_GUIDED_GROUNDING.value,
                "task": "text_guided_region_grounding",
                "question": query,
                "answer": answer,
                "target": target,
                "image_path": str(image_path),
                "count": count,
                "detections": self._make_json_safe(detections),
                "confidence": result.get("confidence", 0.0),
                "visual_evidence": {
                    "type": "remoteclip_grounding",
                    "image": result.get("visual_evidence"),
                },
                "model": result.get("model"),
                "device": result.get("device"),
                "metadata": {
                    "grounding_mode": result.get("grounding_mode"),
                    "tile_size": result.get("tile_size"),
                    "stride": result.get("stride"),
                    "total_tiles": result.get("total_tiles"),
                    "remote_sensing_adapted": result.get(
                        "remote_sensing_adapted",
                        True,
                    ),
                    "confidence_type": "relative_tile_relevance",
                },
                "execution_summary": {
                    "model": result.get("model"),
                    "device": result.get("device"),
                    "input_count": 1,
                    "input_mode": "single_image",
                    "operation": "text_guided_grounding",
                    "target": target,
                    "regions_found": count,
                    "evidence_generated": (
                        result.get("visual_evidence") is not None
                    ),
                    "remote_sensing_adapted": result.get(
                        "remote_sensing_adapted",
                        True,
                    ),
                },
                "limitations": result.get("limitations"),
            }

        except Exception as error:
            return {
                "success": False,
                "intent": Intent.TEXT_GUIDED_GROUNDING.value,
                "task": "text_guided_region_grounding",
                "message": "Text-guided region grounding failed.",
                "error": type(error).__name__,
                "details": str(error),
            }

    # ==========================================================
    # SINGLE-IMAGE VQA
    # ==========================================================

    def preload_vqa(self, warmup: bool = False) -> Dict[str, Any]:
        """Load VQA once, optionally warming it up before serving requests."""
        with self._vqa_lock:
            from ai.vqa import SingleImageVQA

            if self.single_image_vqa is None:
                self.single_image_vqa = SingleImageVQA(device="auto")

            result = {
                "success": True,
                "model": self.single_image_vqa.MODEL_NAME,
                "device": self.single_image_vqa.device,
                "loaded_from_cache": self.single_image_vqa.loaded_from_cache,
                "model_load_ms": self.single_image_vqa.model_load_ms,
            }
            if warmup:
                result["warmup"] = self.single_image_vqa.warmup()
            return result

    def _execute_single_image_vqa(
        self, image_path: Optional[str], query: str,
    ) -> Dict[str, Any]:
        if not image_path:
            return {
                "success": False,
                "intent": Intent.SINGLE_IMAGE_VQA.value,
                "message": "A single satellite image is required for visual question answering.",
                "error": "MissingImagePath",
            }

        self._check_file(image_path)
        try:
            # Share initialization, warmup and cache access with the startup thread.
            with self._vqa_lock:
                self.preload_vqa()
                result = self.single_image_vqa.answer(
                    image_path=image_path, question=query,
                )

            if not result.get("success", False):
                return {
                    "success": False,
                    "intent": Intent.SINGLE_IMAGE_VQA.value,
                    "task": "single_image_vqa",
                    "question": query,
                    "image_path": str(image_path),
                    "message": result.get("message", "Single-image VQA failed."),
                    "error": result.get("error"),
                    "details": result.get("details"),
                }

            metadata = result.get("metadata", {})
            timing = result.get("timing_ms", {})
            return {
                "success": True,
                "intent": Intent.SINGLE_IMAGE_VQA.value,
                "task": "single_image_vqa",
                "question": query,
                "answer": result.get("answer"),
                "image_path": str(image_path),
                "model": result.get("model"),
                "device": result.get("device"),
                "cache_hit": result.get("cache_hit", False),
                "timing_ms": timing,
                "visual_evidence": {"type": "input_image", "image": str(image_path)},
                "metadata": self._make_json_safe(metadata),
                "execution_summary": {
                    "model": result.get("model"),
                    "device": result.get("device"),
                    "input_count": 1,
                    "input_mode": "single_image",
                    "output": "natural_language_answer",
                    "question": query,
                    "answer": result.get("answer"),
                    "cache_hit": result.get("cache_hit", False),
                    "timing_ms": timing,
                    "remote_sensing_adapted": metadata.get("remote_sensing_adapted", False),
                    "baseline": metadata.get("baseline", True),
                },
                "limitations": result.get("limitations"),
            }
        except Exception as error:
            return {
                "success": False,
                "intent": Intent.SINGLE_IMAGE_VQA.value,
                "task": "single_image_vqa",
                "message": "Single-image visual question answering failed.",
                "error": type(error).__name__,
                "details": str(error),
            }

    # ==========================================================
    # OBJECT DETECTION
    # ==========================================================

    def _execute_object_detection(
        self,
        image_path: Optional[str],
        requested_class: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not image_path:

            return {
                "success": False,
                "task": "object_detection",
                "message": (
                    "Object detection requires "
                    "image_path."
                ),
            }

        self._check_file(
            image_path
        )

        # ------------------------------------------------------
        # LOAD ONCE
        # ------------------------------------------------------

        if self.object_detector is None:

            from ai.detection import (
                ObjectDetector
            )

            # Keep the router aligned with ObjectDetector defaults.
            self.object_detector = (
                ObjectDetector(
                    confidence_threshold=0.25,
                    tile_size=512,
                    tile_overlap=0.25,
                    nms_iou_threshold=0.45,
                    tile_batch_size=4,
                    tile_upscale=1.5,
                    inference_size=640,
                    building_tile_size=512,
                    building_tile_overlap=0.25,
                    min_building_area=80,
                )
            )

        detector = (
            self.object_detector
        )

        if not detector.is_loaded():

            return {
                "success": False,
                "task": "object_detection",
                "message": (
                    "Object detection model "
                    "failed to load."
                ),
            }

        # ------------------------------------------------------
        # DETERMINE REQUEST
        # ------------------------------------------------------

        normalized_class = (
            self._normalize_object_class(
                requested_class
            )
        )

        wants_building = (
            self._query_mentions_building(
                query=query,
                requested_class=(
                    normalized_class
                ),
            )
        )

        # ------------------------------------------------------
        # BUILDING ONLY
        # ------------------------------------------------------

        if (
            normalized_class
            == "building"
        ):

            building_result = (
                detector.detect(
                    image_path=image_path,
                    requested_class="building",
                )
            )

            return {
                "success": True,
                "task": "object_detection",
                "object_detection": (
                    building_result
                ),
                "combined_analysis": False,
            }

        # ------------------------------------------------------
        # NORMAL OBJECT DETECTION
        # ------------------------------------------------------

        detection_result = detector.detect(
            image_path=image_path,
            requested_class=normalized_class,
            confidence_threshold=detector.confidence_threshold,
        )

        # ------------------------------------------------------
        # If the caller didn't pin a single class (e.g. a query like
        # "detect cars and buildings"), detect() just ran against every
        # SUPPORTED_OBJECTS class. Keep only the classes the query
        # actually named -- otherwise stray low-confidence COCO
        # misclassifications (a parking row read as "train", a rooftop
        # read as "boat") show up in the count even though the user
        # never asked about them.
        # ------------------------------------------------------

        if normalized_class is None and query:

            target_classes = (
                self._extract_target_classes(query)
            )

            if target_classes:

                detection_result = (
                    self._filter_detection_result(
                        detection_result,
                        target_classes,
                    )
                )

        execution: Dict[str, Any] = {
            "success": True,
            "task": "object_detection",
            "object_detection": (
                detection_result
            ),
            "combined_analysis": False,
        }

        # ------------------------------------------------------
        # BUILDINGS + OBJECTS
        # ------------------------------------------------------

        if wants_building:

            building_result = (
                detector.detect(
                    image_path=image_path,
                    requested_class="building",
                )
            )

            execution[
                "combined_analysis"
            ] = True

            execution[
                "building_detection"
            ] = building_result

            # --------------------------------------------------
            # Combined counts
            # --------------------------------------------------

            object_counts = (
                detection_result.get(
                    "class_counts",
                    {},
                )
            )

            building_counts = (
                building_result.get(
                    "class_counts",
                    {},
                )
            )

            combined_counts = {}

            combined_counts.update(
                object_counts
            )

            combined_counts.update(
                building_counts
            )

            execution[
                "combined_class_counts"
            ] = combined_counts

            execution[
                "combined_detection_count"
            ] = (
                detection_result.get(
                    "detection_count",
                    0,
                )
                + building_result.get(
                    "detection_count",
                    0,
                )
            )

            execution[
                "combined_description"
            ] = (
                self._build_combined_description(
                    combined_counts
                )
            )

        return execution

    # ==========================================================
    # OBJECT CLASS
    # ==========================================================

    @staticmethod
    def _normalize_object_class(
        requested_class: Optional[str],
    ) -> Optional[str]:

        if not requested_class:
            return None

        value = (
            requested_class
            .lower()
            .strip()
        )

        aliases = {
            "cars": "car",
            "car": "car",

            "vehicles": "car",
            "vehicle": "car",

            "trucks": "truck",
            "truck": "truck",

            "buses": "bus",
            "bus": "bus",

            "people": "person",
            "persons": "person",
            "person": "person",

            "ships": "boat",
            "ship": "boat",
            "boats": "boat",
            "boat": "boat",

            "airplanes": "airplane",
            "airplane": "airplane",
            "aircraft": "airplane",

            "motorcycles": "motorcycle",
            "motorcycle": "motorcycle",

            "bicycles": "bicycle",
            "bicycle": "bicycle",

            "laptops": "laptop",
            "laptop": "laptop",

            "buildings": "building",
            "building": "building",

            "houses": "building",
            "house": "building",

            "built up": "building",
            "built-up": "building",
        }

        return aliases.get(
            value,
            value,
        )

    # ==========================================================
    # TARGET CLASS EXTRACTION
    # ==========================================================

    # Keywords that mean "the user wants this COCO class". Deliberately
    # excludes "building" -- that path is already handled separately by
    # _query_mentions_building / the dedicated building detector.
    CLASS_KEYWORDS: Dict[str, List[str]] = {
        "car": ["car", "cars", "vehicle", "vehicles"],
        "truck": ["truck", "trucks"],
        "bus": ["bus", "buses"],
        "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes"],
        "bicycle": ["bicycle", "bicycles", "bike", "bikes"],
        "person": ["person", "people", "persons", "pedestrian", "pedestrians"],
        "boat": ["boat", "boats", "ship", "ships"],
        "airplane": ["airplane", "airplanes", "aircraft", "plane", "planes"],
        "train": ["train", "trains"],
        "laptop": ["laptop", "laptops"],
    }

    @classmethod
    def _extract_target_classes(
        cls,
        query: Optional[str],
    ) -> set:
        """
        Return the set of COCO classes explicitly named in the query
        text. Returns an empty set if nothing matched -- callers should
        treat an empty set as "no filtering possible, leave as-is"
        rather than "detected nothing was asked for".
        """

        if not query:
            return set()

        text = query.lower()

        found = set()

        for class_name, keywords in cls.CLASS_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                found.add(class_name)

        return found

    @staticmethod
    def _filter_detection_result(
        detection_result: Dict[str, Any],
        target_classes: set,
    ) -> Dict[str, Any]:
        """
        Drop any detections/classes not in target_classes from an
        already-computed detect() result, and recompute the derived
        fields (counts, description) so they stay consistent.
        """

        from collections import Counter

        detections = [
            item
            for item in detection_result.get("detections", [])
            if item.get("class") in target_classes
        ]

        counts = dict(Counter(item["class"] for item in detections))

        filtered = dict(detection_result)
        filtered["detections"] = detections
        filtered["detection_count"] = len(detections)
        filtered["class_counts"] = counts
        filtered["description"] = (
            SatQueryOrchestrator._describe_counts(counts)
        )

        return filtered

    @staticmethod
    def _describe_counts(counts: Dict[str, int]) -> str:

        if not counts:
            return "None of the requested objects were detected."

        preferred = [
            "car", "truck", "bus", "motorcycle", "bicycle",
            "person", "boat", "airplane", "train", "laptop",
        ]
        ordered = [c for c in preferred if c in counts]
        ordered += [c for c in counts if c not in ordered]

        parts = []
        for name in ordered:
            count = counts[name]
            noun = name if count == 1 else f"{name}s"
            parts.append(f"{count} {noun}")

        if len(parts) == 1:
            return f"Detected {parts[0]}."
        if len(parts) == 2:
            return f"Detected {parts[0]} and {parts[1]}."
        return f"Detected {', '.join(parts[:-1])}, and {parts[-1]}."

    # ==========================================================
    # BUILDING QUERY
    # ==========================================================

    @staticmethod
    def _query_mentions_building(
        query: Optional[str],
        requested_class: Optional[str],
    ) -> bool:

        if (
            requested_class
            == "building"
        ):
            return True

        if not query:
            return False

        text = (
            query
            .lower()
        )

        building_words = [
            "building",
            "buildings",
            "house",
            "houses",
            "built up",
            "built-up",
        ]

        return any(
            word in text
            for word in building_words
        )

    # ==========================================================
    # COMBINED DESCRIPTION
    # ==========================================================

    @staticmethod
    def _build_combined_description(
        counts: Dict[str, int],
    ) -> str:

        if not counts:

            return (
                "No requested objects "
                "were detected."
            )

        parts = []

        preferred_order = [
            "car",
            "truck",
            "bus",
            "motorcycle",
            "bicycle",
            "person",
            "boat",
            "airplane",
            "building",
        ]

        ordered_keys = []

        for key in preferred_order:

            if key in counts:
                ordered_keys.append(key)

        for key in counts:

            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:

            count = counts[key]

            noun = (
                key
                if count == 1
                else f"{key}s"
            )

            parts.append(
                f"{count} {noun}"
            )

        if len(parts) == 1:

            return (
                "Detected "
                + parts[0]
                + "."
            )

        if len(parts) == 2:

            return (
                "Detected "
                + parts[0]
                + " and "
                + parts[1]
                + "."
            )

        return (
            "Detected "
            + ", ".join(parts[:-1])
            + ", and "
            + parts[-1]
            + "."
        )

    # ==========================================================
    # SERIALIZE OBJECT RESULT
    # ==========================================================

    @staticmethod
    def _serialize_object_result(
        result: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            result,
            dict,
        ):
            return result

        output: Dict[
            str,
            Any
        ] = {}

        if hasattr(
            result,
            "__dict__",
        ):

            output.update(
                vars(result)
            )

        return (
            SatQueryOrchestrator
            ._make_json_safe(
                output
            )
        )

    # ==========================================================
    # JSON SAFE
    # ==========================================================

    @staticmethod
    def _make_json_safe(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            dict,
        ):

            return {
                str(key): (
                    SatQueryOrchestrator
                    ._make_json_safe(
                        item
                    )
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):

            return [
                SatQueryOrchestrator
                ._make_json_safe(item)
                for item in value
            ]

        if hasattr(
            value,
            "__dict__",
        ):

            return (
                SatQueryOrchestrator
                ._make_json_safe(
                    vars(value)
                )
            )

        return value

    # ==========================================================
    # SEMANTIC
    # ==========================================================

    def _execute_semantic(
        self,
        image_path: Optional[str],
        requested_class: Optional[str],
    ) -> Dict[str, Any]:

        if not image_path:

            return {
                "success": False,
                "message": (
                    "Semantic analysis requires "
                    "image_path."
                ),
            }

        self._check_file(
            image_path
        )

        from ai.models import (
            SemanticModel
        )

        from ai.semantic.analyzer import (
            SemanticAnalyzer
        )

        model = SemanticModel()

        if not model.is_loaded():

            return {
                "success": False,
                "message": (
                    "Semantic model failed "
                    "to load."
                ),
            }

        mask = model.predict(
            image_path
        )

        analyzer = (
            SemanticAnalyzer()
        )

        result = analyzer.analyze(
            mask,
            class_map=(
                model.get_class_map()
            ),
            requested_class=(
                requested_class
            ),
        )

        description = (
            analyzer.describe(
                result
            )
        )

        return {
            "success": True,
            "task": "semantic_analysis",
            "image": str(
                image_path
            ),
            "mask_shape": tuple(
                mask.shape
            ),
            "mask_dtype": str(
                mask.dtype
            ),
            "analysis": (
                self._make_json_safe(
                    result
                )
            ),
            "description": description,
        }

    # ==========================================================
    # MULTISPECTRAL
    # ==========================================================

    def _execute_multispectral(
        self,
        image_path: Optional[str],
    ) -> Dict[str, Any]:

        if not image_path:

            return {
                "success": False,
                "message": (
                    "Multispectral analysis "
                    "requires image_path."
                ),
            }

        from ai.multispectral import (
            MultispectralAnalyzer
        )

        path = Path(
            image_path
        )

        self._check_file(
            image_path
        )

        if (
            path.suffix.lower()
            == ".npy"
        ):

            data = np.load(
                path,
                allow_pickle=False,
            )

        else:

            from ai.data import (
                MultispectralLoader
            )

            loader = (
                MultispectralLoader()
            )

            loaded = loader.load(
                str(path)
            )

            data = loaded.data

        if not isinstance(
            data,
            np.ndarray,
        ):

            raise TypeError(
                "Multispectral input "
                "must be a NumPy array."
            )

        if data.ndim != 3:

            raise ValueError(
                "Multispectral data must "
                "have shape (bands, height, width). "
                f"Got {data.shape}."
            )

        analyzer = (
            MultispectralAnalyzer()
        )

        result = analyzer.summary(
            data
        )

        return {
            "success": True,
            "task": "multispectral_analysis",
            "image": str(
                image_path
            ),
            "shape": tuple(
                data.shape
            ),
            "dtype": str(
                data.dtype
            ),
            "analysis": (
                self._make_json_safe(
                    result
                )
            ),
            "description": (
                self._describe_multispectral(
                    result
                )
            ),
        }

    # ==========================================================
    # SAR
    # ==========================================================

    def _execute_sar(
        self,
        image_path: Optional[str],
    ) -> Dict[str, Any]:

        if not image_path:

            return {
                "success": False,
                "message": (
                    "SAR analysis requires "
                    "a SAR image or .npy file."
                ),
            }

        from ai.sar import (
            SARLoader,
            SARAnalyzer,
        )

        self._check_file(
            image_path
        )

        loader = SARLoader()

        sample = loader.load(
            image_path
        )

        analyzer = SARAnalyzer()

        result = analyzer.summary(
            sample.data
        )

        return {
            "success": True,
            "task": "sar_analysis",
            "image": str(
                image_path
            ),
            "shape": tuple(
                sample.data.shape
            ),
            "dtype": str(
                sample.data.dtype
            ),
            "analysis": (
                self._make_json_safe(
                    result
                )
            ),
            "description": (
                self._describe_sar(
                    result
                )
            ),
        }

    # ==========================================================
    # OPTICAL + SAR FUSION
    # ==========================================================

    def _execute_fusion(
        self,
        optical_path: Optional[str],
        sar_path: Optional[str],
    ) -> Dict[str, Any]:

        if not optical_path:

            return {
                "success": False,
                "message": (
                    "Optical-SAR fusion requires "
                    "optical_path."
                ),
            }

        if not sar_path:

            return {
                "success": False,
                "message": (
                    "Optical-SAR fusion requires "
                    "sar_path."
                ),
            }

        self._check_file(
            optical_path
        )

        self._check_file(
            sar_path
        )

        from ai.sar import (
            SARLoader
        )

        from ai.fusion import (
            OpticalSARFusion,
            FusionAnalyzer,
        )

        optical_file = Path(
            optical_path
        )

        if (
            optical_file.suffix.lower()
            == ".npy"
        ):

            optical_data = np.load(
                optical_file,
                allow_pickle=False,
            )

            if optical_data.ndim != 3:

                raise ValueError(
                    "Optical .npy data must "
                    "have shape (bands,H,W)."
                )

        else:

            from ai.data import (
                MultispectralLoader
            )

            optical_loader = (
                MultispectralLoader()
            )

            optical_sample = (
                optical_loader.load(
                    str(optical_file)
                )
            )

            optical_data = (
                optical_sample.data
            )

        sar_loader = SARLoader()

        sar_sample = sar_loader.load(
            str(sar_path)
        )

        sar_data = (
            sar_sample.data
        )

        fusion = (
            OpticalSARFusion()
        )

        fused_result = fusion.fuse(
            optical_data,
            sar_data,
        )

        analyzer = (
            FusionAnalyzer()
        )

        analysis = analyzer.analyze(
            fused_result.fused
        )

        description = (
            analyzer.describe(
                analysis
            )
        )

        return {
            "success": True,
            "task": "optical_sar_fusion",
            "optical": {
                "path": str(
                    optical_path
                ),
                "shape": tuple(
                    optical_data.shape
                ),
                "dtype": str(
                    optical_data.dtype
                ),
            },
            "sar": {
                "path": str(
                    sar_path
                ),
                "shape": tuple(
                    sar_data.shape
                ),
                "dtype": str(
                    sar_data.dtype
                ),
            },
            "fused": {
                "shape": tuple(
                    fused_result.fused.shape
                ),
                "dtype": str(
                    fused_result.fused.dtype
                ),
                "metadata": (
                    self._make_json_safe(
                        fused_result.metadata
                    )
                ),
            },
            "analysis": (
                self._make_json_safe(
                    analysis
                )
            ),
            "description": description,
        }

    # ==========================================================
    # CHANGE DETECTION
    # ==========================================================

    # ai/router/orchestrator.py
# REPLACE ONLY _execute_change_detection() WITH THIS VERSION


    # === SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ===

    @staticmethod
    def _assess_visual_pair(
        before_path: str,
        after_path: str,
    ) -> Dict[str, Any]:
        # Scene-level RGB difference metrics. These complement ChangeFormer;
        # they are not semantic land-cover labels by themselves.
        from PIL import Image

        with Image.open(before_path) as before_image:
            before_rgb = np.asarray(
                before_image.convert("RGB"),
                dtype=np.float32,
            )

        with Image.open(after_path) as after_image:
            after_rgb = np.asarray(
                after_image.convert("RGB"),
                dtype=np.float32,
            )

        if before_rgb.shape != after_rgb.shape:
            return {
                "available": False,
                "reason": (
                    "Visual-difference metrics require aligned images "
                    "with identical dimensions."
                ),
            }

        absolute_difference = np.abs(
            after_rgb - before_rgb
        )

        per_pixel_difference = np.mean(
            absolute_difference,
            axis=2,
        )

        mean_absolute_difference = float(
            np.mean(
                absolute_difference
            )
        )

        changed_gt_10 = float(
            np.mean(
                per_pixel_difference > 10.0
            )
            * 100.0
        )

        changed_gt_20 = float(
            np.mean(
                per_pixel_difference > 20.0
            )
            * 100.0
        )

        changed_gt_30 = float(
            np.mean(
                per_pixel_difference > 30.0
            )
            * 100.0
        )

        if (
            changed_gt_20 >= 25.0
            or mean_absolute_difference >= 20.0
        ):
            level = "significant"
        elif (
            changed_gt_20 >= 10.0
            or mean_absolute_difference >= 10.0
        ):
            level = "meaningful"
        elif (
            changed_gt_20 >= 3.0
            or mean_absolute_difference >= 5.0
        ):
            level = "limited"
        else:
            level = "low"

        return {
            "available": True,
            "level": level,
            "mean_absolute_rgb_difference": round(
                mean_absolute_difference,
                4,
            ),
            "pixels_difference_gt_10_pct": round(
                changed_gt_10,
                4,
            ),
            "pixels_difference_gt_20_pct": round(
                changed_gt_20,
                4,
            ),
            "pixels_difference_gt_30_pct": round(
                changed_gt_30,
                4,
            ),
            "note": (
                "RGB scene differences can include real surface change, "
                "seasonality, illumination, atmosphere, compositing, and "
                "small registration effects. They are not a semantic "
                "land-cover label by themselves."
            ),
        }

    @staticmethod
    def _build_combined_change_answer(
        visual_assessment: Dict[str, Any],
        changed_percentage: float,
        before_year: Optional[int],
        after_year: Optional[int],
        fallback_answer: str,
    ) -> str:
        if before_year is not None and after_year is not None:
            period = (
                f"between {before_year} and {after_year}"
            )
        else:
            period = (
                "between the before and after images"
            )

        visual_available = bool(
            visual_assessment.get("available")
        )

        if not visual_available:
            return fallback_answer

        level = str(
            visual_assessment.get(
                "level",
                "low",
            )
        )

        pct20 = float(
            visual_assessment.get(
                "pixels_difference_gt_20_pct",
                0.0,
            )
        )

        mean_diff = float(
            visual_assessment.get(
                "mean_absolute_rgb_difference",
                0.0,
            )
        )

        if level == "significant":
            visual_text = (
                f"Substantial visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels, with "
                f"a mean absolute RGB difference of {mean_diff:.1f}/255."
            )
        elif level == "meaningful":
            visual_text = (
                f"Meaningful visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels, with "
                f"a mean absolute RGB difference of {mean_diff:.1f}/255."
            )
        elif level == "limited":
            visual_text = (
                f"Limited visual differences are present {period}. "
                f"About {pct20:.1f}% of aligned pixels have an average "
                f"RGB difference greater than 20 intensity levels."
            )
        else:
            visual_text = (
                f"The aligned scenes show only small RGB differences "
                f"{period}."
            )

        if changed_percentage < 0.01:
            structural_text = (
                "ChangeFormerV6 detected no confident LEVIR-style "
                "structural/building change (0.00%)."
            )
        else:
            structural_text = (
                "ChangeFormerV6 detected LEVIR-style structural/building "
                f"change across approximately {changed_percentage:.2f}% "
                "of the analyzed area."
            )

        limitation_text = (
            "Because ChangeFormerV6 was trained on high-resolution "
            "LEVIR-CD building imagery, its structural result should not "
            "override scene-level differences in medium-resolution "
            "Sentinel/Landsat composites. RGB differences can also be "
            "caused by seasonality, illumination, atmosphere, compositing, "
            "or small registration effects, so they should not be treated "
            "as a semantic land-cover transition by themselves."
        )

        return (
            f"{visual_text} "
            f"{structural_text} "
            f"{limitation_text}"
        )

    # === END SATQUERY BI-TEMPORAL VISUAL CHANGE V1 ===


    # === SATQUERY EVIDENCE-GROUNDED BI-TEMPORAL V3 ===

    @staticmethod
    def _canonical_semantic_class_name(name: Any) -> Optional[str]:
        value = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "water": "water",
            "water_body": "water",
            "water_bodies": "water",
            "vegetation": "vegetation",
            "forest": "vegetation",
            "forests": "vegetation",
            "crop": "vegetation",
            "crops": "vegetation",
            "agricultural": "vegetation",
            "built_up": "built_up",
            "builtup": "built_up",
            "urban": "built_up",
            "building": "built_up",
            "buildings": "built_up",
            "road": "built_up",
            "roads": "built_up",
            "bare_land": "bare_land",
            "bareland": "bare_land",
            "barren": "bare_land",
            "soil": "bare_land",
            "unknown": "unknown",
            "ignore": "unknown",
            "background": "unknown",
        }
        return aliases.get(value)

    @classmethod
    def _semantic_percentages_from_mask(
        cls,
        mask: np.ndarray,
        class_map: Any,
    ) -> Dict[str, float]:
        array = np.asarray(mask)
        array = np.squeeze(array)
        if array.ndim != 2:
            raise ValueError(f"Semantic mask must be 2-D. Got {array.shape}.")

        id_to_name: Dict[int, str] = {}
        if isinstance(class_map, dict):
            for key, value in class_map.items():
                try:
                    class_id = int(key)
                    class_name = value
                except Exception:
                    try:
                        class_id = int(value)
                        class_name = key
                    except Exception:
                        continue

                canonical = cls._canonical_semantic_class_name(class_name)
                if canonical:
                    id_to_name[class_id] = canonical

        if not id_to_name:
            raise ValueError("Semantic model class map has no supported land-cover classes.")

        total = int(array.size)
        if total <= 0:
            raise ValueError("Semantic mask is empty.")

        counts: Dict[str, int] = {
            "water": 0,
            "vegetation": 0,
            "built_up": 0,
            "bare_land": 0,
            "unknown": 0,
        }

        mapped_pixels = 0
        for class_id, class_name in id_to_name.items():
            pixels = int(np.count_nonzero(array == class_id))
            mapped_pixels += pixels
            counts[class_name] += pixels

        # Any unexpected model IDs are explicitly treated as unknown rather
        # than silently disappearing from the percentage total.
        if mapped_pixels < total:
            counts["unknown"] += total - mapped_pixels

        percentages = {
            key: round((count / total) * 100.0, 4)
            for key, count in counts.items()
        }
        percentages["known_total"] = round(
            sum(percentages[key] for key in ("water", "vegetation", "built_up", "bare_land")),
            4,
        )
        return percentages

    @classmethod
    def _assess_bitemporal_semantic(
        cls,
        before_path: str,
        after_path: str,
    ) -> Dict[str, Any]:
        try:
            from ai.models import SemanticModel

            model = SemanticModel()
            if not model.is_loaded():
                return {
                    "available": False,
                    "reliable": False,
                    "reason": "Semantic model failed to load.",
                }

            before_mask = model.predict(before_path)
            after_mask = model.predict(after_path)
            class_map = model.get_class_map()

            before_pct = cls._semantic_percentages_from_mask(before_mask, class_map)
            after_pct = cls._semantic_percentages_from_mask(after_mask, class_map)

            land_classes = ("water", "vegetation", "built_up", "bare_land")
            deltas = {
                key: round(after_pct[key] - before_pct[key], 4)
                for key in land_classes
            }

            ranked = sorted(
                deltas.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            quality_gate = cls._evaluate_semantic_quality(before_pct, after_pct, deltas)
            reliable = quality_gate["passed"]

            return {
                "available": True,
                "reliable": reliable,
                "method": "semantic_land_cover_comparison",
                "before_percentages": before_pct,
                "after_percentages": after_pct,
                "delta_percentage_points": deltas,
                "largest_changes": [
                    {"class": key, "delta_percentage_points": value}
                    for key, value in ranked[:4]
                ],
                "quality_gate": {
                    **quality_gate,
                },
                "model": type(model).__name__,
                "note": (
                    "Land-cover percentages are model estimates. They are used as primary "
                    "temporal evidence only when the semantic quality gate passes."
                ),
            }
        except Exception as error:
            return {
                "available": False,
                "reliable": False,
                "reason": f"{type(error).__name__}: {error}",
            }

    @staticmethod
    def _assess_bitemporal_spectral(
        before_path: str,
        after_path: str,
    ) -> Dict[str, Any]:
        try:
            from ai.data import MultispectralLoader
            from ai.multispectral import MultispectralAnalyzer

            def load_data(path_value: str) -> np.ndarray:
                path = Path(path_value)
                if path.suffix.lower() == ".npy":
                    data = np.load(path, allow_pickle=False)
                else:
                    data = MultispectralLoader().load(str(path)).data

                data = np.asarray(data)
                if data.ndim != 3:
                    raise ValueError(f"Expected (bands,H,W), got {data.shape}.")
                if data.shape[0] <= 3:
                    raise ValueError(
                        "Only RGB bands are available; true NDVI/NDBI/NDWI require "
                        "multispectral bands."
                    )
                return data

            analyzer = MultispectralAnalyzer()
            before_summary = analyzer.summary(load_data(before_path))
            after_summary = analyzer.summary(load_data(after_path))

            metrics: Dict[str, Dict[str, float]] = {}
            for key in ("ndvi", "ndwi", "ndbi"):
                before_block = before_summary.get(key)
                after_block = after_summary.get(key)
                if not isinstance(before_block, dict) or not isinstance(after_block, dict):
                    continue
                before_mean = before_block.get("mean")
                after_mean = after_block.get("mean")
                if before_mean is None or after_mean is None:
                    continue
                before_value = float(before_mean)
                after_value = float(after_mean)
                metrics[key] = {
                    "before": round(before_value, 6),
                    "after": round(after_value, 6),
                    "delta": round(after_value - before_value, 6),
                }

            if not metrics:
                return {
                    "available": False,
                    "reason": "No validated spectral index metrics were produced.",
                }

            return {
                "available": True,
                "method": "multispectral_index_comparison",
                "metrics": metrics,
                "note": (
                    "Spectral conclusions depend on correct sensor-specific band mapping "
                    "and comparable preprocessing between dates."
                ),
            }
        except Exception as error:
            return {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }

    @staticmethod
    def _assess_bitemporal_visual_support(
        before_path: str,
        after_path: str,
        evidence_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from PIL import Image

            def read_rgb_and_validity(path_value: str):
                path = Path(path_value)
                if path.suffix.lower() in {".tif", ".tiff"}:
                    import rasterio
                    with rasterio.open(path) as dataset:
                        if dataset.count < 3:
                            raise ValueError("RGB support check requires at least three raster bands.")
                        data = dataset.read([1, 2, 3], masked=True).astype(np.float32)
                        rgb = np.moveaxis(data.filled(np.nan), 0, -1)
                        valid = dataset.dataset_mask() > 0
                        valid &= np.all(np.isfinite(rgb), axis=2)
                        return rgb, valid
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                return rgb, np.all(np.isfinite(rgb), axis=2)

            before_rgb, before_valid = read_rgb_and_validity(before_path)
            after_rgb, after_valid = read_rgb_and_validity(after_path)

            if before_rgb.shape != after_rgb.shape:
                return {
                    "available": False,
                    "reason": "RGB support check requires aligned images with identical dimensions.",
                }

            valid = before_valid & after_valid
            valid_pixel_count = int(valid.sum())
            if valid_pixel_count == 0:
                return {"available": False, "reason": "RGB support check has no valid overlapping pixels."}

            absolute_difference = np.abs(after_rgb - before_rgb)
            per_pixel_difference = np.mean(absolute_difference, axis=2)

            mean_difference = float(np.mean(absolute_difference[valid]))
            gt10 = float(np.mean(per_pixel_difference[valid] > 10.0) * 100.0)
            gt20 = float(np.mean(per_pixel_difference[valid] > 20.0) * 100.0)
            gt30 = float(np.mean(per_pixel_difference[valid] > 30.0) * 100.0)

            if gt20 >= 25.0 or mean_difference >= 20.0:
                level = "significant_visual_difference"
            elif gt20 >= 10.0 or mean_difference >= 10.0:
                level = "meaningful_visual_difference"
            elif gt20 >= 3.0 or mean_difference >= 5.0:
                level = "limited_visual_difference"
            else:
                level = "low_visual_difference"

            saved_path = None
            if evidence_path:
                diff = np.where(valid, np.clip(per_pixel_difference, 0, 255), 0).astype(np.uint8)
                Image.fromarray(diff, mode="L").save(evidence_path)
                saved_path = str(evidence_path)

            return {
                "available": True,
                "level": level,
                "mean_absolute_rgb_difference": round(mean_difference, 4),
                "pixels_difference_gt_10_pct": round(gt10, 4),
                "pixels_difference_gt_20_pct": round(gt20, 4),
                "pixels_difference_gt_30_pct": round(gt30, 4),
                "valid_pixel_count": valid_pixel_count,
                "valid_pixel_percentage": round(valid_pixel_count * 100.0 / valid.size, 4),
                "evidence_path": saved_path,
                "role": "supporting_non_semantic_evidence",
                "note": (
                    "RGB difference is supporting evidence only. It can reflect real surface "
                    "change, seasonality, atmosphere, illumination, compositing or small "
                    "registration errors and is not itself a land-cover label."
                ),
            }
        except Exception as error:
            return {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
            }

    @staticmethod
    def _format_land_cover_name(name: str) -> str:
        return {
            "water": "water",
            "vegetation": "vegetation",
            "built_up": "built-up land",
            "bare_land": "bare land",
        }.get(name, name.replace("_", " "))

    @classmethod
    def _build_evidence_grounded_change_answer(
        cls,
        before_year: Optional[int],
        after_year: Optional[int],
        requested_class: Optional[str],
        spectral: Dict[str, Any],
        semantic: Dict[str, Any],
        visual: Dict[str, Any],
        structural_change_percentage: Optional[float],
        structural_error: Optional[str] = None,
    ) -> str:
        period = (
            f"between {before_year} and {after_year}"
            if before_year is not None and after_year is not None
            else "between the before and after images"
        )

        parts = []
        spectral_available = bool(spectral.get("available"))
        semantic_available = bool(semantic.get("available"))
        semantic_reliable = bool(semantic.get("reliable"))

        if spectral_available:
            metric_parts = []
            for key, label in (
                ("ndvi", "NDVI"),
                ("ndbi", "NDBI"),
                ("ndwi", "NDWI"),
            ):
                metric = spectral.get("metrics", {}).get(key)
                if not metric:
                    continue
                delta = float(metric["delta"])
                direction = (
                    "increased" if delta > 0
                    else "decreased" if delta < 0
                    else "was unchanged"
                )
                if direction == "was unchanged":
                    metric_parts.append(f"{label} was essentially unchanged")
                else:
                    metric_parts.append(f"{label} {direction} by {abs(delta):.3f}")
            if metric_parts:
                parts.append(
                    f"Multispectral comparison {period} shows that "
                    + ", ".join(metric_parts)
                    + "."
                )

        if semantic_available and semantic_reliable:
            deltas = semantic.get("delta_percentage_points", {})
            focus = requested_class
            if focus == "building":
                focus = "built_up"

            selected = []
            if focus in {"water", "vegetation", "built_up", "bare_land"}:
                if focus in deltas:
                    selected = [(focus, float(deltas[focus]))]
            else:
                selected = sorted(
                    ((key, float(value)) for key, value in deltas.items()),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[:3]

            semantic_parts = []
            for key, delta in selected:
                if abs(delta) < 0.1:
                    continue
                direction = "increased" if delta > 0 else "decreased"
                semantic_parts.append(
                    f"{cls._format_land_cover_name(key)} {direction} "
                    f"by {abs(delta):.1f} percentage points"
                )

            if semantic_parts:
                parts.append(
                    "Semantic land-cover estimates indicate "
                    + ", ".join(semantic_parts)
                    + "."
                )
            else:
                parts.append(
                    "Semantic land-cover estimates do not show a large "
                    f"class-proportion shift {period}."
                )

        elif semantic_available and not semantic_reliable:
            gate = semantic.get("quality_gate", {})
            before_unknown = float(gate.get("before_unknown_pct", 0.0))
            after_unknown = float(gate.get("after_unknown_pct", 0.0))
            reasons = gate.get("reasons") or []
            reason_text = "; ".join(str(item) for item in reasons[:3])
            sentence = (
                "Semantic segmentation was not used as primary evidence because "
                "its quality gate failed. "
                f"Unknown-class coverage was {before_unknown:.1f}% in the before image "
                f"and {after_unknown:.1f}% in the after image."
            )
            if reason_text:
                sentence += f" Quality checks: {reason_text}."
            parts.append(sentence)

        if structural_change_percentage is not None:
            if structural_change_percentage < 0.01:
                parts.append(
                    "ChangeFormerV6 detected no confident LEVIR-style "
                    "structural/building change (0.00%)."
                )
            else:
                parts.append(
                    "ChangeFormerV6 detected LEVIR-style structural/building "
                    f"change across approximately {structural_change_percentage:.2f}% "
                    "of the analyzed area."
                )
        elif structural_error:
            parts.append(
                "Structural ChangeFormer evidence was unavailable for this run, so "
                "the result is based on the other available evidence sources."
            )

        if visual.get("available"):
            mean_diff = float(visual.get("mean_absolute_rgb_difference", 0.0))
            pct20 = float(visual.get("pixels_difference_gt_20_pct", 0.0))
            if pct20 >= 25.0 or mean_diff >= 20.0:
                parts.append(
                    f"Scene-level visual differences are present {period}. "
                    "As supporting evidence only, the aligned RGB scenes have a mean "
                    f"absolute difference of {mean_diff:.1f}/255, with {pct20:.1f}% "
                    "of pixels differing by more than 20 intensity levels."
                )
            else:
                parts.append(
                    "As supporting evidence only, the aligned RGB scenes show relatively "
                    f"limited pixel-level difference {period}."
                )

        if spectral_available or (semantic_available and semantic_reliable):
            parts.append(
                "Validated spectral or quality-gated semantic evidence is used for the "
                "main land-cover conclusion; raw RGB difference is not treated as proof "
                "of a semantic transition."
            )
        else:
            parts.append(
                "The available RGB-only pair does not support a definitive semantic "
                "land-cover transition conclusion. Use a validated multispectral pair "
                "or a sensor-calibrated semantic model for a stronger land-cover result."
            )

        return " ".join(parts)

    # === END SATQUERY EVIDENCE-GROUNDED BI-TEMPORAL V3 ===

    def _execute_change_detection(
        self,
        query: str,
        before_path: Optional[str],
        after_path: Optional[str],
        requested_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not before_path:
            return {
                "success": False,
                "intent": "change_detection",
                "message": "A before image is required.",
                "error": "Missing before_path",
            }

        if not after_path:
            return {
                "success": False,
                "intent": "change_detection",
                "message": "An after image is required.",
                "error": "Missing after_path",
            }

        self._check_file(before_path)
        self._check_file(after_path)

        # This legacy UI executor does not reproject or register inputs. Do
        # not let equal-sized but geographically shifted rasters reach pixel
        # comparison, semantic inference, or ChangeFormer.
        from ai.validation import InputValidator
        input_validation = InputValidator().validate_change_pair(before_path, after_path)
        if not input_validation.get("valid"):
            return {
                "success": False,
                "intent": "change_detection",
                "error": "SpatialInputValidationFailed",
                "message": "The before/after pair is not on a valid common comparison grid.",
                "input_validation": input_validation,
            }

        input_before_year = self._extract_year_from_path(before_path)
        input_after_year = self._extract_year_from_path(after_path)
        if input_before_year is not None and input_after_year is not None and input_before_year >= input_after_year:
            return {
                "success": False,
                "intent": "change_detection",
                "error": "InvalidTemporalOrder",
                "message": "Before imagery must precede after imagery when dated input names are supplied.",
                "input_validation": input_validation,
            }

        normalized_class = (
            requested_class.lower().strip()
            if requested_class
            else None
        )
        class_aliases = {
            "built-up": "built_up",
            "built up": "built_up",
            "urban": "built_up",
            "buildings": "building",
            "water body": "water",
            "water bodies": "water",
            "forest": "vegetation",
            "forests": "vegetation",
            "bare land": "bare_land",
            "barren": "bare_land",
        }
        if normalized_class:
            normalized_class = class_aliases.get(normalized_class, normalized_class)

        supported_targets = {
            None,
            "",
            "overall",
            "change",
            "building",
            "built_up",
            "water",
            "vegetation",
            "bare_land",
        }
        if normalized_class not in supported_targets:
            return {
                "success": False,
                "intent": "change_detection",
                "error": "UnsupportedSemanticChangeTarget",
                "message": f"Unsupported temporal target: {normalized_class}",
                "requested_class": normalized_class,
            }

        plan = self.plan(query)
        years = plan.get("years", [])
        before_year = years[0] if len(years) >= 1 else None
        after_year = years[1] if len(years) >= 2 else None

        evidence_dir = Path("outputs") / "evidence" / f"ana_{uuid4().hex}"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        rgb_difference_path = evidence_dir / "bitemporal_rgb_difference.png"

        spectral_assessment = self._assess_bitemporal_spectral(
            before_path=before_path,
            after_path=after_path,
        )
        semantic_assessment = self._assess_bitemporal_semantic(
            before_path=before_path,
            after_path=after_path,
        )
        visual_assessment = self._assess_bitemporal_visual_support(
            before_path=before_path,
            after_path=after_path,
            evidence_path=str(rgb_difference_path),
        )

        structural_change_percentage: Optional[float] = None
        structural_error: Optional[str] = None
        analysis: Dict[str, Any] = {}
        vqa_result: Dict[str, Any] = {}
        detector_metadata: Dict[str, Any] = {}
        visual_evidence: Dict[str, Any] = {}

        if visual_assessment.get("evidence_path"):
            visual_evidence["rgb_difference"] = visual_assessment["evidence_path"]

        should_run_structural = normalized_class in {
            None,
            "",
            "overall",
            "change",
            "building",
            "built_up",
        }

        if should_run_structural:
            try:
                from ai.change import ChangeFormerDetector, ChangeAnalyzer, ChangeVQA
                from ai.evidence import EvidenceOverlay

                change_mask_path = evidence_dir / "changeformer_change_mask.png"
                before_overlay_path = evidence_dir / "changeformer_before_overlay.png"
                after_overlay_path = evidence_dir / "changeformer_after_overlay.png"
                comparison_path = evidence_dir / "changeformer_comparison.png"

                detector = ChangeFormerDetector(tile_size=256, device="cpu")
                detection = detector.detect_with_result(
                    before_path=before_path,
                    after_path=after_path,
                    output_path=str(change_mask_path),
                )

                analyzer = ChangeAnalyzer()
                analysis = analyzer.analyze(detection.mask)
                structural_change_percentage = float(
                    analysis.get("changed_percentage", 0.0)
                )

                vqa = ChangeVQA()
                vqa_result = vqa.answer(
                    query=query,
                    analysis=analysis,
                    before_year=before_year,
                    after_year=after_year,
                )

                detector_metadata = self._make_json_safe(
                    detection.metadata if hasattr(detection, "metadata") else {}
                )

                overlay = EvidenceOverlay()
                overlay.create_binary_mask(
                    mask=detection.mask,
                    output_path=str(change_mask_path),
                )
                overlay.overlay_mask(
                    base_image_path=before_path,
                    mask=detection.mask,
                    output_path=str(before_overlay_path),
                )
                overlay.overlay_mask(
                    base_image_path=after_path,
                    mask=detection.mask,
                    output_path=str(after_overlay_path),
                )
                overlay.create_change_comparison(
                    before_image_path=before_path,
                    after_image_path=after_path,
                    mask=detection.mask,
                    output_path=str(comparison_path),
                )

                visual_evidence.update(
                    {
                        "change_mask": str(change_mask_path),
                        "before_overlay": str(before_overlay_path),
                        "after_overlay": str(after_overlay_path),
                        "comparison": str(comparison_path),
                    }
                )
            except Exception as error:
                structural_error = f"{type(error).__name__}: {error}"

        answer = self._build_evidence_grounded_change_answer(
            before_year=before_year,
            after_year=after_year,
            requested_class=normalized_class,
            spectral=spectral_assessment,
            semantic=semantic_assessment,
            visual=visual_assessment,
            structural_change_percentage=structural_change_percentage,
            structural_error=structural_error,
        )

        evidence_available = any(
            (
                spectral_assessment.get("available"),
                semantic_assessment.get("available"),
                visual_assessment.get("available"),
                structural_change_percentage is not None,
            )
        )

        if not evidence_available:
            return {
                "success": False,
                "intent": "change_detection",
                "message": "No temporal evidence source completed successfully.",
                "error": "NoTemporalEvidence",
                "details": {
                    "spectral": spectral_assessment,
                    "semantic": semantic_assessment,
                    "visual": visual_assessment,
                    "structural_error": structural_error,
                },
            }

        if structural_change_percentage is None:
            structural_level = "unavailable"
        elif structural_change_percentage < 0.01:
            structural_level = "very_low_change"
        elif structural_change_percentage < 1.0:
            structural_level = "minor_change"
        elif structural_change_percentage < 10.0:
            structural_level = "localized_change"
        elif structural_change_percentage < 30.0:
            structural_level = "significant_change"
        else:
            structural_level = "major_change"

        semantic_reliable = bool(
            semantic_assessment.get("available")
            and semantic_assessment.get("reliable")
        )

        if spectral_assessment.get("available"):
            primary_evidence = "spectral"
        elif semantic_reliable:
            primary_evidence = "semantic"
        elif normalized_class in {"building", "built_up"} and structural_change_percentage is not None:
            primary_evidence = "structural"
        else:
            primary_evidence = "qualified_supporting_evidence"

        return {
            "success": True,
            "intent": "change_detection",
            "task": "evidence_grounded_bitemporal_change",
            "model": "SatQuery Temporal Evidence Fusion",
            "models_used": [
                name
                for name, enabled in (
                    ("MultispectralAnalyzer", spectral_assessment.get("available")),
                    (semantic_assessment.get("model", "SemanticModel"), semantic_assessment.get("available")),
                    ("ChangeFormerV6", structural_change_percentage is not None),
                    ("RGB Support Check", visual_assessment.get("available")),
                )
                if enabled
            ],
            "before_image": str(before_path),
            "after_image": str(after_path),
            "input_validation": self._make_json_safe(input_validation),
            "requested_class": normalized_class or "overall",
            "answer": answer,
            "description": answer,
            "primary_evidence": primary_evidence,
            "spectral_change_assessment": self._make_json_safe(spectral_assessment),
            "semantic_change_assessment": self._make_json_safe(semantic_assessment),
            "visual_change_assessment": self._make_json_safe(visual_assessment),
            "structural_change_percentage": structural_change_percentage,
            "structural_analysis": self._make_json_safe(analysis),
            "change_vqa": self._make_json_safe(vqa_result),
            "detector_metadata": detector_metadata,
            "structural_error": structural_error,
            "visual_evidence": self._make_json_safe(visual_evidence),
            "execution_summary": {
                "before_year": before_year,
                "after_year": after_year,
                "temporal_order_evidence": "dated_input_filenames" if input_before_year is not None and input_after_year is not None else "query_or_unspecified",
                "spatial_alignment": "validated_common_grid",
                "primary_evidence": primary_evidence,
                "spectral_available": bool(spectral_assessment.get("available")),
                "semantic_available": bool(semantic_assessment.get("available")),
                "semantic_reliable": semantic_reliable,
                "semantic_quality_score": (
                    semantic_assessment.get("quality_gate", {}).get("score")
                ),
                "semantic_before_unknown_pct": (
                    semantic_assessment.get("quality_gate", {}).get("before_unknown_pct")
                ),
                "semantic_after_unknown_pct": (
                    semantic_assessment.get("quality_gate", {}).get("after_unknown_pct")
                ),
                "structural_available": structural_change_percentage is not None,
                "visual_support_available": bool(visual_assessment.get("available")),
                "structural_change_percentage": structural_change_percentage,
                "structural_change_level": structural_level,
                "rgb_mean_absolute_difference": visual_assessment.get(
                    "mean_absolute_rgb_difference"
                ),
                "rgb_pixels_difference_gt_20_pct": visual_assessment.get(
                    "pixels_difference_gt_20_pct"
                ),
                "evidence_saved": bool(visual_evidence),
            },
            "limitations": (
                "Semantic class proportions are model estimates and are promoted to primary "
                "evidence only when the semantic quality gate passes; multispectral indices are "
                "used only when suitable bands are available; ChangeFormerV6 is limited to "
                "LEVIR-style structural/building change; RGB differences are supporting "
                "non-semantic evidence only."
            ),
        }


    # ==========================================================
    # CHANGE RESULT SERIALIZATION
    # ==========================================================

    @staticmethod
    def _evaluate_semantic_quality(
        before_pct: Dict[str, float],
        after_pct: Dict[str, float],
        deltas: Dict[str, float],
    ) -> Dict[str, Any]:
        """Evaluate semantic evidence with explicit threshold semantics."""
        classes = ("water", "vegetation", "built_up", "bare_land")
        before_unknown = float(before_pct.get("unknown", 0.0))
        after_unknown = float(after_pct.get("unknown", 0.0))
        before_known = float(before_pct.get("known_total", 0.0))
        after_known = float(after_pct.get("known_total", 0.0))
        before_class = max(classes, key=lambda key: float(before_pct.get(key, 0.0)))
        after_class = max(classes, key=lambda key: float(after_pct.get(key, 0.0)))
        before_dominant = float(before_pct.get(before_class, 0.0))
        after_dominant = float(after_pct.get(after_class, 0.0))
        max_delta = max((abs(float(value)) for value in deltas.values()), default=0.0)
        reasons = []
        if before_unknown > 35.0:
            reasons.append(f"before-image unknown coverage is {before_unknown:.1f}% (limit 35.0%)")
        if after_unknown > 35.0:
            reasons.append(f"after-image unknown coverage is {after_unknown:.1f}% (limit 35.0%)")
        if before_known < 65.0:
            reasons.append(f"before-image known-class coverage is only {before_known:.1f}%")
        if after_known < 65.0:
            reasons.append(f"after-image known-class coverage is only {after_known:.1f}%")

        extreme = max(before_dominant, after_dominant) >= 98.0 and max_delta >= 50.0
        if extreme:
            reasons.append(
                "semantic output contains an extreme single-class prediction combined with an extreme temporal class shift"
            )

        score = 1.0 - min(0.65, max(before_unknown, after_unknown) / 100.0 * 0.75)
        if extreme:
            score -= 0.25

        return {
            "passed": not reasons,
            "score": round(max(0.0, min(1.0, score)), 4),
            "reasons": reasons,
            "thresholds": {
                "max_unknown_pct": 35.0,
                "min_known_coverage_pct": 65.0,
                "extreme_dominant_pct": 98.0,
                "extreme_transition_pp": 50.0,
                "semantics": "unknown coverage fails only above 35.0%; known coverage fails only below 65.0%",
            },
            "before_unknown_pct": round(before_unknown, 4),
            "after_unknown_pct": round(after_unknown, 4),
            "before_known_coverage_pct": round(before_known, 4),
            "after_known_coverage_pct": round(after_known, 4),
            "before_dominant_class": before_class,
            "before_dominant_pct": round(before_dominant, 4),
            "after_dominant_class": after_class,
            "after_dominant_pct": round(after_dominant, 4),
            "max_abs_class_delta_pp": round(max_delta, 4),
        }

    @staticmethod
    def _serialize_change_result(
        result: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            result,
            dict,
        ):

            return (
                SatQueryOrchestrator
                ._make_json_safe(
                    result
                )
            )

        return (
            SatQueryOrchestrator
            ._make_json_safe(
                vars(result)
                if hasattr(
                    result,
                    "__dict__",
                )
                else result
            )
        )

    # ==========================================================
    # MULTISPECTRAL DESCRIPTION
    # ==========================================================

    @staticmethod
    def _describe_multispectral(
        result: Dict[str, Any],
    ) -> str:

        parts = []

        if "ndvi" in result:

            mean = result[
                "ndvi"
            ].get(
                "mean"
            )

            if mean is not None:

                parts.append(
                    f"NDVI mean: {mean:.3f}"
                )

        if "ndwi" in result:

            mean = result[
                "ndwi"
            ].get(
                "mean"
            )

            if mean is not None:

                parts.append(
                    f"NDWI mean: {mean:.3f}"
                )

        if "ndbi" in result:

            mean = result[
                "ndbi"
            ].get(
                "mean"
            )

            if mean is not None:

                parts.append(
                    f"NDBI mean: {mean:.3f}"
                )

        if not parts:

            return (
                "Multispectral analysis "
                "completed successfully."
            )

        return (
            "Multispectral analysis completed. "
            + "; ".join(parts)
            + "."
        )

    # ==========================================================
    # SAR DESCRIPTION
    # ==========================================================

    @staticmethod
    def _describe_sar(
        result: Dict[str, Any],
    ) -> str:

        vv = result.get(
            "vv",
            {},
        )

        vh = result.get(
            "vh",
            {},
        )

        ratio = result.get(
            "vv_vh_ratio",
            {},
        )

        return (
            "SAR analysis completed. "
            f"VV mean: "
            f"{vv.get('mean', 0):.3f}; "
            f"VH mean: "
            f"{vh.get('mean', 0):.3f}; "
            f"VV/VH median: "
            f"{ratio.get('median', 0):.3f}."
        )

    # ==========================================================
    # FILE VALIDATION
    # ==========================================================

    @staticmethod
    def _check_file(
        path: str,
    ) -> None:

        file_path = Path(
            path
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        if not file_path.is_file():

            raise ValueError(
                f"Input path is not a file: {path}"
            )

    # ==========================================================
    # EXPLAIN PLAN
    # ==========================================================

    def explain_plan(
        self,
        query: str,
    ) -> str:

        result = self.plan(
            query
        )

        if not result["success"]:

            return (
                "I could not determine which "
                "SatQuery analysis capability "
                "is required for this query."
            )

        tools = ", ".join(
            result[
                "required_tools"
            ]
        )

        return (
            f"Intent: "
            f"{result['intent']}. "
            f"Confidence: "
            f"{result['confidence']:.2f}. "
            f"Required tools: "
            f"{tools}."
        )


__all__ = [
    "SatQueryOrchestrator",
]


