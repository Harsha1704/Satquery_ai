from pathlib import Path
import re
from threading import RLock
from typing import Any, Dict, List, Optional

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
        }

        if normalized_class:
            normalized_class = class_aliases.get(
                normalized_class,
                normalized_class,
            )

        supported_targets = {
            None,
            "",
            "overall",
            "change",
            "building",
            "built_up",
        }

        if normalized_class not in supported_targets:
            return {
                "success": False,
                "intent": "change_detection",
                "error": "UnsupportedSemanticChangeTarget",
                "message": (
                    "The current learned ChangeFormerV6 model is "
                    "trained on LEVIR-CD for binary structural/building "
                    "change detection. "
                    f"'{normalized_class}' semantic change requires the "
                    "semantic change-VQA pipeline."
                ),
                "requested_class": normalized_class,
            }

        try:
            from ai.change import (
                ChangeFormerDetector,
                ChangeAnalyzer,
                ChangeVQA,
            )
            from ai.evidence import EvidenceOverlay

            evidence_dir = Path("outputs") / "evidence"
            evidence_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            evidence_path = (
                evidence_dir
                / "changeformer_change_mask.png"
            )

            detector = ChangeFormerDetector(
                tile_size=256,
                device="cpu",
            )

            detection = detector.detect_with_result(
                before_path=before_path,
                after_path=after_path,
                output_path=str(evidence_path),
            )

            analyzer = ChangeAnalyzer()
            analysis = analyzer.analyze(
                detection.mask
            )

            plan = self.plan(query)
            years = plan.get("years", [])

            before_year = (
                years[0]
                if len(years) >= 1
                else None
            )

            after_year = (
                years[1]
                if len(years) >= 2
                else None
            )

            vqa = ChangeVQA()
            vqa_result = vqa.answer(
                query=query,
                analysis=analysis,
                before_year=before_year,
                after_year=after_year,
            )

            description = vqa_result.get(
                "answer",
                analyzer.describe(analysis),
            )

            changed_percentage = float(
                analysis.get(
                    "changed_percentage",
                    0.0,
                )
            )

            if changed_percentage < 0.01:
                confidence_label = "very_low_change"
            elif changed_percentage < 1.0:
                confidence_label = "minor_change"
            elif changed_percentage < 10.0:
                confidence_label = "localized_change"
            elif changed_percentage < 30.0:
                confidence_label = "significant_change"
            else:
                confidence_label = "major_change"

            overlay = EvidenceOverlay()

            change_mask_path = (
                evidence_dir
                / "changeformer_change_mask.png"
            )
            before_overlay_path = (
                evidence_dir
                / "changeformer_before_overlay.png"
            )
            after_overlay_path = (
                evidence_dir
                / "changeformer_after_overlay.png"
            )
            comparison_path = (
                evidence_dir
                / "changeformer_comparison.png"
            )

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

            visual_evidence = {
                "change_mask": str(change_mask_path),
                "before_overlay": str(before_overlay_path),
                "after_overlay": str(after_overlay_path),
                "comparison": str(comparison_path),
            }

            return {
                "success": True,
                "intent": "change_detection",
                "model": "ChangeFormerV6",
                "training_dataset": "LEVIR-CD",
                "before_image": str(before_path),
                "after_image": str(after_path),
                "requested_class": (
                    normalized_class or "overall"
                ),
                "analysis": self._make_json_safe(
                    analysis
                ),
                "vqa": self._make_json_safe(
                    vqa_result
                ),
                "answer": vqa_result.get(
                    "answer",
                    description,
                ),
                "description": description,
                "detector_metadata": self._make_json_safe(
                    detection.metadata
                    if hasattr(detection, "metadata")
                    else {}
                ),
                "visual_evidence": self._make_json_safe(
                    visual_evidence
                ),
                "execution_summary": {
                    "model": "ChangeFormerV6",
                    "training_dataset": "LEVIR-CD",
                    "device": "cpu",
                    "tile_size": 256,
                    "before_year": before_year,
                    "after_year": after_year,
                    "changed_percentage": changed_percentage,
                    "change_level": confidence_label,
                    "question_type": vqa_result.get(
                        "question_type"
                    ),
                    "evidence_saved": True,
                },
                "limitations": (
                    "This LEVIR-CD checkpoint detects binary "
                    "structural/building changes. It does not yet "
                    "identify semantic transitions such as "
                    "vegetation-to-built-up or water-to-bare-land."
                ),
            }

        except Exception as error:
            return {
                "success": False,
                "intent": "change_detection",
                "message": (
                    "ChangeFormer bi-temporal "
                    "change detection failed."
                ),
                "error": type(error).__name__,
                "details": str(error),
            }

    # ==========================================================
    # CHANGE RESULT SERIALIZATION
    # ==========================================================

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
