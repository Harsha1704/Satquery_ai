from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ai.router import SatQueryOrchestrator
from ai.evidence import EvidenceVisualizer, EvidenceOverlay
from ai.multispectral import MultispectralAnalyzer

from .formatter import ResponseFormatter


class ResponseGenerator:
    """
    End-to-end SatQuery-AI response generator.

    Pipeline:

        Query
          ↓
        Orchestrator
          ↓
        AI Analysis
          ↓
        Evidence Generation
          ↓
        Response Formatter
          ↓
        Final Structured Response

    The generator is responsible for:
        - executing the orchestrator
        - preserving execution metadata
        - generating visual evidence
        - attaching evidence to execution/response
        - returning one consistent response structure
    """

    def __init__(
        self,
        orchestrator: Optional[SatQueryOrchestrator] = None,
        formatter: Optional[ResponseFormatter] = None,
        visualizer: Optional[EvidenceVisualizer] = None,
        overlay: Optional[EvidenceOverlay] = None,
    ):
        self.orchestrator = (
            orchestrator
            or SatQueryOrchestrator()
        )

        self.formatter = (
            formatter
            or ResponseFormatter()
        )

        self.visualizer = (
            visualizer
            or EvidenceVisualizer()
        )

        self.overlay = (
            overlay
            or EvidenceOverlay()
        )

        self.multispectral_analyzer = (
            MultispectralAnalyzer()
        )

    # ==============================================================
    # MAIN GENERATOR
    # ==============================================================

    def generate(
        self,
        query: str,
        image_path: Optional[str] = None,
        before_path: Optional[str] = None,
        after_path: Optional[str] = None,
        optical_path: Optional[str] = None,
        sar_path: Optional[str] = None,
        requested_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete SatQuery-AI request.

        Returns a structured dictionary containing:

            success
            query
            intent
            confidence
            matched_keywords
            required_tools
            execution
            response
        """

        # ----------------------------------------------------------
        # 1. EXECUTE ORCHESTRATOR
        # ----------------------------------------------------------

        try:
            orchestrator_result = self.orchestrator.execute(
                query=query,
                image_path=image_path,
                before_path=before_path,
                after_path=after_path,
                optical_path=optical_path,
                sar_path=sar_path,
                requested_class=requested_class,
            )

        except Exception as exc:
            return self._build_failure_response(
                query=query,
                error=type(exc).__name__,
                message=str(exc),
            )

        # ----------------------------------------------------------
        # 2. VALIDATE ORCHESTRATOR RESULT
        # ----------------------------------------------------------

        if not isinstance(
            orchestrator_result,
            dict,
        ):
            return self._build_failure_response(
                query=query,
                error="InvalidExecution",
                message=(
                    "Orchestrator returned "
                    "an invalid result."
                ),
            )

        # ----------------------------------------------------------
        # 3. EXTRACT EXECUTION RESULT
        # ----------------------------------------------------------

        execution = orchestrator_result.get(
            "execution",
            {},
        )

        if not isinstance(
            execution,
            dict,
        ):
            execution = {
                "success": False,
                "error": "InvalidExecution",
                "message": (
                    "Orchestrator execution "
                    "result is invalid."
                ),
            }

        # ----------------------------------------------------------
        # 4. COPY ROUTER METADATA
        # ----------------------------------------------------------

        self._copy_router_metadata(
            execution=execution,
            orchestrator_result=orchestrator_result,
        )

        # ----------------------------------------------------------
        # 5. GENERATE VISUAL EVIDENCE
        # ----------------------------------------------------------

        evidence = self._generate_evidence(
            execution=execution,
            image_path=image_path,
            before_path=before_path,
            after_path=after_path,
            optical_path=optical_path,
            sar_path=sar_path,
        )

        # Always keep evidence inside execution.
        execution["evidence"] = evidence

        # ----------------------------------------------------------
        # 6. CREATE STRUCTURED RESPONSE
        # ----------------------------------------------------------

        try:
            response = self.formatter.create_response(
                execution=execution,
                query=query,
            )

        except Exception as exc:
            response = {
                "success": False,
                "intent": execution.get("intent"),
                "answer": (
                    "SatQuery-AI completed the analysis "
                    "but could not format the response."
                ),
                "error": type(exc).__name__,
                "message": str(exc),
            }

        # ----------------------------------------------------------
        # 7. NORMALIZE RESPONSE
        # ----------------------------------------------------------

        if not isinstance(
            response,
            dict,
        ):
            response = {
                "success": bool(
                    execution.get(
                        "success",
                        False,
                    )
                ),
                "intent": execution.get(
                    "intent"
                ),
                "answer": str(response),
            }

        # ----------------------------------------------------------
        # 8. ATTACH EVIDENCE
        # ----------------------------------------------------------

        response["evidence"] = evidence

        # ----------------------------------------------------------
        # 9. PRESERVE IMPORTANT ANALYSIS FIELDS
        # ----------------------------------------------------------

        self._preserve_analysis_fields(
            response=response,
            execution=execution,
        )

        # ----------------------------------------------------------
        # 10. FINAL SUCCESS FLAG
        # ----------------------------------------------------------

        execution_success = bool(
            execution.get(
                "success",
                False,
            )
        )

        response_success = bool(
            response.get(
                "success",
                False,
            )
        )

        success = (
            execution_success
            and response_success
        )

        # ----------------------------------------------------------
        # 11. FINAL STRUCTURED RESULT
        # ----------------------------------------------------------

        return {
            "success": success,

            "query": query,

            "intent": orchestrator_result.get(
                "intent",
                execution.get("intent"),
            ),

            "confidence": orchestrator_result.get(
                "confidence",
                execution.get("confidence"),
            ),

            "matched_keywords": orchestrator_result.get(
                "matched_keywords",
                execution.get(
                    "matched_keywords",
                    [],
                ),
            ),

            "required_tools": orchestrator_result.get(
                "required_tools",
                execution.get(
                    "required_tools",
                    [],
                ),
            ),

            "execution": execution,

            "response": response,
        }

    # ==============================================================
    # COPY ROUTER METADATA
    # ==============================================================

    @staticmethod
    def _copy_router_metadata(
        execution: Dict[str, Any],
        orchestrator_result: Dict[str, Any],
    ) -> None:
        """
        Ensure router/planner metadata exists inside execution.
        """

        if "intent" not in execution:
            execution["intent"] = (
                orchestrator_result.get(
                    "intent"
                )
            )

        if "confidence" not in execution:
            execution["confidence"] = (
                orchestrator_result.get(
                    "confidence"
                )
            )

        if "matched_keywords" not in execution:
            execution["matched_keywords"] = (
                orchestrator_result.get(
                    "matched_keywords",
                    [],
                )
            )

        if "required_tools" not in execution:
            execution["required_tools"] = (
                orchestrator_result.get(
                    "required_tools",
                    [],
                )
            )

    # ==============================================================
    # PRESERVE ANALYSIS FIELDS
    # ==============================================================

    @staticmethod
    def _preserve_analysis_fields(
        response: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> None:
        """
        Preserve important execution information in the
        final response.

        This is especially important for change detection,
        where semantic_reasoning contains the detailed
        transition information.
        """

        fields = [
            "analysis",
            "description",
            "semantic_reasoning",
            "evidence",
            "before_image",
            "after_image",
            "before_shape",
            "after_shape",
        ]

        for field in fields:

            if field in execution:
                response[field] = execution.get(
                    field
                )

    # ==============================================================
    # FAILURE RESPONSE
    # ==============================================================

    @staticmethod
    def _build_failure_response(
        query: str,
        error: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Build a consistent failure response.
        """

        execution = {
            "success": False,
            "error": error,
            "message": message,
        }

        response = {
            "success": False,
            "intent": None,
            "answer": (
                "SatQuery-AI could not complete "
                "the requested analysis."
            ),
            "error": error,
            "message": message,
            "evidence": [],
        }

        return {
            "success": False,
            "query": query,
            "intent": None,
            "confidence": 0.0,
            "matched_keywords": [],
            "required_tools": [],
            "execution": execution,
            "response": response,
        }

    # ==============================================================
    # EVIDENCE GENERATION
    # ==============================================================

    def _generate_evidence(
        self,
        execution: Dict[str, Any],
        image_path: Optional[str] = None,
        before_path: Optional[str] = None,
        after_path: Optional[str] = None,
        optical_path: Optional[str] = None,
        sar_path: Optional[str] = None,
    ) -> list:
        """
        Generate visual evidence for the completed analysis.

        Evidence generation is intentionally isolated from the
        main analysis. If evidence generation fails, the main
        analysis result is still returned.
        """

        evidence = []

        # ----------------------------------------------------------
        # ANALYSIS FAILED
        # ----------------------------------------------------------

        if not execution.get(
            "success",
            False,
        ):
            return evidence

        intent = execution.get(
            "intent",
            "",
        )

        output_dir = (
            Path("data") / "evidence"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ==========================================================
        # MULTISPECTRAL ANALYSIS
        # ==========================================================

        if intent == "multispectral_analysis":

            self._generate_multispectral_evidence(
                evidence=evidence,
                output_dir=output_dir,
                image_path=image_path,
            )

        # ==========================================================
        # SEMANTIC ANALYSIS
        # ==========================================================

        elif intent == "semantic_analysis":

            self._append_source_evidence(
                evidence=evidence,
                evidence_type="source_image",
                path=image_path,
            )

        # ==========================================================
        # SAR ANALYSIS
        # ==========================================================

        elif intent == "sar_analysis":

            self._append_source_evidence(
                evidence=evidence,
                evidence_type="sar_source",
                path=image_path,
            )

        # ==========================================================
        # CHANGE DETECTION
        # ==========================================================

        elif intent == "change_detection":

            self._generate_change_evidence(
                evidence=evidence,
                before_path=before_path,
                after_path=after_path,
                execution=execution,
            )

        # ==========================================================
        # OPTICAL + SAR FUSION
        # ==========================================================

        elif intent == "optical_sar_fusion":

            self._append_source_evidence(
                evidence=evidence,
                evidence_type="optical_source",
                path=optical_path,
            )

            self._append_source_evidence(
                evidence=evidence,
                evidence_type="sar_source",
                path=sar_path,
            )

        return evidence

    # ==============================================================
    # MULTISPECTRAL EVIDENCE
    # ==============================================================

    def _generate_multispectral_evidence(
        self,
        evidence: list,
        output_dir: Path,
        image_path: Optional[str],
    ) -> None:
        """
        Generate NDVI, NDWI and NDBI maps.
        """

        if not image_path:
            return

        source = Path(
            image_path
        )

        if not source.exists():
            return

        try:

            data = np.load(
                source,
                allow_pickle=False,
            )

            indices = (
                self.multispectral_analyzer
                .calculate_indices(data)
            )

            # ------------------------------------------------------
            # NDVI
            # ------------------------------------------------------

            ndvi_path = (
                output_dir
                / "ndvi_map.png"
            )

            self.visualizer.create_ndvi_map(
                indices["ndvi"],
                str(ndvi_path),
            )

            evidence.append(
                {
                    "type": "ndvi_map",
                    "path": str(ndvi_path),
                }
            )

            # ------------------------------------------------------
            # NDWI
            # ------------------------------------------------------

            ndwi_path = (
                output_dir
                / "ndwi_map.png"
            )

            self.visualizer.create_ndwi_map(
                indices["ndwi"],
                str(ndwi_path),
            )

            evidence.append(
                {
                    "type": "ndwi_map",
                    "path": str(ndwi_path),
                }
            )

            # ------------------------------------------------------
            # NDBI
            # ------------------------------------------------------

            ndbi_path = (
                output_dir
                / "ndbi_map.png"
            )

            self.visualizer.create_ndbi_map(
                indices["ndbi"],
                str(ndbi_path),
            )

            evidence.append(
                {
                    "type": "ndbi_map",
                    "path": str(ndbi_path),
                }
            )

            # ------------------------------------------------------
            # SOURCE IMAGE
            # ------------------------------------------------------

            evidence.append(
                {
                    "type": "source_image",
                    "path": str(source),
                }
            )

        except Exception as exc:

            # Evidence failure must never break analysis.
            evidence.append(
                {
                    "type": "source_image",
                    "path": str(source),
                    "evidence_error": type(
                        exc
                    ).__name__,
                }
            )

    # ==============================================================
    # CHANGE DETECTION EVIDENCE
    # ==============================================================

    def _generate_change_evidence(
        self,
        evidence: list,
        before_path: Optional[str],
        after_path: Optional[str],
        execution: Dict[str, Any],
    ) -> None:
        """
        Generate evidence for bi-temporal change detection.

        The before/after images are always preserved.

        If the orchestrator has generated semantic reasoning,
        that information is also exposed through evidence
        metadata.
        """

        # ----------------------------------------------------------
        # BEFORE IMAGE
        # ----------------------------------------------------------

        if (
            before_path
            and Path(before_path).exists()
        ):

            evidence.append(
                {
                    "type": "before_image",
                    "path": str(
                        Path(before_path)
                    ),
                }
            )

        # ----------------------------------------------------------
        # AFTER IMAGE
        # ----------------------------------------------------------

        if (
            after_path
            and Path(after_path).exists()
        ):

            evidence.append(
                {
                    "type": "after_image",
                    "path": str(
                        Path(after_path)
                    ),
                }
            )

        # ----------------------------------------------------------
        # SEMANTIC REASONING EVIDENCE
        # ----------------------------------------------------------

        reasoning = execution.get(
            "semantic_reasoning"
        )

        if isinstance(
            reasoning,
            dict,
        ):

            change_detected = reasoning.get(
                "change_detected"
            )

            changed_pixels = reasoning.get(
                "changed_pixels"
            )

            changed_percentage = reasoning.get(
                "changed_percentage"
            )

            stable_pixels = reasoning.get(
                "stable_pixels"
            )

            stable_percentage = reasoning.get(
                "stable_percentage"
            )

            reasoning_evidence = {
                "type": "change_analysis",
            }

            if change_detected is not None:
                reasoning_evidence[
                    "change_detected"
                ] = change_detected

            if changed_pixels is not None:
                reasoning_evidence[
                    "changed_pixels"
                ] = changed_pixels

            if changed_percentage is not None:
                reasoning_evidence[
                    "changed_percentage"
                ] = changed_percentage

            if stable_pixels is not None:
                reasoning_evidence[
                    "stable_pixels"
                ] = stable_pixels

            if stable_percentage is not None:
                reasoning_evidence[
                    "stable_percentage"
                ] = stable_percentage

            # Preserve transition information.
            transitions = reasoning.get(
                "transitions"
            )

            if isinstance(
                transitions,
                dict,
            ):
                reasoning_evidence[
                    "transitions"
                ] = transitions

            # Preserve major changes.
            major_changes = reasoning.get(
                "major_changes"
            )

            if isinstance(
                major_changes,
                list,
            ):
                reasoning_evidence[
                    "major_changes"
                ] = major_changes

            evidence.append(
                reasoning_evidence
            )

    # ==============================================================
    # SOURCE EVIDENCE
    # ==============================================================

    @staticmethod
    def _append_source_evidence(
        evidence: list,
        evidence_type: str,
        path: Optional[str],
    ) -> None:
        """
        Append a source image/file to evidence when it exists.
        """

        if not path:
            return

        file_path = Path(
            path
        )

        if not file_path.exists():
            return

        evidence.append(
            {
                "type": evidence_type,
                "path": str(file_path),
            }
        )

    # ==============================================================
    # ANSWER CONVENIENCE METHOD
    # ==============================================================

    def answer(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Convenience method returning only the final
        natural-language answer.
        """

        result = self.generate(
            query=query,
            **kwargs,
        )

        response = result.get(
            "response",
            {},
        )

        if isinstance(
            response,
            dict,
        ):
            return str(
                response.get(
                    "answer",
                    "No answer was generated.",
                )
            )

        return str(response)

