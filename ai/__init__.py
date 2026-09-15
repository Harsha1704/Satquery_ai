from typing import Any, Dict, Optional

from .router import SatQueryOrchestrator
from .response import ResponseGenerator


class SatQueryAI:
    """
    Main public interface for SatQuery-AI.

    Pipeline:

        User Query
            ↓
        Query Planner
            ↓
        SatQuery Orchestrator
            ↓
        AI Analysis Modules
            ↓
        Response Generator
            ↓
        Final Structured Response
    """

    def __init__(
        self,
        orchestrator: Optional[SatQueryOrchestrator] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.orchestrator = (
            orchestrator
            or SatQueryOrchestrator()
        )

        self.response_generator = (
            response_generator
            or ResponseGenerator(
                orchestrator=self.orchestrator
            )
        )

    def ask(
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

        Parameters
        ----------
        query:
            Natural-language remote-sensing query.

        image_path:
            Single image for semantic,
            multispectral, or SAR analysis.

        before_path:
            Earlier image for change detection.

        after_path:
            Later image for change detection.

        optical_path:
            Optical image for fusion.

        sar_path:
            SAR image for fusion or SAR analysis.

        requested_class:
            Optional semantic class such as:
            water, vegetation, built_up, bare_land.
        """

        # ----------------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------------

        if not query or not query.strip():
            return {
                "success": False,
                "query": query,
                "execution": {
                    "success": False,
                    "message": "Query cannot be empty.",
                },
                "response": {
                    "success": False,
                    "answer": (
                        "Please provide a SatQuery "
                        "remote-sensing question."
                    ),
                },
            }

        # ----------------------------------------------------------
        # EXECUTE RESPONSE GENERATOR
        # ----------------------------------------------------------

        try:

            result = self.response_generator.generate(
                query=query,
                image_path=image_path,
                before_path=before_path,
                after_path=after_path,
                optical_path=optical_path,
                sar_path=sar_path,
                requested_class=requested_class,
            )

        except Exception as exc:

            return {
                "success": False,
                "query": query,
                "execution": {
                    "success": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                "response": {
                    "success": False,
                    "answer": (
                        "SatQuery-AI could not complete "
                        "the requested analysis."
                    ),
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            }

        # ----------------------------------------------------------
        # NORMALIZE RESPONSE
        # ----------------------------------------------------------

        if not isinstance(result, dict):

            return {
                "success": False,
                "query": query,
                "execution": {
                    "success": False,
                    "error": "InvalidResponse",
                    "message": (
                        "ResponseGenerator returned an "
                        "invalid response type."
                    ),
                },
                "response": {
                    "success": False,
                    "answer": str(result),
                    "error": "InvalidResponse",
                },
            }

        # ----------------------------------------------------------
        # GET EXECUTION RESULT
        # ----------------------------------------------------------

        execution = result.get(
            "execution",
            {},
        )

        if not isinstance(execution, dict):
            execution = {
                "success": False,
                "error": "InvalidExecution",
                "message": (
                    "ResponseGenerator returned "
                    "an invalid execution object."
                ),
            }

        # ----------------------------------------------------------
        # GET RESPONSE RESULT
        # ----------------------------------------------------------

        response = result.get(
            "response",
            {},
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        # ResponseGenerator may return a string response.
        #
        # Convert it into the structured response format
        # expected by SatQueryAI.
        # ----------------------------------------------------------

        if isinstance(response, str):

            response = {
                "success": execution.get(
                    "success",
                    False,
                ),
                "intent": execution.get(
                    "intent"
                ),
                "answer": response,
                "confidence": result.get(
                    "confidence"
                ),
                "matched_keywords": result.get(
                    "matched_keywords",
                    [],
                ),
                "required_tools": result.get(
                    "required_tools",
                    [],
                ),
                "analysis": execution.get(
                    "analysis"
                ),
                "description": execution.get(
                    "description"
                ),
                "semantic_reasoning": execution.get(
                    "semantic_reasoning"
                ),
            }

        elif not isinstance(response, dict):

            response = {
                "success": False,
                "intent": execution.get(
                    "intent"
                ),
                "answer": (
                    "SatQuery-AI returned an "
                    "invalid response."
                ),
                "error": "InvalidResponse",
            }

        # ----------------------------------------------------------
        # SUCCESS FLAGS
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

        # ----------------------------------------------------------
        # COMPLETE PIPELINE SUCCESS
        # ----------------------------------------------------------

        success = (
            execution_success
            and response_success
        )

        # ----------------------------------------------------------
        # FINAL STRUCTURED RESULT
        # ----------------------------------------------------------

        return {
            "success": success,
            "query": query,
            "execution": execution,
            "response": response,
        }


__all__ = [
    "SatQueryAI",
]