from typing import Any, Dict


class ResponseFormatter:
    """
    Formats SatQuery-AI execution results into
    clean user-facing responses.
    """

    def format(
        self,
        execution: Dict[str, Any],
    ) -> str:
        """
        Return only the human-readable answer.
        """

        if not execution:
            return "No analysis result was returned."

        if not execution.get("success", False):
            message = execution.get(
                "message",
                "The requested analysis failed.",
            )
            return f"Analysis failed: {message}"

        intent = execution.get("intent", "")
        description = execution.get("description", "")

        # ----------------------------------------------------------
        # MULTISPECTRAL ANALYSIS
        # ----------------------------------------------------------

        if intent == "multispectral_analysis":

            analysis = execution.get("analysis", {})

            if not isinstance(analysis, dict):
                analysis = {}

            parts = []

            for index_name in (
                "ndvi",
                "ndwi",
                "ndbi",
            ):
                index = analysis.get(index_name, {})

                if not isinstance(index, dict):
                    continue

                mean = index.get("mean")

                if mean is not None:
                    parts.append(
                        f"{index_name.upper()} mean: "
                        f"{float(mean):.3f}"
                    )

            if parts:
                return (
                    "Multispectral analysis completed. "
                    + "; ".join(parts)
                    + "."
                )

            return "Multispectral analysis completed."

        # ----------------------------------------------------------
        # SEMANTIC ANALYSIS
        # ----------------------------------------------------------

        if intent == "semantic_analysis":

            if description:
                return (
                    "Semantic analysis completed. "
                    + str(description)
                )

            return "Semantic analysis completed."

        # ----------------------------------------------------------
        # SAR ANALYSIS
        # ----------------------------------------------------------

        if intent == "sar_analysis":

            if description:
                return str(description)

            return "SAR analysis completed."

        # ----------------------------------------------------------
        # OPTICAL + SAR FUSION
        # ----------------------------------------------------------

        if intent == "optical_sar_fusion":

            if description:
                return str(description)

            return "Optical and SAR fusion completed."

        # ----------------------------------------------------------
        # CHANGE DETECTION
        # ----------------------------------------------------------

        if intent == "change_detection":

            if description:
                return (
                    "Change detection completed. "
                    + str(description)
                )

            return "Change detection completed."

        # ----------------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------------

        if description:
            return str(description)

        return "SatQuery-AI analysis completed successfully."

    def create_response(
        self,
        execution: Dict[str, Any],
        query: str = "",
        confidence: Any = None,
        matched_keywords: Any = None,
        required_tools: Any = None,
    ) -> Dict[str, Any]:
        """
        Create the final structured response object.

        This is the method used by ResponseGenerator.
        """

        answer = self.format(execution)

        return {
            "success": bool(
                execution.get(
                    "success",
                    False,
                )
            ),
            "intent": execution.get("intent"),
            "answer": answer,
            "confidence": (
                confidence
                if confidence is not None
                else execution.get("confidence")
            ),
            "matched_keywords": (
                matched_keywords
                if matched_keywords is not None
                else execution.get(
                    "matched_keywords",
                    [],
                )
            ),
            "required_tools": (
                required_tools
                if required_tools is not None
                else execution.get(
                    "required_tools",
                    [],
                )
            ),
            "analysis": execution.get("analysis"),
            "description": execution.get("description"),
            "semantic_reasoning": execution.get(
                "semantic_reasoning"
            ),
            "evidence": execution.get(
                "evidence",
                [],
            ),
            "query": query,
        }