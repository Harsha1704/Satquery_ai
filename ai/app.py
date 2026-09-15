from typing import Any, Dict, Optional

from ai.response import ResponseGenerator


class SatQueryAI:
    """
    High-level SatQuery-AI application interface.

    Pipeline:

        User Query
             ↓
        Response Generator
             ↓
        Orchestrator
             ↓
        AI Analysis Modules
             ↓
        Structured Result
             ↓
        Natural Language Response
    """

    def __init__(
        self,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        self.generator = (
            response_generator
            or ResponseGenerator()
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
        """

        if not query or not query.strip():
            return {
                "success": False,
                "query": query,
                "error": "Query cannot be empty.",
            }

        try:
            result = self.generator.generate(
                query=query,
                image_path=image_path,
                before_path=before_path,
                after_path=after_path,
                optical_path=optical_path,
                sar_path=sar_path,
                requested_class=requested_class,
            )

            return result

        except Exception as exc:
            return {
                "success": False,
                "query": query,
                "error": type(exc).__name__,
                "message": str(exc),
            }

    def answer(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
        Convenience method that returns only
        the final natural-language answer.
        """

        result = self.ask(
            query=query,
            **kwargs,
        )

        if not result.get("success", False):
            response = result.get("response")

            if isinstance(response, dict):
                return response.get(
                    "answer",
                    response.get(
                        "message",
                        "SatQuery-AI could not complete the request.",
                    ),
                )

            return result.get(
                "message",
                "SatQuery-AI could not complete the request.",
            )

        response = result.get("response", {})

        if isinstance(response, dict):
            return response.get(
                "answer",
                "Analysis completed successfully.",
            )

        return str(response)


def create_app() -> SatQueryAI:
    """
    Factory function for creating the SatQuery-AI application.
    """

    return SatQueryAI()


__all__ = [
    "SatQueryAI",
    "create_app",
]