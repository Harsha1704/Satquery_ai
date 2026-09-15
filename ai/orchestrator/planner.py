from ai.query_router import QueryRouter, Intent

from .task import Task


class TaskPlanner:
    """
    Converts a natural-language query into an executable task.
    """

    def __init__(self):
        self.router = QueryRouter()

    def create_task(self, query: str) -> Task:

        intent = self.router.route(query)

        if intent == Intent.SPECTRAL_ANALYSIS:

            return Task(
                query=query,
                intent=intent,
                required_tools=["ndvi_engine"],
                parameters={
                    "analysis": "ndvi"
                }
            )

        if intent == Intent.SEMANTIC_ANALYSIS:
            return Task(
                query=query,
                intent=intent,
                required_tools=["semantic_analyzer"],
                parameters={
                    "analysis": "semantic_analysis"
                }
            )
        
        if intent == Intent.CHANGE_DETECTION:

            return Task(
                query=query,
                intent=intent,
                required_tools=["change_detection_engine"],
                parameters={
                    "analysis": "change_detection"
                }
            )

        if intent == Intent.OBJECT_DETECTION:

            return Task(
                query=query,
                intent=intent,
                required_tools=["object_detection_model"],
                parameters={
                    "analysis": "object_detection"
                }
            )

        if intent == Intent.SEGMENTATION:

            return Task(
                query=query,
                intent=intent,
                required_tools=["segmentation_model"],
                parameters={
                    "analysis": "segmentation"
                }
            )

        if intent == Intent.CLASSIFICATION:

            return Task(
                query=query,
                intent=intent,
                required_tools=["classification_model"],
                parameters={
                    "analysis": "classification"
                }
            )

        if intent == Intent.OPTICAL_SAR_ANALYSIS:

            return Task(
                query=query,
                intent=intent,
                required_tools=["optical_sar_engine"],
                parameters={
                    "analysis": "optical_sar"
                }
            )

        if intent == Intent.VQA:

            return Task(
                query=query,
                intent=intent,
                required_tools=["vqa_model"],
                parameters={
                    "analysis": "visual_question_answering"
                }
            )

        return Task(
            query=query,
            intent=Intent.UNKNOWN,
            required_tools=[],
            parameters={}
        )