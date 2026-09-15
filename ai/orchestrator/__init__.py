from .task import Task
from .result import TaskResult
from .planner import TaskPlanner
from .executor import TaskExecutor


class SatQueryOrchestrator:

    def __init__(self):

        self.planner = TaskPlanner()
        self.executor = TaskExecutor()

    def run(
        self,
        query: str,
        image_path: str = None,
        second_image_path: str = None
    ) -> TaskResult:

        task = self.planner.create_task(
            query
        )

        return self.executor.execute(
            task,
            image_path=image_path,
            second_image_path=second_image_path
        )


__all__ = [
    "Task",
    "TaskResult",
    "TaskPlanner",
    "TaskExecutor",
    "SatQueryOrchestrator"
]