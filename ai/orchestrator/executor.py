from pathlib import Path
from typing import Optional

from ai.spectral import NDVIEngine
from ai.evidence import EvidenceVisualizer
from ai.semantic import SemanticAnalyzer
from ai.change_detection import (
    ChangeDetectionEngine,
    ChangeStatistics
)
from geospatial.raster import RasterLoader

from .task import Task
from .result import TaskResult


class TaskExecutor:
    """
    Executes SatQuery specialist tasks.

    Currently supported:
        - NDVI / spectral analysis
        - Change detection
    """

    def __init__(self):

        self.raster_loader = RasterLoader()

        self.ndvi_engine = NDVIEngine()

        self.visualizer = EvidenceVisualizer()

        self.change_engine = ChangeDetectionEngine()

        self.change_statistics = ChangeStatistics()

        self.semantic_analyzer = SemanticAnalyzer()


    def execute(
        self,
        task: Task,
        image_path: Optional[str] = None,
        second_image_path: Optional[str] = None
    ) -> TaskResult:

        # -----------------------------------------------
        # Spectral Analysis
        # -----------------------------------------------

        if task.required_tools == ["ndvi_engine"]:

            return self._execute_ndvi(
                task,
                image_path
            )

        # -----------------------------------------------
        # Change Detection
        # -----------------------------------------------

        if task.required_tools == [
            "change_detection_engine"
        ]:

            return self._execute_change_detection(
                task,
                image_path,
                second_image_path
            )

                # -----------------------------------------------
                # Semantic Analysis
                # -----------------------------------------------

        if task.required_tools == [
            "semantic_analyzer"
        ]:

            return self._execute_semantic(
                task,
                image_path
            )

        # -----------------------------------------------
        # Unknown / unsupported task
        # -----------------------------------------------

        if not task.required_tools:

            return TaskResult(
                success=False,
                task=task.intent.value,
                message="No executable task was identified.",
                error="Unknown intent"
            )

        return TaskResult(
            success=False,
            task=task.intent.value,
            message=(
                f"Task identified as "
                f"{task.intent.value}, but its specialist "
                f"tool is not implemented yet."
            ),
            data={
                "required_tools": task.required_tools
            }
        )

    # ===================================================
    # NDVI
    # ===================================================

    def _execute_ndvi(
        self,
        task: Task,
        image_path: Optional[str]
    ) -> TaskResult:

        if not image_path:

            return TaskResult(
                success=False,
                task="spectral_analysis",
                message=(
                    "An image is required "
                    "for NDVI analysis."
                ),
                error="Missing image_path"
            )

        if not Path(image_path).exists():

            return TaskResult(
                success=False,
                task="spectral_analysis",
                message="The supplied image does not exist.",
                error=f"File not found: {image_path}"
            )

        try:

            red_band = self.raster_loader.read_band(
                image_path,
                3
            )

            nir_band = self.raster_loader.read_band(
                image_path,
                4
            )

            ndvi = self.ndvi_engine.calculate(
                red_band,
                nir_band
            )

            statistics = self.ndvi_engine.statistics(
                ndvi
            )

            interpretation = self.ndvi_engine.classify(
                statistics["mean"]
            )

            evidence_path = (
                "outputs/evidence/ndvi_analysis.png"
            )

            generated_evidence = (
                self.visualizer.create_ndvi_map(
                    ndvi,
                    evidence_path
                )
            )

            evidence = [
                f"Mean NDVI: {statistics['mean']:.4f}",
                f"Minimum NDVI: {statistics['min']:.4f}",
                f"Maximum NDVI: {statistics['max']:.4f}",
                (
                    "Vegetation coverage: "
                    f"{statistics['vegetation_percentage']:.2f}%"
                ),
                f"Visual evidence: {generated_evidence}"
            ]

            return TaskResult(
                success=True,
                task="spectral_analysis",
                message=(
                    "Vegetation analysis completed successfully. "
                    f"The image shows "
                    f"{interpretation.lower()}."
                ),
                data={
                    "index": "NDVI",
                    "statistics": statistics,
                    "interpretation": interpretation,
                    "evidence_image": generated_evidence
                },
                evidence=evidence
            )

        except Exception as error:

            return TaskResult(
                success=False,
                task="spectral_analysis",
                message="NDVI analysis failed.",
                error=str(error)
            )

    # ===================================================
    # CHANGE DETECTION
    # ===================================================

    def _execute_change_detection(
        self,
        task: Task,
        before_image_path: Optional[str],
        after_image_path: Optional[str]
    ) -> TaskResult:

        # -----------------------------------------------
        # Validate first image
        # -----------------------------------------------

        if not before_image_path:

            return TaskResult(
                success=False,
                task="change_detection",
                message=(
                    "A before image is required "
                    "for change detection."
                ),
                error="Missing before image"
            )

        # -----------------------------------------------
        # Validate second image
        # -----------------------------------------------

        if not after_image_path:

            return TaskResult(
                success=False,
                task="change_detection",
                message=(
                    "An after image is required "
                    "for change detection."
                ),
                error="Missing after image"
            )

        # -----------------------------------------------
        # Validate files
        # -----------------------------------------------

        if not Path(before_image_path).exists():

            return TaskResult(
                success=False,
                task="change_detection",
                message="Before image does not exist.",
                error=(
                    f"File not found: "
                    f"{before_image_path}"
                )
            )

        if not Path(after_image_path).exists():

            return TaskResult(
                success=False,
                task="change_detection",
                message="After image does not exist.",
                error=(
                    f"File not found: "
                    f"{after_image_path}"
                )
            )

        try:

            # -------------------------------------------
            # Load images
            # -------------------------------------------

            before = self.raster_loader.read_band(
                before_image_path,
                1
            )

            after = self.raster_loader.read_band(
                after_image_path,
                1
            )

            # -------------------------------------------
            # Detect changes
            # -------------------------------------------

            result = self.change_engine.detect(
                before,
                after,
                threshold=0.20
            )

            difference = result["difference"]

            change_mask = result["change_mask"]

            # -------------------------------------------
            # Calculate statistics
            # -------------------------------------------

            statistics = (
                self.change_statistics.calculate(
                    difference,
                    change_mask
                )
            )

            classification = (
                self.change_statistics.classify(
                    statistics["change_percentage"]
                )
            )

            # -------------------------------------------
            # Generate visual evidence
            # -------------------------------------------

            evidence_path = (
                "outputs/evidence/change_map.png"
            )

            generated_evidence = (
                self.visualizer.create_grayscale_map(
                    change_mask,
                    evidence_path
                )
            )

            # -------------------------------------------
            # Evidence
            # -------------------------------------------

            evidence = [
                (
                    "Changed pixels: "
                    f"{statistics['changed_pixels']}"
                ),
                (
                    "Change percentage: "
                    f"{statistics['change_percentage']:.2f}%"
                ),
                (
                    "Mean difference: "
                    f"{statistics['mean_difference']:.4f}"
                ),
                (
                    "Maximum difference: "
                    f"{statistics['max_difference']:.4f}"
                ),
                f"Classification: {classification}",
                f"Visual evidence: {generated_evidence}"
            ]

            # -------------------------------------------
            # Final result
            # -------------------------------------------

            return TaskResult(
                success=True,
                task="change_detection",
                message=(
                    "Change detection completed successfully. "
                    f"The analysis indicates "
                    f"{classification.lower()}."
                ),
                data={
                    "analysis": "change_detection",
                    "statistics": statistics,
                    "classification": classification,
                    "threshold": result["threshold"],
                    "evidence_image": generated_evidence
                },
                evidence=evidence
            )

        except Exception as error:

            return TaskResult(
                success=False,
                task="change_detection",
                message="Change detection failed.",
                error=str(error)
            )

    # ===================================================
    # SEMANTIC ANALYSIS
    # ===================================================

    def _execute_semantic(
        self,
        task: Task,
        image_path: Optional[str]
    ) -> TaskResult:

        if not image_path:

            return TaskResult(
                success=False,
                task="semantic_analysis",
                message=(
                    "An image is required "
                    "for semantic analysis."
                ),
                error="Missing image_path"
            )

        if not Path(image_path).exists():

            return TaskResult(
                success=False,
                task="semantic_analysis",
                message="The supplied image does not exist.",
                error=f"File not found: {image_path}"
            )

        try:

            # ------------------------------------------------
            # IMPORTANT
            # ------------------------------------------------
            # At this stage we expect a prediction mask to
            # be supplied through task.parameters.
            #
            # A real trained segmentation model will later
            # generate this mask from the satellite image.
            # ------------------------------------------------

            prediction_mask = task.parameters.get(
                "prediction_mask"
            )

            if prediction_mask is None:

                return TaskResult(
                    success=False,
                    task="semantic_analysis",
                    message=(
                        "Semantic analysis requires a "
                        "prediction mask. A trained semantic "
                        "model has not been connected yet."
                    ),
                    error="Missing prediction_mask"
                )

            class_map = {
                0: "water",
                1: "vegetation",
                2: "built_up",
                3: "bare_land"
            }

            # ------------------------------------------------
            # Analyze semantic mask
            # ------------------------------------------------

            analysis = self.semantic_analyzer.analyze(
                prediction_mask,
                class_map
            )

            description = (
                self.semantic_analyzer.describe(
                    analysis
                )
            )

            # ------------------------------------------------
            # Requested semantic class
            # ------------------------------------------------

            requested_class = task.parameters.get(
                "semantic_class"
            )

            evidence = []

            if requested_class:

                class_result = analysis[
                    "classes"
                ].get(
                    requested_class
                )

                if class_result:

                    evidence.append(
                        f"Requested class: "
                        f"{requested_class}"
                    )

                    evidence.append(
                        f"Pixels: "
                        f"{class_result['pixels']}"
                    )

                    evidence.append(
                        f"Coverage: "
                        f"{class_result['percentage']:.2f}%"
                    )

            evidence.append(
                f"Dominant class: "
                f"{analysis['dominant_class']}"
            )

            evidence.append(
                f"Dominant coverage: "
                f"{analysis['dominant_percentage']:.2f}%"
            )

            # ------------------------------------------------
            # Final result
            # ------------------------------------------------

            return TaskResult(
                success=True,
                task="semantic_analysis",
                message=description,
                data={
                    "analysis": "semantic_analysis",
                    "requested_class": requested_class,
                    "statistics": analysis,
                    "description": description
                },
                evidence=evidence
            )

        except Exception as error:

            return TaskResult(
                success=False,
                task="semantic_analysis",
                message="Semantic analysis failed.",
                error=str(error)
            )