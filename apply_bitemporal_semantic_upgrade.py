from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
if (ROOT / "ai" / "router" / "orchestrator.py").exists():
    PROJECT_ROOT = ROOT
else:
    PROJECT_ROOT = Path.cwd()

ORCHESTRATOR = PROJECT_ROOT / "ai" / "router" / "orchestrator.py"
BACKUP = ORCHESTRATOR.with_name("orchestrator.py.before_semantic_bitemporal_upgrade")

HELPER_MARKER = "# === SATQUERY EVIDENCE-GROUNDED BI-TEMPORAL V2 ==="

HELPERS = r'''
    # === SATQUERY EVIDENCE-GROUNDED BI-TEMPORAL V2 ===

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
            "built_up": "built_up",
            "builtup": "built_up",
            "urban": "built_up",
            "building": "built_up",
            "buildings": "built_up",
            "bare_land": "bare_land",
            "bareland": "bare_land",
            "barren": "bare_land",
            "soil": "bare_land",
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

        result: Dict[str, float] = {
            "water": 0.0,
            "vegetation": 0.0,
            "built_up": 0.0,
            "bare_land": 0.0,
        }

        for class_id, class_name in id_to_name.items():
            pixels = int(np.count_nonzero(array == class_id))
            result[class_name] += (pixels / total) * 100.0

        return {key: round(value, 4) for key, value in result.items()}

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
                    "reason": "Semantic model failed to load.",
                }

            before_mask = model.predict(before_path)
            after_mask = model.predict(after_path)
            class_map = model.get_class_map()

            before_pct = cls._semantic_percentages_from_mask(before_mask, class_map)
            after_pct = cls._semantic_percentages_from_mask(after_mask, class_map)

            deltas = {
                key: round(after_pct[key] - before_pct[key], 4)
                for key in before_pct
            }

            ranked = sorted(
                deltas.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            return {
                "available": True,
                "method": "semantic_land_cover_comparison",
                "before_percentages": before_pct,
                "after_percentages": after_pct,
                "delta_percentage_points": deltas,
                "largest_changes": [
                    {"class": key, "delta_percentage_points": value}
                    for key, value in ranked[:4]
                ],
                "model": type(model).__name__,
                "note": (
                    "Land-cover percentages are model estimates and should be "
                    "validated for the target sensor and region before consequential use."
                ),
            }
        except Exception as error:
            return {
                "available": False,
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

            with Image.open(before_path) as image:
                before_rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            with Image.open(after_path) as image:
                after_rgb = np.asarray(image.convert("RGB"), dtype=np.float32)

            if before_rgb.shape != after_rgb.shape:
                return {
                    "available": False,
                    "reason": "RGB support check requires aligned images with identical dimensions.",
                }

            absolute_difference = np.abs(after_rgb - before_rgb)
            per_pixel_difference = np.mean(absolute_difference, axis=2)

            mean_difference = float(np.mean(absolute_difference))
            gt10 = float(np.mean(per_pixel_difference > 10.0) * 100.0)
            gt20 = float(np.mean(per_pixel_difference > 20.0) * 100.0)
            gt30 = float(np.mean(per_pixel_difference > 30.0) * 100.0)

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
                diff = np.clip(per_pixel_difference, 0, 255).astype(np.uint8)
                Image.fromarray(diff, mode="L").save(evidence_path)
                saved_path = str(evidence_path)

            return {
                "available": True,
                "level": level,
                "mean_absolute_rgb_difference": round(mean_difference, 4),
                "pixels_difference_gt_10_pct": round(gt10, 4),
                "pixels_difference_gt_20_pct": round(gt20, 4),
                "pixels_difference_gt_30_pct": round(gt30, 4),
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

        if spectral.get("available"):
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
                direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
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

        if semantic.get("available"):
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
                    f"{cls._format_land_cover_name(key)} {direction} by {abs(delta):.1f} percentage points"
                )

            if semantic_parts:
                parts.append(
                    "Semantic land-cover estimates indicate "
                    + ", ".join(semantic_parts)
                    + "."
                )
            else:
                parts.append(
                    "Semantic land-cover estimates do not show a large class-proportion shift "
                    f"{period}."
                )

        if structural_change_percentage is not None:
            if structural_change_percentage < 0.01:
                parts.append(
                    "ChangeFormerV6 detected no confident LEVIR-style structural/building "
                    "change (0.00%)."
                )
            else:
                parts.append(
                    "ChangeFormerV6 detected LEVIR-style structural/building change across "
                    f"approximately {structural_change_percentage:.2f}% of the analyzed area."
                )
        elif structural_error:
            parts.append(
                "Structural ChangeFormer evidence was unavailable for this run, so the result "
                "is based on the other available evidence sources."
            )

        if visual.get("available"):
            parts.append(
                "As supporting evidence only, the aligned RGB scenes have a mean absolute "
                f"difference of {float(visual.get('mean_absolute_rgb_difference', 0.0)):.1f}/255, "
                f"with {float(visual.get('pixels_difference_gt_20_pct', 0.0)):.1f}% of pixels "
                "differing by more than 20 intensity levels."
            )

        if semantic.get("available") or spectral.get("available"):
            parts.append(
                "The semantic/spectral evidence is used for the main land-cover conclusion; "
                "raw RGB difference is not treated as proof of a semantic transition."
            )
        else:
            parts.append(
                "No semantic or validated multispectral evidence was available, so RGB "
                "differences should be interpreted cautiously and not as a specific land-cover transition."
            )

        return " ".join(parts)

    # === END SATQUERY EVIDENCE-GROUNDED BI-TEMPORAL V2 ===
'''

NEW_METHOD = r'''    def _execute_change_detection(
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

        evidence_dir = Path("outputs") / "evidence"
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

        primary_evidence = (
            "spectral"
            if spectral_assessment.get("available")
            else "semantic"
            if semantic_assessment.get("available")
            else "structural"
            if structural_change_percentage is not None
            else "visual_support_only"
        )

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
                "primary_evidence": primary_evidence,
                "spectral_available": bool(spectral_assessment.get("available")),
                "semantic_available": bool(semantic_assessment.get("available")),
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
                "Semantic class proportions are model estimates; multispectral indices are used "
                "only when suitable bands are available; ChangeFormerV6 is limited to LEVIR-style "
                "structural/building change; RGB differences are supporting non-semantic evidence only."
            ),
        }
'''


def replace_method(source: str) -> str:
    pattern = re.compile(
        r"^    def _execute_change_detection\(.*?"
        r"(?=^    # ={10,}\n    # CHANGE RESULT SERIALIZATION)",
        re.MULTILINE | re.DOTALL,
    )
    source, count = pattern.subn(NEW_METHOD + "\n\n", source, count=1)
    if count != 1:
        raise RuntimeError("Could not replace _execute_change_detection() safely.")
    return source


def insert_helpers(source: str) -> str:
    if HELPER_MARKER in source:
        return source

    anchor = "    def _execute_change_detection(\n"
    index = source.find(anchor)
    if index < 0:
        raise RuntimeError("Could not find _execute_change_detection() insertion point.")
    return source[:index] + HELPERS + "\n" + source[index:]


def main() -> None:
    if not ORCHESTRATOR.exists():
        raise FileNotFoundError(f"Missing: {ORCHESTRATOR}")

    if not BACKUP.exists():
        shutil.copy2(ORCHESTRATOR, BACKUP)
        print("Backup:", BACKUP)

    source = ORCHESTRATOR.read_text(encoding="utf-8")

    # Replace the active change method first, then inject the new helper methods.
    source = replace_method(source)
    source = insert_helpers(source)

    ORCHESTRATOR.write_text(source, encoding="utf-8")

    print("Updated:", ORCHESTRATOR)
    print("Upgrade: spectral/semantic evidence is now primary; RGB difference is supporting only.")


if __name__ == "__main__":
    main()
