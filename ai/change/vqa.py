# ai/change/vqa.py

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class ChangeVQA:
    """
    Natural-language question answering over the output
    of the bi-temporal ChangeFormer + ChangeAnalyzer pipeline.

    This module answers questions using computed change
    evidence. It does NOT invent semantic land-cover
    transitions that the binary LEVIR-CD model cannot detect.
    """

    def answer(
        self,
        query: str,
        analysis: Dict[str, Any],
        before_year: Optional[int] = None,
        after_year: Optional[int] = None,
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            raise ValueError(
                "Change-VQA query cannot be empty."
            )

        if not analysis:
            raise ValueError(
                "Change analysis result is required."
            )

        text = query.lower().strip()

        total_pixels = int(
            analysis.get(
                "total_pixels",
                0,
            )
        )

        changed_pixels = int(
            analysis.get(
                "changed_pixels",
                0,
            )
        )

        stable_pixels = int(
            analysis.get(
                "stable_pixels",
                0,
            )
        )

        changed_percentage = float(
            analysis.get(
                "changed_percentage",
                0.0,
            )
        )

        stable_percentage = float(
            analysis.get(
                "stable_percentage",
                0.0,
            )
        )

        change_detected = bool(
            analysis.get(
                "change_detected",
                changed_pixels > 0,
            )
        )

        years = self._extract_years(
            text
        )

        if before_year is None and years:
            before_year = years[0]

        if (
            after_year is None
            and len(years) >= 2
        ):
            after_year = years[1]

        period = self._period_text(
            before_year,
            after_year,
        )

        question_type = (
            self._detect_question_type(
                text
            )
        )

        if question_type == "changed_percentage":

            answer = (
                f"Approximately "
                f"{changed_percentage:.2f}% "
                f"of the analyzed area changed"
                f"{period}."
            )

            value = changed_percentage
            unit = "percent"

        elif question_type == "stable_percentage":

            answer = (
                f"Approximately "
                f"{stable_percentage:.2f}% "
                f"of the analyzed area remained "
                f"stable{period}."
            )

            value = stable_percentage
            unit = "percent"

        elif question_type == "changed_pixels":

            answer = (
                f"{changed_pixels:,} pixels "
                f"were identified as changed"
                f"{period}."
            )

            value = changed_pixels
            unit = "pixels"

        elif question_type == "stable_pixels":

            answer = (
                f"{stable_pixels:,} pixels "
                f"remained unchanged"
                f"{period}."
            )

            value = stable_pixels
            unit = "pixels"

        elif question_type == "total_pixels":

            answer = (
                f"The analysis covers "
                f"{total_pixels:,} pixels."
            )

            value = total_pixels
            unit = "pixels"

        elif question_type == "did_change":

            if change_detected:

                answer = (
                    "Yes. Change was detected "
                    f"across approximately "
                    f"{changed_percentage:.2f}% "
                    f"of the analyzed area"
                    f"{period}."
                )

            else:

                answer = (
                    "No significant change was "
                    f"detected{period}."
                )

            value = change_detected
            unit = "boolean"

        elif question_type == "change_level":

            level = self._change_level(
                changed_percentage
            )

            answer = (
                f"The detected change is "
                f"classified as {level.replace('_', ' ')}. "
                f"Approximately "
                f"{changed_percentage:.2f}% "
                f"of the analyzed area changed"
                f"{period}."
            )

            value = level
            unit = "category"

        elif question_type == "where_change":

            answer = (
                "The changed locations are shown "
                "in the generated binary change "
                "mask. Changed pixels are marked "
                "as foreground while stable pixels "
                "remain background."
            )

            value = None
            unit = "visual_evidence"

        elif question_type == "semantic_transition":

            answer = (
                "The current ChangeFormerV6 "
                "LEVIR-CD model detects binary "
                "structural/building change, but "
                "it cannot reliably determine "
                "semantic transitions such as "
                "vegetation-to-built-up, "
                "water-to-bare-land, or "
                "vegetation loss. A semantic "
                "change model is required for "
                "that question."
            )

            value = None
            unit = "unsupported"

        else:

            answer = self._general_description(
                changed_pixels=changed_pixels,
                changed_percentage=(
                    changed_percentage
                ),
                stable_percentage=(
                    stable_percentage
                ),
                period=period,
            )

            value = changed_percentage
            unit = "percent"

        return {
            "success": True,
            "question": query,
            "question_type": question_type,
            "answer": answer,
            "value": value,
            "unit": unit,
            "before_year": before_year,
            "after_year": after_year,
            "evidence": {
                "total_pixels": total_pixels,
                "changed_pixels": (
                    changed_pixels
                ),
                "stable_pixels": (
                    stable_pixels
                ),
                "changed_percentage": (
                    changed_percentage
                ),
                "stable_percentage": (
                    stable_percentage
                ),
                "change_detected": (
                    change_detected
                ),
            },
        }

    def describe(
        self,
        analysis: Dict[str, Any],
        before_year: Optional[int] = None,
        after_year: Optional[int] = None,
    ) -> str:

        changed_pixels = int(
            analysis.get(
                "changed_pixels",
                0,
            )
        )

        changed_percentage = float(
            analysis.get(
                "changed_percentage",
                0.0,
            )
        )

        stable_percentage = float(
            analysis.get(
                "stable_percentage",
                0.0,
            )
        )

        period = self._period_text(
            before_year,
            after_year,
        )

        return self._general_description(
            changed_pixels=changed_pixels,
            changed_percentage=(
                changed_percentage
            ),
            stable_percentage=(
                stable_percentage
            ),
            period=period,
        )

    @staticmethod
    def _detect_question_type(
        text: str,
    ) -> str:

        semantic_terms = [
            "water",
            "vegetation",
            "forest",
            "agriculture",
            "farmland",
            "bare land",
            "bare_land",
            "bare soil",
        ]

        transition_terms = [
            "converted to",
            "converted into",
            "changed to",
            "changed into",
            "became",
            "replaced by",
            "transformed into",
            "turned into",
            "loss of",
            "lost",
        ]

        if (
            any(
                term in text
                for term in semantic_terms
            )
            and any(
                term in text
                for term in transition_terms
            )
        ):
            return "semantic_transition"

        if any(
            phrase in text
            for phrase in [
                "where did",
                "where has",
                "where are",
                "where is",
                "show change",
                "show changes",
                "highlight change",
                "highlight changes",
                "locate change",
                "change map",
            ]
        ):
            return "where_change"

        if any(
            phrase in text
            for phrase in [
                "how much changed",
                "what percentage changed",
                "percentage changed",
                "percent changed",
                "change percentage",
                "percentage of change",
                "how much area changed",
            ]
        ):
            return "changed_percentage"

        if any(
            phrase in text
            for phrase in [
                "stable percentage",
                "percentage stable",
                "percent stable",
                "percentage unchanged",
                "percent unchanged",
                "how much remained stable",
                "how much remained unchanged",
            ]
        ):
            return "stable_percentage"

        if any(
            phrase in text
            for phrase in [
                "how many pixels changed",
                "changed pixels",
                "number of changed pixels",
            ]
        ):
            return "changed_pixels"

        if any(
            phrase in text
            for phrase in [
                "stable pixels",
                "unchanged pixels",
                "how many pixels remained",
            ]
        ):
            return "stable_pixels"

        if any(
            phrase in text
            for phrase in [
                "total pixels",
                "how many pixels are analyzed",
                "how many pixels were analyzed",
            ]
        ):
            return "total_pixels"

        if any(
            phrase in text
            for phrase in [
                "was there change",
                "is there change",
                "did anything change",
                "did the area change",
                "has anything changed",
                "was change detected",
            ]
        ):
            return "did_change"

        if any(
            phrase in text
            for phrase in [
                "change level",
                "level of change",
                "significant change",
                "how significant",
                "severity",
                "major change",
                "minor change",
            ]
        ):
            return "change_level"

        return "general_change"

    @staticmethod
    def _extract_years(
        text: str,
    ):

        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )

        result = []

        for year in years:

            value = int(year)

            if value not in result:
                result.append(value)

        return result

    @staticmethod
    def _period_text(
        before_year: Optional[int],
        after_year: Optional[int],
    ) -> str:

        if (
            before_year is not None
            and after_year is not None
        ):
            return (
                f" between {before_year} "
                f"and {after_year}"
            )

        return ""

    @staticmethod
    def _change_level(
        percentage: float,
    ) -> str:

        if percentage < 0.01:
            return "very_low_change"

        if percentage < 1.0:
            return "minor_change"

        if percentage < 10.0:
            return "localized_change"

        if percentage < 30.0:
            return "significant_change"

        return "major_change"

    @staticmethod
    def _general_description(
        changed_pixels: int,
        changed_percentage: float,
        stable_percentage: float,
        period: str,
    ) -> str:

        if changed_pixels == 0:

            return (
                "No significant change was "
                f"detected{period}."
            )

        return (
            f"Bi-temporal analysis detected "
            f"change across approximately "
            f"{changed_percentage:.2f}% of the "
            f"analyzed area{period}. "
            f"{changed_pixels:,} pixels were "
            f"classified as changed, while "
            f"{stable_percentage:.2f}% of the "
            f"area remained stable."
        )


__all__ = [
    "ChangeVQA",
]