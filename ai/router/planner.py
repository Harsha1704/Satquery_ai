# ai/router/planner.py

import re
from typing import Dict, List, Tuple

from .intent import Intent, IntentResult


class QueryPlanner:
    """
    Natural-language query planner for SatQuery-AI.

    Extracts:
    - intent
    - confidence
    - matched keywords
    - target
    - multiple targets
    - operation
    - change direction
    - temporal years
    - land-cover transitions
    """

    RULES: Dict[Intent, List[str]] = {

        Intent.CHANGE_DETECTION: [
            "change",
            "changes",
            "changed",
            "difference",
            "differences",
            "compare",
            "comparison",
            "before and after",
            "before-after",
            "before after",
            "compare images",
            "compare image",
            "compare these images",
            "compare these two images",
            "compare the images",
            "compare two images",
            "bi-temporal",
            "bi temporal",
            "bitemporal",
            "temporal change",
            "temporal",
            "time series",
            "change detection",
            "detect change",
            "detect changes",
            "find changes",
            "identify changes",
            "expansion",
            "expanded",
            "loss",
            "lost",
            "increase",
            "increased",
            "decrease",
            "decreased",
            "growth",
            "grew",
            "converted to",
            "converted into",
            "changed to",
            "changed into",
            "became",
            "replaced by",
            "transformed into",
            "turned into",
            "new construction",
            "construction change",
            "over time",
            "between these dates",
            "between two dates",
        ],

        Intent.OPTICAL_SAR_FUSION: [
            "optical and sar",
            "optical + sar",
            "optical plus sar",
            "sar and optical",
            "sar + optical",
            "sar plus optical",
            "sar optical",
            "optical sar",
            "optical-sar",
            "sar-optical",
            "fusion",
            "data fusion",
            "image fusion",
            "sensor fusion",
            "cross modal",
            "cross-modal",
            "multimodal",
            "multi modal",
            "multimodal analysis",
        ],

        Intent.SAR_ANALYSIS: [
            "sar",
            "sar image",
            "sar imagery",
            "sar analysis",
            "analyze sar",
            "analyse sar",
            "sentinel-1",
            "sentinel 1",
            "sentinel1",
            "radar",
            "radar image",
            "radar imagery",
            "radar analysis",
            "backscatter",
            "backscatter analysis",
            "vv",
            "vh",
            "vv polarization",
            "vh polarization",
            "vv vh",
            "sar polarization",
        ],

        Intent.MULTISPECTRAL_ANALYSIS: [
            "ndvi",
            "ndwi",
            "ndbi",
            "savi",
            "evi",
            "gndvi",
            "spectral index",
            "spectral indices",
            "spectral analysis",
            "multispectral",
            "multispectral analysis",
            "multispectral image",
            "multispectral imagery",
            "sentinel-2",
            "sentinel 2",
            "sentinel2",
            "landsat",
            "landsat image",
            "landsat imagery",
            "bands",
            "spectral bands",
            "vegetation index",
            "vegetation indices",
            "water index",
            "water indices",
            "built-up index",
            "built up index",
            "spectral reflectance",
        ],

        Intent.OBJECT_DETECTION: [
            "object detection",
            "object detector",
            "detect object",
            "detect objects",
            "find object",
            "find objects",
            "locate object",
            "locate objects",
            "identify object",
            "identify objects",
            "count object",
            "count objects",

            "detect car",
            "detect cars",
            "find car",
            "find cars",
            "locate car",
            "locate cars",
            "count car",
            "count cars",

            "detect vehicle",
            "detect vehicles",
            "find vehicle",
            "find vehicles",
            "locate vehicle",
            "locate vehicles",
            "count vehicle",
            "count vehicles",

            "detect truck",
            "detect trucks",
            "find truck",
            "find trucks",
            "count truck",
            "count trucks",

            "detect bus",
            "detect buses",
            "find bus",
            "find buses",
            "count bus",
            "count buses",

            "detect building",
            "detect buildings",
            "find building",
            "find buildings",
            "locate building",
            "locate buildings",
            "count building",
            "count buildings",

            "detect house",
            "detect houses",
            "find house",
            "find houses",
            "count house",
            "count houses",

            "detect ship",
            "detect ships",
            "find ship",
            "find ships",
            "count ship",
            "count ships",

            "detect boat",
            "detect boats",
            "find boat",
            "find boats",
            "count boat",
            "count boats",

            "detect aircraft",
            "detect airplane",
            "detect airplanes",
            "find aircraft",
            "find airplane",
            "count aircraft",
            "count airplane",
            "count airplanes",

            "detect person",
            "detect people",
            "find person",
            "find people",
            "count person",
            "count people",

            "detect road",
            "detect roads",
            "find road",
            "find roads",
            "locate road",
            "locate roads",

            "detect motorcycle",
            "detect motorcycles",
            "detect bicycle",
            "detect bicycles",

            "object_detection_model",
            "object_analyzer",
            "object_detector",
            "building_extractor",
        ],

        Intent.SINGLE_IMAGE_VQA: [
            "are there",
            "is there",
            "do you see",
            "can you see",
            "can you identify",
            "what is visible",
            "what can you see",
            "what is shown",
            "what does this image contain",
            "what does the image contain",
            "what is in this image",
            "what is in the image",
            "what is in this satellite image",
            "what is in the satellite image",
            "what does this image show",
            "what does the image show",
            "what does this satellite image show",
            "what does the satellite image show",
            "is this area urban",
            "is this area rural",
            "urban or rural",
            "is this urban or rural",
            "is this area urban or rural",
            "does this image contain",
            "does the image contain",
            "does this satellite image contain",
            "does the satellite image contain",
            "are buildings present",
            "are roads present",
            "are vehicles present",
            "are water bodies present",
            "is vegetation present",
            "what type of area is this",
            "what kind of area is this",
            "what type of land is this",
            "which objects are visible",
            "which features are visible",
            "answer this question",
            "visual question",
            "vqa",
        ],

        Intent.SEMANTIC_ANALYSIS: [
            "analyze this satellite image",
            "analyse this satellite image",
            "analyze satellite image",
            "analyse satellite image",
            "analyze this image",
            "analyse this image",
            "analyze the satellite image",
            "analyse the satellite image",

            "describe this satellite image",
            "describe the satellite image",
            "describe satellite image",
            "describe this image",

            "give me an analysis",
            "give me analysis",
            "provide analysis",
            "provide an analysis",
            "general analysis",
            "image analysis",
            "satellite image analysis",

            "understand this image",
            "interpret this image",
            "interpret the satellite image",
            "understand the satellite image",

            "semantic analysis",
            "semantic segmentation",
            "semantic",
            "land cover",
            "land-cover",
            "land cover classification",
            "land-cover classification",
            "classify",
            "classification",
            "land classification",

            "water",
            "water body",
            "water bodies",
            "waterbody",

            "vegetation",
            "vegetation area",
            "vegetation regions",
            "forest",
            "forests",
            "agriculture",
            "agricultural",
            "farmland",
            "crop",
            "crops",

            "urban",
            "urban area",
            "urban areas",
            "built up",
            "built-up",
            "built up area",
            "built-up area",

            "bare land",
            "bare soil",
            "soil",
            "open land",
            "barren land",
            "barren",
        ],
        
        Intent.TEXT_GUIDED_GROUNDING: [
            "locate ",
            "highlight ",
            "highlight the ",
            "find ",
             "find the ",
            "show me where",
            "show where",
            "where are the",
            "where is the",
            "mark ",
            "mark the ",
            "ground ",
            "ground the ",
            "point out",
            "identify regions",
            "show regions",
        ],
    }

    PRIORITY = [
        Intent.CHANGE_DETECTION,
        Intent.OPTICAL_SAR_FUSION,
        Intent.SAR_ANALYSIS,
        Intent.MULTISPECTRAL_ANALYSIS,
        Intent.TEXT_GUIDED_GROUNDING,
        Intent.OBJECT_DETECTION,
        Intent.SINGLE_IMAGE_VQA,
        Intent.SEMANTIC_ANALYSIS,
    ]

    def _contains_phrase(
        self,
        text: str,
        phrase: str,
    ) -> bool:

        phrase = phrase.strip().lower()

        if not phrase:
            return False

        pattern = (
            r"(?<!\w)"
            + re.escape(phrase)
            + r"(?!\w)"
        )

        return bool(
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
        )

    def _extract_years(
        self,
        text: str,
    ) -> List[int]:

        matches = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )

        years: List[int] = []

        for match in matches:
            year = int(match)

            if year not in years:
                years.append(year)

        return years

    def _extract_targets(
        self,
        text: str,
    ) -> List[str]:

        target_keywords = {

            "water": [
                "water bodies",
                "water body",
                "waterbody",
                "water",
                "river",
                "lake",
                "pond",
                "reservoir",
                "flood",
            ],

            "vegetation": [
                "vegetation",
                "forests",
                "forest",
                "trees",
                "tree",
                "crops",
                "crop",
                "agriculture",
                "agricultural",
                "farmland",
                "greenery",
            ],

            "built_up": [
                "built-up area",
                "built up area",
                "built-up",
                "built up",
                "builtup",
                "urban areas",
                "urban area",
                "urban",
                "developed area",
                "development",
                "construction",
                "buildings",
                "building",
            ],

            "bare_land": [
                "bare land",
                "bare soil",
                "open land",
                "barren land",
                "barren",
                "soil",
            ],

            "ndvi": [
                "ndvi",
                "vegetation index",
            ],

            "ndwi": [
                "ndwi",
                "water index",
            ],

            "ndbi": [
                "ndbi",
                "built-up index",
                "built up index",
            ],

            "sar": [
                "sentinel-1",
                "sentinel 1",
                "sentinel1",
                "backscatter",
                "radar",
                "sar",
            ],

            "vehicle": [
                "vehicles",
                "vehicle",
                "cars",
                "car",
                "trucks",
                "truck",
                "buses",
                "bus",
            ],
        }

        found_targets: List[str] = []

        for target, keywords in target_keywords.items():
            for keyword in keywords:
                if self._contains_phrase(
                    text,
                    keyword,
                ):
                    if target not in found_targets:
                        found_targets.append(target)

                    break

        return found_targets

    def _extract_transition(
        self,
        text: str,
        targets: List[str],
    ) -> Tuple[str, str]:

        if len(targets) < 2:
            return "none", "none"

        transition_phrases = [
            "converted to",
            "converted into",
            "changed to",
            "changed into",
            "became",
            "replaced by",
            "transformed into",
            "turned into",
        ]

        transition_match = None
        transition_position = None

        for phrase in transition_phrases:

            position = text.find(
                phrase
            )

            if position != -1:
                if (
                    transition_position is None
                    or position
                    < transition_position
                ):
                    transition_position = position
                    transition_match = phrase

        if transition_match is None:
            return "none", "none"

        before_text = text[
            :transition_position
        ]

        after_start = (
            transition_position
            + len(
                transition_match
            )
        )

        after_text = text[
            after_start:
        ]

        target_aliases = {

            "water": [
                "water",
                "river",
                "lake",
                "pond",
                "reservoir",
            ],

            "vegetation": [
                "vegetation",
                "forest",
                "forests",
                "crop",
                "crops",
                "agriculture",
                "farmland",
            ],

            "built_up": [
                "built-up",
                "built up",
                "builtup",
                "urban",
                "building",
                "buildings",
                "construction",
            ],

            "bare_land": [
                "bare land",
                "bare soil",
                "open land",
                "barren land",
                "barren",
                "soil",
            ],
        }

        transition_from = "none"
        transition_to = "none"

        best_from_position = -1

        for target in targets:

            aliases = target_aliases.get(
                target,
                [target],
            )

            for alias in aliases:

                position = (
                    before_text.rfind(
                        alias
                    )
                )

                if (
                    position
                    > best_from_position
                ):
                    best_from_position = (
                        position
                    )
                    transition_from = target

        best_to_position = None

        for target in targets:

            aliases = target_aliases.get(
                target,
                [target],
            )

            for alias in aliases:

                position = (
                    after_text.find(
                        alias
                    )
                )

                if position != -1:

                    if (
                        best_to_position
                        is None
                        or position
                        < best_to_position
                    ):
                        best_to_position = (
                            position
                        )
                        transition_to = target

        if (
            transition_from == "none"
            or transition_to == "none"
            or transition_from
            == transition_to
        ):
            return "none", "none"

        return (
            transition_from,
            transition_to,
        )

    def _extract_operation(
        self,
        text: str,
    ) -> str:

        if any(
            phrase in text
            for phrase in [
                "percentage",
                "percent",
                "how much",
                "coverage",
                "area percentage",
                "what percentage",
            ]
        ):
            return "percentage"

        if any(
            phrase in text
            for phrase in [
                "show",
                "highlight",
                "locate",
                "mark",
                "display",
            ]
        ):
            return "show"

        if any(
            phrase in text
            for phrase in [
                "count",
                "how many",
                "number of",
            ]
        ):
            return "count"

        if any(
            phrase in text
            for phrase in [
                "calculate",
                "compute",
            ]
        ):
            return "calculate"

        if any(
            phrase in text
            for phrase in [
                "decrease",
                "decreased",
                "decreasing",
                "loss",
                "lost",
                "reduced",
                "reduction",
                "decline",
                "declined",
                "shrink",
                "shrunk",
            ]
        ):
            return "decrease"

        if any(
            phrase in text
            for phrase in [
                "increase",
                "increased",
                "increasing",
                "growth",
                "grew",
                "expanded",
                "expansion",
                "gain",
                "gained",
            ]
        ):
            return "increase"

        if any(
            phrase in text
            for phrase in [
                "compare",
                "comparison",
                "difference",
                "differences",
                "versus",
                " vs ",
                "between",
            ]
        ):
            return "compare"

        if any(
            phrase in text
            for phrase in [
                "change",
                "changed",
                "changes",
                "what changed",
                "converted to",
                "converted into",
                "became",
                "replaced by",
                "transformed into",
                "turned into",
            ]
        ):
            return "change"

        if any(
            phrase in text
            for phrase in [
                "describe",
                "explain",
                "what is visible",
                "what is in",
                "what does",
            ]
        ):
            return "describe"

        return "analyze"

    def _extract_change_direction(
        self,
        text: str,
    ) -> str:

        decrease_keywords = [
            "decrease",
            "decreased",
            "decreasing",
            "loss",
            "lost",
            "reduce",
            "reduced",
            "reduction",
            "decline",
            "declined",
            "declining",
            "shrink",
            "shrunk",
            "smaller",
            "less",
        ]

        increase_keywords = [
            "increase",
            "increased",
            "increasing",
            "growth",
            "grew",
            "grown",
            "expand",
            "expanded",
            "expansion",
            "gain",
            "gained",
            "larger",
            "more",
        ]

        for keyword in decrease_keywords:
            if keyword in text:
                return "decrease"

        for keyword in increase_keywords:
            if keyword in text:
                return "increase"

        return "none"

    def _has_change_signal(
        self,
        text: str,
        years: List[int],
        transition_from: str = "none",
        transition_to: str = "none",
    ) -> bool:

        if len(years) >= 2:
            return True

        if (
            transition_from != "none"
            and transition_to != "none"
        ):
            return True

        change_terms = [
            "change",
            "changed",
            "changes",
            "difference",
            "differences",
            "compare",
            "comparison",
            "increase",
            "increased",
            "increasing",
            "decrease",
            "decreased",
            "decreasing",
            "loss",
            "lost",
            "growth",
            "grew",
            "grown",
            "expand",
            "expanded",
            "expansion",
            "gain",
            "gained",
            "converted to",
            "converted into",
            "changed to",
            "changed into",
            "became",
            "replaced by",
            "transformed into",
            "turned into",
            "before and after",
            "before-after",
            "before after",
            "between two dates",
            "between these dates",
            "over time",
            "temporal",
            "bi-temporal",
            "bi temporal",
            "bitemporal",
        ]

        return any(
            term in text
            for term in change_terms
        )

    def _has_optical_sar_fusion_signal(
        self,
        text: str,
    ) -> bool:

        fusion_phrases = [
            "optical and sar",
            "optical + sar",
            "optical plus sar",
            "sar and optical",
            "sar + optical",
            "sar plus optical",
            "sar optical",
            "optical sar",
            "optical-sar",
            "sar-optical",
            "optical and radar",
            "radar and optical",
            "optical radar",
            "radar optical",

            "optical sar fusion",
            "sar optical fusion",
            "optical and sar fusion",
            "sar and optical fusion",

            "fuse optical and sar",
            "fuse optical with sar",
            "fuse sar with optical",
            "fuse sar and optical",

            "fuse sentinel-2 optical with sar",
            "fuse sentinel 2 optical with sar",
            "fuse sentinel2 optical with sar",

            "combine optical and sar",
            "combine optical with sar",
            "combine sar with optical",
            "combine sar and optical",

            "integrate optical and sar",
            "integrate optical with sar",
            "integrate sar with optical",

            "sensor fusion",
            "image fusion",
            "data fusion",
            "cross modal",
            "cross-modal",
            "multimodal",
            "multi modal",
        ]

        if any(
            phrase in text
            for phrase in fusion_phrases
        ):
            return True

        optical_signals = [
            "optical",
            "sentinel-2",
            "sentinel 2",
            "sentinel2",
            "landsat",
            "multispectral",
        ]

        sar_signals = [
            "sar",
            "radar",
            "sentinel-1",
            "sentinel 1",
            "sentinel1",
        ]

        fusion_actions = [
            "fuse",
            "fusion",
            "combine",
            "combined",
            "integrate",
            "integration",
            "together",
            "joint",
            "jointly",
            "multimodal",
            "multi modal",
        ]

        has_optical = any(
            signal in text
            for signal in optical_signals
        )

        has_sar = any(
            signal in text
            for signal in sar_signals
        )

        has_fusion_action = any(
            action in text
            for action in fusion_actions
        )

        return (
            has_optical
            and has_sar
            and has_fusion_action
        )

    def plan(
        self,
        query: str,
    ) -> IntentResult:

        if not query or not query.strip():

            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                matched_keywords=[],
                target="overall",
                targets=[],
                operation="analyze",
                change_direction="none",
                years=[],
                original_query=query or "",
                transition_from="none",
                transition_to="none",
            )

        text = query.lower().strip()

        years = self._extract_years(
            text
        )

        targets = self._extract_targets(
            text
        )

        (
            transition_from,
            transition_to,
        ) = self._extract_transition(
            text=text,
            targets=targets,
        )

        if targets:
            target = targets[0]
        else:
            target = "overall"

        operation = (
            self._extract_operation(
                text
            )
        )

        change_direction = (
            self._extract_change_direction(
                text
            )
        )

        scores: Dict[
            Intent,
            List[str],
        ] = {
            intent: []
            for intent in self.RULES
        }

        for (
            intent,
            keywords,
        ) in self.RULES.items():

            for keyword in keywords:

                if self._contains_phrase(
                    text,
                    keyword,
                ):

                    if (
                        keyword
                        not in scores[
                            intent
                        ]
                    ):
                        scores[
                            intent
                        ].append(
                            keyword
                        )

        force_change = (
            self._has_change_signal(
                text=text,
                years=years,
                transition_from=(
                    transition_from
                ),
                transition_to=(
                    transition_to
                ),
            )
        )

        force_fusion = (
            self._has_optical_sar_fusion_signal(
                text
            )
        )

        best_intent = Intent.UNKNOWN
        best_matches: List[str] = []

        for intent in self.PRIORITY:

            matches = scores.get(
                intent,
                [],
            )

            if len(matches) > len(
                best_matches
            ):

                best_intent = intent
                best_matches = matches

        if force_change:

            best_intent = (
                Intent.CHANGE_DETECTION
            )

            change_matches = scores.get(
                Intent.CHANGE_DETECTION,
                [],
            )

            if change_matches:
                best_matches = (
                    change_matches
                )

            else:
                best_matches = [
                    "temporal_change"
                ]

        elif force_fusion:

            best_intent = (
                Intent.OPTICAL_SAR_FUSION
            )

            fusion_matches = scores.get(
                Intent.OPTICAL_SAR_FUSION,
                [],
            )

            if fusion_matches:
                best_matches = fusion_matches
            else:
                best_matches = [
                    "optical_sar_fusion"
                ]

        if (
            best_intent
            == Intent.UNKNOWN
            or not best_matches
        ):

            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                matched_keywords=[],
                target=target,
                targets=targets,
                operation=operation,
                change_direction=(
                    change_direction
                ),
                years=years,
                original_query=query,
                transition_from=(
                    transition_from
                ),
                transition_to=(
                    transition_to
                ),
            )

        confidence = min(
            0.60
            + (
                0.10
                * len(
                    best_matches
                )
            ),
            0.99,
        )

        if (
            best_intent
            == Intent.CHANGE_DETECTION
            and len(years) >= 2
        ):
            confidence = max(
                confidence,
                0.95,
            )

        if (
            best_intent
            == Intent.CHANGE_DETECTION
            and transition_from
            != "none"
            and transition_to
            != "none"
        ):
            confidence = max(
                confidence,
                0.95,
            )

        if (
            best_intent
            == Intent.SINGLE_IMAGE_VQA
        ):
            confidence = max(
                confidence,
                0.85,
            )

        if (
            best_intent
            == Intent.TEXT_GUIDED_GROUNDING
        ):
            confidence = max(
                confidence,
                0.86,
            )

        if (
            best_intent
            == Intent.OPTICAL_SAR_FUSION
        ):
            confidence = max(
                confidence,
                0.95,
            )

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            matched_keywords=(
                best_matches
            ),
            target=target,
            targets=targets,
            operation=operation,
            change_direction=(
                change_direction
            ),
            years=years,
            original_query=query,
            transition_from=(
                transition_from
            ),
            transition_to=(
                transition_to
            ),
        )

    def required_tools(
        self,
        intent: Intent,
    ) -> List[str]:

        mapping: Dict[
            Intent,
            List[str],
        ] = {

            Intent.SEMANTIC_ANALYSIS: [
                "semantic_model",
                "semantic_analyzer",
            ],

            Intent.SINGLE_IMAGE_VQA: [
                "blip_vqa",
                "single_image_vqa",
            ],

            Intent.TEXT_GUIDED_GROUNDING: [
                "remoteclip",
                "text_guided_region_grounder",
            ],

            Intent.OBJECT_DETECTION: [
                "object_detection_model",
                "object_detector",
                "building_extractor",
                "object_analyzer",
            ],

            Intent.MULTISPECTRAL_ANALYSIS: [
                "multispectral_analyzer",
            ],

            Intent.SAR_ANALYSIS: [
                "sar_loader",
                "sar_analyzer",
            ],

            Intent.OPTICAL_SAR_FUSION: [
                "multispectral_loader",
                "sar_loader",
                "optical_sar_fusion",
                "fusion_analyzer",
            ],

            Intent.CHANGE_DETECTION: [
                "changeformer_detector",
                "change_analyzer",
                "change_vqa",
            ],

            Intent.UNKNOWN: [],
        }

        return mapping.get(
            intent,
            [],
        )


__all__ = [
    "QueryPlanner",
]
