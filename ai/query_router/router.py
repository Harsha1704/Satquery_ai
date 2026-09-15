from .intent import Intent


class QueryRouter:
    """
    Lightweight natural-language router for SatQuery AI.

    First-stage routing layer.
    Later this can be augmented with an ML/VLM-based router.
    """

    KEYWORDS = {

        # ---------------------------------------------
        # Optical / SAR
        # ---------------------------------------------

        Intent.OPTICAL_SAR_ANALYSIS: [
            "sar",
            "radar",
            "optical and sar",
            "optical + sar",
            "sentinel-1",
            "sentinel 1",
            "risat"
        ],

        # ---------------------------------------------
        # Change Detection
        # ---------------------------------------------

        Intent.CHANGE_DETECTION: [
            "change",
            "changed",
            "changes",
            "before and after",
            "before vs after",
            "difference",
            "compare",
            "comparison",
            "growth",
            "decline"
        ],

        # ---------------------------------------------
        # Spectral Analysis
        # ---------------------------------------------

        Intent.SPECTRAL_ANALYSIS: [
            "ndvi",
            "vegetation",
            "vegetation health",
            "vegetation coverage",
            "greenery",
            "spectral index",
            "spectral analysis",
            "water index",
            "ndwi",
            "ndbi"
        ],

        # ---------------------------------------------
        # Semantic Land Cover
        # ---------------------------------------------

        Intent.SEMANTIC_ANALYSIS: [
            "water bodies",
            "water body",
            "lake",
            "lakes",
            "river",
            "rivers",
            "pond",
            "ponds",

            "agricultural land",
            "agriculture",
            "crop",
            "crops",
            "farmland",
            "forest",
            "forests",

            "built-up areas",
            "built up areas",
            "built-up area",
            "built up area",
            "urban area",
            "urban areas",
            "settlement",
            "settlements",

            "bare land",
            "bare-land",
            "exposed land"
        ],

        # ---------------------------------------------
        # Object Detection
        # ---------------------------------------------

        Intent.OBJECT_DETECTION: [
            "detect buildings",
            "detect building",
            "find buildings",
            "find building",
            "detect roads",
            "find roads",
            "detect vehicles",
            "find vehicles",
            "count buildings",
            "count vehicles",
            "objects"
        ],

        # ---------------------------------------------
        # Segmentation
        # ---------------------------------------------

        Intent.SEGMENTATION: [
            "segment",
            "segmentation",
            "segment the image",
            "create a mask",
            "mask",
            "pixel level",
            "pixel-level"
        ],

        # ---------------------------------------------
        # Classification
        # ---------------------------------------------

        Intent.CLASSIFICATION: [
            "classify",
            "classification",
            "land cover",
            "land-cover",
            "land use",
            "land-use",
            "what type of land",
            "identify land"
        ],

        # ---------------------------------------------
        # VQA
        # ---------------------------------------------

        Intent.VQA: [
            "what is",
            "what are",
            "where is",
            "where are",
            "how many",
            "is there",
            "are there",
            "describe",
            "describe the image",
            "what do you see"
        ]
    }

    def route(self, query: str) -> Intent:

        if not query or not query.strip():
            return Intent.UNKNOWN

        normalized_query = query.lower().strip()

        # Specific intents must be checked first.
        priority_order = [

            Intent.OPTICAL_SAR_ANALYSIS,

            Intent.CHANGE_DETECTION,

            Intent.OBJECT_DETECTION,

            Intent.SEGMENTATION,

            Intent.SEMANTIC_ANALYSIS,

            Intent.SPECTRAL_ANALYSIS,

            Intent.CLASSIFICATION,

            Intent.VQA
        ]

        for intent in priority_order:

            keywords = self.KEYWORDS.get(
                intent,
                []
            )

            for keyword in keywords:

                if keyword in normalized_query:
                    return intent

        return Intent.UNKNOWN

    def explain(self, query: str) -> dict:

        intent = self.route(query)

        return {
            "query": query,
            "intent": intent.value,
            "confidence": self._confidence(
                query,
                intent
            ),
            "routing_reason": self._reason(
                intent
            )
        }

    def _confidence(
        self,
        query: str,
        intent: Intent
    ) -> float:

        if intent == Intent.UNKNOWN:
            return 0.0

        normalized_query = query.lower()

        keywords = self.KEYWORDS.get(
            intent,
            []
        )

        matches = sum(
            keyword in normalized_query
            for keyword in keywords
        )

        if matches >= 2:
            return 0.95

        return 0.85

    def _reason(
        self,
        intent: Intent
    ) -> str:

        reasons = {

            Intent.SPECTRAL_ANALYSIS:
                "The query requests spectral or vegetation-index analysis.",

            Intent.SEMANTIC_ANALYSIS:
                "The query requests identification of a semantic land-cover category.",

            Intent.CHANGE_DETECTION:
                "The query indicates comparison or temporal change analysis.",

            Intent.OPTICAL_SAR_ANALYSIS:
                "The query explicitly references SAR or optical-SAR analysis.",

            Intent.OBJECT_DETECTION:
                "The query requests detection or counting of objects.",

            Intent.SEGMENTATION:
                "The query requests pixel-level segmentation or masking.",

            Intent.CLASSIFICATION:
                "The query requests land-cover or land-use classification.",

            Intent.VQA:
                "The query is a general visual question about the image.",

            Intent.UNKNOWN:
                "No supported analysis intent was detected."
        }

        return reasons[intent]