from ai.query_router import QueryRouter, Intent


def test_query_router():

    router = QueryRouter()

    test_queries = [

        (
            "How much vegetation is present?",
            Intent.SPECTRAL_ANALYSIS
        ),

        (
            "Calculate NDVI for this image",
            Intent.SPECTRAL_ANALYSIS
        ),

        (
            "Detect buildings in this image",
            Intent.OBJECT_DETECTION
        ),

        (
            "Segment the water bodies",
            Intent.SEGMENTATION
        ),

        (
            "What type of land is visible?",
            Intent.CLASSIFICATION
        ),

        (
            "What is visible in this image?",
            Intent.VQA
        ),

        (
            "What changed between these two images?",
            Intent.CHANGE_DETECTION
        ),

        (
            "Compare the optical and SAR images",
            Intent.OPTICAL_SAR_ANALYSIS
        ),

        (
            "Tell me something about this image",
            Intent.UNKNOWN
        )
    ]

    print("\nSatQuery Query Router Test")
    print("-" * 50)

    passed = 0

    for query, expected in test_queries:

        result = router.explain(query)

        actual = router.route(query)

        status = "PASS" if actual == expected else "FAIL"

        if actual == expected:
            passed += 1

        print(f"\nQuery      : {query}")
        print(f"Intent     : {actual.value}")
        print(f"Expected   : {expected.value}")
        print(f"Confidence : {result['confidence']}")
        print(f"Status     : {status}")

    print("\n" + "-" * 50)
    print(
        f"Router tests: {passed}/{len(test_queries)} passed"
    )

    assert passed == len(test_queries)


if __name__ == "__main__":
    test_query_router()