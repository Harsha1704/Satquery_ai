# tests/test_input_validation.py

from pprint import pprint

from ai.validation import InputValidator


validator = InputValidator()


print("\n" + "=" * 70)
print("1. MULTISPECTRAL NPY TEST")
print("=" * 70)

result = validator.validate_multispectral(
    r".\data\multispectral\data\S2A_MSIL2A_20170613T101031_17_70_all_bands.npy"
)

pprint(result)


print("\n" + "=" * 70)
print("2. SAR NPY TEST")
print("=" * 70)

result = validator.validate_single_image(
    r".\data\test\sar_sample.npy",
    expected_modality="sar",
)

pprint(result)


print("\n" + "=" * 70)
print("3. OPTICAL + SAR COMPATIBILITY TEST")
print("=" * 70)

result = validator.validate_optical_sar_pair(
    optical_path=(
        r".\data\multispectral\data"
        r"\S2A_MSIL2A_20170613T101031_17_70_all_bands.npy"
    ),
    sar_path=r".\data\test\sar_sample.npy",
)

pprint(result)


print("\n" + "=" * 70)
print("4. CHANGE PAIR TEST")
print("=" * 70)

result = validator.validate_change_pair(
    before_path=r".\data\LEVIR-CD\test\A\test_1.png",
    after_path=r".\data\LEVIR-CD\test\B\test_1.png",
)

pprint(result)


print("\n" + "=" * 70)
print("PRIORITY 10 VALIDATION TEST COMPLETE")
print("=" * 70)