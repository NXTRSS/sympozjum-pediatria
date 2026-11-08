import sys


def test_python_311():
    assert sys.version_info[:2] == (3, 11)


def test_kluczowe_biblioteki_sie_importuja():
    import cv2, einops, numpy, SimpleITK, torch  # noqa: F401
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: F401
    assert numpy.__version__.startswith("1.")
    assert torch.__version__.startswith("2.2")


def test_pakiet_usg_masa():
    import usg_masa
    assert usg_masa.__doc__
