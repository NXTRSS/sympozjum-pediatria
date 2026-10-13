import sys


def test_python_311():
    assert sys.version_info[:2] == (3, 11)


def test_kluczowe_biblioteki():
    import matplotlib, numpy, pandas, torch  # noqa: F401
    assert torch.__version__.startswith("2.3.0")
    assert numpy.__version__.startswith("1.")


def test_pakiet_delphi_demo():
    import delphi_demo
    assert delphi_demo.__doc__
