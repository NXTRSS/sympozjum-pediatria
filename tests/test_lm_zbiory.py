from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    import importlib.util
    import sys

    nb = ROOT / "notebook"
    if str(nb) not in sys.path:
        sys.path.insert(0, str(nb))
    spec = importlib.util.spec_from_file_location(
        "lm_zbiory", nb / "lm_zbiory.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("zbior", ["choroby", "leki", "dinozaury"])
def test_wczytaj_zbior(zbior):
    mod = _mod()
    nazwy, df, zrodlo, meta = mod.wczytaj_zbior(zbior)
    assert len(nazwy) == len(df) > 100
    assert meta["emoji"]
    assert str(zrodlo)


def test_zly_zbior():
    mod = _mod()
    with pytest.raises(ValueError, match="ZBIOR"):
        mod.wczytaj_zbior("krzesla")