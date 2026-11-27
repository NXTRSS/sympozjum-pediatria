"""Logika środowiska notebooków (lokalnie, bez Colaba)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "notebook" / "colab_sciezka.py"


def test_colab_sciezka_w_repo():
    assert CS.is_file()


def test_korzen_repo_w_workspace():
    spec = __import__("importlib.util").util.spec_from_file_location("colab_sciezka", CS)
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.korzen_repo_lokalnie() == ROOT


@pytest.mark.parametrize(
    "marker,folder",
    [
        ("notebook/lm_utils.py", "notebook"),
        ("delphi/pyproject.toml", "delphi/notebook"),
        ("usg/pyproject.toml", "usg/notebook"),
    ],
)
def test_lokalizuj_colab_bootstrap():
    spec = __import__("importlib.util").util.spec_from_file_location("colab_sciezka", CS)
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod._lokalizuj_colab_bootstrap("delphi/notebook")
    assert p.name == "colab_bootstrap.py" and p.is_file()


def test_upewnij_lokalnie(marker, folder):
    spec = __import__("importlib.util").util.spec_from_file_location("colab_sciezka", CS)
    mod = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    nb = mod.upewnij_srodowisko(marker, folder)
    assert (ROOT / folder).samefile(nb) or nb.resolve() == (ROOT / folder).resolve()