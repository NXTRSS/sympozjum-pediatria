from pathlib import Path
from unittest import mock

import pytest

from usg_masa import colab_przygotowanie as cp

WYCINEK = cp.WYCINEK_NAZWA


@pytest.fixture
def mini_usg(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "przygotuj_demo.py").write_text("# stub\n")
    (tmp_path / "pamiec").mkdir(parents=True)
    (tmp_path / "dane" / "colab").mkdir(parents=True)
    return tmp_path


def test_gotowy_wycinek_z_colab_mha(mini_usg):
    for suf in (".npz", "_fuvai.npz", "_sano.json"):
        (mini_usg / "pamiec" / f"{WYCINEK}{suf}").write_text("x")
    mha = mini_usg / "dane" / "colab" / f"{WYCINEK}.mha"
    mha.write_bytes(b"x")
    assert cp._gotowy_wycinek(mini_usg)


def test_uruchom_tylko_release_bez_prep(mini_usg, monkeypatch):
    monkeypatch.setattr(cp, "_pobierz_z_release", lambda _d: True)
    for suf in (".npz", "_fuvai.npz", "_sano.json"):
        (mini_usg / "pamiec" / f"{WYCINEK}{suf}").write_text("x")
    (mini_usg / "dane" / "colab" / f"{WYCINEK}.mha").write_bytes(b"x")
    monkeypatch.setattr(cp, "_zainstaluj", lambda *_a, **_k: None)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)
    cp.uruchom_jesli_potrzeba(mini_usg)
    check.assert_not_called()


def test_brak_release_bez_force_wywala(mini_usg, monkeypatch):
    monkeypatch.setattr(cp, "_pobierz_z_release", lambda _d: False)
    monkeypatch.setattr(cp, "_zainstaluj", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="Release"):
        cp.uruchom_jesli_potrzeba(mini_usg)