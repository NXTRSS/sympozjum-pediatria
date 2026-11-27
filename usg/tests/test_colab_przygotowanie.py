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


def test_brak_release_uruchamia_przygotowanie(mini_usg, monkeypatch):
    monkeypatch.setattr(cp, "_pobierz_z_release", lambda _d: False)
    monkeypatch.setattr(cp, "_zainstaluj", lambda *_a, **_k: None)
    monkeypatch.delenv("USG_COLAB_BEZ_PRZYGOTOWANIA", raising=False)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)
    cp.uruchom_jesli_potrzeba(mini_usg)
    check.assert_called_once()
    assert "przygotuj_demo.py" in check.call_args.args[0][1]


def test_brak_release_bez_przygotowania_jasny_blad(mini_usg, monkeypatch):
    monkeypatch.setattr(cp, "_pobierz_z_release", lambda _d: False)
    monkeypatch.setattr(cp, "_zainstaluj", lambda *_a, **_k: None)
    monkeypatch.setenv("USG_COLAB_BEZ_PRZYGOTOWANIA", "1")
    with pytest.raises(RuntimeError, match="USG_COLAB_BEZ_PRZYGOTOWANIA"):
        cp.uruchom_jesli_potrzeba(mini_usg)