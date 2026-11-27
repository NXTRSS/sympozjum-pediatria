from pathlib import Path
from unittest import mock

import pytest

from usg_masa.colab_przygotowanie import _ma_wyniki_wycinka, uruchom_jesli_potrzeba


@pytest.fixture
def mini_usg(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "przygotuj_demo.py").write_text("# stub\n")
    (tmp_path / "pamiec").mkdir()
    return tmp_path


def test_ma_wyniki_wycinka_false_bez_plikow(mini_usg, monkeypatch):
    monkeypatch.setattr(
        "usg_masa.dane.pobierz_przypadek_startowy",
        lambda: mini_usg / "brak.mha",
    )
    assert _ma_wyniki_wycinka(mini_usg) is False


def test_uruchom_pomija_gdy_sa_wyniki(mini_usg, monkeypatch):
    nazwa = "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000"
    start = mini_usg / f"{nazwa}.mha"
    start.parent.mkdir(parents=True, exist_ok=True)
    start.write_bytes(b"x")
    for suf in (".npz", "_fuvai.npz", "_sano.json"):
        (mini_usg / "pamiec" / f"{nazwa}{suf}").write_text("x")
    monkeypatch.setattr("usg_masa.dane.pobierz_przypadek_startowy", lambda: start)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)
    monkeypatch.setattr("usg_masa.colab_przygotowanie._zainstaluj_zaleznosci", lambda: None)

    uruchom_jesli_potrzeba(mini_usg)
    check.assert_not_called()


def test_uruchom_woła_przygotuj(mini_usg, monkeypatch):
    monkeypatch.setattr(
        "usg_masa.dane.pobierz_przypadek_startowy",
        lambda: mini_usg / "brak.mha",
    )
    monkeypatch.setattr("usg_masa.colab_przygotowanie._zainstaluj_zaleznosci", lambda: None)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)
    monkeypatch.setenv("USG_COLAB_ILE_ZENODO", "0")

    uruchom_jesli_potrzeba(mini_usg)
    check.assert_called_once()
    assert "przygotuj_demo.py" in str(check.call_args[0][0])