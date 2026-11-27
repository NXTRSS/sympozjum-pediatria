from pathlib import Path
from unittest import mock

import pytest

from delphi_demo import trening, zasoby
from delphi_demo.colab_przygotowanie import uruchom_jesli_potrzeba


@pytest.fixture
def mini_delphi(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "przygotuj_demo.py").write_text("# stub\n")
    monkeypatch.setattr(zasoby, "KATALOG_DELPHI", tmp_path)
    monkeypatch.setattr(zasoby, "KATALOG_ZEWN", tmp_path / "zewnetrzne")
    monkeypatch.setattr(zasoby, "KATALOG_PAMIECI", tmp_path / "pamiec")
    monkeypatch.setattr(trening, "SCIEZKA_MODELU", tmp_path / "pamiec" / "delphi_syntetyczny.pt")
    return tmp_path


def test_colab_przygotowanie_pomija_gdy_model_jest(mini_delphi, monkeypatch):
    cel = zasoby.katalog_kodu()
    cel.mkdir(parents=True)
    (cel / "model.py").write_text("# stub\n")
    mini_delphi.joinpath("pamiec").mkdir(parents=True)
    trening.SCIEZKA_MODELU.write_bytes(b"x")

    sklonuj = mock.Mock()
    monkeypatch.setattr(zasoby, "sklonuj", sklonuj)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)

    uruchom_jesli_potrzeba(mini_delphi)
    sklonuj.assert_not_called()
    check.assert_not_called()


def test_colab_przygotowanie_woła_sklonuj_i_przygotuj(mini_delphi, monkeypatch):
    sklonuj = mock.Mock(side_effect=lambda: zasoby.katalog_kodu().mkdir(parents=True) or (zasoby.katalog_kodu() / "model.py").write_text("#\n"))
    monkeypatch.setattr(zasoby, "sklonuj", sklonuj)
    check = mock.Mock()
    monkeypatch.setattr("subprocess.check_call", check)
    monkeypatch.setenv("DELPHI_COLAB_KROKI", "100")

    uruchom_jesli_potrzeba(mini_delphi)
    sklonuj.assert_called_once()
    check.assert_called_once()
    args = check.call_args[0][0]
    assert any("przygotuj_demo.py" in str(a) for a in args)
    assert "--kroki" in args and "100" in args