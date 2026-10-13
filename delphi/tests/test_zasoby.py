import runpy
import subprocess

import pandas as pd
import pytest

from delphi_demo import tokeny, zasoby


def test_brak_kodu_czytelny_blad(monkeypatch, tmp_path):
    monkeypatch.setattr(zasoby, "KATALOG_ZEWN", tmp_path)
    with pytest.raises(FileNotFoundError, match="przygotuj_demo"):
        zasoby.sprawdz_kod()


def test_modul_modelu_ma_klasy():
    m = zasoby.modul_modelu()
    assert hasattr(m, "Delphi") and hasattr(m, "DelphiConfig")


def test_stale_zgodne_z_etykietami_autorow():
    etyk = pd.read_csv(zasoby.sciezka_etykiet(), header=None, sep="\t")[0].tolist()
    assert len(etyk) == tokeny.LICZBA_TOKENOW
    assert etyk[tokeny.PADDING] == "Padding" and etyk[tokeny.BRAK_ZDARZENIA] == "No event"
    assert etyk[tokeny.KOBIETA] == "Female" and etyk[tokeny.MEZCZYZNA] == "Male"
    assert etyk[tokeny.SMIERC] == "Death"
    assert etyk[tokeny.PIERWSZA_CHOROBA].startswith("A00")


def test_ignorowane_jak_w_configu_autorow():
    cfg = runpy.run_path(str(zasoby.sprawdz_kod() / "config" / "train_delphi_demo.py"))
    assert tuple(cfg["ignore_tokens"]) == tokeny.IGNOROWANE
    assert cfg["no_event_token_rate"] == tokeny.CO_ILE_LAT_BRAK_ZDARZENIA
    assert cfg["vocab_size"] == tokeny.LICZBA_TOKENOW


def test_maly_model_dziala(maly_model):
    import torch
    x = torch.tensor([[tokeny.KOBIETA, 215, tokeny.BRAK_ZDARZENIA]])
    a = torch.tensor([[0.0, 50 * 365.25, 60 * 365.25]])
    logity, _, _ = maly_model(x, a)
    assert logity.shape == (1, 3, tokeny.LICZBA_TOKENOW)


@pytest.mark.siec
def test_przypiety_commit():
    sciezka = zasoby.sklonuj()
    head = subprocess.run(["git", "-C", str(sciezka), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == zasoby.COMMIT
