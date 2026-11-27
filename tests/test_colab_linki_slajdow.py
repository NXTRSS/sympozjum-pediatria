"""Linki Colab na slajdach (HTML źródłowy)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELPHI = "https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/delphi/notebook/delphi_2m.ipynb"
USG = "https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/usg/notebook/usg_masa_plodu.ipynb"
LM = "https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/notebook/model_jezyka_choroby.ipynb"


def _czyta(sciezka: str) -> str:
    return (ROOT / sciezka).read_text(encoding="utf-8")


def test_delphi_linki():
    for plik in ("slides/n09_delphi_b.html", "slides/zd02_trajektoria.html"):
        html = _czyta(plik)
        assert DELPHI in html
        assert 'data-export-link' in html


def test_usg_linki():
    for plik in ("slides/n13b_demo_usg.html", "slides/zu06_aplikacja.html"):
        html = _czyta(plik)
        assert USG in html
        assert 'data-export-link' in html


def test_materialy_i_lm():
    html = _czyta("slides/n15_materialy.html")
    assert DELPHI in html and USG in html and LM in html