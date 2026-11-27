"""Zgodność zapisanych kroków generowania z wynikową nazwą (pod wykresy Gradio)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebook"))

from demo_model import TEMPERATURA, generuj, przebieg, ustaw_ziarna  # noqa: E402
from lm_utils import figura_prawdopodobienstw  # noqa: E402
from lm_wizualizacja import figura_histogramy_krokow, kroki_do_histogramu  # noqa: E402
from model_cache import wyczysc as wyczysc_cache  # noqa: E402


@pytest.fixture(scope="module")
def model_tokenizer():
    wyczysc_cache()
    wynik = przebieg(force_train=True, use_cache=True)
    return wynik["model"], wynik["tokenizer"]


def test_kroki_składają_pierwszą_nazwę(model_tokenizer):
    model, tokenizer = model_tokenizer
    ustaw_ziarna(42)
    nazwa, pelny, kroki = generuj(
        model, tokenizer, temperatura=TEMPERATURA, zbierz_kroki=True
    )
    assert pelny.startswith("%")
    assert pelny.endswith("!")
    z_krokow = "".join(k["znak"] for k in kroki)
    assert z_krokow.endswith("!")
    assert z_krokow[:-1] == pelny[1:-1]
    assert nazwa == pelny[1:-1]
    assert len(kroki) <= 30


def test_kroki_idx_zgadza_się_ze_znakiem(model_tokenizer):
    model, tokenizer = model_tokenizer
    ustaw_ziarna(7)
    _, _, kroki = generuj(model, tokenizer, temperatura=0.4, zbierz_kroki=True)
    for k in kroki:
        assert tokenizer.index_word[k["idx"]] == k["znak"]


def test_histogram_zbiorczy_dla_krokow(model_tokenizer):
    import matplotlib.pyplot as plt

    model, tokenizer = model_tokenizer
    ustaw_ziarna(3)
    _, _, kroki = generuj(model, tokenizer, zbierz_kroki=True)
    assert kroki
    hist = kroki_do_histogramu(kroki)
    fig = figura_histogramy_krokow(hist, tokenizer.index_word, len(tokenizer.index_word))
    assert len(fig.axes) >= len(kroki)
    plt.close(fig)


def test_figura_prawdopodobienstw_zamyka_sie_po_użyciu(model_tokenizer):
    import matplotlib.pyplot as plt

    _, tokenizer = model_tokenizer
    ustaw_ziarna(1)
    _, _, kroki = generuj(model_tokenizer[0], tokenizer, zbierz_kroki=True)
    assert kroki
    k = kroki[0]
    fig = figura_prawdopodobienstw(
        np.asarray(k["rozklad"]), k["krok"], tokenizer, k["prompt"]
    )
    assert fig.axes
    plt.close(fig)