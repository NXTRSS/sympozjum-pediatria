import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from collections import namedtuple

from matplotlib.figure import Figure

from delphi_demo import tokeny, wykresy
from delphi_demo.tokeny import Zdarzenie

Krok = namedtuple("Krok", "wiek_lat token kandydaci")   # jak prognoza.Krok (Task 4)


def nazwa(t):
    return {215: "cukrzyca typu 2", 499: "nadciśnienie", 505: "zawał", tokeny.SMIERC: "zgon"}.get(t, f"t{t}")


def test_os_zycia():
    h = [Zdarzenie(tokeny.KOBIETA, 0), Zdarzenie(215, 50), Zdarzenie(tokeny.BRAK_ZDARZENIA, 55)]
    fig = wykresy.os_zycia(h, nazwa, kolor=lambda t: "#ff0000", wygenerowane=[Zdarzenie(505, 63)])
    assert isinstance(fig, Figure)
    teksty = [t.get_text() for t in fig.axes[0].texts]
    assert "zawał  (63 l.)" in teksty and len(teksty) == 3     # „brak zdarzenia" nie jest rysowany


def test_ranking_slupki():
    df = pd.DataFrame({"nazwa": ["zawał", "zgon"], "ryzyko_1_rok": [0.02, 0.01]})
    fig = wykresy.ranking_slupki(df)
    assert isinstance(fig, Figure)
    assert any(wykresy.ATRYBUCJA in t.get_text() and t.get_fontsize() >= 8 for t in fig.texts)
    assert [t.get_text() for t in fig.axes[0].texts] == [" 1,0%", " 2,0%"]   # przecinek dziesiętny


def test_krzywe_pomija_nan_i_rysuje_porownanie():
    a = pd.DataFrame({"lata": [0, 1, 2], 505: [0, 0.1, 0.2], 215: [np.nan] * 3})
    b = pd.DataFrame({"lata": [0, 1, 2], 505: [0, 0.05, 0.1], 215: [0, 0.1, 0.2]})
    fig = wykresy.krzywe(a, nazwa, df_porownanie=b, etykiety=("z cukrzycą", "bez"))
    etyk = [l.get_label() for l in fig.axes[0].get_lines()]
    assert etyk == ["zawał (z cukrzycą)", "zawał (bez)"]


def test_krzywa_uczenia():
    ewal = pd.DataFrame({"krok": [250, 500], "trening": [80.0, 60.0], "walidacja": [82.0, 63.0]})
    fig = wykresy.krzywa_uczenia(ewal, czas_min=40)
    assert "40 min" in fig.axes[0].get_title()
    assert any("krok 500" in l.get_label() for l in fig.axes[0].get_lines())   # minimum walidacji


def test_slad_generowania():
    kroki = [Krok(60.0, tokeny.BRAK_ZDARZENIA, []),
             Krok(61.2, 505, [(tokeny.BRAK_ZDARZENIA, 0.4), (505, 0.3), (tokeny.SMIERC, 0.1), (499, 0.05)]),
             Krok(70.0, tokeny.SMIERC, [(tokeny.SMIERC, 0.5)])]
    slad = wykresy.slad_generowania(kroki, nazwa)
    linie = slad.splitlines()
    assert len(linie) == 2 and "zawał" in linie[0] and "30,0%" in linie[0] and "zgon" in linie[1]
    assert "wiek  61,2" in linie[0]                             # przecinek dziesiętny
    assert "t1 " not in slad and "40,0%" not in slad            # „brak zdarzenia" nie jest kandydatem na sali


def test_slad_pokazuje_p_wylosowanego():
    KrokP = namedtuple("KrokP", "wiek_lat token kandydaci p")  # jak prognoza.Krok z polem p
    kroki = [KrokP(61.2, 499, [(505, 0.3), (tokeny.SMIERC, 0.1)], 0.004)]
    linia = wykresy.slad_generowania(kroki, nazwa)
    assert "→ nadciśnienie (0,4%)" in linia and "zawał 30,0%" in linia
    rzadki = wykresy.slad_generowania([KrokP(61.2, 499, [(505, 0.3)], 0.0001)], nazwa)
    assert "→ nadciśnienie (<0,1%)" in rzadki                  # nie „0,0%": wylosowany nie miał szansy 0


def test_slad_skraca_dlugie_nazwy():
    kroki = [Krok(61.0, 7, [(7, 0.2)])]
    slad = wykresy.slad_generowania(kroki, lambda t: "x" * 60)
    assert "x" * 37 + "…" in slad and "x" * 39 not in slad
