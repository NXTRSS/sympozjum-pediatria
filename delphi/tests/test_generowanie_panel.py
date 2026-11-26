import pytest

from delphi_demo import prognoza
from delphi_demo.generowanie_panel import _domyslne_dopasowanie, panel_generowania
from delphi_demo.tokeny import Zdarzenie

E11, I10 = 215, 505
HIST = [Zdarzenie(E11, 50), Zdarzenie(I10, 55)]


@pytest.fixture(scope="module")
def sl():
    from delphi_demo.slownik import wczytaj_slownik
    return wczytaj_slownik()


def test_domyslne_dopasowanie():
    d = dict(historia="L20 2, J45 6", plec="M", wiek_teraz=45)
    assert _domyslne_dopasowanie(d, "L20 2, J45 6", "M", 45, 0, 1.0, 80)
    assert not _domyslne_dopasowanie(d, "L20 2", "M", 45, 0, 1.0, 80)
    assert not _domyslne_dopasowanie(d, "L20 2, J45 6", "M", 45, 1, 1.0, 80)


def test_temperatura_1_nie_zmienia_wyniku(maly_model):
    a = prognoza.generuj(maly_model, HIST, "K", 60, maks_krokow=8, ziarno=4, temperatura=1.0)
    b = prognoza.generuj(maly_model, HIST, "K", 60, maks_krokow=8, ziarno=4, temperatura=1.0)
    assert [k.token for k in a] == [k.token for k in b]


def test_panel_generuj_wywoluje_callback(sl):
    wywolania = []

    def pokaz(h, pl, wiek, zi, temp, maks, ile, kroki_z_pamieci=None):
        wywolania.append((h, pl, wiek, zi, temp, maks, ile, kroki_z_pamieci))
        return None

    d = dict(historia="L20 2, J45 6", plec="M", wiek_teraz=45)
    p = panel_generowania(sl, pokaz, historia=d["historia"], plec="M", wiek_teraz=45)
    assert p.generuj.disabled is False
    p.uruchom_domyslnie()
    assert len(wywolania) == 1
    assert wywolania[0][0] == d["historia"]
    assert wywolania[0][2] == 45
    p.ziarno.value = 3
    p.generuj.click()
    assert len(wywolania) == 2
    assert wywolania[1][3] == 3


def test_panel_blokuje_rownolegle(sl):
    licznik = {"n": 0}

    def pokaz(*_a, **_k):
        licznik["n"] += 1
        return None

    p = panel_generowania(sl, pokaz, historia="")
    p.uruchom_domyslnie()
    assert licznik["n"] == 1
    assert p.generuj.disabled is False