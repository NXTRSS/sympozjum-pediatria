import pytest

from delphi_demo import pamiec, prognoza
from delphi_demo.generowanie_panel import _domyslne_dopasowanie, panel_generowania
from delphi_demo.prognoza import Krok
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


def test_panel_wczytuje_slad_z_tego_samego_klucza_co_jest(sl, tmp_path):
    """Zgodność z przygotuj_demo: zapis slad_{nazwa}, odczyt bez magicznego obcięcia prefiksu."""
    kroki = [Krok(45.0, 499, [(505, 0.3)], 0.01)]
    pamiec.zapisz_kroki("slad_astma_od_dziecka", kroki, tmp_path)
    odebrane = []

    def pokaz(h, pl, wiek, zi, temp, maks, ile, kroki_z_pamieci=None):
        odebrane.append(kroki_z_pamieci)
        return None

    domyslny = dict(historia="J45 6, J30 8", plec="M", wiek_teraz=45)
    p = panel_generowania(
        sl,
        pokaz,
        tryb="z_pamieci",
        klucz_pamieci="astma_od_dziecka",
        domyslny=domyslny,
        historia=domyslny["historia"],
        plec="M",
        wiek_teraz=45,
    )
    # Monkeypatch katalog pamięci panelu nie jest — testujemy wczytaj przez ten sam klucz co jest():
    import delphi_demo.generowanie_panel as gp

    orig_jest = pamiec.jest
    orig_wczytaj = pamiec.wczytaj_kroki

    def jest(nazwa, katalog=None):
        return orig_jest(nazwa, katalog or tmp_path)

    def wczytaj(nazwa, katalog=None):
        return orig_wczytaj(nazwa, katalog or tmp_path)

    gp.pamiec.jest = jest
    gp.pamiec.wczytaj_kroki = wczytaj
    try:
        p.uruchom_domyslnie()
    finally:
        gp.pamiec.jest = orig_jest
        gp.pamiec.wczytaj_kroki = orig_wczytaj
    assert len(odebrane) == 1
    assert odebrane[0] == kroki


def test_panel_blokuje_rownolegle(sl):
    licznik = {"n": 0}

    def pokaz(*_a, **_k):
        licznik["n"] += 1
        return None

    p = panel_generowania(sl, pokaz, historia="")
    p.uruchom_domyslnie()
    assert licznik["n"] == 1
    assert p.generuj.disabled is False