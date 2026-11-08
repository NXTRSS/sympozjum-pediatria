import numpy as np
import pytest

from usg_masa import segmentacja as s


def test_wybierz_najlepsza_najwiekszy_obszar():
    idx = np.array([0, 4, 8, 12])
    assert s.wybierz_najlepsza(idx, np.array([0, 50, 10, 0]), np.array([0, 0, 0, 80])) == 12
    assert s.wybierz_najlepsza(idx, np.array([0, 90, 10, 0]), np.array([0, 0, 0, 80])) == 4


def test_wybierz_najlepsza_brak():
    idx = np.arange(3)
    assert s.wybierz_najlepsza(idx, np.zeros(3), np.zeros(3)) == -1


def test_zapis_i_odczyt(tmp_path):
    w = s.WynikSweepu(indeksy=np.array([0, 4]), pole_opt=np.array([0, 9]),
                      pole_subopt=np.array([1, 0]), najlepsza_klatka=4,
                      maska=np.ones((3, 3), np.uint8), czas_s=1.5)
    s.zapisz_wynik(w, tmp_path / "w.npz")
    w2 = s.wczytaj_wynik(tmp_path / "w.npz")
    assert w2.najlepsza_klatka == 4 and w2.czas_s == 1.5
    np.testing.assert_array_equal(w2.maska, w.maska)


def test_zapis_bez_maski(tmp_path):
    w = s.WynikSweepu(np.array([0]), np.array([0]), np.array([0]), -1, None, 0.1)
    s.zapisz_wynik(w, tmp_path / "w.npz")
    assert s.wczytaj_wynik(tmp_path / "w.npz").maska is None


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwy_sweep_fragment():
    from usg_masa import dane, zasoby
    sweep = dane.wczytaj_mha(dane.pobierz_przypadek_startowy())
    pred = s.zaladuj_predyktor(zasoby.katalog_modelu_nnunet())
    fragment = dane.Sweep(sweep.klatki[:60], sweep.odstep_mm, "fragment")   # plik ma 128 klatek
    w = s.segmentuj_sweep(pred, fragment, co_ktora=10, paczka=8)
    assert len(w.indeksy) == len(range(0, len(fragment.klatki), 10)) and w.czas_s > 0
    assert w.najlepsza_klatka == -1 or w.maska.shape == sweep.klatki.shape[1:]


@pytest.mark.siec
@pytest.mark.wolny
def test_szybka_zgodna_z_pelna():
    from usg_masa import dane, zasoby
    sweep = dane.wczytaj_mha(dane.pobierz_przypadek_startowy())
    pred = s.zaladuj_predyktor(zasoby.katalog_modelu_nnunet())
    pelna = s.segmentuj_sweep(pred, sweep, co_ktora=2)
    szybka = s.segmentuj_sweep_szybko(pred, sweep, co_ktora=2)
    assert szybka.czas_s < pelna.czas_s
    assert (pelna.najlepsza_klatka == -1) == (szybka.najlepsza_klatka == -1)
    if pelna.maska is not None and szybka.maska is not None:
        przeciecie = np.logical_and(pelna.maska, szybka.maska).sum()
        dice = 2 * przeciecie / (pelna.maska.sum() + szybka.maska.sum())
        assert dice > 0.8


def _stub_wszystko_brzuchem():
    """Podstawka sieci: każda przekazana klatka dostaje wielką maskę klasy 1 (3, n, H, W)."""
    wywolane = []

    def licz(predyktor, klatki, odstep_mm):
        wywolane.append(klatki.copy())
        n, H, W = klatki.shape
        prob = np.zeros((3, n, H, W), np.float32)
        prob[0] = 0.1
        for j in range(n):
            # klatki o większej jasności dostają mniejszy obszar; czarne (gdyby dotarły) – największy
            bok = 3 if klatki[j].max() > 0 else 7
            prob[1, j, :bok, :bok] = 0.9
            prob[0, j, :bok, :bok] = 0.05
        return prob

    return licz, wywolane


def test_czarne_klatki_to_tlo_i_nie_trafiaja_do_sieci():
    klatki = np.zeros((8, 10, 10), np.uint8)
    klatki[[1, 4, 5]] = 100                      # reszta: całkiem czarna
    sweep = s.Sweep(klatki, 1.0, "t")
    licz, wywolane = _stub_wszystko_brzuchem()
    w = s._segmentuj(licz, None, sweep, co_ktora=1, paczka=4)
    assert len(w.indeksy) == 8                    # indeksy bez zmian
    assert all(w.pole_opt[i] == 0 and w.pole_subopt[i] == 0 for i in (0, 2, 3, 6, 7))
    assert all(w.pole_opt[i] == 9 for i in (1, 4, 5))
    assert sum(len(x) for x in wywolane) == 3 and all(x.max(axis=(1, 2)).min() > 0 for x in wywolane)
    assert w.najlepsza_klatka in (1, 4, 5) and w.maska is not None


def test_calkiem_czarny_sweep_nie_wola_sieci():
    sweep = s.Sweep(np.zeros((6, 10, 10), np.uint8), 1.0, "czarny")
    licz, wywolane = _stub_wszystko_brzuchem()
    w = s._segmentuj(licz, None, sweep, co_ktora=2, paczka=2)
    assert wywolane == []
    assert w.najlepsza_klatka == -1 and w.maska is None and not w.pole_opt.any()


def test_po_paczce_dostaje_numery_klatek_i_maski():
    klatki = np.zeros((8, 10, 10), np.uint8)
    klatki[[1, 4, 5]] = 100
    licz, _ = _stub_wszystko_brzuchem()
    wywolania = []
    s._segmentuj(licz, None, s.Sweep(klatki, 1.0, "t"), co_ktora=1, paczka=4,
                 po_paczce=lambda numery, maski, zrobione, wszystkie: wywolania.append(
                     (list(numery), maski.shape[0], zrobione, wszystkie)))
    assert wywolania == [([1], 1, 4, 8), ([4, 5], 2, 8, 8)]


def test_po_paczce_takze_dla_paczek_samych_czarnych_klatek():
    klatki = np.zeros((8, 10, 10), np.uint8)
    klatki[[5, 6]] = 100
    licz, _ = _stub_wszystko_brzuchem()
    wywolania = []
    s._segmentuj(licz, None, s.Sweep(klatki, 1.0, "t"), co_ktora=1, paczka=4,
                 po_paczce=lambda numery, maski, zrobione, wszystkie: wywolania.append(
                     (list(numery), maski.shape[0], zrobione, wszystkie)))
    assert wywolania == [([], 0, 4, 8), ([5, 6], 2, 8, 8)]   # postęp idzie równo, bez obrysów
