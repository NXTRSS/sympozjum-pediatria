import cv2
import numpy as np
import pytest
import torch

from usg_masa import fuvai
from usg_masa.dane import Sweep


def _elipsa(ksztalt=(120, 160), srodek=(80, 60), osie=(40, 30)):
    m = np.zeros(ksztalt, np.uint8)
    cv2.ellipse(m, srodek, osie, 0, 0, 360, 1, -1)
    return m


# ── stałe i przygotowanie klatki ─────────────────────────────────────────────

def test_stale():
    assert fuvai.KLASY == ("głowa", "brzuch", "kość udowa", "tło")
    assert fuvai.KLASY[fuvai.BRZUCH] == "brzuch"
    assert (fuvai.ROZMIAR, fuvai.PROG_MASKI, fuvai.PROG_ARTYKULU, fuvai.GORNE_PROCENT) == (224, 0.6, 0.9, 10)


def test_przygotuj_klatke_ksztalt_i_zakres():
    klatka = np.full((120, 160), 255, np.uint8)
    klatka[:, :80] = 51
    x = fuvai.przygotuj_klatke(klatka)
    assert x.shape == (1, 1, 224, 224) and x.dtype == torch.float32
    assert float(x.max()) == pytest.approx(1.0) and float(x.min()) == pytest.approx(0.2, abs=1e-6)


# ── postprocessing maski ─────────────────────────────────────────────────────

def test_maska_ksztalt_i_typ():
    prob = np.zeros((224, 224), np.float32)
    prob[60:160, 60:160] = 0.9
    maska = fuvai.maska_z_prawdopodobienstwa(prob, (120, 160))
    assert maska.shape == (120, 160) and maska.dtype == np.uint8
    assert set(np.unique(maska)) <= {0, 1} and maska.sum() > 0


def test_maska_prog_0_6():
    pod = np.zeros((224, 224), np.float32)
    pod[60:160, 60:160] = 0.55          # poniżej progu 0,6
    nad = pod + 0.1                      # 0,65 w obszarze
    nad[pod == 0] = 0
    assert fuvai.maska_z_prawdopodobienstwa(pod, (224, 224)).sum() == 0
    assert fuvai.maska_z_prawdopodobienstwa(nad, (224, 224)).sum() > 0


def test_maska_usuwa_drobne_plamki():
    prob = np.zeros((224, 224), np.float32)
    prob[100:104, 100:104] = 0.95        # 4×4 px: erozja krzyżem 5×5 to usuwa
    assert fuvai.maska_z_prawdopodobienstwa(prob, (224, 224)).sum() == 0


# ── zgodność elipsy ──────────────────────────────────────────────────────────

def test_zgodnosc_elipsy_elipsa():
    assert fuvai.zgodnosc_elipsy(_elipsa()) > 0.95


def test_zgodnosc_elipsy_kwadrat_gorsza():
    kwadrat = np.zeros((120, 160), np.uint8)
    kwadrat[30:90, 50:110] = 1
    z = fuvai.zgodnosc_elipsy(kwadrat)
    assert 0 < z < fuvai.zgodnosc_elipsy(_elipsa()) and z < 0.99


def test_zgodnosc_elipsy_pusta():
    assert fuvai.zgodnosc_elipsy(np.zeros((50, 50), np.uint8)) == 0.0


# ── wybór klatki ─────────────────────────────────────────────────────────────

def _maski(slownik, ksztalt=(120, 160)):
    return lambda poz: slownik.get(poz, np.zeros(ksztalt, np.uint8))


def _p_klas(p_brzuch):
    """(n,) p(brzuch) → (n, 4): reszta rozdzielona po równo na głowę, kość udową i tło.
    Przy p(brzuch) > 0,25 brzuch jest najbardziej prawdopodobną klasą."""
    p_brzuch = np.asarray(p_brzuch, float)
    reszta = (1 - p_brzuch) / 3
    return np.stack([reszta, p_brzuch, reszta, reszta], axis=1)


def test_wybor_bierze_gorne_10_procent():
    n = 20
    indeksy = np.arange(n) * 2
    p = np.linspace(0.0, 0.5, n)              # górne 10% = 2 najwyższe (poz. 18, 19)
    ocenione = np.ones(n, bool)
    wolane = []

    def maska_dla(poz):
        wolane.append(poz)
        return _elipsa()

    klatka, maska, ac, pb = fuvai.wybierz_klatke(indeksy, _p_klas(p), ocenione, maska_dla, 1.0)
    assert set(wolane) <= {18, 19}
    assert klatka in (36, 38) and maska is not None and ac is not None
    assert pb == pytest.approx(p[indeksy.tolist().index(klatka)])


def test_wybor_odrzuca_maske_dotykajaca_krawedzi():
    n = 10
    indeksy = np.arange(n)
    p = np.zeros(n)
    p[7], p[3] = 0.9, 0.85                     # kandydaci: 7 (maska uciętą) i 3 (dobra); górne 10% z 10 → tylko 7
    ocenione = np.ones(n, bool)
    ucieta = np.zeros((120, 160), np.uint8)
    ucieta[:, :60] = 1
    # przy 10% kandydatem jest tylko poz. 7 → brak wyniku
    klatka, maska, ac, _ = fuvai.wybierz_klatke(indeksy, _p_klas(p), ocenione, _maski({7: ucieta, 3: _elipsa()}), 1.0)
    assert (klatka, maska, ac) == (-1, None, None)
    # przy dwóch kandydatach (górne 20%) wygrywa dobra maska z poz. 3
    klatka, maska, ac, pb = fuvai.wybierz_klatke(indeksy, _p_klas(p), ocenione, _maski({7: ucieta, 3: _elipsa()}), 1.0,
                                                 gorne_procent=20)
    assert klatka == 3 and ac is not None and pb == pytest.approx(0.85)


def test_wybor_brak_kandydatow():
    indeksy = np.arange(5)
    klatka, maska, ac, pb = fuvai.wybierz_klatke(indeksy, _p_klas(np.full(5, 0.5)), np.ones(5, bool),
                                                 _maski({}), 1.0)
    assert (klatka, maska, ac, pb) == (-1, None, None, 0.0)


def test_wybor_bez_ocenionych_klatek():
    klatka, maska, ac, _ = fuvai.wybierz_klatke(np.arange(3), _p_klas(np.zeros(3)), np.zeros(3, bool), _maski({}), 1.0)
    assert (klatka, maska, ac) == (-1, None, None)


def test_wybor_kwantyl_liczony_tylko_z_ocenionych():
    # czarne klatki (nieocenione, p=0) nie mogą obniżać progu górnych 10%
    n = 10
    p = np.array([0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ocenione = np.array([False] * 5 + [True] * 5)
    wolane = []

    def maska_dla(poz):
        wolane.append(poz)
        return _elipsa()

    fuvai.wybierz_klatke(np.arange(n), _p_klas(p), ocenione, maska_dla, 1.0)
    assert wolane == [9]


def test_wybor_wynik_to_srednia_p_i_zgodnosci():
    # słabszy p, ale lepsza elipsa vs mocniejszy p i kwadrat: wygrywa wyższa średnia
    kwadrat = np.zeros((120, 160), np.uint8)
    kwadrat[30:90, 50:110] = 1
    indeksy = np.arange(4)
    p = np.array([0.0, 0.0, 0.50, 0.51])
    klatka, _, _, _ = fuvai.wybierz_klatke(indeksy, _p_klas(p), np.ones(4, bool), _maski({2: _elipsa(), 3: kwadrat}), 1.0,
                                           gorne_procent=50)
    assert klatka == 2


# klatka 3: najwyższe p(brzuch), ale model uważa ją za głowę; klatka 2: niższe p(brzuch), ale brzuch wygrywa
_GLOWA_I_BRZUCH = np.array([[0.00, 0.00, 0.00, 1.00],
                            [0.00, 0.00, 0.00, 1.00],
                            [0.25, 0.30, 0.20, 0.25],
                            [0.60, 0.35, 0.00, 0.05]])


def test_wybor_odrzuca_klatke_gdzie_wygrywa_glowa():
    wolane = []

    def maska_dla(poz):
        wolane.append(poz)
        return _elipsa()

    # górne 10% z 4 klatek → tylko klatka 3 (głowa) → nie znaleziono, maski nawet nie liczymy
    wynik = fuvai.wybierz_klatke(np.arange(4), _GLOWA_I_BRZUCH, np.ones(4, bool), maska_dla, 1.0)
    assert wynik == (-1, None, None, 0.0) and wolane == []


def test_wybor_nizsze_p_ale_brzuch_najbardziej_prawdopodobny_wygrywa():
    klatka, maska, ac, pb = fuvai.wybierz_klatke(np.arange(4), _GLOWA_I_BRZUCH, np.ones(4, bool),
                                                 _maski({2: _elipsa(), 3: _elipsa()}), 1.0, gorne_procent=50)
    assert klatka == 2 and maska is not None and ac is not None and pb == pytest.approx(0.30)


# ── analizuj_sweep ze zmyślonym modelem ──────────────────────────────────────

class _Atrapa(torch.nn.Module):
    """Zamiast YNet: zapisuje, ile klatek dostała; brzuch = jasna elipsa tylko dla klatek o średniej > 100."""

    def __init__(self):
        super().__init__()
        self.waga = torch.nn.Parameter(torch.zeros(1))
        self.paczki = []

    def forward(self, x):
        self.paczki.append(tuple(x.shape))
        b = x.shape[0]
        fc = torch.zeros(b, 4)
        fc[:, 1] = 3.0 * (x.mean(dim=(1, 2, 3, 4)) > 0.2).float()      # p(brzuch) rośnie dla jasnych klatek
        out = torch.zeros(b, 1, 224, 224)
        out[:, :, 60:170, 50:180] = 0.95
        return fc, out


def _sweep(n=12, ksztalt=(120, 160)):
    klatki = np.full((n, *ksztalt), 60, np.uint8)
    klatki[:, :, :] += np.arange(n, dtype=np.uint8)[:, None, None]
    klatki[[0, 1, 5]] = 0                                   # całkiem czarne
    return Sweep(klatki, 0.5, "atrapa")


def test_analizuj_sweep_czarne_nie_trafiaja_do_modelu():
    model = _Atrapa()
    wynik = fuvai.analizuj_sweep(model, _sweep(), co_ktora=1, paczka=4)
    assert sum(k[0] for k in model.paczki) == 12 - 3
    assert all(k[1:] == (1, 1, 224, 224) for k in model.paczki)        # T = 1
    assert wynik.p_klas.shape == (12, 4) and len(wynik.indeksy) == 12
    np.testing.assert_array_equal(wynik.p_klas[[0, 1, 5]], np.tile([0, 0, 0, 1.0], (3, 1)))
    np.testing.assert_allclose(wynik.p_klas.sum(axis=1), 1.0, atol=1e-5)
    assert wynik.czas_s > 0


def test_analizuj_sweep_co_ktora_i_wynik():
    model = _Atrapa()
    sweep = _sweep(n=40)
    wynik = fuvai.analizuj_sweep(model, sweep, co_ktora=2, paczka=8)
    np.testing.assert_array_equal(wynik.indeksy, np.arange(0, 40, 2))
    assert wynik.najlepsza_klatka in wynik.indeksy and wynik.najlepsza_klatka >= 0
    assert wynik.maska.shape == (120, 160) and wynik.ac_mm is not None and wynik.ac_mm > 0
    assert wynik.p_brzuch_najlepszej == pytest.approx(wynik.p_klas[wynik.indeksy == wynik.najlepsza_klatka, 1][0])


def test_analizuj_sweep_brak_brzucha():
    klatki = np.zeros((6, 50, 60), np.uint8)               # same czarne klatki
    wynik = fuvai.analizuj_sweep(_Atrapa(), Sweep(klatki, 1.0, "czarne"), co_ktora=1)
    assert wynik.najlepsza_klatka == -1 and wynik.maska is None and wynik.ac_mm is None


# ── zapis ────────────────────────────────────────────────────────────────────

def test_zapis_i_odczyt(tmp_path):
    w = fuvai.WynikFUVAI(indeksy=np.array([0, 2]), p_klas=np.array([[0, .7, 0, .3], [0, 0, 0, 1.]]),
                         najlepsza_klatka=0, maska=np.ones((3, 3), np.uint8), ac_mm=242.0,
                         p_brzuch_najlepszej=0.7, czas_s=1.5)
    fuvai.zapisz_wynik(w, tmp_path / "w.npz")
    w2 = fuvai.wczytaj_wynik(tmp_path / "w.npz")
    assert w2.najlepsza_klatka == 0 and w2.czas_s == 1.5 and w2.ac_mm == 242.0
    assert w2.p_brzuch_najlepszej == pytest.approx(0.7)
    np.testing.assert_array_equal(w2.maska, w.maska)
    np.testing.assert_allclose(w2.p_klas, w.p_klas)


def test_zapis_bez_maski(tmp_path):
    w = fuvai.WynikFUVAI(np.array([0]), np.array([[0, 0, 0, 1.]]), -1, None, None, 0.0, 0.1)
    fuvai.zapisz_wynik(w, tmp_path / "w.npz")
    w2 = fuvai.wczytaj_wynik(tmp_path / "w.npz")
    assert w2.maska is None and w2.ac_mm is None and w2.najlepsza_klatka == -1


# ── poprawka init_hidden ─────────────────────────────────────────────────────

def _falszywy_modul_autorow():
    """Kopia błędu autorów: stan początkowy powstaje na CPU, niezależnie od urządzenia wag."""
    import types

    class ConvLSTMCell(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 3, 3, padding=1)
            self.hidden_dim, self.height, self.width = 3, 4, 4

        def init_hidden(self, batch_size):
            return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                    torch.zeros(batch_size, self.hidden_dim, self.height, self.width))

    return types.SimpleNamespace(ConvLSTMCell=ConvLSTMCell)


def test_poprawka_init_hidden_idempotentna():
    modul = _falszywy_modul_autorow()
    fuvai._napraw_init_hidden(modul)
    pierwsza = modul.ConvLSTMCell.init_hidden
    fuvai._napraw_init_hidden(modul)             # drugi raz nie owija ponownie
    assert modul.ConvLSTMCell.init_hidden is pierwsza
    komorka = modul.ConvLSTMCell()
    h, c = komorka.init_hidden(2)
    assert h.shape == (2, 3, 4, 4) and c.shape == h.shape and h.device == komorka.conv.weight.device


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="brak MPS")
def test_poprawka_init_hidden_na_mps():
    modul = _falszywy_modul_autorow()
    fuvai._napraw_init_hidden(modul)
    komorka = modul.ConvLSTMCell().to("mps")
    h, c = komorka.init_hidden(2)
    assert h.device.type == "mps" and c.device.type == "mps"


# ── prawdziwy model (sieć + wagi + MPS) ──────────────────────────────────────

@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwe_wagi_i_strict_load():
    model = fuvai.zbuduj_fuvai(torch.device("cpu"))
    assert sum(p.numel() for p in model.parameters()) == 69_157_401
    assert not model.training


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwy_sweep_fragment():
    from usg_masa import dane
    sweep = dane.wczytaj_mha(dane.pobierz_przypadek_startowy())
    fragment = Sweep(sweep.klatki[:60], sweep.odstep_mm, "fragment")
    wynik = fuvai.analizuj_sweep(fuvai.zbuduj_fuvai(), fragment, co_ktora=10, paczka=8)
    assert len(wynik.indeksy) == 6 and wynik.p_klas.shape == (6, 4) and wynik.czas_s > 0
    assert wynik.najlepsza_klatka == -1 or wynik.maska.shape == sweep.klatki.shape[1:]
