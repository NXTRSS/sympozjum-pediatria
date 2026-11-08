import json

import numpy as np
import pytest
import torch

from usg_masa import model_sano as ms
from usg_masa.dane import Sweep
from usg_masa.segmentacja import WynikSweepu


def _chunks_autorow(L, n, skip):
    """Wzór `chunks` z src/video_data_loader.py autorów (przepisany do porównania, nie importowany)."""
    if skip > 0:
        return [L[x: x + (skip + 1) * (n - 1) + 1: skip + 1] for x in range(0, len(L), (skip + 1) * (n - 1) + 1)]
    return [L[x: x + n: skip + 1] for x in range(0, len(L), n)]


def _wynik_nnunet(indeksy, pole_opt, pole_subopt=None):
    indeksy = np.asarray(indeksy)
    pole_opt = np.asarray(pole_opt, np.int64)
    pole_subopt = np.zeros_like(pole_opt) if pole_subopt is None else np.asarray(pole_subopt, np.int64)
    return WynikSweepu(indeksy, pole_opt, pole_subopt, -1, None, 0.0)


# ── stałe ────────────────────────────────────────────────────────────────────

def test_stale():
    assert (ms.KLATKI, ms.SKIP, ms.ROZMIAR, ms.MAX_W, ms.PIKSEL_MM) == (16, 2, (128, 128), 1680, 0.2)
    assert ms.LICZBA_PARAMETROW == 33_147_976


# ── wybór klatek brzucha ─────────────────────────────────────────────────────

def test_klatki_brzucha_z_wyniku_co_ktorej():
    klatki = np.full((10, 4, 5), 50, np.uint8)
    sweep = Sweep(klatki, 0.28, "atrapa")
    # nnU-Net co 2. klatkę: 0, 2, 4, 6, 8; brzuch w 2 (opt) i 6 (subopt)
    wynik = _wynik_nnunet([0, 2, 4, 6, 8], [0, 30, 0, 0, 0], [0, 0, 0, 12, 0])
    assert ms.klatki_brzucha(sweep, wynik) == [2, 3, 6, 7]


def test_klatki_brzucha_pomija_czarne_i_zachowuje_kolejnosc():
    klatki = np.full((12, 4, 5), 50, np.uint8)
    klatki[[5, 9]] = 0                                       # całkiem czarne
    sweep = Sweep(klatki, 0.28, "atrapa")
    wynik = _wynik_nnunet([0, 4, 8], [10, 10, 10])           # co 4., brzuch wszędzie
    assert ms.klatki_brzucha(sweep, wynik) == [0, 1, 2, 3, 4, 6, 7, 8, 10, 11]


def test_klatki_brzucha_wzor_plan():
    """j jest „brzuchem”, gdy w klatce (j // k) * k pole > 0 (k = co_ktora)."""
    rng = np.random.default_rng(0)
    n, k = 50, 3
    klatki = np.full((n, 2, 2), 9, np.uint8)
    indeksy = np.arange(0, n, k)
    pole = rng.integers(0, 2, len(indeksy)) * 100
    sweep = Sweep(klatki, 0.28, "atrapa")
    oczekiwane = [j for j in range(n) if pole[(j // k)] > 0]
    assert ms.klatki_brzucha(sweep, _wynik_nnunet(indeksy, pole)) == oczekiwane


def test_klatki_brzucha_brak():
    sweep = Sweep(np.full((6, 3, 3), 7, np.uint8), 0.28, "atrapa")
    assert ms.klatki_brzucha(sweep, _wynik_nnunet([0, 2, 4], [0, 0, 0])) == []


# ── segmentacja na klipy ─────────────────────────────────────────────────────

@pytest.mark.parametrize("dlugosc", [0, 1, 15, 16, 45, 46, 47, 91, 92, 93, 137, 138, 300, 841])
def test_segmenty_jak_chunks_autorow(dlugosc):
    lista = list(range(1000, 1000 + dlugosc))
    oczekiwane = [c for c in _chunks_autorow(lista, 16, 2) if len(c) == 16]
    seg, dopelniony = ms.segmenty(lista)
    if oczekiwane:
        assert seg == oczekiwane and dopelniony is False
    for s in seg:
        assert len(s) <= ms.KLATKI


def test_segmenty_krok_46_co_trzecia():
    seg, dopelniony = ms.segmenty(list(range(100)))
    assert not dopelniony
    assert [s[0] for s in seg] == [0, 46]
    assert seg[0] == list(range(0, 46, 3)) and len(seg[0]) == 16


def test_segmenty_pomija_niepelne():
    seg, dopelniony = ms.segmenty(list(range(60)))       # 0..45 pełny, 46..59 niepełny
    assert seg == [list(range(0, 46, 3))] and not dopelniony


def test_segmenty_fallback_dopelnienia():
    lista = [3, 5, 8, 13, 21, 34, 55]                       # za mało na pełny segment
    seg, dopelniony = ms.segmenty(lista)
    assert dopelniony is True
    assert seg == [_chunks_autorow(lista, 16, 2)[0]] == [[3, 13, 55]]


def test_segmenty_pusta_lista():
    assert ms.segmenty([]) == ([], False)


# ── geometria i normalizacja ─────────────────────────────────────────────────

def test_przygotuj_klatke_geometria_acouslic():
    klatka = np.full((562, 744), 200, np.uint8)            # ACOUSLIC-AI: 744×562 px, 0,28 mm/px
    x = ms.przygotuj_klatke(klatka, 0.28)
    assert x.shape == (128, 128) and x.dtype == np.float32
    wiersze = np.flatnonzero(x.max(axis=1) > 0)
    kolumny = np.flatnonzero(x.max(axis=0) > 0)
    # 562·1,4/1680·128 = 59,9 → 59; 744·1,4/1680·128 = 79,4 → 79
    assert len(wiersze) == 59 and len(kolumny) == 79
    # wyśrodkowanie jak A.PadIfNeeded: góra (128−59)//2 = 34, lewo (128−79)//2 = 24
    assert (wiersze[0], wiersze[-1]) == (34, 92)
    assert (kolumny[0], kolumny[-1]) == (24, 102)
    assert float(x.max()) == pytest.approx(200.0)


def test_przygotuj_klatke_za_duza():
    with pytest.raises(ValueError):
        ms.przygotuj_klatke(np.ones((100, 3000), np.uint8), 0.2)


def test_klip_tensor_ksztalt_i_normalizacja():
    klatki = [np.full((562, 744), v, np.uint8) for v in range(10, 170, 10)]   # 16 klatek
    x = ms.klip_tensor(klatki, 0.28)
    assert x.shape == (1, 1, 16, 128, 128) and x.dtype == torch.float32
    assert float(x.max()) == pytest.approx(1.0)
    assert float(x.min()) == pytest.approx(-1.0)          # dopełnienie zerami → −1
    # klatka 0 (wartość 10, max 160): (10 − 80)/80
    assert float(x[0, 0, 0, 64, 64]) == pytest.approx((10 - 80) / 80)


def test_klip_tensor_dopelnia_zerami_na_koncu():
    klatki = [np.full((562, 744), 100, np.uint8)] * 3
    x = ms.klip_tensor(klatki, 0.28)
    assert x.shape == (1, 1, 16, 128, 128)
    assert torch.all(x[0, 0, 3:] == -1.0)                  # zera po normalizacji (x − 0,5m)/(0,5m)
    assert float(x[0, 0, 0, 64, 64]) == pytest.approx(1.0)


# ── przewidywanie ze zmyślonym modelem ───────────────────────────────────────

class _Atrapa(torch.nn.Module):
    """Zamiast 3D ResNet-18: zwraca 1000 · (średnia klipu + 2) i zapisuje kształty wejść."""

    def __init__(self):
        super().__init__()
        self.waga = torch.nn.Parameter(torch.zeros(1))
        self.wejscia = []

    def forward(self, x, tab):
        self.wejscia.append((tuple(x.shape), tuple(tab.shape)))
        return 1000.0 * (x.mean(dim=(1, 2, 3, 4)) + 2.0)


def _sweep(n, ksztalt=(56, 74)):
    rng = np.random.default_rng(1)
    klatki = rng.integers(1, 255, (n, *ksztalt), dtype=np.uint8)
    return Sweep(klatki, 0.28, "atrapa")


def test_przewidz_srednia_z_segmentow():
    sweep = _sweep(120)
    wynik_nn = _wynik_nnunet(np.arange(0, 120, 2), np.full(60, 50))   # wszystko „brzuchem”
    model = _Atrapa()
    w = ms.przewidz(model, sweep, wynik_nn)
    assert w.n_klatek_brzucha == 120 and not w.dopelniony
    assert w.starty == [0, 46]
    assert len(w.oceny_g) == 2
    assert w.masa_g == pytest.approx(np.mean(w.oceny_g))
    assert all(k[0][1:] == (1, 16, 128, 128) for k in model.wejscia)
    assert sum(k[0][0] for k in model.wejscia) == 2
    assert w.czas_s > 0


def test_przewidz_z_listy_mapuje_starty_na_klatki_surowe():
    sweep = _sweep(200)
    lista = list(range(100, 200, 2))                       # 50 klatek → jeden pełny segment
    w = ms.przewidz_z_listy(_Atrapa(), sweep, lista)
    assert w.starty == [100] and len(w.oceny_g) == 1 and w.n_klatek_brzucha == 50


def test_przewidz_fallback_dopelniony():
    sweep = _sweep(30)
    wynik_nn = _wynik_nnunet(np.arange(0, 30, 2), [0] * 10 + [9] * 5)  # brzuch tylko w 20..29
    w = ms.przewidz(_Atrapa(), sweep, wynik_nn)
    assert w.dopelniony and w.n_klatek_brzucha == 10 and w.starty == [20] and len(w.oceny_g) == 1


def test_przewidz_brak_brzucha():
    sweep = _sweep(10)
    model = _Atrapa()
    w = ms.przewidz(model, sweep, _wynik_nnunet(np.arange(0, 10, 2), np.zeros(5)))
    assert w.masa_g is None and w.oceny_g == [] and w.starty == [] and w.n_klatek_brzucha == 0
    assert model.wejscia == []


# ── zapis ────────────────────────────────────────────────────────────────────

def test_zapis_i_odczyt(tmp_path):
    w = ms.WynikSano(masa_g=3012.5, oceny_g=[3000.0, 3025.0], starty=[4, 50], n_klatek_brzucha=97,
                     dopelniony=False, czas_s=1.25)
    sciezka = tmp_path / "x_sano.json"
    ms.zapisz_wynik(w, sciezka)
    assert json.loads(sciezka.read_text())["masa_g"] == 3012.5
    assert ms.wczytaj_wynik(sciezka) == w


def test_zapis_bez_wyniku(tmp_path):
    w = ms.WynikSano(None, [], [], 0, False, 0.01)
    ms.zapisz_wynik(w, tmp_path / "pod" / "y_sano.json")
    assert ms.wczytaj_wynik(tmp_path / "pod" / "y_sano.json") == w


# ── prawdziwy model (publiczny checkpoint autorów) ───────────────────────────

@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwe_wagi_strict_load():
    model = ms.zbuduj_model_sano()
    assert sum(p.numel() for p in model.parameters()) == ms.LICZBA_PARAMETROW
    assert not model.training and next(model.parameters()).device.type == "cpu"


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwy_wycinek_startowy():
    from usg_masa import dane, segmentacja as seg, zasoby
    sweep = dane.wczytaj_mha(dane.pobierz_przypadek_startowy())
    pamiec = seg.KATALOG_PAMIECI / f"{sweep.nazwa}.npz"
    if pamiec.exists():
        wynik_nn = seg.wczytaj_wynik(pamiec)
    else:
        wynik_nn = seg.segmentuj_sweep_szybko(seg.zaladuj_predyktor(zasoby.katalog_modelu_nnunet()),
                                              sweep, co_ktora=2)
    w = ms.przewidz(ms.zbuduj_model_sano(), sweep, wynik_nn)
    assert w.n_klatek_brzucha > 0 and len(w.oceny_g) == len(w.starty) >= 1
    assert 500 < w.masa_g < 6000 and w.czas_s > 0
