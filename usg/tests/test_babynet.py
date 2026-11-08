import numpy as np
import pytest
import torch

from usg_masa import babynet as bn
from usg_masa.dane import Sweep


def _sweep(n=100):
    return Sweep(klatki=np.random.default_rng(0).integers(0, 255, (n, 120, 160), dtype=np.uint8),
                 odstep_mm=0.28, nazwa="sztuczny")


def test_klip_ksztalt_i_zakres():
    k = bn.klip(_sweep(), 0)
    assert k.shape == (1, 1, 16, 64, 64) and k.dtype == torch.float32
    assert -1.0 <= float(k.min()) and float(k.max()) <= 1.0 + 1e-6
    assert float(k.max()) == pytest.approx(1.0)


def test_klip_zachowuje_proporcje():
    k = bn.klip(_sweep(), 0)            # klatki 120×160 → 48×64 + po 8 wierszy dopełnienia u góry i dołu
    klatka = k[0, 0, 0]
    assert torch.all(klatka[:8] == -1.0) and torch.all(klatka[-8:] == -1.0)   # zera po normalizacji = −1
    assert not torch.all(klatka[8:56] == -1.0)


def test_klip_na_koncu_dopelnia():
    sw = _sweep(30)
    k = bn.klip(sw, 20)          # zostało 10 klatek → reszta to powtórzona ostatnia
    assert k.shape == (1, 1, 16, 64, 64)
    assert torch.equal(k[0, 0, -1], k[0, 0, -2])


def test_starty_klipow_tylko_pelne_klipy():
    st = bn.starty_klipow(840)
    assert bn.KROK_KLIPU == 76
    assert st[0] == 0 and len(st) == 11 and st[-1] == 760
    assert all(s + 75 <= 839 for s in st)
    assert all(b - a == bn.KROK_KLIPU for a, b in zip(st, st[1:]))


def test_starty_klipow_krotkie_nagranie():
    assert bn.starty_klipow(30) == [0]
    assert bn.starty_klipow(76) == [0]


def test_cechy_kolejnosc():
    c = bn.cechy_kliniczne(AC=300, GA=38)
    assert c.shape == (1, 6) and float(c[0, 0]) == 300 and float(c[0, 5]) == 38


def test_cechy_nieznana_nazwa():
    with pytest.raises(KeyError):
        bn.cechy_kliniczne(XYZ=1)


@pytest.mark.siec
def test_model_przechodzi_caly_sweep():
    model = bn.zbuduj_babynet()
    assert bn.liczba_parametrow(model) > 1_000_000
    srednia, oceny = bn.przewidz_caly_sweep(model, _sweep(200), bn.cechy_kliniczne())
    assert len(oceny) == len(bn.starty_klipow(200)) and np.isfinite(srednia)


@pytest.mark.siec
def test_wczytaj_wagi_odrzuca_niepasujace(tmp_path):
    model = bn.zbuduj_babynet()
    torch.save({"zly": torch.zeros(1)}, tmp_path / "w.pth")
    with pytest.raises(RuntimeError):
        bn.wczytaj_wagi(model, tmp_path / "w.pth")
