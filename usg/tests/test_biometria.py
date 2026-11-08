import math

import cv2
import numpy as np
import pytest

from usg_masa import biometria as b


def _elipsa(h=400, w=500, polosie=(100, 60), srodek=(250, 200)):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, srodek, polosie, 0, 0, 360, 1, -1)
    return m


def test_obwod_elipsy_ramanujan():
    # półosie 100 i 60 px przy 0,5 mm/px → a=50 mm, b=30 mm → obwód ≈ 255,3 mm
    assert b.obwod_z_maski(_elipsa(), 0.5) == pytest.approx(255.3, rel=0.02)


def test_obwod_pusta_maska():
    assert b.obwod_z_maski(np.zeros((50, 50), np.uint8), 0.28) is None


def test_obwod_maly_obszar():
    m = np.zeros((50, 50), np.uint8)
    m[10:13, 10:13] = 1
    assert b.obwod_z_maski(m, 0.28) is None


def test_obwod_uciety_przy_krawedzi():
    # środek przesunięty tak, że prawa część elipsy wychodzi poza kadr (szerokość 500 px)
    m = _elipsa(srodek=(450, 200))
    assert b.obwod_z_maski(m, 0.5) is None
    o = b.oszacuj(m, 0.5)
    assert o.ac_mm is None and not o.wiarygodne and "nie znaleziono wiarygodnego obrysu" in o.komunikat


def test_obwod_dwa_kawalki_bierze_wiekszy():
    m = _elipsa()
    m[5:15, 5:15] = 1
    assert b.obwod_z_maski(m, 0.5) == pytest.approx(255.3, rel=0.02)


def test_masa_campbell():
    assert b.masa_campbell_g(300) == pytest.approx(2502, abs=2)
    assert b.masa_campbell_g(350) == pytest.approx(3495, abs=2)


def test_tydzien_hadlock():
    assert b.tydzien_hadlock(300) == pytest.approx(33.97, abs=0.01)


def test_poza_zakresem():
    assert not b.w_zakresie(80) and not b.w_zakresie(450) and b.w_zakresie(300)
    o = b.oszacuj(_elipsa(polosie=(20, 15)), 0.5)   # AC ≈ 55 mm → poza zakresem
    assert o.ac_mm is not None and not o.wiarygodne and "poza zakresem" in o.komunikat


def test_oszacuj_brak_maski():
    o = b.oszacuj(None, 0.28)
    assert o.ac_mm is None and not o.wiarygodne and "nie znaleziono" in o.komunikat


def test_oszacuj_typowy():
    o = b.oszacuj(_elipsa(polosie=(220, 160), h=600, w=700, srodek=(350, 300)), 0.28)
    assert o.wiarygodne and 250 < o.ac_mm < 350 and math.isfinite(o.masa_g)
    assert ", ok." in o.komunikat and "tc." in o.komunikat   # przecinek w zdaniu zostaje
