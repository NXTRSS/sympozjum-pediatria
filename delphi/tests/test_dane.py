import numpy as np

from delphi_demo import dane, tokeny, zasoby
from delphi_demo.tokeny import Zdarzenie

MALE = np.array([[7, 0, 1], [7, 3000, 602], [7, 20000, 214],      # pacjent 7: K, astma w 8. r.ż.
                 [9, 0, 2], [9, 25000, 498]], dtype=np.uint32)     # pacjent 9: M, nadciśnienie


def test_pacjent_przesuwa_tokeny_i_liczy_lata():
    h = dane.pacjent(MALE, 7)
    assert h[0] == Zdarzenie(tokeny.KOBIETA, 0.0)
    assert h[1].token == 603 and abs(h[1].wiek_lat - 3000 / 365.25) < 1e-9


def test_statystyki():
    s = dane.statystyki(MALE)
    assert s["pacjentow"] == 2 and s["zdarzen"] == 5
    # Wiersze płci (tokeny 1 i 2 w pliku, wiek 0) nie liczą się: zostają 3 choroby (602, 214, 498),
    # z czego jedna (602, 3000 dni ≈ 8,2 r.ż.) przed 18. r.ż. → 1/3.
    assert abs(s["odsetek_zdarzen_przed_18"] - 1 / 3) < 1e-9


def test_pacjenci_z_astma_w_dziecinstwie():
    assert dane.pacjenci_z(MALE, 603, przed_wiekiem=18) == [7]
    assert dane.pacjenci_z(MALE, 603, przed_wiekiem=5) == []
    assert dane.pacjenci_z(MALE, 603, min_zdarzen=4) == []


def test_prawdziwe_dane_autorow():
    d = dane.wczytaj_bin(zasoby.katalog_danych() / "val.bin")
    s = dane.statystyki(d)
    assert d.shape == (181293, 3) and s["pacjentow"] == 7144
    assert 0.0120 < s["odsetek_zdarzen_przed_18"] < 0.0130        # zmierzone: 0,012472 (same choroby i zgon)
    assert len(dane.pacjenci_z(d, 603, przed_wiekiem=18, min_zdarzen=15)) > 10
