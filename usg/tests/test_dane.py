from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from usg_masa import dane


def _zapisz_mha(sciezka, n=5, h=20, w=30, spacing=(0.28, 0.28, 1.0)):
    arr = (np.arange(n * h * w) % 256).astype(np.uint8).reshape(n, h, w)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(sciezka), useCompression=True)
    return arr


def test_wczytaj_mha_ksztalt_i_odstep(tmp_path):
    arr = _zapisz_mha(tmp_path / "s.mha")
    s = dane.wczytaj_mha(tmp_path / "s.mha")
    assert s.klatki.shape == (5, 20, 30) and s.klatki.dtype == np.uint8
    np.testing.assert_array_equal(s.klatki, arr)
    assert s.odstep_mm == pytest.approx(0.28)
    assert s.nazwa == "s"


def test_wczytaj_mha_bez_odstepu_uzywa_acouslic(tmp_path):
    _zapisz_mha(tmp_path / "s.mha", spacing=(1.0, 1.0, 1.0))
    with pytest.warns(UserWarning, match="0,28"):
        s = dane.wczytaj_mha(tmp_path / "s.mha")
    assert s.odstep_mm == pytest.approx(dane.ODSTEP_ACOUSLIC_MM)


def test_wybierz_przypadki_deterministycznie():
    P = dane.PREFIKS_OBRAZOW
    nazwy = [P + "c.mha", "acouslic-ai-train-set/masks/stacked_fetal_ultrasound/x.mha",
             P + "a.mha", dane.CSV_OBWODOW, P + "b.mha"]
    assert dane.wybierz_przypadki(nazwy, 2) == [P + "a.mha", P + "b.mha"]


def test_obwody_referencyjne(tmp_path):
    csv = tmp_path / "ac.csv"
    csv.write_text("uuid,subject_id,sweep_1_ac_mm,sweep_2_ac_mm,sweep_3_ac_mm,"
                   "sweep_4_ac_mm,sweep_5_ac_mm,sweep_6_ac_mm\n"
                   "aaa,000,289.6,,,,,\nbbb,001,,,,,259.7,301.0\n")
    df = dane.wczytaj_obwody_referencyjne(csv)
    assert dane.ac_referencyjne(df, "bbb") == [259.7, 301.0]
    assert dane.ac_referencyjne(df, "aaa_1_0000") == [289.6]   # nazwa pliku z repo baseline
    assert dane.ac_referencyjne(df, "nieznany") == []


@pytest.mark.siec
@pytest.mark.wolny
def test_przypadek_startowy_prawdziwy():
    s = dane.wczytaj_mha(dane.pobierz_przypadek_startowy())
    assert s.klatki.shape == (128, 562, 744)
    assert 0.2 < s.odstep_mm < 0.4


@pytest.mark.siec
def test_lista_zip_zenodo_zawiera_obrazy_i_obwody():
    nazwy = dane.lista_plikow_zip()
    assert any(n.endswith(".mha") for n in nazwy)
    assert any(n.endswith(".csv") for n in nazwy)


# ── atrapy ZIP-a (bez sieci) ───────────────────────────────

class _Strumien:
    """Odczyt członka ZIP-a: każdy element listy to kawałek danych albo wyjątek do zgłoszenia."""

    def __init__(self, kawalki):
        self.kawalki = list(kawalki)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, rozmiar=-1):
        kawalek = self.kawalki.pop(0)
        if isinstance(kawalek, Exception):
            raise kawalek
        return kawalek


class _FalszywyZip:
    """Atrapa RemoteZip: indeks podaje `rozmiar`, a `open` zwraca strumień z `kawalki`."""

    def __init__(self, kawalki, rozmiar):
        self._kawalki = kawalki
        self._rozmiar = rozmiar

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def getinfo(self, n):
        return SimpleNamespace(file_size=self._rozmiar)

    def open(self, n):
        return _Strumien(self._kawalki)


def test_pobierz_z_zip_gotowe_pliki_bez_sieci(tmp_path, monkeypatch):
    nazwy = [dane.PREFIKS_OBRAZOW + "a.mha", dane.PREFIKS_OBRAZOW + "b.mha"]
    monkeypatch.setattr(dane, "KATALOG_DANYCH", tmp_path)
    for n in nazwy:
        cel = tmp_path / "zenodo" / n
        cel.parent.mkdir(parents=True, exist_ok=True)
        cel.write_bytes(b"x")
    monkeypatch.setattr(dane, "RemoteZip", lambda *a, **k: pytest.fail("nie powinno otwierać ZIP-a"))
    assert dane.pobierz_z_zip(nazwy) == [tmp_path / "zenodo" / n for n in nazwy]


def test_pobierz_z_zip_przerwane_nie_zostawia_pliku(tmp_path, monkeypatch):
    n = dane.PREFIKS_OBRAZOW + "a.mha"
    monkeypatch.setattr(dane, "KATALOG_DANYCH", tmp_path)
    monkeypatch.setattr(dane, "RemoteZip",
                        lambda *a, **k: _FalszywyZip([b"x" * 40, OSError("przerwane")], 100))
    with pytest.raises(OSError, match="przerwane"):
        dane.pobierz_z_zip([n])
    cel = tmp_path / "zenodo" / n
    assert not cel.exists()
    assert not dane._czesc(cel).exists()


def test_pobierz_z_zip_za_krotki_plik_odrzucony(tmp_path, monkeypatch):
    n = dane.PREFIKS_OBRAZOW + "a.mha"
    monkeypatch.setattr(dane, "KATALOG_DANYCH", tmp_path)
    monkeypatch.setattr(dane, "RemoteZip", lambda *a, **k: _FalszywyZip([b"x" * 40, b""], 100))
    with pytest.raises(ValueError, match="Niepełny"):
        dane.pobierz_z_zip([n])
    cel = tmp_path / "zenodo" / n
    assert not cel.exists()
    assert not dane._czesc(cel).exists()


def test_pobierz_z_zip_pozostalosc_czesc_wymusza_pobranie(tmp_path, monkeypatch):
    n = dane.PREFIKS_OBRAZOW + "a.mha"
    monkeypatch.setattr(dane, "KATALOG_DANYCH", tmp_path)
    cel = tmp_path / "zenodo" / n
    cel.parent.mkdir(parents=True, exist_ok=True)
    cel.write_bytes(b"x" * 40)                      # ucięty plik po dawnym pobraniu
    dane._czesc(cel).write_bytes(b"x" * 40)         # pozostałość przerwanego pobierania
    monkeypatch.setattr(dane, "RemoteZip",
                        lambda *a, **k: _FalszywyZip([b"y" * 100, b""], 100))
    assert dane.pobierz_z_zip([n]) == [cel]
    assert cel.read_bytes() == b"y" * 100
    assert not dane._czesc(cel).exists()
