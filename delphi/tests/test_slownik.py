import pandas as pd
import pytest

from delphi_demo import icd10_pl, tokeny
from delphi_demo.slownik import BladHistorii, parsuj_historie, wczytaj_slownik
from delphi_demo.tokeny import Zdarzenie


@pytest.fixture(scope="module")
def sl():
    return wczytaj_slownik()


def test_kody_i_nazwy(sl):
    assert sl.kod_do_tokenu["E11"] == 215 and sl.kod_do_tokenu["I10"] == 499
    assert sl.kod_do_tokenu["O01"] == 936                 # duplikat w labels.csv: pierwszy wygrywa
    assert sl.kod(215) == "E11" and sl.nazwa(215) == "cukrzyca typu 2"
    assert sl.nazwa(tokeny.SMIERC) == "zgon"
    assert sl.nazwa(sl.kod_do_tokenu["A00"]) == "choroba zakaźna wywołana przez Vibrio cholerae [cholera]"
    assert len(sl.rozdzial) == len(sl.kolor) == tokeny.LICZBA_TOKENOW


def test_nazwy_oficjalne_cez(sl):
    """dane/icd10_pl.csv (Centrum e-Zdrowia, ICD-10-2008-PL, CC BY 4.0): 2050 kategorii, 1248 z 1254 kodów autorów."""
    if not icd10_pl.SCIEZKA_CSV.exists():
        pytest.skip("brak dane/icd10_pl.csv; najpierw: uv run python scripts/przygotuj_demo.py")
    assert len(sl.oficjalne) == 2050
    assert sl.nazwa(sl.kod_do_tokenu["E78"]) == "zaburzenia lipidowe"            # NAZWY_PL ma pierwszeństwo
    assert sl.nazwa(sl.kod_do_tokenu["K62"]) == "inne choroby odbytu i odbytnicy"
    assert sl.nazwa(sl.kod_do_tokenu["F00"]) == "otępienie w chorobie Alzheimera"   # bez odsyłacza „(G30.–†)"
    assert sl.nazwa(sl.kod_do_tokenu["B59"]) == "pneumocystoza"                    # bez „(J17.3*)"
    assert sl.nazwa(sl.kod_do_tokenu["A97"]) == "dengue (ang.)"                    # brak w wydaniu 2008 → angielski
    kody = {k for k in sl.kod_do_tokenu if len(k) == 3 and k[0].isalpha() and k[1:].isdigit()}
    assert len(kody) == 1254
    assert {k for k in kody if k not in sl.oficjalne} == {"A97", "B98", "C42", "C86", "G14", "K64"}
    assert not any(n.endswith(("†)", "*)")) for n in sl.oficjalne.values())


def test_bez_pliku_oficjalnych_nazw_dziala(tmp_path):
    sl2 = wczytaj_slownik(sciezka_nazw_pl=tmp_path / "nie_ma.csv")
    assert sl2.oficjalne == {} and sl2.nazwa(sl2.kod_do_tokenu["A00"]) == "cholera (ang.)"


def test_opisz_dokleja_kolumny(sl):
    df = sl.opisz(pd.DataFrame({"token": [215, tokeny.SMIERC], "p": [0.1, 0.2]}))
    assert list(df.columns[:3]) == ["kod", "nazwa", "token"]
    assert df["kod"].tolist() == ["E11", "Death"]


def test_parsuj_proste(sl):
    assert parsuj_historie("E11 50, I10 55", sl) == [Zdarzenie(215, 50.0), Zdarzenie(499, 55.0)]


def test_parsuj_po_polsku_przecinek_podkod_male_litery(sl):
    h = parsuj_historie("i10 w wieku 55,5; e11.9 50 lat", sl)
    assert h == [Zdarzenie(215, 50.0), Zdarzenie(499, 55.5)]   # posortowane wg wieku


@pytest.mark.parametrize("tekst", ["Smoking_high 40", "smoking_high 40", "E11 50, BMI_high 45",
                                   "ALCOHOL_HIGH 30", "BMI 45"])
def test_styl_zycia_odrzucony_z_wyjasnieniem(sl, tekst):
    """W train.bin i val.bin jest 0 tokenów 4–12: model nie zna stylu życia, więc go nie przyjmujemy."""
    with pytest.raises(BladHistorii, match="Stylu życia .* nie da się wpisać: w danych syntetycznych"):
        parsuj_historie(tekst, sl)


@pytest.mark.parametrize("tekst", ["E11 50 r.ż., I10 55 r.ż.", "E11 50 rż, I10 55 rż", "E11 50 r. ż.; I10 55 r. ż.",
                                   "E11 od 50 roku życia, I10 od 55 roku życia", "E11 od 50 r.ż., I10 od 55 r.ż.",
                                   "E11 od 50 lat, I10 od 55 lat", "E11 od 50. r.ż., I10 w wieku 55 lat",
                                   "E11 50r.ż., I10 55 R.Ż."])
def test_wiek_opisany_slowami(sl, tekst):
    assert parsuj_historie(tekst, sl) == [Zdarzenie(215, 50.0), Zdarzenie(499, 55.0)]


@pytest.mark.parametrize("tekst", ["E11 r.ż.", "E11 od r.ż. I10 55", "E11 od roku życia", "E11 od, I10 55"])
def test_slowa_bez_liczby_to_nadal_brak_wieku(sl, tekst):
    with pytest.raises(BladHistorii, match="Brak wieku przy E11"):
        parsuj_historie(tekst, sl)


def test_parsuj_duplikat_zostaje_najwczesniejszy(sl):
    assert parsuj_historie("E11 50, E11 45", sl) == [Zdarzenie(215, 45.0)]


def test_parsuj_pusty_tekst(sl):
    assert parsuj_historie("   ", sl) == []


def test_kod_spoza_slownika_czytelny_blad(sl):
    with pytest.raises(BladHistorii, match="R50, Z99"):
        parsuj_historie("E11 50, R50 52, Z99 53", sl)


def test_smieci_w_tekscie(sl):
    with pytest.raises(BladHistorii, match="cukrzyca"):
        parsuj_historie("cukrzyca 50", sl)


def test_wiek_poza_zakresem(sl):
    with pytest.raises(BladHistorii, match="0–110"):
        parsuj_historie("E11 150", sl)


@pytest.mark.parametrize("tekst", ["Death 70", "death 70", "zgon 70", "E11 50, zgon 70"])
def test_nie_mozna_wpisac_zgonu(sl, tekst):
    with pytest.raises(BladHistorii, match="Zgonu nie wpisuje"):      # dedykowany komunikat, nie echo wejścia
        parsuj_historie(tekst, sl)


@pytest.mark.parametrize("tekst", ["E11 I10", "E11, I10", "E11.9", "E111", "E1150", "E11 -5"])
def test_zly_wiek_to_blad_a_nie_po_cichu_zly_wiek(sl, tekst):
    with pytest.raises(BladHistorii):
        parsuj_historie(tekst, sl)


def test_brak_wieku_przy_kodzie_nazwany_w_komunikacie(sl):
    with pytest.raises(BladHistorii, match="Brak wieku przy E11"):
        parsuj_historie("E11 I10 55", sl)


@pytest.mark.parametrize("tekst", ["E11: 50", "E11 = 50", "E11 – 50", "E11 - 50", "E11, 50", "E11.9 w wieku 50 lat"])
def test_dozwolone_separatory(sl, tekst):
    assert parsuj_historie(tekst, sl) == [Zdarzenie(215, 50.0)]


def test_nazwa_zawsze_niepusta(sl):
    for t in range(tokeny.LICZBA_TOKENOW):
        n = sl.nazwa(t)
        assert n and not n.startswith(" "), (t, sl.etykiety[t], n)
    assert sl.nazwa(1142) == "nowotwór złośliwy o nieokreślonym umiejscowieniu"
    assert sl.nazwa(tokeny.PADDING) == "Padding (ang.)"
