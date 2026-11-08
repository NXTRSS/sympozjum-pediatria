import pytest

from delphi_demo import icd10_pl
from delphi_demo.slownik import parsuj_historie, wczytaj_slownik
from delphi_demo.wyszukiwarka import bez_ogonkow, dopisz, opis_historii, panel, szukaj


@pytest.fixture(scope="module")
def sl():
    return wczytaj_slownik()


def kody(wyniki):
    return [k for k, _ in wyniki]


def test_bez_ogonkow():
    assert bez_ogonkow("Zawał Serca, ŁÓDŹ") == "zawal serca, lodz"


def test_szukaj_po_polskiej_nazwie(sl):
    assert kody(szukaj("cukrzyca typu 2", sl))[0] == "E11"
    assert "E10" in kody(szukaj("cukrzyca", sl)) and "E11" in kody(szukaj("cukrzyca", sl))
    assert kody(szukaj("astma", sl))[0] == "J45"
    assert "I21" in kody(szukaj("zawał", sl))


def test_szukaj_bez_polskich_znakow_i_wielkosci_liter(sl):
    assert kody(szukaj("NADCISNIENIE", sl))[0] == "I10"
    assert kody(szukaj("zawal", sl)) == kody(szukaj("zawał", sl))


def test_szukaj_po_kodzie(sl):
    assert kody(szukaj("e11", sl))[0] == "E11"
    assert kody(szukaj("I10", sl))[0] == "I10"


def test_szukaj_kilka_slow_naraz(sl):
    wyniki = kody(szukaj("niewydol serc", sl))
    assert wyniki and wyniki[0] == "I50"


def test_szukaj_potoczne_nazwy_i_odmiana(sl):
    assert kody(szukaj("depresja", sl))[0] == "F32"
    assert kody(szukaj("autyzm", sl))[0] == "F84"
    assert kody(szukaj("celiakia", sl))[0] == "K90"
    assert kody(szukaj("mukowiscydoza", sl))[0] == "E84"
    assert kody(szukaj("ADHD", sl))[0] == "F90"
    assert kody(szukaj("otyłości", sl))[0] == "E66"
    assert "I50" in kody(szukaj("serce", sl, limit=50))
    assert "Q54" not in kody(szukaj("ospa", sl))          # angielskie „Hypospadias” nie przeszukujemy
    assert "A05" not in kody(szukaj("zawal", sl, limit=50))   # rdzeń „zawa” nie łapie „zawarte”


def test_synonimy_tylko_dla_kodow_modelu(sl):
    from delphi_demo.wyszukiwarka import SYNONIMY
    assert set(SYNONIMY) <= set(sl.kod_do_tokenu)


def test_szukaj_po_nazwie_oficjalnej_gdy_jest_skrot(sl):
    """E78 ma krótką nazwę „zaburzenia lipidowe”; oficjalna CeZ mówi o lipoproteinach."""
    if not icd10_pl.SCIEZKA_CSV.exists():
        pytest.skip("brak dane/icd10_pl.csv")
    assert "E78" in kody(szukaj("lipoprotein", sl))


def test_szukaj_tylko_kody_modelu_bez_tokenow_specjalnych(sl):
    for fraza in ("zgon", "palenie", "BMI", "kobieta"):
        assert all(len(k) == 3 and k[0].isalpha() and k[1:].isdigit() for k in kody(szukaj(fraza, sl)))
    assert szukaj("", sl) == [] and szukaj("   ", sl) == []
    assert szukaj("xyzxyz", sl) == []


def test_szukaj_limit(sl):
    assert len(szukaj("choroba", sl, limit=5)) == 5


def test_dopisz_i_opis(sl):
    h = dopisz("", "E11", 50)
    h = dopisz(h, "I10", 55.5)
    assert h == "E11 50, I10 55,5"
    assert [z.wiek_lat for z in parsuj_historie(h, sl)] == [50.0, 55.5]
    opis = opis_historii(h, sl)
    assert "cukrzyca typu 2" in opis and "nadciśnienie tętnicze" in opis and "55,5" in opis
    assert "UWAGA" in opis_historii("E11", sl)          # błąd parsera pokazany po polsku, bez wyjątku
    assert "pusta" in opis_historii("", sl)


def test_panel_dodaje_wybrana_chorobe(sl):
    wywolania = []
    p = panel(sl, lambda h, plec, wiek: wywolania.append((h, plec, wiek)), historia="E11 50",
              plec="K", wiek_teraz=60)
    p.szukaj.value = "astma"
    assert p.wyniki.options and p.wyniki.options[0][1] == "J45"
    p.wyniki.value = "J45"
    p.wiek.value = 6
    p.dodaj.click()
    assert p.historia.value == "E11 50, J45 6"
    assert "astma" in p.opis.value
    p.pokaz.click()
    assert wywolania == [("E11 50, J45 6", "K", 60)]
    p.wyczysc.click()
    assert p.historia.value == ""
