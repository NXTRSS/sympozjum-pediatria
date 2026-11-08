"""Aplikacja pokazowa: logika (usg_masa.aplikacja), zdarzenia i budowa interfejsu Gradio bez uruchamiania serwera."""
import importlib.util
import os
import socket
import types
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from usg_masa import aplikacja as A
from usg_masa import fuvai, model_sano, segmentacja as seg, zasoby

APP = Path(__file__).resolve().parents[1] / "app" / "demo_usg.py"
NR1 = A.NAGRANIA["Nr 1"]
# Zmienne, które app/demo_usg.py ustawia przy imporcie; po teście wracają do stanu sprzed niego.
ZMIENNE_APLIKACJI = ("GRADIO_ANALYTICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_OFFLINE")
LICZBA_WYJSC_ANALIZY = 13   # analiza, suwak, podgląd, podpis, wybrana, podpis, wykres, 3 karty, FUVAI, podpis, status


@pytest.fixture
def app(monkeypatch):
    """Moduł app/demo_usg.py; monkeypatch zapamiętuje zmienne środowiska i przywraca je po teście."""
    for zmienna in ZMIENNE_APLIKACJI:
        if zmienna in os.environ:
            monkeypatch.setenv(zmienna, os.environ[zmienna])
        else:
            monkeypatch.delenv(zmienna, raising=False)
    spec = importlib.util.spec_from_file_location("demo_usg", APP)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def _bez_postepu(*_a, **_k):
    return None


def _dane(etykieta: str) -> bool:
    return etykieta in A.dostepne_nagrania() and A.ma_wyniki_w_pamieci(A.NAGRANIA[etykieta])


def _dane_nr1() -> bool:
    return _dane("Nr 1")


def _pomin_bez(etykieta: str):
    return pytest.mark.skipif(not _dane(etykieta),
                              reason=f"brak nagrania {etykieta} lub wyników w pamiec/ (scripts/przygotuj_demo.py)")


def _zapisz_mha(sciezka: Path, klatki: np.ndarray, odstep: float = 0.28) -> Path:
    obraz = sitk.GetImageFromArray(klatki)
    obraz.SetSpacing((odstep, odstep, 1.0))
    sitk.WriteImage(obraz, str(sciezka))
    return sciezka


def test_liczby_po_polsku():
    assert A.po_polsku(34.42) == "34,4"
    assert A.po_polsku(0.9) == "0,9"
    assert A.gramy(3550.84) == "3 551"
    assert A.odmiana(1, "klatka", "klatki", "klatek") == "klatka"
    assert A.odmiana(22, "klatka", "klatki", "klatek") == "klatki"
    assert A.odmiana(12, "klatka", "klatki", "klatek") == "klatek"


def test_przeciagniecie():
    assert A.przeciagniecie(66, 840) == 1
    assert A.przeciagniecie(839, 840) == 6
    assert A.przeciagniecie(66, 128) is None   # wycinek nie ma przeciągnięć
    assert A.przeciagniecie(-1, 840) is None


def _fuvai_wynik(p_brzuch, klatka=4, maska=True):
    p = np.zeros((3, 4))
    p[:, fuvai.TLO] = 1.0
    p[1] = [1 - p_brzuch - 0.01, p_brzuch, 0.0, 0.01]
    m = np.zeros((40, 40), np.uint8)
    m[10:30, 8:32] = 1
    return fuvai.WynikFUVAI(np.array([0, 4, 8]), p, klatka if maska else -1, m if maska else None,
                            70.0 if maska else None, p_brzuch if maska else 0.0, 1.0)


def _nnunet_wynik(pole_w_klatce_4):
    return seg.WynikSweepu(np.array([0, 4, 8]), np.array([0, pole_w_klatce_4, 0]), np.array([0, 0, 500]),
                           8, np.ones((40, 40), np.uint8), 1.0)


def test_fuvai_bez_masy_gdy_niepewny_i_nnunet_nie_widzi_brzucha():
    o = A.ocen_fuvai(_fuvai_wynik(0.6), _nnunet_wynik(0), 1.0)
    assert o.znalazl and o.bez_masy and o.masa_g is None and o.pole_nnunet == 0
    assert "Masy nie podajemy" in o.komunikat
    assert o.rozklad[0][0] == "brzuch"


def test_fuvai_z_masa_gdy_nnunet_widzi_brzuch():
    o = A.ocen_fuvai(_fuvai_wynik(0.6), _nnunet_wynik(300), 5.0)   # 5 mm/px → AC w zakresie fizjologicznym
    assert o.znalazl and not o.bez_masy and o.pole_nnunet == 300
    assert o.masa_g is not None and o.ac_mm is not None


def test_fuvai_nie_znalazl():
    o = A.ocen_fuvai(_fuvai_wynik(0.2, maska=False), _nnunet_wynik(0), 1.0)
    assert not o.znalazl and o.masa_g is None and o.ac_mm is None
    assert "nie znalazł" in o.komunikat


def test_z_pamieci_bez_wynikow(tmp_path):
    obraz = sitk.GetImageFromArray(np.zeros((3, 20, 20), np.uint8))
    obraz.SetSpacing((0.28, 0.28, 1.0))
    plik = tmp_path / "nieznane-nagranie.mha"
    sitk.WriteImage(obraz, str(plik))
    with pytest.raises(A.BrakWynikowWPamieci):
        A.analizuj(plik, A.Z_PAMIECI)


def test_interfejs_buduje_sie_bez_uruchamiania(app):
    import gradio as gr
    demo = app.zbuduj_interfejs()
    assert isinstance(demo, gr.Blocks)
    assert app.motyw() is not None
    for karta in (app.karta_nnunet(None), app.karta_fuvai(None), app.karta_sano(None)):
        assert "karta" in karta
    assert "Nie jest wyrobem medycznym" in app.html_naglowek()
    assert "Zenodo 12697994" in app.html_podpis("x")
    zrodla = app.html_zrodla()
    # Atrybucja publiczna: cytowania prac i podziękowanie zespołom (bez prywatnych podziękowań imiennych).
    for fraza in ("ACOUSLIC-AI", "FUVAI", "TabAttention", "BabyNet",
                  "Płotka i in.", "Grzeszczyk", "MICCAI 2023", "zespołom Sano i WUM"):
        assert fraza in zrodla
    assert "Płotce" not in zrodla and "Grzeszczykowi" not in zrodla
    assert "obraz + pomiary lekarza" in zrodla and "prognoz" not in zrodla
    assert "masa w dniu porodu" in app.karta_sano(None) and "prognoz" not in app.html_naglowek()


@_pomin_bez("Nr 1")
def test_nr1_z_pamieci_te_same_liczby_co_notebook(app):
    a = A.analizuj("Nr 1", A.Z_PAMIECI)
    assert a.nnunet.najlepsza_klatka == 66
    assert round(a.pomiar.ac_mm) == 304
    assert round(a.masa_teraz_g, -1) == 2590
    assert A.po_polsku(a.pomiar.tydzien) == "34,4"
    assert a.plaszczyzna == "suboptymalna (skośna)"
    assert a.eksperci == [312.3, 293.8] and a.przeciagniecie == 1
    assert round(a.ekspert_przeciagniecia) == 312
    # FUVAI: klatka 464, p(brzuch) 0,77 < 0,9 i nnU-Net nie widzi tam brzucha → bez masy
    f = a.ocena_fuvai
    assert f.klatka == 464 and round(f.ac_mm) == 242 and f.bez_masy and f.masa_g is None
    # model Sano: 3 551 g, pełny segment
    assert round(a.sano.masa_g) == 3551 and not a.sano.dopelniony

    assert "3 551" in app.karta_sano(a) and "nie wie, kiedy nie wie" in app.karta_sano(a)
    assert "wychodzi poza dane" in app.karta_sano(a) and "prognoz" not in app.karta_sano(a)
    assert "304" in app.karta_nnunet(a) and "2 590" in app.karta_nnunet(a)
    assert "masy nie podajemy" in app.karta_fuvai(a)

    sweep = A.wczytaj_sweep("Nr 1")
    obraz, opis = A.obraz_klatki(sweep, 66, a)
    assert obraz.shape[1] == sweep.klatki.shape[2] and obraz.shape[2] == 3 and "nnU-Net" in opis
    assert A.wykres(a, 66).ndim == 3


# ── zgodność z notebookiem: pozostałe nagrania (z pamięci) ──────────────────

@_pomin_bez("Nr 2")
def test_nr2_z_pamieci_jak_notebook(app):
    a = A.analizuj("Nr 2", A.Z_PAMIECI)
    assert round(a.pomiar.ac_mm) == 257 and a.eksperci == [257.2]
    assert round(a.sano.masa_g) == 3855 and not a.sano.dopelniony
    f = a.ocena_fuvai
    assert f.klatka == 282 and f.bez_masy and f.masa_g is None   # p(brzuch) 0,53 i nnU-Net nie widzi brzucha
    assert A.gramy(3855) in app.karta_sano(a) and A.gramy(3855) + "*" not in app.karta_sano(a)
    assert "masy nie podajemy" in app.karta_fuvai(a)


@_pomin_bez("Nr 3")
def test_nr3_z_pamieci_jak_notebook(app):
    a = A.analizuj("Nr 3", A.Z_PAMIECI)
    assert round(a.pomiar.ac_mm) == 242 and a.eksperci == [217.0]
    assert round(a.sano.masa_g) == 3898 and not a.sano.dopelniony
    f = a.ocena_fuvai
    assert f.klatka == 50 == a.nnunet.najlepsza_klatka   # FUVAI i nnU-Net wybrały tę samą klatkę
    assert round(f.ac_mm) == 209 and not f.bez_masy and round(f.masa_g, -1) == 890
    karta = app.karta_fuvai(a)
    assert "209" in karta and f"ok. {A.gramy(890)} g" in karta and "masy nie podajemy" not in karta
    assert A.gramy(3898) in app.karta_sano(a)
    assert "Przekrój skośny może zawyżać AC" in app.karta_nnunet(a)


@_pomin_bez("Wycinek")
def test_wycinek_z_pamieci_jak_notebook(app):
    a = A.analizuj("Wycinek", A.Z_PAMIECI)
    assert a.n_klatek == 128 and a.przeciagniecie is None
    assert round(a.sano.masa_g) == 4000 and a.sano.dopelniony
    assert not a.ocena_fuvai.znalazl and a.ocena_fuvai.masa_g is None
    sano = app.karta_sano(a)
    assert A.gramy(4000) + "*" in sano and f"potrzeba {A.KLATEK_NA_SEGMENT}" in sano and A.KLATEK_NA_SEGMENT == 46
    assert "nie znalazł" in app.karta_fuvai(a)
    assert "przeciągnięcie" not in app.karta_nnunet(a)


# ── zdarzenia aplikacji ──────────────────────────────────────────────────────

def _sprawdz_wyjscia_analizy(app, wyjscia, tryb):
    import gradio as gr
    assert len(wyjscia) == LICZBA_WYJSC_ANALIZY
    a, suwak, podglad, podpis, wybrana, podpis_w, wykres, k1, k2, k3, obraz_f, podpis_f, status = wyjscia
    assert isinstance(a, A.Analiza) and a.tryb == tryb
    assert isinstance(suwak, gr.Slider)
    assert isinstance(podglad, np.ndarray) and podglad.ndim == 3
    assert wybrana is None or (isinstance(wybrana, np.ndarray) and wybrana.shape[2] == 3)
    assert isinstance(wykres, np.ndarray) and wykres.ndim == 3
    assert obraz_f is None or isinstance(obraz_f, np.ndarray)
    for html_ in (podpis, podpis_w, k1, k2, k3, podpis_f, status):
        assert isinstance(html_, str) and "Traceback" not in html_
    return a, status


def _zablokuj_siec_i_modele(monkeypatch):
    def _blokada(*_a, **_k):
        raise RuntimeError("sieć zablokowana w teście")

    def _bez_modelu(*_a, **_k):
        raise AssertionError("w trybie z pamięci nie wolno budować modeli")

    monkeypatch.setattr(socket, "socket", _blokada)
    monkeypatch.setattr(socket, "create_connection", _blokada)
    monkeypatch.setattr(zasoby, "sklonuj", _bez_modelu)            # Wycinek: bez gita
    for modul, nazwa in ((seg, "zaladuj_predyktor"), (fuvai, "zbuduj_fuvai"),
                         (model_sano, "zbuduj_model_sano"), (A, "_model")):
        monkeypatch.setattr(modul, nazwa, _bez_modelu)


@_pomin_bez("Nr 1")
def test_klik_z_pamieci_bez_sieci_i_bez_modeli(app, monkeypatch):
    _zablokuj_siec_i_modele(monkeypatch)
    wyjscia = app.analizuj_klik("Nr 1", A.Z_PAMIECI, progress=_bez_postepu)
    a, status = _sprawdz_wyjscia_analizy(app, wyjscia, A.Z_PAMIECI)
    assert "Z pamięci" in status and a.wbudowane
    assert "Zenodo 12697994" in wyjscia[3]


@_pomin_bez("Wycinek")
def test_wycinek_z_pamieci_bez_gita_i_bez_sieci(app, monkeypatch):
    _zablokuj_siec_i_modele(monkeypatch)
    assert "Wycinek" in A.dostepne_nagrania()
    assert A.sciezka_nagrania("Wycinek").exists()
    _sprawdz_wyjscia_analizy(app, app.analizuj_klik("Wycinek", A.Z_PAMIECI, progress=_bez_postepu), A.Z_PAMIECI)


def test_zly_plik_krotki_polski_komunikat(app, tmp_path):
    import gradio as gr
    smieci = tmp_path / "zepsuty.mha"
    smieci.write_bytes(b"to nie jest plik MetaImage \x00\x01\x02" * 10)
    plaski = _zapisz_mha(tmp_path / "plaski.mha", np.zeros((64, 64), np.uint8))           # 2D
    czarny = _zapisz_mha(tmp_path / "czarny.mha", np.zeros((4, 64, 64), np.uint8))        # same czarne klatki
    for plik in (smieci, plaski, czarny):
        with pytest.raises(gr.Error) as blad:
            app.wgraj_plik(str(plik))
        assert blad.value.message == app.BLAD_PLIKU.format(plik.name)
        assert "oczekiwany format ACOUSLIC-AI: .mha, 3D" in blad.value.message
        assert "SimpleITK" not in blad.value.message and "Exception" not in blad.value.message


def test_dobry_plik_wgrany_bez_atrybucji_acouslic(app, tmp_path):
    klatki = np.random.default_rng(0).integers(0, 255, (6, 64, 80), dtype=np.uint8)
    plik = _zapisz_mha(tmp_path / "moj.mha", klatki)
    wyjscia = app.wgraj_plik(str(plik))
    assert len(wyjscia) == 6 + 8
    assert wyjscia[0] == str(plik) and wyjscia[3].shape == (64, 80, 3)   # bez paska atrybucji ACOUSLIC-AI
    assert A.ATRYBUCJA_WGRANEGO in wyjscia[4] and "Zenodo" not in wyjscia[4]
    assert not A.wbudowane(str(plik)) and A.wbudowane("Nr 1")


def _sweep_testowy(tmp_path) -> Path:
    klatki = np.zeros((24, 64, 64), np.uint8)
    klatki[4:20] = np.random.default_rng(1).integers(1, 255, (16, 64, 64), dtype=np.uint8)
    return _zapisz_mha(tmp_path / "na-zywo-test.mha", klatki)


def _zaslepki_modeli(monkeypatch):
    """Na żywo bez prawdziwych sieci: budowanie modeli i przewidywanie zastąpione zaślepkami."""
    monkeypatch.setattr(A, "_MODELE", {})
    zbudowane = []
    monkeypatch.setattr(zasoby, "katalog_modelu_nnunet", lambda: Path("/nie/istnieje"))
    monkeypatch.setattr(seg, "zaladuj_predyktor", lambda *_a, **_k: zbudowane.append("nnunet") or "pred")
    monkeypatch.setattr(fuvai, "zbuduj_fuvai", lambda *_a, **_k: zbudowane.append("fuvai") or "fuvai")
    monkeypatch.setattr(model_sano, "zbuduj_model_sano", lambda *_a, **_k: zbudowane.append("sano") or "sano")

    def segmentuj(pred, sweep, co_ktora, po_paczce):
        assert pred == "pred"
        indeksy = np.arange(0, len(sweep.klatki), co_ktora)
        maska = np.zeros(sweep.klatki.shape[1:], np.uint8)
        maska[20:44, 16:48] = 1
        for p in range(0, len(indeksy), 2):
            numery = indeksy[p:p + 2]
            po_paczce(numery, np.stack([maska] * len(numery)), min(p + 2, len(indeksy)), len(indeksy))
        pole = np.array([600 if 4 <= i < 20 else 0 for i in indeksy])
        return seg.WynikSweepu(indeksy, pole, np.zeros_like(pole), 8, maska, 0.1)

    def fuvai_sweep(model, sweep, co_ktora):
        indeksy = np.arange(0, len(sweep.klatki), co_ktora)
        p = np.zeros((len(indeksy), 4))
        p[:, fuvai.TLO] = 1.0
        return fuvai.WynikFUVAI(indeksy, p, -1, None, None, 0.0, 0.05)

    monkeypatch.setattr(seg, "segmentuj_sweep_szybko", segmentuj)
    monkeypatch.setattr(fuvai, "analizuj_sweep", fuvai_sweep)
    monkeypatch.setattr(model_sano, "przewidz",
                        lambda model, sweep, W: model_sano.WynikSano(3700.0, [3700.0], [4], 16, True, 0.01))
    return zbudowane


def test_na_zywo_z_zaslepkami_modeli(app, monkeypatch, tmp_path):
    zbudowane = _zaslepki_modeli(monkeypatch)
    plik = _sweep_testowy(tmp_path)
    postep = []
    a = A.analizuj(plik, A.NA_ZYWO, postep=lambda u, opis: postep.append((u, opis)))
    assert zbudowane == ["nnunet", "fuvai", "sano"] and A.modele_wczytane()
    ulamki = [u for u, _ in postep]
    assert ulamki == sorted(ulamki) and ulamki[0] == 0.0 and postep[-1] == (1.0, "Gotowe")
    assert any("nnU-Net" in opis for _, opis in postep)
    assert {"wczytanie", "nnunet", "fuvai", "sano", "razem"} <= set(a.czasy)
    assert a.obrysy and set(a.obrysy) <= set(int(i) for i in a.nnunet.indeksy)
    assert not a.wbudowane and a.sano.masa_g == 3700.0 and not a.ocena_fuvai.znalazl

    wyjscia = app.analizuj_klik(str(plik), A.NA_ZYWO, progress=_bez_postepu)
    _, status = _sprawdz_wyjscia_analizy(app, wyjscia, A.NA_ZYWO)
    assert "Na żywo" in status
    karta = app.karta_sano(wyjscia[0])
    assert "każdemu z naszych nagrań" not in karta and "co 4. klatkę" in karta
    assert "Zenodo" not in wyjscia[3] and A.ATRYBUCJA_WGRANEGO in wyjscia[3]


def test_na_zywo_blad_krotki_polski_komunikat(app, monkeypatch, tmp_path):
    import gradio as gr
    _zaslepki_modeli(monkeypatch)

    def awaria(*_a, **_k):
        raise RuntimeError("MPS backend out of memory (techniczny tekst)")

    monkeypatch.setattr(seg, "segmentuj_sweep_szybko", awaria)
    with pytest.raises(gr.Error) as blad:
        app.analizuj_klik(str(_sweep_testowy(tmp_path)), A.NA_ZYWO, progress=_bez_postepu)
    assert blad.value.message == app.BLAD_NA_ZYWO and "MPS" not in blad.value.message
    assert "z pamięci" in blad.value.message


def test_rozgrzewanie_z_bledem_ustawia_polski_status(app, monkeypatch):
    monkeypatch.setattr(A, "_MODELE", {})

    def awaria():
        raise RuntimeError("brak wag")

    monkeypatch.setattr(A, "rozgrzej", awaria)
    status, _ = app.zmien_tryb(A.NA_ZYWO, None)
    assert "wczytuję modele" in status
    app._ROZGRZEWANIE["watek"].join(5)
    status, _ = app.sprawdz_modele(A.NA_ZYWO, None)
    assert status == app.html_status(None, app.BLAD_MODELI) and "brak wag" not in status


@_pomin_bez("Wycinek")
def test_status_na_zywo_mowi_o_wlasciwej_liczbie_klatek(app, monkeypatch):
    monkeypatch.setattr(A, "modele_wczytane", lambda: True)
    status, _ = app.zmien_tryb(A.NA_ZYWO, "Wycinek")
    assert "128 klatek" in status and "840" not in status
    assert round(A.szacowany_czas_na_zywo_s(840)) == 25   # zmierzone: ok. 25 s z wczytaniem modeli


def test_karta_fuvai_ac_poza_zakresem_i_brak_ac(app):
    nnunet = types.SimpleNamespace(najlepsza_klatka=10)
    poza = A.OcenaFUVAI(True, 30, 450.0, None, None, 0.95, 0.95, [("brzuch", 0.95)], 500, False,
                        "AC = 450 mm jest poza zakresem fizjologicznym — wynik niewiarygodny.")
    karta = app.karta_fuvai(types.SimpleNamespace(ocena_fuvai=poza, nnunet=nnunet))
    assert "poza zakresem fizjologicznym" in karta and "450" in karta
    bez_ac = A.OcenaFUVAI(True, 30, None, None, None, 0.95, 0.95, [("brzuch", 0.95)], 500, False,
                          "Maska zbyt mała lub postrzępiona — nie znaleziono wiarygodnego obrysu.")
    karta = app.karta_fuvai(types.SimpleNamespace(ocena_fuvai=bez_ac, nnunet=nnunet))
    assert "postrzępiona" in karta and "None" not in karta


def test_aplikacja_nie_zmienia_backendu_matplotlib():
    import subprocess
    import sys
    kod = ("import matplotlib; b = matplotlib.rcParams._get_backend_or_none(); "
           "import usg_masa.aplikacja; assert matplotlib.rcParams._get_backend_or_none() == b, b")
    subprocess.run([sys.executable, "-c", kod], check=True)
