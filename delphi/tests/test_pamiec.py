import numpy as np
import pandas as pd

from delphi_demo import pamiec, przyklady
from delphi_demo.prognoza import Krok
from delphi_demo.slownik import parsuj_historie, wczytaj_slownik


def test_tabela_tam_i_z_powrotem(tmp_path):
    df = pd.DataFrame({"lata": [0, 1], 505: [0.0, 0.1], 215: [np.nan, np.nan]})
    pamiec.zapisz_tabele("krzywe_x", df, tmp_path)
    wynik = pamiec.wczytaj_tabele("krzywe_x", tmp_path)
    assert list(wynik.columns) == ["lata", 505, 215]
    assert wynik[505].tolist() == [0.0, 0.1] and wynik[215].isna().all()


def test_kroki_tam_i_z_powrotem(tmp_path):
    kroki = [Krok(61.5, 505, [(505, 0.3), (1269, 0.1)], 0.3), Krok(62.0, 1, [(505, 0.3)])]
    pamiec.zapisz_kroki("slad", kroki, tmp_path)
    assert pamiec.wczytaj_kroki("slad", tmp_path) == kroki
    assert pamiec.jest("slad", tmp_path) and not pamiec.jest("brak", tmp_path)


def test_wczytaj_kroki_z_prefiksem_slad_lub_bez(tmp_path):
    kroki = [Krok(45.0, 499, [(505, 0.3)], 0.01)]
    pamiec.zapisz_kroki("slad_astma_od_dziecka", kroki, tmp_path)
    assert pamiec.jest("astma_od_dziecka", tmp_path)
    assert pamiec.wczytaj_kroki("astma_od_dziecka", tmp_path) == kroki


def test_kroki_wbudowane_astma():
    kroki = pamiec.kroki_wbudowane("astma_od_dziecka")
    assert kroki and len(kroki) >= 8
    assert pamiec.wczytaj_kroki_lub_wbudowane("astma_od_dziecka") == kroki


def test_zsynchronizuj_wzorce(tmp_path, monkeypatch):
    zrodlo = tmp_path / "wzorce"
    zrodlo.mkdir()
    (zrodlo / "slad_astma_od_dziecka.json").write_text('[{"wiek_lat":1,"token":1,"kandydaci":[[1,1]],"p":1}]')
    cel = tmp_path / "wyniki"
    monkeypatch.setattr(pamiec, "KATALOG_WZORCOW", zrodlo)
    monkeypatch.setattr(pamiec, "KATALOG_WYNIKOW", cel)
    skopiowane = pamiec.zsynchronizuj_wzorce()
    assert len(skopiowane) == 2  # slad_* + legacy astma_od_dziecka.json
    assert pamiec.jest("astma_od_dziecka")


def test_json_tam_i_z_powrotem(tmp_path):
    pamiec.zapisz_json("krzywe_info", {"n_probek": 1000}, tmp_path)
    assert pamiec.wczytaj_json("krzywe_info", tmp_path) == {"n_probek": 1000}


def test_przyklady_sa_poprawne():
    sl = wczytaj_slownik()
    for p in przyklady.PRZYKLADY.values():
        historia = parsuj_historie(p["historia"], sl)
        assert historia and all(z.wiek_lat <= p["wiek_teraz"] for z in historia)
        assert p["plec"] in ("K", "M")
    for kod in przyklady.CELE + przyklady.CELE_ASTMA:
        assert kod in sl.kod_do_tokenu
