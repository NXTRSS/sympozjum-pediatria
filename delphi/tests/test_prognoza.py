import numpy as np
import pytest
import torch

from delphi_demo import prognoza, tokeny
from delphi_demo.tokeny import Zdarzenie

E11, I10, I21 = 215, 499, 505
HIST = [Zdarzenie(E11, 50.0), Zdarzenie(I10, 55.0)]


def test_wejscie_plec_kontrolne_i_teraz():
    idx, wiek = prognoza.przygotuj_wejscie(HIST, "K", 60)
    t, w = idx[0].tolist(), (wiek[0] / tokeny.DNI_W_ROKU).tolist()
    assert t[0] == tokeny.KOBIETA and w[0] == 0
    assert t[-1] == tokeny.BRAK_ZDARZENIA and w[-1] == pytest.approx(60)
    assert w == sorted(w)
    assert t.count(tokeny.BRAK_ZDARZENIA) == 12          # 5, 10, …, 55 oraz „teraz" = 60
    assert E11 in t and I10 in t


def test_wejscie_bledy():
    with pytest.raises(ValueError, match="po wieku"):
        prognoza.przygotuj_wejscie([Zdarzenie(E11, 70.0)], "K", 60)
    with pytest.raises(ValueError, match="plec"):
        prognoza.przygotuj_wejscie(HIST, "X", 60)


def test_ostrzezenia_dziecko_i_senior():
    assert "dzieci" in prognoza.ostrzezenia([], 8)[0]
    assert "80" in prognoza.ostrzezenia([], 85)[0]
    assert prognoza.ostrzezenia(HIST, 60) == []


def test_ranking(maly_model):
    df = prognoza.ranking(maly_model, HIST, "K", 60, top=10)
    assert len(df) == 10 and list(df.columns) == ["token", "p_nastepna", "ryzyko_1_rok", "ryzyko_5_lat"]
    zakazane = set(tokeny.IGNOROWANE) | {tokeny.BRAK_ZDARZENIA, E11, I10}
    assert not zakazane & set(df["token"])
    assert (np.diff(df["p_nastepna"]) <= 0).all()
    assert (df["ryzyko_5_lat"] >= df["ryzyko_1_rok"]).all() and df["p_nastepna"].sum() <= 1


def test_ranking_dziecko_dziala_z_ostrzezeniem(maly_model):
    h = [Zdarzenie(603, 3.0)]                                  # astma od 3. r.ż., chłopiec 8 lat
    assert len(prognoza.ranking(maly_model, h, "M", 8)) == 10
    assert prognoza.ostrzezenia(h, 8)


def test_ranking_pusta_historia(maly_model):
    assert len(prognoza.ranking(maly_model, [], "M", 45, top=3)) == 3


def test_generuj_reguly(maly_model):
    kroki = prognoza.generuj(maly_model, HIST, "K", 60, do_wieku=80, maks_krokow=30, ziarno=1)
    assert kroki
    tokeny_gen = [k.token for k in kroki]
    choroby = [t for t in tokeny_gen if t != tokeny.BRAK_ZDARZENIA]
    assert not set(tokeny_gen) & set(tokeny.IGNOROWANE)
    assert len(choroby) == len(set(choroby)) and not {E11, I10} & set(choroby)
    wieki = [k.wiek_lat for k in kroki]
    assert wieki == sorted(wieki) and wieki[0] >= 60 and wieki[-1] <= 80
    assert tokeny.SMIERC not in tokeny_gen[:-1]
    assert len(kroki[0].kandydaci) == 5


def test_generuj_powtarzalne(maly_model):
    a = prognoza.generuj(maly_model, HIST, "K", 60, maks_krokow=10, ziarno=7)
    b = prognoza.generuj(maly_model, HIST, "K", 60, maks_krokow=10, ziarno=7)
    assert [k.token for k in a] == [k.token for k in b]


def test_zgodnosc_z_generate_autorow(maly_model):
    idx, wiek = prognoza.przygotuj_wejscie(HIST, "K", 60)
    torch.manual_seed(123)
    gi, ga, _ = maly_model.generate(idx.clone(), wiek.clone(), max_new_tokens=1, max_age=1e9,
                                    termination_tokens=[tokeny.SMIERC])
    torch.manual_seed(123)
    kroki = prognoza.generuj(maly_model, HIST, "K", 60, do_wieku=1e6, maks_krokow=1, ziarno=None)
    assert kroki[0].token == int(gi[0, -1])
    assert kroki[0].wiek_lat * tokeny.DNI_W_ROKU == pytest.approx(float(ga[0, -1]), rel=1e-5)


def test_kandydaci_w_sladzie_jak_w_rankingu(maly_model):
    """Kandydaci liczeni bez „brak zdarzenia", więc pierwszy krok = top rankingu dla tego samego stanu."""
    kroki = prognoza.generuj(maly_model, HIST, "K", 60, maks_krokow=20, ziarno=3)
    df = prognoza.ranking(maly_model, HIST, "K", 60, top=5)
    assert [t for t, _ in kroki[0].kandydaci] == df["token"].tolist()
    assert [p for _, p in kroki[0].kandydaci] == pytest.approx(df["p_nastepna"].tolist(), rel=1e-5)
    for k in kroki:
        assert tokeny.BRAK_ZDARZENIA not in [t for t, _ in k.kandydaci]
        if k.token == tokeny.BRAK_ZDARZENIA:
            assert k.p is None
        else:
            assert 0 < k.p <= 1


def test_krzywe_ryzyka(maly_model):
    df = prognoza.krzywe_ryzyka(maly_model, HIST, "K", 60, cele=[I21, tokeny.SMIERC, E11],
                                horyzont_lat=5, n_probek=16, maks_krokow=8)
    assert df["lata"].tolist() == [0, 1, 2, 3, 4, 5]
    for cel in (I21, tokeny.SMIERC):
        kol = df[cel].to_numpy()
        assert (np.diff(kol) >= 0).all() and ((kol >= 0) & (kol <= 1)).all()
    assert df[E11].isna().all()                          # już w historii → NaN, nie mylące 0


def test_ryzyko_stale_tempo(maly_model):
    df = prognoza.ryzyko_stale_tempo(maly_model, HIST, "K", 60, cele=[I21, E11], horyzont_lat=3)
    kol = df[I21].to_numpy()
    assert kol[0] == 0 and (np.diff(kol) >= 0).all() and (kol <= 1).all()
    assert df[E11].isna().all()
