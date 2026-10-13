"""Budżet czasu na sali (M1 Pro, CPU): komórka notebooka ≤ 30 s."""
import time

import pytest

from delphi_demo import przyklady, prognoza, trening
from delphi_demo.slownik import parsuj_historie, wczytaj_slownik


N_PROBEK_TEST = 100   # = N_PROBEK_NA_ZYWO w notebooku


@pytest.fixture(scope="module")
def model_i_slownik():
    if not trening.SCIEZKA_MODELU.exists():
        pytest.fail("najpierw: uv run python scripts/przygotuj_demo.py")
    return trening.zaladuj_model(), wczytaj_slownik()


@pytest.mark.wolny
def test_czasy_na_sali(model_i_slownik):
    model, sl = model_i_slownik
    p = przyklady.PRZYKLADY["cukrzyca"]
    h = parsuj_historie(p["historia"], sl)
    cele = [sl.kod_do_tokenu[k] for k in przyklady.CELE]
    t = time.time(); prognoza.ranking(model, h, "K", 60); t_rank = time.time() - t
    t = time.time(); prognoza.generuj(model, h, "K", 60); t_gen = time.time() - t
    t = time.time(); prognoza.krzywe_ryzyka(model, h, "K", 60, cele, n_probek=N_PROBEK_TEST); t_mc = time.time() - t
    print(f"ranking {t_rank:.3f} s, generuj {t_gen:.2f} s, krzywe({N_PROBEK_TEST}) {t_mc:.1f} s")
    assert t_rank < 0.5 and t_gen < 5 and t_mc < 15     # notebook liczy 2 pacjentów → ≤ 30 s


@pytest.mark.wolny
def test_model_nauczony(model_i_slownik):
    """Po treningu model ma sens: dla 60-latki z cukrzycą i nadciśnieniem w top-10 są choroby
    typowe dla wieku (a nie wady wrodzone i choroby noworodków, jak przy losowych wagach)."""
    model, sl = model_i_slownik
    h = parsuj_historie("E11 50, I10 55", sl)
    df = sl.opisz(prognoza.ranking(model, h, "K", 60))
    rozdzialy = {k[0] for k in df["kod"]}
    assert not rozdzialy & {"P", "Q", "O"}
    assert df["p_nastepna"].iloc[0] > 0.01
