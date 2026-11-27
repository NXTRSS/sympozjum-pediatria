from delphi_demo import wzorce_sladu


def test_astma_od_dziecka_jest_w_paczce():
    kroki = wzorce_sladu.dla_klucza("astma_od_dziecka")
    assert kroki is not None
    assert len(kroki) >= 8
    assert kroki[0].wiek_lat > 44