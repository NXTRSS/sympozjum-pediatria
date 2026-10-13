from delphi_demo import pamiec, wzorce_sladu


def test_astma_od_dziecka_jest_w_paczce():
    kroki = pamiec.kroki_wbudowane("astma_od_dziecka")
    assert wzorce_sladu.dla_klucza("astma_od_dziecka") == kroki
    assert kroki is not None
    assert len(kroki) >= 8
    assert kroki[0].wiek_lat > 44