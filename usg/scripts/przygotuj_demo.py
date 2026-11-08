"""Przed prelekcją: pobierz kod, wagi, nagrania i policz segmentację (tryb offline).

uv run python scripts/przygotuj_demo.py --ile-zenodo 3
"""
import argparse

from usg_masa import babynet, biometria, dane, fuvai, model_sano, segmentacja as seg, zasoby

parser = argparse.ArgumentParser()
parser.add_argument("--ile-zenodo", type=int, default=3)
parser.add_argument("--co-ktora", type=int, default=2)
args = parser.parse_args()

zasoby.sklonuj("BabyNetPlusPlus")
zasoby.sklonuj("FUVAI")
zasoby.sklonuj("TabAttention")
siec_sano = model_sano.zbuduj_model_sano()   # polski model masy w dniu porodu (CPU); wagi 133 MB
model_fuvai = fuvai.zbuduj_fuvai()   # pobiera wagi (277 MB), jeśli ich jeszcze nie ma
predyktor = seg.zaladuj_predyktor(zasoby.katalog_modelu_nnunet())
sciezki = [dane.pobierz_przypadek_startowy()]
if args.ile_zenodo:
    try:
        spis = dane.lista_plikow_zip()
    except Exception as blad:   # brak sieci: bierzemy nagrania, które już są na dysku
        zenodo = dane.KATALOG_DANYCH / "zenodo"
        nagrania = sorted(p for p in zenodo.rglob("*.mha") if "masks" not in p.parts)[:args.ile_zenodo]
        print(f"Brak dostępu do Zenodo ({type(blad).__name__}). Biorę nagrania już pobrane: {len(nagrania)}"
              + ("" if (zenodo / dane.CSV_OBWODOW).exists() else " (bez pliku CSV z AC ekspertów)"), flush=True)
    else:
        wybrane = dane.wybierz_przypadki(spis, args.ile_zenodo)
        *nagrania, _csv = dane.pobierz_z_zip(wybrane + [dane.CSV_OBWODOW])   # CSV z AC ekspertów zostaje w dane/zenodo/
    sciezki += nagrania
for sciezka in sciezki:
    sweep = dane.wczytaj_mha(sciezka)
    wynik = seg.segmentuj_sweep(predyktor, sweep, co_ktora=args.co_ktora)   # pełny nnU-Net: referencja
    seg.zapisz_wynik(wynik, seg.KATALOG_PAMIECI / f"{sweep.nazwa}.npz")
    print(f"{sweep.nazwa}: {len(sweep.klatki)} klatek, najlepsza {wynik.najlepsza_klatka}, "
          f"{wynik.czas_s:.0f} s", flush=True)
    wynik_f = fuvai.analizuj_sweep(model_fuvai, sweep, co_ktora=args.co_ktora)   # polski FUVAI: opcja pośrednia
    fuvai.zapisz_wynik(wynik_f, seg.KATALOG_PAMIECI / f"{sweep.nazwa}_fuvai.npz")
    ac_nnunet = biometria.obwod_z_maski(wynik.maska, sweep.odstep_mm) if wynik.maska is not None else None
    print(f"   FUVAI: "
          + (f"klatka {wynik_f.najlepsza_klatka}, p(brzuch) {wynik_f.p_brzuch_najlepszej:.2f}, AC {wynik_f.ac_mm:.0f} mm"
             if wynik_f.ac_mm is not None else "nie znaleziono klatki brzucha")
          + (f" (nnU-Net: {ac_nnunet:.0f} mm)" if ac_nnunet is not None else "")
          + f", {wynik_f.czas_s:.0f} s", flush=True)
    wynik_s = model_sano.przewidz(siec_sano, sweep, wynik)   # klatki brzucha wg nnU-Netu → segmenty → średnia
    model_sano.zapisz_wynik(wynik_s, seg.KATALOG_PAMIECI / f"{sweep.nazwa}_sano.json")
    print(f"   model Sano: "
          + (f"{wynik_s.masa_g:.0f} g (masa w dniu porodu), {len(wynik_s.oceny_g)} segm."
             + (" dopełniony" if wynik_s.dopelniony else "")
             if wynik_s.masa_g is not None else "brak klatek brzucha")
          + f", {wynik_s.n_klatek_brzucha} klatek brzucha, {wynik_s.czas_s:.1f} s", flush=True)
babynet.zbuduj_babynet()
print("Gotowe. Notebook może działać w trybie z_pamieci bez internetu.")
