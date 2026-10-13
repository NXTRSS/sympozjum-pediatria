"""Przed prelekcją (z internetem): kod autorów, trening na danych syntetycznych, wyniki do trybu offline.

uv run python scripts/przygotuj_demo.py                 # pełne: klon + polska ICD-10 + trening 2500 kroków + wyniki
uv run python scripts/przygotuj_demo.py --kroki 5000    # wariant z wykresem przeuczenia (~36–50 min)
uv run python scripts/przygotuj_demo.py --tylko-wyniki  # model już jest: tylko przelicz wyniki

Buduje też cache fontów matplotlib (pierwszy import trwa kilka–kilkanaście sekund; lepiej teraz niż na sali).
Uwaga: N=1000 losowań to szczyt ok. 10 GB RAM (generate autorów trzyma uwagę wszystkich warstw);
nie uruchamiaj równolegle z treningiem ani innym ciężkim procesem.
"""
import argparse
import json
import shutil
import time

from delphi_demo import icd10_pl, pamiec, prognoza, przyklady, trening, zasoby
from delphi_demo.slownik import parsuj_historie, wczytaj_slownik

parser = argparse.ArgumentParser()
parser.add_argument("--kroki", type=int, default=2500)   # D5: ten sam najlepszy model co 5000, 2× szybciej
parser.add_argument("--urzadzenie", default="mps")
parser.add_argument("--n-probek", type=int, default=1000)
parser.add_argument("--tylko-wyniki", action="store_true")
args = parser.parse_args()

if not args.tylko_wyniki:
    print("Kod autorów:", zasoby.sklonuj())
    print("Polskie nazwy ICD-10 (Centrum e-Zdrowia):", icd10_pl.przygotuj(), "kategorii")
    if not trening.SCIEZKA_MODELU.exists():
        print(f"Trening {args.kroki} kroków na {args.urzadzenie} (M1 Pro: ok. {'36' if args.kroki >= 5000 else '18'} min)…")
        ckpt = trening.trenuj(trening.KATALOG_TRENINGU, kroki=args.kroki, urzadzenie=args.urzadzenie)
        trening.odchudz_checkpoint(ckpt)
        czas = json.loads((trening.KATALOG_TRENINGU / "czas.json").read_text())["czas_s"]
        print(f"Trening: {czas / 60:.1f} min → {trening.SCIEZKA_MODELU}")

model = trening.zaladuj_model()
sl = wczytaj_slownik()
for nazwa, p in przyklady.PRZYKLADY.items():
    start = time.time()
    historia = parsuj_historie(p["historia"], sl)
    cele = [sl.kod_do_tokenu[k] for k in (przyklady.CELE_ASTMA if "astma" in nazwa else przyklady.CELE)]
    pamiec.zapisz_tabele(f"ranking_{nazwa}", prognoza.ranking(model, historia, p["plec"], p["wiek_teraz"]))
    pamiec.zapisz_tabele(f"krzywe_{nazwa}", prognoza.krzywe_ryzyka(
        model, historia, p["plec"], p["wiek_teraz"], cele, n_probek=args.n_probek))
    pamiec.zapisz_kroki(f"slad_{nazwa}", prognoza.generuj(model, historia, p["plec"], p["wiek_teraz"]))
    print(f"{nazwa}: {time.time() - start:.0f} s")
pamiec.zapisz_json("krzywe_info", {"n_probek": args.n_probek})   # notebook drukuje „(z pamięci, N=…)”
log = trening.KATALOG_TRENINGU / "trening.log"
if log.exists():
    shutil.copy(log, pamiec.KATALOG_WYNIKOW / "trening.log")
    shutil.copy(trening.KATALOG_TRENINGU / "czas.json", pamiec.KATALOG_WYNIKOW / "czas_treningu.json")
print("Gotowe. Notebook działa teraz bez internetu (TRYB = \"z_pamieci\").")
