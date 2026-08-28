"""Weryfikacja zwycięskiej konfiguracji na 3 ziarnach + lista nazw do oceny „na oko".

Uruchomienie:  uv run python eksperymenty/final.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from strojenie import trenuj_i_oceniaj  # noqa: E402

ZWYCIEZCA = {"units": 128, "emb": 64, "dropout": 0.3}
EPOKI = [30]
TEMPERATURY = [0.5, 0.6]

if __name__ == "__main__":
    wszystkie = []
    for seed in (7, 21, 99):
        print(f"\n─── ziarno {seed} " + "─" * 50)
        wszystkie += trenuj_i_oceniaj(
            [ZWYCIEZCA], EPOKI, TEMPERATURY, ile=200, seed=seed,
            wyniki_plik=os.path.join(os.path.dirname(__file__), f"final_seed{seed}.json"),
        )

    print("\n═══ PODSUMOWANIE (średnia z 3 ziaren) " + "═" * 30)
    for t in TEMPERATURY:
        g = [r for r in wszystkie if r["temperatura"] == t]
        print(f"T={t}: nowe {np.mean([r['nowe_%'] for r in g]):.1f}%  "
              f"świeże {np.mean([r['swieze_%'] for r in g]):.1f}%  "
              f"wiarygodność {np.mean([r['wiarygodnosc_%'] for r in g]):.1f}%  "
              f"końcówki {np.mean([r['konc_lac_%'] for r in g]):.1f}%  "
              f"val_loss {np.mean([r['val_loss'] for r in g]):.3f}")

    print("\n═══ PRZYKŁADOWE NAZWY (T=0.5, do oceny na oko) " + "═" * 20)
    for r in wszystkie:
        if r["temperatura"] == 0.5:
            for n in r["przyklady"]:
                print("  •", n)
