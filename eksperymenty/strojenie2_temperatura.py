"""Faza 2: doprecyzowanie finalistów — niskie temperatury (0.5–0.9) i 25/30/40 epok.

Faza 1 pokazała, że T=0.8 bije 1.0 i 1.2 pod względem wiarygodności, a val_loss
ma minimum przy 20–30 epokach. Sprawdzamy, jak nisko można zejść z temperaturą,
zanim model zacznie odtwarzać nazwy ze zbioru.

Uruchomienie:  uv run python eksperymenty/strojenie2_temperatura.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from strojenie import trenuj_i_oceniaj  # noqa: E402  (import odpala też przygotowanie danych)

KONFIGI = [
    {"units": 256, "emb": 128, "dropout": 0.3},
    {"units": 128, "emb": 64, "dropout": 0.3},
    {"units": 128, "emb": 64, "dropout": 0.0},
]
PUNKTY = [25, 30, 40]
TEMPERATURY = [0.5, 0.6, 0.7, 0.8, 0.9]

if __name__ == "__main__":
    trenuj_i_oceniaj(
        KONFIGI, PUNKTY, TEMPERATURY,
        ile=200,  # więcej próbek = stabilniejsze odsetki
        wyniki_plik=os.path.join(os.path.dirname(__file__), "wyniki_faza2.json"),
    )
