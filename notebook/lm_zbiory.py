"""Trzy zbiory do notatnika LM: choroby, leki, dinozaury — jeden przełącznik ZBIOR."""
from __future__ import annotations

import pandas as pd

from lm_sciezki import dopisz_pythonpath

dopisz_pythonpath()

from demo_model import (
    BATCH_SIZE,
    DROPOUT,
    EMBEDDING_DIM,
    GRU_UNITS,
    LICZBA_EPOK,
    TEMPO_UCZENIA,
    TEMPERATURA,
    wczytaj_nazwy,
)

DOZWOLONE = ("choroby", "leki", "dinozaury")

# Ten sam zestaw hiperparametrów co na sali (dobór pod choroby — README).
# Dla „leki” / „dinozaury” zwykle wystarcza; przy słabszych wynikach podnieś LICZBA_EPOK w komórce poniżej.
HIEPER = {
    "liczba_epok": LICZBA_EPOK,
    "learning_rate": TEMPO_UCZENIA,
    "temperatura": TEMPERATURA,
    "embedding_dim": EMBEDDING_DIM,
    "gru_units": GRU_UNITS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
}

META: dict[str, dict[str, str]] = {
    "choroby": {
        "emoji": "🩺",
        "opis": "łacińskie nazwy chorób (Wikipedia / Morbi)",
        "jezyk_opis": "łacińskich nazw chorób",
        "plik": "data/choroby.csv",
        "gen_etykieta": "choroby",
    },
    "leki": {
        "emoji": "💊",
        "opis": "nazwy handlowe z Rejestru Produktów Leczniczych (URPL)",
        "jezyk_opis": "nazw leków",
        "plik": "data/leki.csv",
        "gen_etykieta": "leku",
    },
    "dinozaury": {
        "emoji": "🦕",
        "opis": "nazwy gatunków dinozaurów (junosuarez/dinosaurs)",
        "jezyk_opis": "nazw dinozaurów",
        "plik": "data/dinozaury.csv",
        "gen_etykieta": "dinozaura",
    },
}


def sprawdz_zbior(zbior: str) -> str:
    z = zbior.strip().lower()
    if z not in DOZWOLONE:
        raise ValueError(f"ZBIOR musi być jednym z {DOZWOLONE}, jest: {zbior!r}")
    return z


def wczytaj_zbior(zbior: str = "choroby") -> tuple[list[str], pd.DataFrame, str, dict[str, str]]:
    """Nazwy (przetasowane, ziarno 42), ramka, ścieżka/URL źródła, meta do wypisów w notebooku."""
    zbior = sprawdz_zbior(zbior)
    nazwy, zrodlo, _ = wczytaj_nazwy(zbior)
    return nazwy, pd.DataFrame({"nazwa": nazwy}), zrodlo, META[zbior]