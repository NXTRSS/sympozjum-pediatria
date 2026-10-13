"""Dane syntetyczne autorów (train.bin / val.bin): pacjenci „wymyśleni" przez wytrenowany Delphi-2M."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from delphi_demo import tokeny
from delphi_demo.tokeny import Zdarzenie


def wczytaj_bin(sciezka: Path) -> np.ndarray:
    """Plik .bin autorów → tablica (n, 3) uint32: id pacjenta, wiek w dniach, token (przesunięty o −1)."""
    return np.fromfile(sciezka, dtype=np.uint32).reshape(-1, 3)


def pacjent(dane: np.ndarray, pid: int) -> list[Zdarzenie]:
    """Historia jednego pacjenta jako tokeny MODELU (w pliku są o 1 mniejsze)."""
    wiersze = dane[dane[:, 0] == pid]
    wiersze = wiersze[np.argsort(wiersze[:, 1], kind="stable")]
    return [Zdarzenie(int(t) + 1, float(d) / tokeny.DNI_W_ROKU) for _, d, t in wiersze]


def statystyki(dane: np.ndarray) -> dict:
    """Liczby do pokazania na sali. `pacjentow`, `zdarzen` i `zdarzen_na_pacjenta` liczą wszystkie wiersze
    (także płeć; tokenów stylu życia 4–12 w danych syntetycznych autorów nie ma ani jednego).
    `odsetek_zdarzen_przed_18` liczy TYLKO choroby i zgon (token modelu ≥ PIERWSZA_CHOROBA): płeć ma wiek 0,
    więc zawyżałaby ten odsetek. Mianownik to te same zdarzenia."""
    pid = dane[:, 0]
    n_pac = len(np.unique(pid))
    choroby = dane[dane[:, 2] >= tokeny.PIERWSZA_CHOROBA - 1]     # w pliku .bin tokeny są o 1 mniejsze
    przed_18 = (choroby[:, 1] < 18 * tokeny.DNI_W_ROKU).mean() if len(choroby) else 0.0
    return {
        "pacjentow": n_pac,
        "zdarzen": len(dane),
        "zdarzen_na_pacjenta": len(dane) / n_pac,
        "odsetek_zdarzen_przed_18": float(przed_18),
    }


def pacjenci_z(dane: np.ndarray, token: int, przed_wiekiem: float | None = None,
               min_zdarzen: int = 1) -> list[int]:
    """Id pacjentów, którzy mają dany token (modelu), opcjonalnie przed danym wiekiem.

    `min_zdarzen` liczy wszystkie wiersze pacjenta w pliku, także płeć (nie tylko choroby).
    """
    maska = dane[:, 2] == token - 1
    if przed_wiekiem is not None:
        maska &= dane[:, 1] < przed_wiekiem * tokeny.DNI_W_ROKU
    kandydaci = np.unique(dane[maska, 0])
    pid, liczby = np.unique(dane[:, 0], return_counts=True)
    duze = set(pid[liczby >= min_zdarzen].tolist())
    return [int(p) for p in kandydaci if int(p) in duze]
