"""Domyślne ślady generacji w pakiecie — panel działa bez plików w delphi/pamiec/."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from delphi_demo.prognoza import Krok

_NAZWA_PLIKU = {"astma_od_dziecka": "slad_astma_od_dziecka.json"}


def _z_surowego(surowe: list[dict]) -> list[Krok]:
    return [
        Krok(k["wiek_lat"], k["token"], [tuple(x) for x in k["kandydaci"]], k.get("p"))
        for k in surowe
    ]


@lru_cache(maxsize=8)
def dla_klucza(klucz: str) -> list[Krok] | None:
    nazwa = _NAZWA_PLIKU.get(klucz)
    if not nazwa:
        return None
    try:
        pakiet = resources.files("delphi_demo.wzorce_wyniki").joinpath(nazwa)
        return _z_surowego(json.loads(pakiet.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None