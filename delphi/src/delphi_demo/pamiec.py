"""Wyniki policzone przed prelekcją (tryb „z_pamieci": notebook nie liczy Monte Carlo na sali)."""
from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import pandas as pd

from delphi_demo import zasoby
from delphi_demo.prognoza import Krok

KATALOG_WYNIKOW = zasoby.KATALOG_PAMIECI / "wyniki"


def _plik(nazwa: str, katalog: Path | None) -> Path:
    return Path(katalog or KATALOG_WYNIKOW) / f"{nazwa}.json"


def zapisz_tabele(nazwa: str, df: pd.DataFrame, katalog: Path | None = None) -> Path:
    plik = _plik(nazwa, katalog)
    plik.parent.mkdir(parents=True, exist_ok=True)
    df = df.rename(columns=str)                      # tokeny jako nazwy kolumn → tekst w JSON
    plik.write_text(df.to_json(orient="split"))
    return plik


def wczytaj_tabele(nazwa: str, katalog: Path | None = None) -> pd.DataFrame:
    df = pd.read_json(StringIO(_plik(nazwa, katalog).read_text()), orient="split")
    return df.rename(columns=lambda c: int(c) if str(c).isdigit() else c)


def zapisz_kroki(nazwa: str, kroki: list[Krok], katalog: Path | None = None) -> Path:
    plik = _plik(nazwa, katalog)
    plik.parent.mkdir(parents=True, exist_ok=True)
    plik.write_text(json.dumps([asdict(k) for k in kroki]))
    return plik


def wczytaj_kroki(nazwa: str, katalog: Path | None = None) -> list[Krok]:
    surowe = json.loads(_plik(nazwa, katalog).read_text())
    return [Krok(k["wiek_lat"], k["token"], [tuple(x) for x in k["kandydaci"]], k.get("p")) for k in surowe]


def jest(nazwa: str, katalog: Path | None = None) -> bool:
    return _plik(nazwa, katalog).exists()
