"""Wyniki policzone przed prelekcją (tryb „z_pamieci": notebook nie liczy Monte Carlo na sali)."""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from io import StringIO
from pathlib import Path

import pandas as pd

from delphi_demo import zasoby
from delphi_demo.prognoza import Krok

KATALOG_WYNIKOW = zasoby.KATALOG_PAMIECI / "wyniki"
KATALOG_WZORCOW = zasoby.KATALOG_DELPHI / "wzorce_wyniki"


def _plik_zapis(nazwa: str, katalog: Path | None) -> Path:
    return Path(katalog or KATALOG_WYNIKOW) / f"{nazwa}.json"


def _kandydaci_odczytu(nazwa: str) -> list[str]:
    """Kolejność szukania pliku (spójne z przygotuj_demo: slad_{klucz})."""
    if nazwa.startswith("slad_"):
        bez = nazwa.removeprefix("slad_")
        return [nazwa, bez] if bez else [nazwa]
    return [f"slad_{nazwa}", nazwa] if not nazwa.startswith(("krzywe_", "ranking_")) else [nazwa]


def _plik_odczyt(nazwa: str, katalog: Path | None) -> Path:
    kat = Path(katalog or KATALOG_WYNIKOW)
    for kandydat in _kandydaci_odczytu(nazwa):
        sciezka = kat / f"{kandydat}.json"
        if sciezka.exists():
            return sciezka
    return kat / f"{_kandydaci_odczytu(nazwa)[0]}.json"


def zsynchronizuj_wzorce(katalog: Path | None = None) -> list[Path]:
    """Po `git pull`: skopiuj lekkie wzorce do pamiec/wyniki (nie nadpisuje pełnych wyników z przygotuj_demo)."""
    kat = Path(katalog or KATALOG_WYNIKOW)
    kat.mkdir(parents=True, exist_ok=True)
    skopiowane: list[Path] = []
    if not KATALOG_WZORCOW.is_dir():
        return skopiowane
    for zrodlo in sorted(KATALOG_WZORCOW.glob("*.json")):
        cel = kat / zrodlo.name
        if cel.exists():
            continue
        shutil.copy2(zrodlo, cel)
        skopiowane.append(cel)
    return skopiowane


def zapisz_tabele(nazwa: str, df: pd.DataFrame, katalog: Path | None = None) -> Path:
    plik = _plik_zapis(nazwa, katalog)
    plik.parent.mkdir(parents=True, exist_ok=True)
    df = df.rename(columns=str)                      # tokeny jako nazwy kolumn → tekst w JSON
    plik.write_text(df.to_json(orient="split"))
    return plik


def wczytaj_tabele(nazwa: str, katalog: Path | None = None) -> pd.DataFrame:
    df = pd.read_json(StringIO(_plik_odczyt(nazwa, katalog).read_text()), orient="split")
    return df.rename(columns=lambda c: int(c) if str(c).isdigit() else c)


def zapisz_kroki(nazwa: str, kroki: list[Krok], katalog: Path | None = None) -> Path:
    plik = _plik_zapis(nazwa, katalog)
    plik.parent.mkdir(parents=True, exist_ok=True)
    plik.write_text(json.dumps([asdict(k) for k in kroki]))
    return plik


def wczytaj_kroki(nazwa: str, katalog: Path | None = None) -> list[Krok]:
    surowe = json.loads(_plik_odczyt(nazwa, katalog).read_text())
    return [Krok(k["wiek_lat"], k["token"], [tuple(x) for x in k["kandydaci"]], k.get("p")) for k in surowe]


def jest(nazwa: str, katalog: Path | None = None) -> bool:
    return _plik_odczyt(nazwa, katalog).exists()


def zapisz_json(nazwa: str, obiekt, katalog: Path | None = None) -> Path:
    """Drobne metadane wyników (np. {"n_probek": 1000} dla krzywych), żeby notebook mówił prawdę o N."""
    plik = _plik_zapis(nazwa, katalog)
    plik.parent.mkdir(parents=True, exist_ok=True)
    plik.write_text(json.dumps(obiekt))
    return plik


def wczytaj_json(nazwa: str, katalog: Path | None = None):
    return json.loads(_plik_odczyt(nazwa, katalog).read_text())