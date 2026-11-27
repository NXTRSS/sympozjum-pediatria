#!/usr/bin/env python3
"""Uruchom w katalogu delphi: uv run python scripts/diagnoza_delphi.py"""
from __future__ import annotations

import importlib.metadata
import sys

from delphi_demo import pamiec


def main() -> int:
    print("Interpreter:", sys.executable)
    try:
        print("Pakiet delphi-demo:", importlib.metadata.version("delphi-demo"))
    except importlib.metadata.PackageNotFoundError:
        print("BŁĄD: delphi-demo nie jest w tym Pythonie. Uruchom: cd delphi && uv sync")
        return 1
    print("pamiec.WERSJA_PAMIECI:", pamiec.WERSJA_PAMIECI)
    print("katalog wyników:", pamiec.KATALOG_WYNIKOW)
    kroki = pamiec.kroki_wbudowane("astma_od_dziecka")
    n = len(kroki or [])
    print("wbudowane kroki astma:", n)
    if n < 8:
        print("BŁĄD: oczekiwano ≥8 kroków — zrób git pull i uv sync")
        return 1
    print("OK — panel astmy może działać bez plików w pamiec/wyniki/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())