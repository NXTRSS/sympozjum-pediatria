#!/usr/bin/env python3
"""Jednorazowo po git pull: skopiuj wzorce śladu do delphi/pamiec/wyniki/."""
from delphi_demo import pamiec

if __name__ == "__main__":
    skopiowane = pamiec.zsynchronizuj_wzorce()
    if skopiowane:
        for p in skopiowane:
            print("Skopiowano:", p)
    else:
        print("Wyniki w:", pamiec.KATALOG_WYNIKOW)
        print("Wzorce z:", pamiec.KATALOG_WZORCOW)
        print("slad astma:", pamiec.jest("astma_od_dziecka"))
        sciezka = pamiec.KATALOG_WYNIKOW / "slad_astma_od_dziecka.json"
        print("Plik na dysku:", sciezka, "→", sciezka.exists())