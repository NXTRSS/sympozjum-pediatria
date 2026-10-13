"""Polska ICD-10 (Centrum e-Zdrowia, dane.gov.pl, CC BY 4.0) → słownik: kod 3-znakowy → nazwa.

Pobieranie wymaga internetu i odbywa się tylko w przygotowaniu (scripts/przygotuj_demo.py).
Pliki lądują w delphi/dane/ (poza gitem); notebook tylko czyta CSV.
Źródło: zbiór „Rejestr Systemów Kodowania", zasób 55725 „Międzynarodowa Statystyczna Klasyfikacja Chorób
i Problemów Zdrowotnych - ICD-10" (ICD-10-2008-PL), https://dane.gov.pl/pl/dataset/27,rejestr-systemow-kodowania
"""
from __future__ import annotations

import csv
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from delphi_demo import zasoby

URL = ("https://api.dane.gov.pl/resources/55725,miedzynarodowa-statystyczna-klasyfikacja-chorob-"
       "i-problemow-zdrowotnych-icd-10/file")
SCIEZKA_XML = zasoby.KATALOG_DELPHI / "dane" / "icd10_pl.xml"
SCIEZKA_CSV = zasoby.KATALOG_DELPHI / "dane" / "icd10_pl.csv"
_NS = "{http://rsk.rejestrymedyczne.csioz.gov.pl}"
_NAGLOWEK = ("# Polskie nazwy kategorii ICD-10 (3 znaki). Źródło: Centrum e-Zdrowia, Rejestr Systemów Kodowania, "
             "ICD-10-2008-PL (dane.gov.pl, zasób 55725), licencja CC BY 4.0. Kody z gwiazdką/krzyżykiem bez znacznika.\n")


def parsuj(xml: Path) -> dict[str, str]:
    """XML CeZ → {„A00": „Cholera", …}: tylko kategorie 3-znakowe, pierwsza nazwa wygrywa."""
    nazwy: dict[str, str] = {}
    for wezel in ET.parse(xml).getroot().iter(_NS + "node"):
        kod = (wezel.get("code") or "").strip().rstrip("*†")
        if re.fullmatch(r"[A-Z]\d{2}", kod):
            nazwy.setdefault(kod, (wezel.findtext(_NS + "name") or "").strip())
    return nazwy


def przygotuj(sciezka_xml: Path = SCIEZKA_XML, sciezka_csv: Path = SCIEZKA_CSV,
              min_kategorii: int = 2000) -> int:
    """Pobiera XML (jeśli go nie ma; wymaga internetu) i zapisuje CSV „kod;nazwa_pl". Zwraca liczbę kategorii."""
    sciezka_xml, sciezka_csv = Path(sciezka_xml), Path(sciezka_csv)
    if not sciezka_xml.exists():
        sciezka_xml.parent.mkdir(parents=True, exist_ok=True)
        print(f"Pobieram {URL} → {sciezka_xml}")
        urllib.request.urlretrieve(URL, sciezka_xml)
    nazwy = parsuj(sciezka_xml)
    if len(nazwy) < min_kategorii:
        raise RuntimeError(f"Za mało kategorii ({len(nazwy)}): plik {sciezka_xml} wygląda na niekompletny.")
    sciezka_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(sciezka_csv, "w", newline="", encoding="utf-8") as f:
        f.write(_NAGLOWEK)
        w = csv.writer(f, delimiter=";")
        w.writerow(["kod", "nazwa_pl"])
        for kod in sorted(nazwy):
            w.writerow([kod, nazwy[kod]])
    return len(nazwy)
