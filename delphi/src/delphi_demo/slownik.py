"""Kody ICD-10 ↔ tokeny modelu, polskie nazwy i parser historii wpisywanej przez lekarza."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from delphi_demo import icd10_pl, zasoby
from delphi_demo.tokeny import Zdarzenie

# Krótkie polskie nazwy dla kodów, które pokazujemy na sali (pierwszeństwo przed oficjalnymi z CeZ).
# Reszta: oficjalna polska ICD-10 (dane/icd10_pl.csv), a gdy jej brak, nazwa angielska z dopiskiem „(ang.)".
NAZWY_PL = {
    "Female": "kobieta", "Male": "mężczyzna", "Death": "zgon", "No event": "brak zdarzenia",
    "BMI_low": "BMI niskie", "BMI_mid": "BMI średnie", "BMI_high": "BMI wysokie",
    "Smoking_low": "palenie: nie", "Smoking_mid": "palenie: dawniej", "Smoking_high": "palenie: obecnie",
    "Alcohol_low": "alkohol: mało", "Alcohol_mid": "alkohol: średnio", "Alcohol_high": "alkohol: dużo",
    "B01": "ospa wietrzna", "B05": "odra", "B06": "różyczka", "B26": "świnka",
    "C18": "rak jelita grubego", "C34": "rak oskrzela i płuca", "C50": "rak piersi",
    "C61": "rak gruczołu krokowego",
    "E10": "cukrzyca typu 1", "E11": "cukrzyca typu 2", "E66": "otyłość",
    "E78": "zaburzenia lipidowe", "E03": "niedoczynność tarczycy", "E14": "cukrzyca nieokreślona",
    "F03": "otępienie", "F32": "epizod depresyjny", "F41": "zaburzenia lękowe",
    "G30": "choroba Alzheimera", "G43": "migrena",
    "H26": "zaćma", "H66": "zapalenie ucha środkowego",
    "I10": "nadciśnienie tętnicze", "I20": "dławica piersiowa", "I47": "częstoskurcz napadowy", "I21": "ostry zawał serca",
    "I25": "przewlekła choroba niedokrwienna serca", "I48": "migotanie przedsionków",
    "I50": "niewydolność serca", "I63": "udar niedokrwienny mózgu", "I64": "udar mózgu",
    "J03": "ostre zapalenie migdałków", "J06": "infekcja górnych dróg oddechowych",
    "J18": "zapalenie płuc", "J22": "zakażenie dolnych dróg oddechowych", "J30": "alergiczny nieżyt nosa", "J44": "POChP", "J45": "astma",
    "K21": "refluks żołądkowo-przełykowy", "K37": "zapalenie wyrostka robaczkowego",
    "K29": "zapalenie żołądka i dwunastnicy", "K30": "dyspepsja", "K44": "przepuklina przeponowa",
    "K57": "uchyłkowatość jelit", "K80": "kamica żółciowa",
    "L20": "atopowe zapalenie skóry",
    "M10": "dna moczanowa", "M17": "choroba zwyrodnieniowa kolana", "M19": "inna choroba zwyrodnieniowa stawów",
    "M25": "inne choroby stawów", "M54": "bóle kręgosłupa", "M79": "inne choroby tkanek miękkich",
    "N18": "przewlekła choroba nerek", "N39": "inne choroby układu moczowego",
    "N40": "rozrost gruczołu krokowego",
    "P07": "wcześniactwo / mała masa urodzeniowa", "Q21": "wady przegród serca",
    "CXX": "nowotwór złośliwy o nieokreślonym umiejscowieniu",   # token 1142 „CXX Unknown Cancer"
}

_KOD_ETYKIETY = r"^([A-Z]\d{2}|CXX)\b"          # początek etykiety autorów: „E11 …", „CXX Unknown Cancer"
_KOD_WPISU = r"(?<![A-Za-z0-9_])(?:[A-Za-z]\d{2}(?!\d)|BMI_\w+|Smoking_\w+|Alcohol_\w+)"
# Między kodem (z opcjonalnym podkodem „.9") a wiekiem wolno tylko: białe znaki, „: , ; = –", myślnik ze spacją
# po nim oraz słowa „w" / „wieku". Wiek to 1–3 cyfry (opcjonalnie z częścią dziesiętną), bez cyfr po bokach.
_WPIS = re.compile(
    r"(?P<kod>" + _KOD_WPISU + r")(?:\.\d+)?"                                      # E11, E11.9, Smoking_high
    r"(?:\s|[:,;=–]|-(?=\s)|\b(?:w|wieku)\b)*"                                   # „ w wieku ", „: ", " – "
    r"(?<![\d.])(?P<wiek>\d{1,3}(?:[.,]\d+)?)(?![\d])"                          # 50 lub 50,5
)
_KOD_BEZ_WIEKU = re.compile(_KOD_WPISU)
_ZGON = re.compile(r"\b(death|zgon\w*)\b", re.IGNORECASE)
_SLOWA_DOZWOLONE = re.compile(r"\b(w|wieku|lat|lata|roku|rż)\b", re.IGNORECASE)


class BladHistorii(ValueError):
    """Historia wpisana przez użytkownika jest niepoprawna; komunikat jest po polsku, dla sali."""


@dataclass
class Slownik:
    etykiety: list[str]               # indeks = token modelu
    kod_do_tokenu: dict[str, int]     # "E11" → 215; tokeny specjalne pod nazwą autorów („Smoking_high")
    rozdzial: list[str]               # skrócona nazwa rozdziału ICD-10 (z tabeli autorów)
    kolor: list[str]                  # kolor rozdziału (#rrggbb)
    oficjalne: dict[str, str] = field(default_factory=dict)   # „E78" → polska nazwa z ICD-10-2008-PL (CeZ, CC BY 4.0)

    def kod(self, token: int) -> str:
        m = re.match(_KOD_ETYKIETY, self.etykiety[token])
        return m.group(1) if m else self.etykiety[token]

    def nazwa(self, token: int) -> str:
        kod = self.kod(token)
        if kod in NAZWY_PL:               # 1. krótka nazwa dla sali
            return NAZWY_PL[kod]
        if kod in self.oficjalne:          # 2. oficjalna polska ICD-10 (Centrum e-Zdrowia)
            return self.oficjalne[kod]
        opis = self.etykiety[token][len(kod):].strip().strip("()")
        opis = re.sub(r"\d+$", "", opis).strip()   # błąd w pliku autorów: „O01 Hydatidiform mole38"
        return f"{opis or self.etykiety[token]} (ang.)"   # nigdy pusta nazwa (np. „Padding")

    def opisz(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dokleja kolumny „kod" i „nazwa" na początku tabeli z kolumną „token"."""
        wynik = df.copy()
        wynik.insert(0, "nazwa", [self.nazwa(int(t)) for t in df["token"]])
        wynik.insert(0, "kod", [self.kod(int(t)) for t in df["token"]])
        return wynik


_ODSYLACZ = re.compile(r"\s*\([A-Z]\d{2}[^)]*[†*]\)\s*$")   # „Pneumocystoza (J17.3*)" → „Pneumocystoza"


def wczytaj_nazwy_oficjalne(sciezka: Path | None = None) -> dict[str, str]:
    """dane/icd10_pl.csv (icd10_pl.przygotuj()) → {„A00": „choroba zakaźna …"}. Brak pliku → {}."""
    sciezka = Path(sciezka or icd10_pl.SCIEZKA_CSV)
    if not sciezka.exists():
        return {}
    with open(sciezka, encoding="utf-8") as f:
        wiersze = [linia for linia in f if not linia.startswith("#")]
    nazwy: dict[str, str] = {}
    for r in csv.DictReader(wiersze, delimiter=";"):
        n = _ODSYLACZ.sub("", r["nazwa_pl"]).strip()
        if n:
            nazwy[r["kod"]] = n[0].lower() + n[1:] if n[1:2].islower() else n   # „Cukrzyca…" → „cukrzyca…", „BMI" zostaje
    return nazwy


def wczytaj_slownik(sciezka_etykiet: Path | None = None,
                    sciezka_rozdzialow: Path | None = None,
                    sciezka_nazw_pl: Path | None = None) -> Slownik:
    etykiety = pd.read_csv(sciezka_etykiet or zasoby.sciezka_etykiet(),
                           header=None, sep="\t")[0].astype(str).tolist()
    kod_do_tokenu: dict[str, int] = {}
    for token, etykieta in enumerate(etykiety):
        m = re.match(_KOD_ETYKIETY, etykieta)
        klucz = m.group(1) if m else etykieta
        kod_do_tokenu.setdefault(klucz, token)          # „O01" występuje 2×: zostaje pierwszy (936)
    roz = pd.read_csv(sciezka_rozdzialow or zasoby.sciezka_rozdzialow())
    return Slownik(etykiety, kod_do_tokenu,
                   roz["ICD-10 Chapter (short)"].astype(str).tolist(), roz["color"].astype(str).tolist(),
                   wczytaj_nazwy_oficjalne(sciezka_nazw_pl))


def parsuj_historie(tekst: str, slownik: Slownik) -> list[Zdarzenie]:
    """„E11 50, I10 w wieku 55,5" → [Zdarzenie(215, 50.0), Zdarzenie(499, 55.5)], posortowane wg wieku.

    Powtórzony kod: zostaje najwcześniejszy wpis (Delphi liczy pierwsze wystąpienie choroby).
    """
    if not tekst.strip():
        return []
    if _ZGON.search(tekst):
        raise BladHistorii("Zgonu nie wpisuje się do historii: to model przewiduje, czy i kiedy nastąpi.")
    reszta = " ".join(_SLOWA_DOZWOLONE.sub(" ", _WPIS.sub(" ", tekst)).split())
    if re.search(r"\w", reszta):
        bez_wieku = _KOD_BEZ_WIEKU.search(reszta)
        if bez_wieku:
            raise BladHistorii(f"Brak wieku przy {bez_wieku.group(0)}: wpisz go po kodzie, "
                               "np. „E11 50” (wiek w latach, bez znaku minus).")
        raise BladHistorii(f"Nie rozumiem fragmentu: „{reszta}”. "
                           "Wpisz np.: „E11 50, I10 55” (kod ICD-10 i wiek w latach).")
    zdarzenia: dict[int, Zdarzenie] = {}
    nieznane = []
    for m in _WPIS.finditer(tekst):
        kod = m.group("kod")
        kod = kod.upper() if re.fullmatch(r"[A-Za-z]\d{2}", kod) else kod
        wiek = float(m.group("wiek").replace(",", "."))
        if not 0 <= wiek <= 110:
            raise BladHistorii(f"Wiek {wiek:g} przy {kod} jest poza zakresem 0–110 lat.")
        token = slownik.kod_do_tokenu.get(kod)
        if token is None:
            nieznane.append(kod)
            continue
        if token not in zdarzenia or wiek < zdarzenia[token].wiek_lat:
            zdarzenia[token] = Zdarzenie(token, wiek)
    if nieznane:
        raise BladHistorii(
            f"Model nie zna kodów: {', '.join(nieznane)}. Delphi-2M ma tylko 3-znakowe kody ICD-10 "
            "z rozdziałów A–Q (bez objawów R, urazów S–T i czynników Z).")
    return sorted(zdarzenia.values(), key=lambda z: z.wiek_lat)
