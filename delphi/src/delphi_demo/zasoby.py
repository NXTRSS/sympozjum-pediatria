"""Kod autorów Delphi (gerstung-lab/Delphi) w przypiętym commicie.

Kod jest na licencji MIT, ale go nie kopiujemy: klonujemy do delphi/zewnetrzne/ (poza gitem),
razem z danymi syntetycznymi autorów. Notebook tylko CZYTA te pliki, nie dotyka sieci.
"""
from __future__ import annotations

import functools
import importlib.util
import subprocess
from pathlib import Path

KATALOG_DELPHI = Path(__file__).resolve().parents[2]
KATALOG_ZEWN = KATALOG_DELPHI / "zewnetrzne"
KATALOG_PAMIECI = KATALOG_DELPHI / "pamiec"
REPO_URL = "https://github.com/gerstung-lab/Delphi"
COMMIT = "fb72166be6b29d8db819227a59487e51c1df1454"   # main z 2026-08-07 („Update README.md")

KOMUNIKAT_BRAK_KODU = (
    "Brak kodu Delphi w delphi/zewnetrzne/Delphi. Uruchom najpierw (z internetem): "
    "cd delphi && uv run python scripts/przygotuj_demo.py"
)


def katalog_kodu() -> Path:
    return KATALOG_ZEWN / "Delphi"


def sklonuj() -> Path:
    """Klonuje repo autorów (jeśli go nie ma) i ustawia przypięty commit. Wymaga internetu."""
    cel = katalog_kodu()
    if not (cel / ".git").exists():
        cel.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO_URL, str(cel)], check=True)
    subprocess.run(["git", "-C", str(cel), "checkout", "-q", COMMIT], check=True)
    return cel


def sprawdz_kod() -> Path:
    """Ścieżka do kodu autorów; czytelny błąd, gdy przygotowanie nie było uruchomione."""
    cel = katalog_kodu()
    if not (cel / "model.py").exists():
        raise FileNotFoundError(KOMUNIKAT_BRAK_KODU)
    return cel


@functools.lru_cache(maxsize=1)
def modul_modelu():
    """Moduł model.py autorów (klasy Delphi i DelphiConfig), wczytany bez zmiany sys.path."""
    sciezka = sprawdz_kod() / "model.py"
    spec = importlib.util.spec_from_file_location("delphi_autorzy_model", sciezka)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def katalog_danych() -> Path:
    """Dane syntetyczne autorów: train.bin, val.bin, labels.csv."""
    return sprawdz_kod() / "data" / "ukb_simulated_data"


def sciezka_etykiet() -> Path:
    return katalog_danych() / "labels.csv"


def sciezka_rozdzialow() -> Path:
    """Tabela autorów: token → rozdział ICD-10 i kolor (do wykresów)."""
    return sprawdz_kod() / "delphi_labels_chapters_colours_icd.csv"
