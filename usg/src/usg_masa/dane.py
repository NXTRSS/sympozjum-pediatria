"""Nagrania USG (sweepy) w formacie .mha: wczytywanie i wybiórcze pobieranie z Zenodo.

Dane ACOUSLIC-AI: Radboudumc, Zenodo 12697994 (v1.1), licencja CC BY-NC-SA 4.0.
ZIP ma 38 GB, więc pobieramy z niego tylko wybrane pliki (zapytania HTTP Range).
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from remotezip import RemoteZip

from usg_masa import zasoby

KATALOG_DANYCH = Path(__file__).resolve().parents[2] / "dane"
ZENODO_ZIP = "https://zenodo.org/api/records/12697994/files/acouslic-ai-train-set.zip/content"
ODSTEP_ACOUSLIC_MM = 0.28   # wg opisu zbioru na acouslic-ai.grand-challenge.org
PREFIKS_OBRAZOW = "acouslic-ai-train-set/images/stacked_fetal_ultrasound/"
CSV_OBWODOW = "acouslic-ai-train-set/circumferences/fetal_abdominal_circumferences_per_sweep.csv"
ATRYBUCJA = "Dane: ACOUSLIC-AI (Radboudumc, Zenodo 12697994), CC BY-NC-SA 4.0"

_STARTOWY = ("test/input/images/stacked-fetal-ultrasound/"
             "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000.mha")


@dataclass
class Sweep:
    klatki: np.ndarray   # (N, H, W), uint8
    odstep_mm: float     # rozmiar piksela w mm
    nazwa: str


def wczytaj_mha(sciezka: Path) -> Sweep:
    obraz = sitk.ReadImage(str(sciezka))
    klatki = sitk.GetArrayFromImage(obraz)
    if klatki.dtype != np.uint8:
        klatki = np.clip(klatki, 0, 255).astype(np.uint8)
    odstep = float(obraz.GetSpacing()[0])
    if abs(odstep - 1.0) < 1e-6:
        warnings.warn("Plik nie ma rozmiaru piksela — przyjmuję 0,28 mm (ACOUSLIC-AI).", UserWarning)
        odstep = ODSTEP_ACOUSLIC_MM
    return Sweep(klatki=klatki, odstep_mm=odstep, nazwa=Path(sciezka).name.removesuffix(".mha"))


def pobierz_przypadek_startowy() -> Path:
    """128-klatkowy wycinek przypadku 04a04f2e-… — zwykły plik w repo baseline (już sklonowanym).

    Uwaga: to DANE ACOUSLIC-AI (CC BY-NC-SA 4.0), mimo że leżą w repo z kodem na Apache-2.0.
    """
    return zasoby.sklonuj("ACOUSLIC-AI-baseline") / _STARTOWY


def lista_plikow_zip(url: str = ZENODO_ZIP) -> list[str]:
    with RemoteZip(url) as z:
        return z.namelist()


def wybierz_przypadki(nazwy: list[str], ile: int) -> list[str]:
    """Pierwsze `ile` nagrań (alfabetycznie) z katalogu obrazów — powtarzalnie."""
    return sorted(n for n in nazwy if n.startswith(PREFIKS_OBRAZOW) and n.endswith(".mha"))[:ile]


def wczytaj_obwody_referencyjne(csv: Path) -> pd.DataFrame:
    """AC zmierzone przez ekspertów ACOUSLIC-AI (mm), po jednej kolumnie na przeciągnięcie głowicą."""
    return pd.read_csv(csv, dtype={"uuid": str, "subject_id": str}).set_index("uuid")


def ac_referencyjne(df: pd.DataFrame, nazwa: str) -> list[float]:
    """Wszystkie dostępne referencyjne AC dla nagrania; nazwa może mieć przyrostek `_1_0000`."""
    uuid = nazwa.split("_")[0]
    if uuid not in df.index:
        return []
    wiersz = df.loc[uuid, [f"sweep_{i}_ac_mm" for i in range(1, 7)]]
    return [round(float(v), 1) for v in wiersz if pd.notna(v)]


def _czesc(cel: Path) -> Path:
    """Plik tymczasowy obok celu — dopóki istnieje, pobieranie nie jest skończone."""
    return cel.with_suffix(cel.suffix + ".czesc")


def _gotowy(cel: Path) -> bool:
    """Plik jest gotowy, gdy istnieje i nie ma po nim pozostałości przerwanego pobierania.

    Nie sprawdzamy rozmiaru bez indeksu ZIP-a: dzięki `replace()` w `_pobierz_czlonka`
    plik pod ostateczną nazwą pojawia się tylko po pełnym zapisie.
    """
    return cel.exists() and not _czesc(cel).exists()


def _pobierz_czlonka(z: RemoteZip, n: str, cel: Path) -> None:
    """Zapisuje jeden członek ZIP-a do `cel` przez plik tymczasowy; przy błędzie nic nie zostaje."""
    cel.parent.mkdir(parents=True, exist_ok=True)
    tymczasowy = _czesc(cel)
    oczekiwany = z.getinfo(n).file_size
    try:
        with z.open(n) as zrodlo, open(tymczasowy, "wb") as ujscie:
            shutil.copyfileobj(zrodlo, ujscie, 1 << 20)
        # zipfile sprawdza CRC przy czytaniu do końca; tu dodatkowo porównujemy rozmiar
        if tymczasowy.stat().st_size != oczekiwany:
            raise ValueError(f"Niepełny plik {n}: {tymczasowy.stat().st_size} z {oczekiwany} B")
    except BaseException:
        tymczasowy.unlink(missing_ok=True)
        raise
    tymczasowy.replace(cel)


def pobierz_z_zip(nazwy: list[str], url: str = ZENODO_ZIP) -> list[Path]:
    """Zwraca ścieżki do plików z ZIP-a; sieć (RemoteZip) otwiera się tylko przy braku plików."""
    cele = [KATALOG_DANYCH / "zenodo" / n for n in nazwy]
    brakujace = [(n, c) for n, c in zip(nazwy, cele) if not _gotowy(c)]
    if brakujace:
        with RemoteZip(url) as z:
            for n, cel in brakujace:
                _pobierz_czlonka(z, n, cel)
    return cele
