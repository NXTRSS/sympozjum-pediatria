"""BabyNet++ (Płotka i in., Sano Kraków; Computers in Biology and Medicine 2023).

Jedno przejście sieci widzi KLIP: 16 klatek (co 5. klatka nagrania) w rozdzielczości 64×64
oraz 6 cech klinicznych. Całe nagranie to ciąg takich klipów (co 76 klatek); wynik dla
pacjentki to średnia ocen z klipów, jak w train_video.py autorów.
Zgodnie z autorami: krok klipu (76), normalizacja (x − 0,5·max) / (0,5·max) → [−1, 1],
średnia po klipach oraz pomijanie niepełnych klipów na końcu nagrania (padding=False w
video_data_loader.py). Krótkie nagranie (< 76 klatek) daje jeden klip dopełniony powtórzeniem
ostatniej klatki.
Nasza adaptacja: klatki skalujemy tak, by dłuższy bok miał 64 px (autorzy skalują do stałej
szerokości 1680 px), z zachowaniem proporcji i dopełnieniem zerami do 64×64.
Całkiem czarnych klatek nie usuwamy przed cięciem na klipy (autorzy pomijają klatki tła
według pliku background_frames.csv).

Kod modelu pobieramy z repo autorów (brak licencji → nie kopiujemy go do naszego repo).
Publiczny kod to architektura bazowa (3D-ResNet-18 + MHSA3D + 6 cech), bez modułu DAFT z artykułu.
Wag autorzy nie opublikowali: bez nich model zwraca liczby losowe.
Conv3D nie działa na MPS w torch 2.2 — model liczymy na CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from usg_masa import zasoby
from usg_masa.dane import Sweep

KLATKI = 16
ROZMIAR = (64, 64)   # (wysokość, szerokość), domyślne x/y_img_size w train_video.py
KROK = 5             # skip_frames=4 → co 5. klatka
KROK_KLIPU = (KLATKI - 1) * KROK + 1   # 76 — jak chunks() autorów
CECHY = ("AC", "HC", "BPD", "FL", "Age", "GA")


def zbuduj_babynet() -> torch.nn.Module:
    src = str(zasoby.sklonuj("BabyNetPlusPlus") / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from models.resnet import r3d_18   # noqa: E402  (kod autorów)
    model = r3d_18(msha=True, n_frames=KLATKI, input_size=ROZMIAR)
    return model.eval()


def liczba_parametrow(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _dopasuj(klatka: np.ndarray) -> np.ndarray:
    """Skalowanie z zachowaniem proporcji (dłuższy bok = 64) + dopełnienie zerami do 64×64."""
    h, w = klatka.shape
    skala = min(ROZMIAR[0] / h, ROZMIAR[1] / w)
    nh, nw = max(1, round(h * skala)), max(1, round(w * skala))
    mala = cv2.resize(klatka, (nw, nh), interpolation=cv2.INTER_LINEAR)
    wynik = np.zeros(ROZMIAR, dtype=klatka.dtype)
    g, l = (ROZMIAR[0] - nh) // 2, (ROZMIAR[1] - nw) // 2
    wynik[g:g + nh, l:l + nw] = mala
    return wynik


def klip(sweep: Sweep, start: int) -> torch.Tensor:
    n = len(sweep.klatki)
    numery = [min(start + i * KROK, n - 1) for i in range(KLATKI)]
    arr = np.stack([_dopasuj(sweep.klatki[i]) for i in numery]).astype(np.float32)
    m = max(float(arr.max()), 1e-8)
    arr = (arr - 0.5 * m) / (0.5 * m)                 # [−1, 1], jak video_data_loader.py
    return torch.from_numpy(arr)[None, None]          # (1, 1, T, H, W)


def starty_klipow(n_klatek: int) -> list[int]:
    """Starty PEŁNYCH klipów (start + 75 ≤ n − 1); niepełne na końcu pomijamy jak autorzy.
    Nagranie krótsze niż jeden klip daje [0] (klip() dopełni je ostatnią klatką)."""
    starty = list(range(0, n_klatek - KROK_KLIPU + 1, KROK_KLIPU))
    return starty or [0]


def cechy_kliniczne(**wartosci: float) -> torch.Tensor:
    nieznane = set(wartosci) - set(CECHY)
    if nieznane:
        raise KeyError(f"Nieznane cechy: {sorted(nieznane)}; dozwolone: {CECHY}")
    return torch.tensor([[float(wartosci.get(c, 0.0)) for c in CECHY]], dtype=torch.float32)


@torch.no_grad()
def przewidz_caly_sweep(model: torch.nn.Module, sweep: Sweep,
                        cechy: torch.Tensor) -> tuple[float, list[float]]:
    oceny = [float(model(klip(sweep, s), cechy)) for s in starty_klipow(len(sweep.klatki))]
    return float(np.mean(oceny)), oceny


def wczytaj_wagi(model: torch.nn.Module, sciezka: Path) -> torch.nn.Module:
    stan = torch.load(sciezka, map_location="cpu")
    if isinstance(stan, dict) and "state_dict" in stan:
        stan = stan["state_dict"]
    model.load_state_dict(stan, strict=True)
    return model.eval()
