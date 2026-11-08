"""Od maski brzucha do masy płodu.

Obwód brzucha (AC) liczymy jak w ACOUSLIC-AI: dopasowujemy elipsę do konturu,
a jej obwód wyznaczamy wzorem Ramanujana.
Masa z samego AC: Campbell i Wilkin (1975), ln(EFW[kg]) = −4,564 + 0,282·AC − 0,00331·AC², AC w cm.
Wiek ciążowy odpowiadający AC (nie datowanie ciąży): Hadlock (1984), GA = 8,14 + 0,753·AC + 0,0036·AC², AC w cm.
Oba wzory to przybliżenia z jednej miary — pełna biometria (HC, BPD, FL) jest dokładniejsza.
Maska dotykająca krawędzi kadru jest ucięta, więc nie dajemy z niej wyniku (zwracamy None).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

ZAKRES_AC_MM = (100.0, 420.0)
_MIN_POLE_PX = 50


def obwod_z_maski(maska: np.ndarray, odstep_mm: float) -> float | None:
    m = (np.asarray(maska) > 0).astype(np.uint8)
    kontury, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not kontury:
        return None
    kontur = max(kontury, key=cv2.contourArea)
    if len(kontur) < 5 or cv2.contourArea(kontur) < _MIN_POLE_PX:
        return None
    # Obrys dotykający krawędzi kadru jest ucięty — elipsa zaniżyłaby obwód.
    wys, szer = m.shape
    x, y = kontur[:, 0, 0], kontur[:, 0, 1]
    if np.any((x == 0) | (y == 0) | (x == szer - 1) | (y == wys - 1)):
        return None
    _, (d1, d2), _ = cv2.fitEllipse(kontur)
    a, b = d1 / 2 * odstep_mm, d2 / 2 * odstep_mm
    h = ((a - b) / (a + b)) ** 2
    return math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))


def masa_campbell_g(ac_mm: float) -> float:
    ac = ac_mm / 10
    return 1000 * math.exp(-4.564 + 0.282 * ac - 0.00331 * ac ** 2)


def tydzien_hadlock(ac_mm: float) -> float:
    ac = ac_mm / 10
    return 8.14 + 0.753 * ac + 0.0036 * ac ** 2


def w_zakresie(ac_mm: float) -> bool:
    return ZAKRES_AC_MM[0] <= ac_mm <= ZAKRES_AC_MM[1]


@dataclass
class Oszacowanie:
    ac_mm: float | None
    masa_g: float | None
    tydzien: float | None
    wiarygodne: bool
    komunikat: str


def oszacuj(maska: np.ndarray | None, odstep_mm: float) -> Oszacowanie:
    if maska is None or not np.any(maska):
        return Oszacowanie(None, None, None, False, "Model nie znalazł brzucha — nie znaleziono klatki do pomiaru.")
    ac = obwod_z_maski(maska, odstep_mm)
    if ac is None:
        return Oszacowanie(None, None, None, False, "Maska zbyt mała lub postrzępiona — nie znaleziono wiarygodnego obrysu.")
    masa, tydzien = masa_campbell_g(ac), tydzien_hadlock(ac)
    if not w_zakresie(ac):
        return Oszacowanie(ac, masa, tydzien, False,
                           f"AC = {ac:.0f} mm jest poza zakresem fizjologicznym — wynik niewiarygodny.")
    masa_txt = f"{masa:,.0f}".replace(",", " ")
    tydzien_txt = f"{tydzien:.1f}".replace(".", ",")
    return Oszacowanie(ac, masa, tydzien, True,
                       f"AC = {ac:.0f} mm → ok. {masa_txt} g, ok. {tydzien_txt} tc.")
