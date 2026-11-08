"""Polski model masy w dniu porodu (Sano): 3D ResNet-18 z pracy TabAttention.

Praca: Grzeszczyk, Płotka i in., „TabAttention: Learning Attention Conditionally on Tabular Data”,
MICCAI 2023 (arXiv 2310.18129); Sano – Centrum Medycyny Obliczeniowej (Kraków) + WUM.
Model dostaje 16 klatek nagrania brzucha płodu i zwraca masę W DNIU PORODU w gramach: uczono go
na badaniach wykonanych < 24 h przed porodem, więc to nie jest prognoza na tygodnie do przodu.

Skąd wagi i dlaczego to model bazowy:
Publiczny checkpoint z README autorów zawiera obrazowy 3D ResNet-18, bez modułu TabAttention (mMAE 289 g, mMAPE 8,5% w pracy).
  • kod sieci (src/models/resnet.py) pochodzi z repo autorów SanoScience/Tab-Attention; repo nie ma
    licencji, więc NIE kopiujemy go do naszego repo — wczytujemy go z klonu w usg/zewnetrzne/ pod
    własną nazwą modułu (pakiet „models” autorów zderzyłby się z „models” BabyNet++ i FUVAI).

Preprocessing wg pracy (§3) i src/video_data_loader.py autorów w trybie „val”; skala 0,2 mm/px
w kroku 3 to nasza interpretacja pracy (patrz „Świadome odstępstwa i interpretacje”):
  1. tylko klatki brzucha, w kolejności nagrania;
  2. segmenty jak `chunks(L, n=16, skip=2)`: początek co 46 klatek listy, w segmencie co 3. klatka;
     niepełne segmenty pomijamy (u autorów padding=False);
  3. skala do 0,2 mm/px (NASZA interpretacja §3 pracy; loader autorów tego nie robi), potem
     proporcjonalnie do 128 px względem szerokości 1680 px
     (`int(h/1680·128)`, cv2.INTER_LINEAR), wyśrodkowane dopełnienie zerami do 128×128;
  4. normalizacja całego segmentu: m = max, (x − 0,5·m)/(0,5·m);
  5. wynik nagrania = średnia z segmentów (praca: średnia z wszystkich 16-klatkowych segmentów).

Świadome odstępstwa i interpretacje:
  • przeskalowanie do 0,2 mm/px to NASZA interpretacja pracy (§3 podaje rozmiar piksela 0,2 mm);
    loader autorów (src/video_data_loader.py) nie przeskalowuje obrazu, tylko zmniejsza go względem
    szerokości 1680 px. Bez tego kroku wynik przesuwa się o 350–800 g (README, tabela wyników);
  • „brzuch” wskazuje u nas nnU-Net (wynik offline co k-tą klatkę), a nie ręczne etykiety
    „abdomen”: klatka surowa j jest brzuchem, gdy w klatce (j // k)·k pole brzucha > 0;
    całkiem czarne klatki (max == 0) zawsze odpadają;
  • skalowanie 0,28 → 0,2 mm/px i 1680 → 128 px robimy jednym cv2.resize (iloczyn obu skal),
    a nie dwoma — jedna interpolacja zamiast dwóch, rozmiar wycinka ten sam;
  • gdy nagranie nie ma ani jednego pełnego segmentu, bierzemy jeden segment dopełniony zerami
    (jak padding=True u autorów) i zaznaczamy to flagą `dopelniony`;
  • cechy kliniczne podajemy jako zera (forward autorów i tak ich nie używa); liczymy na CPU.

Ograniczenia (mówimy je na głos):
  • model uczono na ciążach donoszonych (masa ur. 3495 ± 507 g, 38,3 ± 1,5 tc.), nagraniach
    < 24 h przed porodem, jednym protokole, aparatach GE. ACOUSLIC-AI to inne nagrania
    („na ślepo”, 26–35 tc.) — model wychodzi tu poza dane, na których się uczył (ekstrapolacja);
  • poza swoimi danymi model podaje bez cienia wątpliwości ok. 3,5–4,0 kg — powyżej średniej zbioru uczącego (3 495 g), prawie
    niezależnie od wieku ciążowego, silnie zależnie od skali obrazu;
  • nie znamy mas urodzeniowych dzieci z ACOUSLIC-AI, więc nie możemy sprawdzić, czy wynik jest trafny.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from usg_masa import zasoby
from usg_masa.dane import Sweep
from usg_masa.segmentacja import WynikSweepu

KLATKI = 16                 # frames_num
SKIP = 2                    # skip_frames: w segmencie co 3. klatka
ROZMIAR = (128, 128)        # input_size (wysokość, szerokość)
MAX_W = 1680                # max_width z loadera autorów
PIKSEL_MM = 0.2             # rozmiar piksela w danych autorów (praca §3)
LICZBA_PARAMETROW = 33_147_976

_NAZWA_MODULU = "tabattention_resnet_autorow"


@dataclass
class WynikSano:
    masa_g: float | None        # średnia z segmentów; None, gdy nie ma klatek brzucha
    oceny_g: list[float]        # ocena każdego segmentu
    starty: list[int]           # numer surowej klatki, od której zaczyna się każdy segment
    n_klatek_brzucha: int       # długość listy klatek, z której cięto segmenty
    dopelniony: bool            # True: jedyny segment dopełniony zerami (za mało klatek)
    czas_s: float               # całość: przygotowanie klatek + sieć


# ── kod autorów i model ──────────────────────────────────────────────────────

def _zaladuj_kod_autorow():
    """Importuje src/models/resnet.py z klonu autorów pod własną nazwą (patrz docstring modułu)."""
    if _NAZWA_MODULU in sys.modules:
        return sys.modules[_NAZWA_MODULU]
    plik = zasoby.sklonuj("TabAttention") / "src" / "models" / "resnet.py"
    spec = importlib.util.spec_from_file_location(_NAZWA_MODULU, plik)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    sys.modules[_NAZWA_MODULU] = modul
    return modul


def zbuduj_model_sano() -> torch.nn.Module:
    """3D ResNet-18 autorów z wagami TabAttention.pt (strict), w trybie eval, na CPU."""
    modul = _zaladuj_kod_autorow()
    model = modul.r3d_18(msha=False, n_frames=KLATKI, input_size=ROZMIAR)
    stan = torch.load(zasoby.wagi_sano(), map_location="cpu", weights_only=True)
    model.load_state_dict(stan, strict=True)
    return model.eval()


# ── wybór klatek i segmenty ──────────────────────────────────────────────────

def klatki_brzucha(sweep: Sweep, wynik_nnunet: WynikSweepu) -> list[int]:
    """Numery klatek surowych, w których (wg nnU-Netu) widać brzuch; kolejność nagrania.

    nnU-Net analizował co k-tą klatkę (wynik_nnunet.indeksy); klatka j dziedziczy ocenę
    ostatniej przeanalizowanej klatki ≤ j, czyli (j // k)·k. Czarne klatki (max == 0) odpadają.
    """
    indeksy = np.asarray(wynik_nnunet.indeksy)
    if indeksy.size == 0:
        return []
    brzuch = np.maximum(wynik_nnunet.pole_opt, wynik_nnunet.pole_subopt) > 0
    wynik = []
    for j in range(len(sweep.klatki)):
        poz = int(np.searchsorted(indeksy, j, side="right")) - 1   # = j // k przy indeksach co k
        if poz >= 0 and brzuch[poz] and sweep.klatki[j].max() > 0:
            wynik.append(j)
    return wynik


def segmenty(lista: Sequence[int], n: int = KLATKI, skip: int = SKIP) -> tuple[list[list[int]], bool]:
    """Cięcie listy klatek jak `chunks(L, n, skip)` autorów. Zwraca (segmenty, dopelniony).

    Zwracamy tylko pełne segmenty (n klatek). Gdy nie ma żadnego pełnego, a lista nie jest pusta,
    zwracamy jeden niepełny segment (do dopełnienia zerami) i dopelniony=True.
    """
    lista = list(lista)
    dlugosc = (skip + 1) * (n - 1) + 1                 # 46 przy n=16, skip=2
    wszystkie = [lista[x: x + dlugosc: skip + 1] for x in range(0, len(lista), dlugosc)]
    pelne = [s for s in wszystkie if len(s) == n]
    if pelne:
        return pelne, False
    if wszystkie:
        return [wszystkie[0]], True
    return [], False


# ── preprocessing ────────────────────────────────────────────────────────────

def przygotuj_klatke(klatka: np.ndarray, odstep_mm: float) -> np.ndarray:
    """(H, W) uint8 → (128, 128) float32: skala 0,2 mm/px i 128/1680, wyśrodkowane zera dookoła."""
    h, w = klatka.shape
    skala = odstep_mm / PIKSEL_MM / MAX_W * ROZMIAR[1]   # u autorów obie osie względem x_image_size
    nh, nw = int(h * skala), int(w * skala)
    if not (0 < nh <= ROZMIAR[0] and 0 < nw <= ROZMIAR[1]):
        raise ValueError(f"Klatka {w}×{h} px przy {odstep_mm} mm/px nie mieści się w {ROZMIAR[1]}×{ROZMIAR[0]} "
                         f"(po przeskalowaniu {nw}×{nh}).")
    mala = cv2.resize(klatka, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    plansza = np.zeros(ROZMIAR, np.float32)
    y, x = (ROZMIAR[0] - nh) // 2, (ROZMIAR[1] - nw) // 2   # jak A.PadIfNeeded (środek)
    plansza[y:y + nh, x:x + nw] = mala
    return plansza


def klip_tensor(klatki: Sequence[np.ndarray], odstep_mm: float) -> torch.Tensor:
    """Do 16 klatek → (1, 1, 16, 128, 128); brakujące na końcu = zera; normalizacja (x − 0,5m)/(0,5m)."""
    klip = np.zeros((KLATKI, *ROZMIAR), np.float32)
    for i, klatka in enumerate(klatki):
        klip[i] = przygotuj_klatke(klatka, odstep_mm)
    m = float(klip.max())
    if m > 0:
        klip = (klip - 0.5 * m) / (0.5 * m)
    return torch.from_numpy(klip)[None, None]


# ── przewidywanie ────────────────────────────────────────────────────────────

def przewidz_z_listy(model: torch.nn.Module, sweep: Sweep, lista: Sequence[int], paczka: int = 4) -> WynikSano:
    """Masa z dowolnej listy klatek (np. wszystkich niepustych) — segmenty, sieć, średnia."""
    start = time.perf_counter()
    lista = list(lista)
    segm, dopelniony = segmenty(lista)
    oceny: list[float] = []
    with torch.no_grad():
        for p in range(0, len(segm), paczka):
            x = torch.cat([klip_tensor([sweep.klatki[j] for j in s], sweep.odstep_mm) for s in segm[p:p + paczka]])
            tab = torch.zeros((x.shape[0], 1, 6))          # 6 cech klinicznych; forward autorów ich nie używa
            oceny += [float(v) for v in model(x, tab).reshape(-1)]
    masa = float(np.mean(oceny)) if oceny else None
    return WynikSano(masa, oceny, [int(s[0]) for s in segm], len(lista), dopelniony,
                     time.perf_counter() - start)


def przewidz(model: torch.nn.Module, sweep: Sweep, wynik_nnunet: WynikSweepu) -> WynikSano:
    """Masa urodzeniowa wg modelu Sano: klatki brzucha z nnU-Netu → segmenty → średnia."""
    return przewidz_z_listy(model, sweep, klatki_brzucha(sweep, wynik_nnunet))


# ── zapis ────────────────────────────────────────────────────────────────────

def zapisz_wynik(wynik: WynikSano, sciezka: Path) -> None:
    """Zapis do JSON (zwykle pamiec/<nazwa>_sano.json)."""
    Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
    Path(sciezka).write_text(json.dumps(asdict(wynik), ensure_ascii=False, indent=1), encoding="utf-8")


def wczytaj_wynik(sciezka: Path) -> WynikSano:
    return WynikSano(**json.loads(Path(sciezka).read_text(encoding="utf-8")))
