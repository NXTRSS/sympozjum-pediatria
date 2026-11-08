"""FUVAI (Płotka i in., Sano + WUM; Physics in Medicine & Biology 2022, 67(4):045013).

Polski model z publicznymi wagami: z nagrania sam rozpoznaje, co widać w klatce
(głowa / brzuch / kość udowa / tło), i rysuje maskę widocznej struktury. Gałąź segmentacji
ma jedno wyjście (n_class=1), niezależne od klasy: maska to „struktura”, a nie koniecznie brzuch.
Dlatego klatkę do pomiaru bierzemy tylko wtedy, gdy klasyfikator uważa brzuch za najbardziej
prawdopodobną klasę (argmax == BRZUCH). Daje to „opcję pośrednią”
między holenderskim nnU-Netem (działa, ale nie jest polski) a BabyNet++ (polski, ale bez wag).

Co bierzemy od autorów (artykuł §2.1, §2.3.1): klatka skalowana do 224×224 BEZ zachowania
proporcji, wartości 0–1 (÷255); maska → rozmiar oryginału (bilinear), próg 0,6, erozja i
dylatacja krzyżem 5×5, mediana 13×13, kontur, elipsa, obwód. Klasy: 0 głowa, 1 brzuch,
2 kość udowa, 3 tło (potwierdzone na naszych nagraniach: p(brzuch) najwyższe w klatce
brzucha wskazanej przez nnU-Net).

Świadome odstępstwa od artykułu:
  • T = 1: każdą klatkę podajemy osobno (sieć ma warstwę ConvLSTM, ale seria klatek
    pogarsza p(brzuch) na naszych danych);
  • wybór klatki (§2.3.2) to NASZA adaptacja. Autorzy biorą klatki z p > 0,9, a na nagraniach
    „na ślepo” (ACOUSLIC-AI) p(brzuch) nigdy nie przekracza 0,9: model był uczony na nagraniach
    „z ręki” w płaszczyznach standardowych (GE Voluson, 700 ciąż), więc jest poza swoją domeną.
    Dlatego kandydatami są klatki z p(brzuch) w górnych 10% danego nagrania, w których brzuch
    jest najbardziej prawdopodobną klasą (inaczej model „widzi” raczej głowę, kość udową albo tło),
    z maską nie dotykającą krawędzi kadru; wygrywa najwyższa średnia z p(brzuch) i zgodności
    elipsy z maską. Brak takiej klatki = „nie znaleziono”;
  • masa z samego AC (Campbell), nie Hadlock III: ACOUSLIC-AI zawiera tylko brzuch (bez HC i FL);
  • maska FUVAI zwykle zaniża dolną krawędź brzucha, więc AC bywa mniejszy niż u nnU-Netu.
Czarne klatki (max == 0) pomijamy jak w segmentacji: p = [0, 0, 0, 1], bez wysyłania do sieci.

Kod sieci pobieramy z repo autorów (brak licencji → nie kopiujemy go do naszego repo).
Wczytujemy go z pliku pod własną nazwą modułu, bo pakiet „models” autorów zderzyłby się
z pakietem „models” z BabyNet++.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from usg_masa import biometria, segmentacja, zasoby
from usg_masa.dane import Sweep

KLASY = ("głowa", "brzuch", "kość udowa", "tło")
BRZUCH = 1
TLO = 3
ROZMIAR = 224
PROG_MASKI = 0.6
PROG_ARTYKULU = 0.9      # próg p(klasa) z artykułu; na nagraniach „na ślepo” nieosiągalny
GORNE_PROCENT = 10       # nasza adaptacja: kandydaci = górne 10% p(brzuch) w nagraniu
LICZBA_PARAMETROW = 69_157_401

_KRZYZ = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
_NAZWA_MODULU = "fuvai_kod_autorow"


@dataclass
class WynikFUVAI:
    indeksy: np.ndarray          # numery przeanalizowanych klatek
    p_klas: np.ndarray           # (n, 4) prawdopodobieństwa klas; czarne klatki = [0, 0, 0, 1]
    najlepsza_klatka: int        # -1, gdy brak
    maska: np.ndarray | None     # (H, W) uint8 dla najlepszej klatki
    ac_mm: float | None
    p_brzuch_najlepszej: float   # 0,0, gdy brak najlepszej klatki
    czas_s: float                # całość analizy (sieć + wybór klatki)


# ── kod autorów i model ──────────────────────────────────────────────────────

def _zaladuj_kod_autorow():
    """Importuje models/fuvai.py z klonu autorów pod własną nazwą (patrz docstring modułu)."""
    if _NAZWA_MODULU in sys.modules:
        return sys.modules[_NAZWA_MODULU]
    plik = zasoby.sklonuj("FUVAI") / "models" / "fuvai.py"
    spec = importlib.util.spec_from_file_location(_NAZWA_MODULU, plik)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    sys.modules[_NAZWA_MODULU] = modul
    return modul


def _napraw_init_hidden(modul) -> None:
    """Podmienia ConvLSTMCell.init_hidden (idempotentnie), nie ruszając kodu autorów.

    Błąd autorów: init_hidden tworzy stan początkowy zawsze na CPU, więc na MPS/CUDA
    kończy się „RuntimeError: ... mps:0 and cpu”. Nasza owijka przenosi stan na urządzenie wag.
    """
    komorka = modul.ConvLSTMCell
    if getattr(komorka.init_hidden, "_naprawiona", False):
        return
    oryginal = komorka.init_hidden

    def init_hidden(self, batch_size):
        urzadzenie = self.conv.weight.device
        return tuple(t.to(urzadzenie) for t in oryginal(self, batch_size))

    init_hidden._naprawiona = True
    komorka.init_hidden = init_hidden


def zbuduj_fuvai(urz: torch.device | None = None) -> torch.nn.Module:
    """YNet z wagami autorów (strict), w trybie eval, na urządzeniu urz (domyślnie MPS/CUDA/CPU)."""
    urz = urz or segmentacja.urzadzenie()
    modul = _zaladuj_kod_autorow()
    _napraw_init_hidden(modul)
    model = modul.YNet(1, 64, 1)
    stan = torch.load(zasoby.wagi_fuvai(), map_location="cpu", weights_only=True)
    model.load_state_dict(stan, strict=True)
    return model.eval().to(urz)


# ── pre- i postprocessing ────────────────────────────────────────────────────

def przygotuj_klatke(klatka: np.ndarray) -> torch.Tensor:
    """(H, W) uint8 → (1, 1, 224, 224) float32 0–1; skalowanie bez zachowania proporcji."""
    mala = cv2.resize(klatka, (ROZMIAR, ROZMIAR), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(mala.astype(np.float32) / 255.0)[None, None]


def maska_z_prawdopodobienstwa(prob224: np.ndarray, ksztalt: tuple[int, int]) -> np.ndarray:
    """Mapa p(struktura) 224×224 → maska (H, W) uint8 0/1, jak w artykule §2.3.1.

    Gałąź segmentacji nie zna klasy (n_class=1): maska pokazuje strukturę, którą sieć widzi
    (głowę, brzuch albo kość udową). Stąd reguła argmax == BRZUCH w wybierz_klatke.
    """
    prob = cv2.resize(np.asarray(prob224, np.float32), (ksztalt[1], ksztalt[0]),
                      interpolation=cv2.INTER_LINEAR)
    maska = (prob > PROG_MASKI).astype(np.uint8)
    maska = cv2.dilate(cv2.erode(maska, _KRZYZ), _KRZYZ)
    return cv2.medianBlur(maska * 255, 13) // 255


def zgodnosc_elipsy(maska: np.ndarray) -> float:
    """IoU elipsy dopasowanej do największego konturu i samej maski; 0, gdy nie ma konturu."""
    m = (np.asarray(maska) > 0).astype(np.uint8)
    kontury, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not kontury:
        return 0.0
    kontur = max(kontury, key=cv2.contourArea)
    if len(kontur) < 5:
        return 0.0
    elipsa = np.zeros_like(m)
    cv2.ellipse(elipsa, cv2.fitEllipse(kontur), 1, -1)
    suma = np.logical_or(elipsa, m).sum()
    return float(np.logical_and(elipsa, m).sum() / suma) if suma else 0.0


# ── wybór klatki ─────────────────────────────────────────────────────────────

def wybierz_klatke(indeksy: np.ndarray, p_klas: np.ndarray, ocenione: np.ndarray,
                   maska_dla: Callable[[int], np.ndarray], odstep_mm: float,
                   gorne_procent: float = GORNE_PROCENT):
    """Nasza adaptacja §2.3.2 (patrz docstring modułu). Zwraca (klatka, maska, ac_mm, p_brzuch).

    p_klas: (n, 4) prawdopodobieństwa klas (KLASY) dla każdej pozycji.
    ocenione: które klatki przeszły przez sieć (czarne nie). Próg górnych X% p(brzuch) liczymy tylko z nich.
    Kandydat = oceniona klatka z p(brzuch) w górnych X%, w której brzuch jest najbardziej
    prawdopodobną klasą (argmax p_klas == BRZUCH).
    maska_dla(poz): maska (H, W) klatki na pozycji poz; wołana wyłącznie dla kandydatów.
    Brak kandydatów (albo żadna ich maska nie przeszła przez obwod_z_maski) → (-1, None, None, 0.0).
    """
    p_klas = np.asarray(p_klas, float)
    p_brzuch = p_klas[:, BRZUCH]
    ocenione = np.asarray(ocenione, bool)
    if not ocenione.any():
        return -1, None, None, 0.0
    prog = np.quantile(p_brzuch[ocenione], 1 - gorne_procent / 100)
    brzuch_wygrywa = p_klas.argmax(axis=1) == BRZUCH
    najlepszy = None
    for poz in np.flatnonzero(ocenione & (p_brzuch >= prog) & brzuch_wygrywa):
        maska = maska_dla(int(poz))
        ac = biometria.obwod_z_maski(maska, odstep_mm)
        if ac is None:
            continue
        wynik = (p_brzuch[poz] + zgodnosc_elipsy(maska)) / 2
        if najlepszy is None or wynik > najlepszy[0]:
            najlepszy = (wynik, int(poz), maska, ac)
    if najlepszy is None:
        return -1, None, None, 0.0
    _, poz, maska, ac = najlepszy
    return int(indeksy[poz]), maska, float(ac), float(p_brzuch[poz])


def analizuj_sweep(model: torch.nn.Module, sweep: Sweep, co_ktora: int = 4, paczka: int = 16) -> WynikFUVAI:
    """Co co_ktora-ta klatka → p(klasy) i mapa struktury; potem wybór klatki i AC.

    Zmierzone na M1 Pro (MPS, co 4. klatka, 840 klatek): ≈ 6–11 s na nagranie.
    """
    start = time.perf_counter()
    urz = next(model.parameters()).device
    indeksy = np.arange(0, len(sweep.klatki), co_ktora)
    p_klas = np.zeros((len(indeksy), len(KLASY)))
    p_klas[:, TLO] = 1.0                                                   # czarne klatki zostają tłem
    mapy: list[np.ndarray | None] = [None] * len(indeksy)
    ocenione = np.array([sweep.klatki[i].max() > 0 for i in indeksy], bool)   # czarne klatki = tło
    pozycje = np.flatnonzero(ocenione)
    with torch.no_grad():
        for p in range(0, len(pozycje), paczka):
            poz = pozycje[p:p + paczka]
            x = torch.cat([przygotuj_klatke(sweep.klatki[indeksy[k]]) for k in poz])[:, None]   # (B, T=1, 1, 224, 224)
            fc, wyjscie = model(x.to(urz))
            p_klas[poz] = torch.softmax(fc, 1).cpu().numpy()
            probs = wyjscie[:, 0].float().cpu().numpy()
            for j, k in enumerate(poz):
                mapy[k] = probs[j]
    ksztalt = sweep.klatki.shape[1:]
    klatka, maska, ac, p_najlepszej = wybierz_klatke(
        indeksy, p_klas, ocenione,
        lambda poz: maska_z_prawdopodobienstwa(mapy[poz], ksztalt), sweep.odstep_mm)
    return WynikFUVAI(indeksy, p_klas, klatka, maska, ac, p_najlepszej, time.perf_counter() - start)


# ── zapis ────────────────────────────────────────────────────────────────────

def zapisz_wynik(wynik: WynikFUVAI, sciezka: Path) -> None:
    Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sciezka, indeksy=wynik.indeksy, p_klas=wynik.p_klas,
                        najlepsza_klatka=wynik.najlepsza_klatka,
                        maska=wynik.maska if wynik.maska is not None else np.zeros((0, 0), np.uint8),
                        ac_mm=np.nan if wynik.ac_mm is None else wynik.ac_mm,
                        p_brzuch_najlepszej=wynik.p_brzuch_najlepszej, czas_s=wynik.czas_s)


def wczytaj_wynik(sciezka: Path) -> WynikFUVAI:
    d = np.load(sciezka)
    ac = float(d["ac_mm"])
    return WynikFUVAI(d["indeksy"], d["p_klas"], int(d["najlepsza_klatka"]),
                      d["maska"] if d["maska"].size else None,
                      None if np.isnan(ac) else ac, float(d["p_brzuch_najlepszej"]), float(d["czas_s"]))
