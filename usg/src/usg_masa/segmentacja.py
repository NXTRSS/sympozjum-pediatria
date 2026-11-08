"""Segmentacja brzucha płodu w CAŁYM nagraniu: model nnU-Net z baseline'u ACOUSLIC-AI.

Wybór klatki odtwarza baseline (DIAGNijmegen/ACOUSLIC-AI-baseline, Apache-2.0):
próg 0,5 → argmax → największa spójna składowa → klatka o największym polu
klasy 1 (płaszczyzna optymalna) lub 2 (suboptymalna).

Świadome odstępstwa od baseline'u:
  • największą składową wybieramy osobno w każdej klatce (2D), a nie w całym stosie (3D),
    bo przy analizie co n-tej klatki spójność między klatkami traci sens;
  • lustrzane odbicia (mirroring) domyślnie wyłączone — 4× szybciej;
  • segmentuj_sweep_szybko pomija kafelkowanie i ważenie gaussowskie nnU-Netu;
  • całkiem czarne klatki (max == 0) traktujemy jako tło i nie wysyłamy do sieci: po
    normalizacji min-max sieć „widziałaby” w nich ogromny brzuch (pełne nagrania mają ich ~127).
Uwaga: dataset.json w folderze modelu nazywa etykiety head/torso (przestarzała kopia z
Dataset107); obowiązuje znaczenie z danych: 1 = optymalna, 2 = suboptymalna.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.postprocessing.remove_connected_components import (
    remove_all_but_largest_component_from_segmentation,
)

from usg_masa.dane import Sweep

KATALOG_PAMIECI = Path(__file__).resolve().parents[2] / "pamiec"


@dataclass
class WynikSweepu:
    indeksy: np.ndarray      # numery przeanalizowanych klatek
    pole_opt: np.ndarray     # liczba pikseli klasy 1 na klatkę
    pole_subopt: np.ndarray  # liczba pikseli klasy 2 na klatkę
    najlepsza_klatka: int    # -1, gdy brak
    maska: np.ndarray | None # (H, W) uint8 dla najlepszej klatki
    czas_s: float


def wybierz_najlepsza(indeksy, pole_opt, pole_subopt) -> int:
    wynik = np.maximum(pole_opt, pole_subopt)
    if wynik.size == 0 or wynik.max() <= 0:
        return -1
    return int(indeksy[int(np.argmax(wynik))])


def urzadzenie() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")


def zaladuj_predyktor(katalog_modelu: Path, urz: torch.device | None = None,
                      lustrzane: bool = False) -> nnUNetPredictor:
    urz = urz or urzadzenie()
    predyktor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=lustrzane,
                                device=urz, verbose=False, verbose_preprocessing=False,
                                allow_tqdm=False)
    predyktor.initialize_from_trained_model_folder(str(katalog_modelu), use_folds=(0,),
                                                   checkpoint_name="checkpoint_final.pth")
    return predyktor


def _maski_z_prawdopodobienstw(prob: np.ndarray) -> np.ndarray:
    """prob: (3, n, H, W) → maski (n, H, W) uint8, największa składowa osobno w każdej klatce."""
    prob = prob.copy()
    prob[prob < 0.5] = 0
    maski = np.argmax(prob, axis=0).astype(np.uint8)
    return np.stack([remove_all_but_largest_component_from_segmentation(m, labels_or_regions=[1, 2])
                     for m in maski])


def _prawdopodobienstwa_pelne(predyktor: nnUNetPredictor, klatki: np.ndarray,
                              odstep_mm: float) -> np.ndarray:
    """Pełny nnU-Net (kafle + gauss). klatki (n, H, W) → prob (3, n, H, W)."""
    obraz = klatki[np.newaxis].astype(np.float32)                    # (1, n, H, W)
    wlasciwosci = {"spacing": [1.0, odstep_mm, odstep_mm]}            # docstring nnU-Net: wystarczy 'spacing'
    _, prob = predyktor.predict_single_npy_array(obraz, copy.deepcopy(wlasciwosci), None, None, True)
    return prob


def _prawdopodobienstwa_bezposrednio(predyktor: nnUNetPredictor, klatki: np.ndarray,
                                     odstep_mm: float) -> np.ndarray:
    """Sieć wprost, bez kafli (≈6× szybciej). klatki (n, H, W) → prob (3, n, H, W)."""
    urz = predyktor.device
    net = predyktor.network
    net.load_state_dict(predyktor.list_of_parameters[0])            # wagi folda 0 (na wszelki wypadek)
    net = net.to(urz).eval()
    x = klatki.astype(np.float32)
    mn = x.min(axis=(1, 2), keepdims=True)
    x = (x - mn) / np.clip(x.max(axis=(1, 2), keepdims=True) - mn, 1e-8, None)   # jak RescaleTo01Normalization
    x = torch.from_numpy(x)[:, None]                                  # (n, 1, H, W)
    H, W = klatki.shape[1:]
    H64, W64 = -(-H // 64) * 64, -(-W // 64) * 64
    x = torch.nn.functional.pad(x, (0, W64 - W, 0, H64 - H))
    with torch.no_grad():
        y = net(x.to(urz))
        y = y[0] if isinstance(y, (list, tuple)) else y              # deep supervision → pełna rozdzielczość
        prob = torch.softmax(y.float(), 1)[..., :H, :W].cpu().numpy()  # (n, 3, H, W)
    return prob.transpose(1, 0, 2, 3)                                 # (3, n, H, W)


# po_paczce(numery_klatek, maski, zrobione, wszystkie): opcjonalne wywołanie po KAŻDEJ paczce, także
# złożonej z samych czarnych klatek (wtedy numery i maski są puste), żeby pasek postępu szedł równo;
# maski (n, H, W) uint8 dla niepustych klatek paczki, `zrobione` z `wszystkie` oglądanych klatek.
PoPaczce = Callable[[np.ndarray, np.ndarray, int, int], None]


def _segmentuj(licz_prob, predyktor, sweep: Sweep, co_ktora: int, paczka: int,
               po_paczce: PoPaczce | None = None) -> WynikSweepu:
    start = time.perf_counter()
    indeksy = np.arange(0, len(sweep.klatki), co_ktora)
    pole_opt = np.zeros(len(indeksy), np.int64)
    pole_subopt = np.zeros(len(indeksy), np.int64)
    najlepsza_wartosc, najlepsza_maska = 0, None
    for p in range(0, len(indeksy), paczka):
        idx = indeksy[p:p + paczka]
        niepuste = np.array([sweep.klatki[i].max() > 0 for i in idx])   # czarne klatki = tło
        if not niepuste.any():
            if po_paczce is not None:
                po_paczce(idx[:0], np.zeros((0, *sweep.klatki.shape[1:]), np.uint8),
                          min(p + paczka, len(indeksy)), len(indeksy))
            continue
        maski = _maski_z_prawdopodobienstw(
            licz_prob(predyktor, sweep.klatki[idx[niepuste]], sweep.odstep_mm))
        for j, k in enumerate(np.flatnonzero(niepuste)):
            pole_opt[p + k] = int((maski[j] == 1).sum())
            pole_subopt[p + k] = int((maski[j] == 2).sum())
            wartosc = max(pole_opt[p + k], pole_subopt[p + k])
            if wartosc > najlepsza_wartosc:
                najlepsza_wartosc, najlepsza_maska = wartosc, (maski[j] > 0).astype(np.uint8)
        if po_paczce is not None:
            po_paczce(idx[niepuste], maski, min(p + paczka, len(indeksy)), len(indeksy))
    najlepsza = wybierz_najlepsza(indeksy, pole_opt, pole_subopt)
    return WynikSweepu(indeksy, pole_opt, pole_subopt, najlepsza,
                       najlepsza_maska if najlepsza >= 0 else None, time.perf_counter() - start)


def segmentuj_sweep(predyktor: nnUNetPredictor, sweep: Sweep, co_ktora: int = 4,
                    paczka: int = 64, po_paczce: PoPaczce | None = None) -> WynikSweepu:
    """Referencja: pełny nnU-Net (zmierzone na M1 Pro: ≈ 0,25 s/klatkę). Do przygotowania offline."""
    return _segmentuj(_prawdopodobienstwa_pelne, predyktor, sweep, co_ktora, paczka, po_paczce)


def segmentuj_sweep_szybko(predyktor: nnUNetPredictor, sweep: Sweep, co_ktora: int = 4,
                           paczka: int = 8, po_paczce: PoPaczce | None = None) -> WynikSweepu:
    """Na żywo: sieć bezpośrednio (zmierzone na M1 Pro: ≈ 0,16–0,21 s/klatkę; Dice ≈ 0,9 względem pełnej)."""
    return _segmentuj(_prawdopodobienstwa_bezposrednio, predyktor, sweep, co_ktora, paczka, po_paczce)


def zapisz_wynik(wynik: WynikSweepu, sciezka: Path) -> None:
    Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sciezka, indeksy=wynik.indeksy, pole_opt=wynik.pole_opt,
                        pole_subopt=wynik.pole_subopt, najlepsza_klatka=wynik.najlepsza_klatka,
                        maska=wynik.maska if wynik.maska is not None else np.zeros((0, 0), np.uint8),
                        czas_s=wynik.czas_s)


def wczytaj_wynik(sciezka: Path) -> WynikSweepu:
    d = np.load(sciezka)
    maska = d["maska"] if d["maska"].size else None
    return WynikSweepu(d["indeksy"], d["pole_opt"], d["pole_subopt"],
                       int(d["najlepsza_klatka"]), maska, float(d["czas_s"]))
