"""Logika aplikacji pokazowej (app/demo_usg.py): ten sam pipeline co notebook, bez Gradio.

analizuj(zrodlo, tryb) → Analiza: nnU-Net (klatka z największym polem brzucha → AC → masa teraz),
FUVAI (z regułą „bez masy, gdy niepewny” jak w notebooku) i model Sano (masa w dniu porodu).
Tryb „z_pamieci” czyta wyniki z usg/pamiec/ (bez sieci i bez modeli), „na_zywo” liczy je teraz:
nnU-Net skróconą ścieżką co 4. klatkę (MPS), FUVAI co 4. klatkę, model Sano na CPU.

Modele wczytujemy leniwie i trzymamy w pamięci (tylko tryb „na_zywo”). Blokada pilnuje, żeby
na MPS nie liczyły dwa zadania naraz. Tu są też rysunki (klatka z obrysem, wykres pola brzucha):
czyste funkcje numpy/OpenCV/matplotlib (Figure bez pyplot, bez zmiany backendu), które aplikacja
tylko wyświetla.
"""
from __future__ import annotations

import contextlib
import io
import logging
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

# nnU-Net bez tych zmiennych drukuje ostrzeżenia (jak w notebooku); muszą być przed importem segmentacji.
_PAMIEC = Path(__file__).resolve().parents[2] / "pamiec"
for _zmienna in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(_zmienna, str(_PAMIEC))
os.environ.setdefault("nnUNet_n_proc_DA", "12")

import cv2  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from usg_masa import biometria, dane, fuvai, model_sano, segmentacja as seg, zasoby  # noqa: E402

# Oznaczenia nagrań jak na slajdach.
NAGRANIA = {
    "Nr 1": "0199616b-bdeb-4119-97a3-a5a3571bd641",
    "Nr 2": "02d3a9bc-63e2-4deb-9dc1-dba17e7d54c1",
    "Nr 3": "02ee26a5-a665-4531-bec1-8bac83345a94",
    "Wycinek": "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000",
}
Z_PAMIECI, NA_ZYWO = "z_pamieci", "na_zywo"
CO_KTORA_NA_ZYWO = 4
KLATEK_PRZECIAGNIECIA = 140      # pełne nagranie: 840 klatek = 6 przeciągnięć głowicą po 140
MASA_UR_UCZACA, SD_UCZACA = 3495, 507    # masa urodzeniowa w zbiorze uczącym modelu Sano: średnia ± SD
# Ile klatek brzucha potrzeba na jeden pełny segment modelu Sano (16 klatek co 3. → 46).
KLATEK_NA_SEGMENT = (model_sano.SKIP + 1) * (model_sano.KLATKI - 1) + 1
# Zmierzone na M1 Pro (aplikacja, na żywo, co 4. klatka): 840 klatek ≈ 25 s razem z wczytaniem
# modeli (ok. 2 s), wycinek 128 klatek ≈ 5–10 s; pierwsze liczenie po starcie bywa wolniejsze (MPS).
CZAS_WCZYTANIA_MODELI_S = 2.0
CZAS_STALY_S = 3.0
CZAS_NA_KLATKE_S = 20.0 / 840
ATRYBUCJA_WGRANEGO = "Dane: plik wgrany przez użytkownika"

# Kolory (RGB) jak w slajdach: cyjan nnU-Net, bursztyn elipsa, fiolet FUVAI.
CYJAN, BURSZTYN, FIOLET = (0, 188, 222), (255, 200, 87), (201, 166, 255)

# Rozsądne granice dla wgranego pliku (ACOUSLIC-AI: 840 × 562 × 744).
MAX_KLATEK, MIN_BOK_PX, MAX_BOK_PX = 5000, 32, 4096

log = logging.getLogger("usg_masa.aplikacja")


class BrakWynikowWPamieci(FileNotFoundError):
    """Dla tego nagrania nie ma wyników policzonych wcześniej (usg/pamiec/)."""


class NiepoprawnyPlik(ValueError):
    """Wgrany plik nie jest nagraniem w formacie ACOUSLIC-AI (.mha, 3D, skala szarości)."""


# ── liczby po polsku ─────────────────────────────────────────────────────────

def po_polsku(x: float, miejsca: int = 1) -> str:
    """Liczba z przecinkiem dziesiętnym."""
    return f"{x:.{miejsca}f}".replace(".", ",")


def gramy(x: float) -> str:
    """3551,2 → „3 551” (twarda spacja tysięcy)."""
    return f"{x:,.0f}".replace(",", " ")


def odmiana(n: int, jeden: str, kilka: str, wiele: str) -> str:
    if n == 1:
        return jeden
    return kilka if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else wiele


# ── nagrania ─────────────────────────────────────────────────────────────────

def sciezka_nagrania(etykieta: str) -> Path:
    """„Nr 1”…„Nr 3” → pełne nagranie z Zenodo; „Wycinek” → 128 klatek z repo baseline'u.

    Tylko składa ścieżkę (bez sieci i bez gita); czy plik jest, mówi dostepne_nagrania().
    """
    if etykieta == "Wycinek":
        return zasoby.KATALOG_ZEWN / "ACOUSLIC-AI-baseline" / dane._STARTOWY
    return dane.KATALOG_DANYCH / "zenodo" / dane.PREFIKS_OBRAZOW / f"{NAGRANIA[etykieta]}.mha"


def dostepne_nagrania() -> list[str]:
    """Etykiety nagrań, których pliki są na dysku (wycinek: tylko gdy repo baseline'u jest sklonowane)."""
    return [etykieta for etykieta in NAGRANIA if sciezka_nagrania(etykieta).exists()]


def wbudowane(zrodlo: str | Path | None) -> bool:
    """True dla nagrań z pokazu (dane ACOUSLIC-AI), False dla pliku wgranego przez użytkownika."""
    return isinstance(zrodlo, str) and zrodlo in NAGRANIA


def atrybucja(zrodlo: str | Path | None) -> str:
    return dane.ATRYBUCJA if wbudowane(zrodlo) else ATRYBUCJA_WGRANEGO


def szacowany_czas_na_zywo_s(n_klatek: int, z_wczytaniem: bool = True) -> float:
    """Ile mniej więcej trwa „na żywo” (M1 Pro, co 4. klatka): 840 klatek ≈ 25 s z wczytaniem modeli."""
    return CZAS_STALY_S + CZAS_NA_KLATKE_S * n_klatek + (CZAS_WCZYTANIA_MODELI_S if z_wczytaniem else 0.0)


@lru_cache(maxsize=2)
def _wczytaj(sciezka: str) -> dane.Sweep:
    return dane.wczytaj_mha(Path(sciezka))


def wczytaj_sweep(zrodlo: str | Path) -> dane.Sweep:
    """Etykieta („Nr 1”, „Wycinek”) albo ścieżka do pliku .mha; dwa ostatnie nagrania w pamięci."""
    sciezka = sciezka_nagrania(zrodlo) if wbudowane(zrodlo) else Path(zrodlo)
    return _wczytaj(str(sciezka))


def sprawdz_plik(sciezka: str | Path) -> dane.Sweep:
    """Wczytuje plik wgrany przez użytkownika albo rzuca NiepoprawnyPlik (szczegóły po angielsku/technicznie,
    do logu; na ekran idzie krótki polski komunikat z aplikacji).

    Wymagamy: plik czytelny dla SimpleITK, obraz 3D (klatki × wysokość × szerokość), jeden kanał,
    wartości liczbowe dające się przenieść do 0–255 i rozsądne wymiary.
    """
    sciezka = Path(sciezka)
    try:
        czytnik = sitk.ImageFileReader()
        czytnik.SetFileName(str(sciezka))
        czytnik.ReadImageInformation()
    except Exception as blad:   # SimpleITK rzuca RuntimeError z długim tekstem
        raise NiepoprawnyPlik(f"{sciezka.name}: SimpleITK nie czyta pliku: {blad}") from blad
    wymiary = czytnik.GetSize()                       # (szerokość, wysokość, klatki)
    if czytnik.GetDimension() != 3:
        raise NiepoprawnyPlik(f"{sciezka.name}: obraz {czytnik.GetDimension()}D, a potrzebny 3D")
    if czytnik.GetNumberOfComponents() != 1:
        raise NiepoprawnyPlik(f"{sciezka.name}: {czytnik.GetNumberOfComponents()} kanały na piksel, a potrzebny 1")
    szer, wys, n = wymiary
    if not (1 <= n <= MAX_KLATEK and MIN_BOK_PX <= szer <= MAX_BOK_PX and MIN_BOK_PX <= wys <= MAX_BOK_PX):
        raise NiepoprawnyPlik(f"{sciezka.name}: nietypowe wymiary {wymiary}")
    try:
        sweep = wczytaj_sweep(sciezka)
    except Exception as blad:
        raise NiepoprawnyPlik(f"{sciezka.name}: nie da się wczytać klatek: {blad!r}") from blad
    if sweep.klatki.ndim != 3 or sweep.klatki.dtype != np.uint8 or not np.isfinite(sweep.odstep_mm) \
            or sweep.odstep_mm <= 0:
        _wczytaj.cache_clear()
        raise NiepoprawnyPlik(f"{sciezka.name}: klatki {sweep.klatki.shape} {sweep.klatki.dtype}, "
                              f"odstęp {sweep.odstep_mm}")
    if not sweep.klatki.any():
        _wczytaj.cache_clear()
        raise NiepoprawnyPlik(f"{sciezka.name}: wszystkie klatki są czarne")
    return sweep


@lru_cache(maxsize=1)
def referencja():
    """AC ekspertów ACOUSLIC-AI (CSV z Zenodo) albo None, gdy pliku nie ma."""
    csv = dane.KATALOG_DANYCH / "zenodo" / dane.CSV_OBWODOW
    return dane.wczytaj_obwody_referencyjne(csv) if csv.exists() else None


def ma_wyniki_w_pamieci(nazwa: str) -> bool:
    return all((seg.KATALOG_PAMIECI / f"{nazwa}{k}").exists() for k in (".npz", "_fuvai.npz", "_sano.json"))


def przeciagniecie(klatka: int, n_klatek: int) -> int | None:
    """Numer przeciągnięcia głowicą (1–6) w pełnym nagraniu; None dla wycinka i innych plików."""
    return klatka // KLATEK_PRZECIAGNIECIA + 1 if n_klatek == 6 * KLATEK_PRZECIAGNIECIA and klatka >= 0 else None


# ── modele (tylko na żywo) ───────────────────────────────────────────────────

_BLOKADA_MPS = threading.Lock()      # jedno zadanie na MPS naraz
_BLOKADA_MODELI = threading.Lock()
_MODELE: dict[str, object] = {}


def _model(nazwa: str):
    with _BLOKADA_MODELI:
        if nazwa not in _MODELE:
            if nazwa == "nnunet":
                with contextlib.redirect_stdout(io.StringIO()):   # techniczne komunikaty nnU-Netu
                    _MODELE[nazwa] = seg.zaladuj_predyktor(zasoby.katalog_modelu_nnunet())
            elif nazwa == "fuvai":
                _MODELE[nazwa] = fuvai.zbuduj_fuvai()
            elif nazwa == "sano":
                _MODELE[nazwa] = model_sano.zbuduj_model_sano()
            else:
                raise KeyError(nazwa)
        return _MODELE[nazwa]


def modele_wczytane() -> bool:
    return {"nnunet", "fuvai", "sano"} <= set(_MODELE)


def rozgrzej() -> float:
    """Wczytuje wszystkie trzy modele (np. w tle po przełączeniu na „na żywo”). Zwraca czas w s."""
    start = time.perf_counter()
    for nazwa in ("nnunet", "fuvai", "sano"):
        _model(nazwa)
    return time.perf_counter() - start


# ── wynik analizy ────────────────────────────────────────────────────────────

@dataclass
class OcenaFUVAI:
    znalazl: bool
    klatka: int
    ac_mm: float | None
    masa_g: float | None          # None także wtedy, gdy masy świadomie nie podajemy
    tydzien: float | None
    p_brzuch: float               # p(brzuch) wybranej klatki (0, gdy nie znalazł)
    p_max: float                  # najwyższe p(brzuch) w całym nagraniu
    rozklad: list[tuple[str, float]]   # co FUVAI widzi w wybranej klatce, malejąco
    pole_nnunet: int | None       # pole brzucha wg nnU-Netu w tej klatce; None = nnU-Net jej nie oglądał
    bez_masy: bool                # p < 0,9 i nnU-Net nie widzi tu brzucha → masy nie podajemy
    komunikat: str


def ocen_fuvai(F: fuvai.WynikFUVAI, W: seg.WynikSweepu, odstep_mm: float) -> OcenaFUVAI:
    """Ta sama reguła co w notebooku: bez masy, gdy p(brzuch) < 0,9, a nnU-Net nie widzi w tej klatce brzucha."""
    p_max = float(F.p_klas[:, fuvai.BRZUCH].max()) if len(F.p_klas) else 0.0
    if F.najlepsza_klatka < 0 or F.maska is None:
        return OcenaFUVAI(False, -1, None, None, None, 0.0, p_max, [], None, True,
                          f"FUVAI nie znalazł w tym nagraniu klatki brzucha (najwyższa pewność "
                          f"{po_polsku(p_max, 2)}). Nie podajemy AC ani masy.")
    o = biometria.oszacuj(F.maska, odstep_mm)
    if W.najlepsza_klatka >= 0 and W.maska is not None and W.najlepsza_klatka == F.najlepsza_klatka:
        pole = int(np.count_nonzero(W.maska))
    elif F.najlepsza_klatka in W.indeksy:
        j = int(np.flatnonzero(W.indeksy == F.najlepsza_klatka)[0])
        pole = int(W.pole_opt[j] + W.pole_subopt[j])
    else:
        pole = None
    bez_masy = F.p_brzuch_najlepszej < fuvai.PROG_ARTYKULU and pole == 0
    p_wybranej = F.p_klas[int(np.flatnonzero(F.indeksy == F.najlepsza_klatka)[0])]
    rozklad = [(fuvai.KLASY[k], float(p_wybranej[k])) for k in np.argsort(-p_wybranej)]
    masa = None if bez_masy or not o.wiarygodne else o.masa_g
    if bez_masy:
        komunikat = (f"Masy nie podajemy: FUVAI nie jest pewny (p(brzuch) {po_polsku(F.p_brzuch_najlepszej, 2)} "
                     f"< {po_polsku(fuvai.PROG_ARTYKULU)}), a nnU-Net nie widzi w tej klatce brzucha. "
                     "To może być inna struktura (np. główka).")
    else:
        komunikat = o.komunikat
    return OcenaFUVAI(True, F.najlepsza_klatka, o.ac_mm, masa, o.tydzien if masa is not None else None,
                      F.p_brzuch_najlepszej, p_max, rozklad, pole, bez_masy, komunikat)


@dataclass
class Analiza:
    nazwa: str
    tryb: str
    n_klatek: int
    odstep_mm: float
    nnunet: seg.WynikSweepu
    pomiar: biometria.Oszacowanie
    plaszczyzna: str | None           # „optymalna” / „suboptymalna (skośna)”
    eksperci: list[float]
    przeciagniecie: int | None        # z którego przeciągnięcia głowicą jest wybrana klatka
    ekspert_przeciagniecia: float | None
    fuvai: fuvai.WynikFUVAI
    ocena_fuvai: OcenaFUVAI
    sano: model_sano.WynikSano
    czasy: dict[str, float] = field(default_factory=dict)
    obrysy: dict[int, list[np.ndarray]] = field(default_factory=dict)   # na żywo: kontury nnU-Netu w każdej klatce
    wbudowane: bool = True            # nagranie z pokazu (ACOUSLIC-AI); False = plik wgrany przez użytkownika

    @property
    def krok(self) -> int:
        i = self.nnunet.indeksy
        return int(i[1] - i[0]) if len(i) > 1 else 1

    @property
    def masa_teraz_g(self) -> float | None:
        return self.pomiar.masa_g if self.pomiar.wiarygodne else None


def _kontury(maska: np.ndarray) -> list[np.ndarray]:
    kontury, _ = cv2.findContours((maska > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(kontury)


def analizuj(zrodlo: str | Path, tryb: str = Z_PAMIECI, co_ktora: int = CO_KTORA_NA_ZYWO,
             postep: Callable[[float, str], None] | None = None) -> Analiza:
    """Cały pipeline dla nagrania (etykieta jak „Nr 1” albo ścieżka .mha).

    postep(ułamek 0–1, opis) — opcjonalnie, do paska postępu (tylko na żywo).
    """
    postep = postep or (lambda *_: None)
    sweep = wczytaj_sweep(zrodlo)
    czasy: dict[str, float] = {}
    obrysy: dict[int, list[np.ndarray]] = {}
    if tryb == Z_PAMIECI:
        if not ma_wyniki_w_pamieci(sweep.nazwa):
            raise BrakWynikowWPamieci(sweep.nazwa)
        W = seg.wczytaj_wynik(seg.KATALOG_PAMIECI / f"{sweep.nazwa}.npz")
        F = fuvai.wczytaj_wynik(seg.KATALOG_PAMIECI / f"{sweep.nazwa}_fuvai.npz")
        S = model_sano.wczytaj_wynik(seg.KATALOG_PAMIECI / f"{sweep.nazwa}_sano.json")
    elif tryb == NA_ZYWO:
        if _BLOKADA_MPS.locked():
            postep(0.0, "Czekam, aż skończy się poprzednie liczenie…")
        with _BLOKADA_MPS:
            start = time.perf_counter()
            postep(0.0, "Wczytuję modele…")
            pred, model_f, model_s = _model("nnunet"), _model("fuvai"), _model("sano")
            czasy["wczytanie"] = time.perf_counter() - start

            def po_paczce(numery, maski, zrobione, wszystkie):   # także dla paczek samych czarnych klatek
                for i, m in zip(numery, maski):
                    if m.any():
                        obrysy[int(i)] = _kontury(m)
                postep(0.85 * zrobione / wszystkie, "nnU-Net szuka brzucha…")

            W = seg.segmentuj_sweep_szybko(pred, sweep, co_ktora=co_ktora, po_paczce=po_paczce)
            czasy["nnunet"] = W.czas_s
            postep(0.87, "FUVAI ocenia klatki…")
            with warnings.catch_warnings():   # ostrzeżenia PyTorcha o przestarzałych funkcjach w kodzie autorów
                warnings.simplefilter("ignore", UserWarning)
                F = fuvai.analizuj_sweep(model_f, sweep, co_ktora=co_ktora)
            czasy["fuvai"] = F.czas_s
            postep(0.97, "Model Sano szacuje masę w dniu porodu…")
            S = model_sano.przewidz(model_s, sweep, W)
            czasy["sano"] = S.czas_s
            czasy["razem"] = time.perf_counter() - start
            postep(1.0, "Gotowe")
    else:
        raise ValueError(f"Nieznany tryb: {tryb}")

    o = biometria.oszacuj(W.maska, sweep.odstep_mm)
    plaszczyzna = None
    if W.najlepsza_klatka >= 0:
        i = int(np.flatnonzero(W.indeksy == W.najlepsza_klatka)[0])
        plaszczyzna = "optymalna" if W.pole_opt[i] >= W.pole_subopt[i] else "suboptymalna (skośna)"
    ref = referencja()
    eksperci = dane.ac_referencyjne(ref, sweep.nazwa) if ref is not None else []
    nr = przeciagniecie(W.najlepsza_klatka, len(sweep.klatki))
    ekspert_nr = None
    if ref is not None and nr is not None and eksperci:
        v = ref.loc[sweep.nazwa.split("_")[0], f"sweep_{nr}_ac_mm"]
        ekspert_nr = float(v) if v == v else None   # NaN ≠ NaN
    return Analiza(sweep.nazwa, tryb, len(sweep.klatki), sweep.odstep_mm, W, o, plaszczyzna, eksperci, nr,
                   ekspert_nr, F, ocen_fuvai(F, W, sweep.odstep_mm), S, czasy, obrysy, wbudowane(zrodlo))


# ── rysunki ──────────────────────────────────────────────────────────────────

_PASEK_ATRYBUCJI = 30


def _z_atrybucja(rgb: np.ndarray, atrybucja: bool = True) -> np.ndarray:
    """Dokleja pod obrazem pasek z atrybucją danych ACOUSLIC-AI (tekst ASCII, więc OpenCV wystarczy).

    atrybucja=False (plik wgrany przez użytkownika): bez paska, to nie są dane ACOUSLIC-AI.
    """
    if not atrybucja:
        return rgb
    h, w = rgb.shape[:2]
    pasek = np.full((_PASEK_ATRYBUCJI, w, 3), (16, 16, 28), np.uint8)
    skala = 0.5 if w >= 700 else 0.4
    (tw, th), _ = cv2.getTextSize(dane.ATRYBUCJA, cv2.FONT_HERSHEY_SIMPLEX, skala, 1)
    cv2.putText(pasek, dane.ATRYBUCJA, ((w - tw) // 2, (_PASEK_ATRYBUCJI + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, skala, (200, 200, 210), 1, cv2.LINE_AA)
    return np.vstack([rgb, pasek])


def _elipsa_przerywana(rgb: np.ndarray, maska: np.ndarray, kolor, grubosc: int = 2) -> None:
    kontury = _kontury(maska)
    if not kontury:
        return
    kontur = max(kontury, key=cv2.contourArea)
    if len(kontur) < 5:
        return
    (x, y), (d1, d2), kat = cv2.fitEllipse(kontur)
    punkty = cv2.ellipse2Poly((int(round(x)), int(round(y))), (int(round(d1 / 2)), int(round(d2 / 2))),
                              int(round(kat)), 0, 360, 2)
    for k in range(0, len(punkty) - 1, 6):   # kreska 3 odcinki, przerwa 3
        cv2.polylines(rgb, [punkty[k:k + 4]], False, kolor, grubosc, cv2.LINE_AA)


def obraz_klatki(sweep: dane.Sweep, i: int, analiza: Analiza | None = None,
                 elipsa: bool = False, atrybucja: bool = True) -> tuple[np.ndarray, str]:
    """Klatka i (RGB, z paskiem atrybucji) i krótki opis, co na niej zaznaczono.

    Obrysy: wybrana klatka nnU-Netu (cyjan, z elipsą przerywaną gdy elipsa=True), klatka FUVAI (fiolet),
    a po analizie na żywo także obrys nnU-Netu z najbliższej przeanalizowanej klatki ≤ i.
    """
    i = int(np.clip(i, 0, len(sweep.klatki) - 1))
    rgb = cv2.cvtColor(sweep.klatki[i], cv2.COLOR_GRAY2RGB)
    opis = ""
    if analiza is not None and analiza.nazwa == sweep.nazwa:
        W, F = analiza.nnunet, analiza.fuvai
        if i == W.najlepsza_klatka and W.maska is not None:
            cv2.drawContours(rgb, _kontury(W.maska), -1, CYJAN, 3, cv2.LINE_AA)
            if elipsa and analiza.pomiar.ac_mm is not None:
                _elipsa_przerywana(rgb, W.maska, BURSZTYN, 2)
            opis = "klatka wybrana przez nnU-Net (największe pole brzucha)"
        elif analiza.obrysy:
            poz = int(np.searchsorted(W.indeksy, i, side="right")) - 1
            k = int(W.indeksy[poz]) if poz >= 0 else -1
            if k in analiza.obrysy:
                cv2.drawContours(rgb, analiza.obrysy[k], -1, CYJAN, 2, cv2.LINE_AA)
                opis = "nnU-Net widzi tu brzuch" + ("" if k == i else f" (obrys z klatki {k})")
        else:
            poz = int(np.searchsorted(W.indeksy, i, side="right")) - 1
            if poz >= 0 and max(W.pole_opt[poz], W.pole_subopt[poz]) > 0:
                opis = f"nnU-Net widzi w tej części nagrania brzuch (klatka {int(W.indeksy[poz])}: " \
                       f"{gramy(max(W.pole_opt[poz], W.pole_subopt[poz]))} px)"
        if i == F.najlepsza_klatka and F.maska is not None:
            cv2.drawContours(rgb, _kontury(F.maska), -1, FIOLET, 3, cv2.LINE_AA)
            opis = (opis + " · " if opis else "") + "klatka wybrana przez FUVAI"
    return _z_atrybucja(rgb, atrybucja), opis


def obraz_wybranej(sweep: dane.Sweep, analiza: Analiza) -> np.ndarray | None:
    """Klatka wybrana przez nnU-Net: obrys (cyjan) i dopasowana elipsa (bursztyn, przerywana)."""
    if analiza.nnunet.najlepsza_klatka < 0:
        return None
    return obraz_klatki(sweep, analiza.nnunet.najlepsza_klatka, analiza, elipsa=True,
                        atrybucja=analiza.wbudowane)[0]


def obraz_fuvai(sweep: dane.Sweep, analiza: Analiza) -> np.ndarray | None:
    """Klatka wybrana przez FUVAI z jego obrysem (fiolet)."""
    F = analiza.fuvai
    if F.najlepsza_klatka < 0 or F.maska is None:
        return None
    rgb = cv2.cvtColor(sweep.klatki[F.najlepsza_klatka], cv2.COLOR_GRAY2RGB)
    cv2.drawContours(rgb, _kontury(F.maska), -1, FIOLET, 3, cv2.LINE_AA)
    return _z_atrybucja(rgb, analiza.wbudowane)


def wykres(analiza: Analiza, biezaca: int | None = None) -> np.ndarray:
    """Pole brzucha wg nnU-Netu w kolejnych klatkach (+ p(brzuch) FUVAI pod spodem), RGB."""
    tlo, tekst, siatka = "#13131f", "#e8e8f0", "#3a3a52"
    fig = Figure(figsize=(12, 3.2), dpi=110, facecolor=tlo)
    FigureCanvasAgg(fig)
    gora, dol = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12})
    W, F = analiza.nnunet, analiza.fuvai
    gora.fill_between(W.indeksy, W.pole_opt, color="#00BCDE", alpha=0.25, lw=0)
    gora.plot(W.indeksy, W.pole_opt, color="#00BCDE", lw=2, label="płaszczyzna optymalna")
    gora.fill_between(W.indeksy, W.pole_subopt, color="#FFAF7B", alpha=0.2, lw=0)
    gora.plot(W.indeksy, W.pole_subopt, color="#FFAF7B", lw=2, label="suboptymalna (skośna)")
    gora.set_ylabel("pole brzucha\n[tys. px]", color=tekst, fontsize=12)
    gora.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: po_polsku(v / 1000, 0)))
    p = F.p_klas[:, fuvai.BRZUCH]
    dol.plot(F.indeksy, p, color="#C9A6FF", lw=1.6, label="p(brzuch) wg FUVAI")
    dol.axhline(fuvai.PROG_ARTYKULU, color="#9a9ab0", ls="--", lw=1.2)
    dol.text(analiza.n_klatek * 0.995, fuvai.PROG_ARTYKULU + 0.03, "próg 0,9 z artykułu", color="#b8b8cc",
             fontsize=10.5, ha="right", va="bottom")
    dol.set_ylim(0, 1.05)
    dol.set_yticks([0, 0.5, 1], ["0", "0,5", "1"])
    dol.set_ylabel("FUVAI\np(brzuch)", color=tekst, fontsize=12)
    dol.set_xlabel("numer klatki", color=tekst, fontsize=12)
    for os_ in (gora, dol):
        os_.set_facecolor(tlo)
        os_.tick_params(colors=tekst, labelsize=11)
        for s in os_.spines.values():
            s.set_color(siatka)
        os_.spines[["top", "right"]].set_visible(False)
        os_.set_xlim(0, analiza.n_klatek - 1)
        if analiza.n_klatek == 6 * KLATEK_PRZECIAGNIECIA:
            for g in range(KLATEK_PRZECIAGNIECIA, analiza.n_klatek, KLATEK_PRZECIAGNIECIA):
                os_.axvline(g, color=siatka, lw=1)
        if W.najlepsza_klatka >= 0:
            os_.axvline(W.najlepsza_klatka, color="#ffffff", ls=(0, (4, 3)), lw=1.8)
        if F.najlepsza_klatka >= 0 and os_ is dol:
            os_.axvline(F.najlepsza_klatka, color="#C9A6FF", ls=":", lw=2)
        if biezaca is not None:
            os_.axvline(biezaca, color="#D76D77", lw=1.4, alpha=0.9)
    if analiza.n_klatek == 6 * KLATEK_PRZECIAGNIECIA:
        for nr in range(6):
            gora.text((nr + 0.5) * KLATEK_PRZECIAGNIECIA, 1.0, f"przeciągnięcie {nr + 1}", color="#9a9ab4",
                      fontsize=10, ha="center", va="bottom", transform=gora.get_xaxis_transform())
    if W.najlepsza_klatka >= 0:
        gora.annotate(f"wybrana klatka {W.najlepsza_klatka}", (W.najlepsza_klatka, 0.93), xytext=(8, 0),
                      textcoords="offset points", xycoords=("data", "axes fraction"), color="#ffffff",
                      fontsize=12, fontweight="bold", va="top",
                      bbox=dict(facecolor=tlo, edgecolor="none", pad=1.5, alpha=0.85))
    gora.legend(loc="upper right", bbox_to_anchor=(1.0, 0.85), fontsize=11.5, frameon=False, labelcolor=tekst)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.16)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
