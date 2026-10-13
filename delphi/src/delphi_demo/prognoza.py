"""Prognoza z historii chorób: ranking następnych zdarzeń, krzywe ryzyka, generowanie życia token po tokenie.

Matematyka jak u autorów (sampling_trajectories.ipynb, evaluate_delphi.ipynb):
- logit l_k = log tempa (na dzień) zdarzenia k; P(następne = k) = softmax(l)_k,
- ryzyko k w ciągu t lat przy stałym tempie: 1 − exp(−exp(l_k) · 365,25 · t),
- losowanie: dla każdego k czas T_k ~ Exp(exp(l_k)); wygrywa najkrótszy (konkurujące wykładnicze).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from delphi_demo import tokeny
from delphi_demo.tokeny import Zdarzenie

MIN_WIEK_DANYCH = 40.0    # UK Biobank: zapraszano 40–69-latków (faktycznie 37–73 w chwili rekrutacji); wcześniejsze choroby tylko z wywiadu
MAKS_WIEK_DANYCH = 80.0   # dane syntetyczne kończą się na 80. roku życia


@dataclass
class Krok:
    """Jeden krok generowania: wylosowany token, wiek i 5 najbardziej prawdopodobnych kandydatów.

    Kandydaci i `p` liczone są jak `p_nastepna` w `ranking`: spośród chorób i zgonu, bez „brak zdarzenia"
    (czyli „jeśli coś się wydarzy, to z jaką szansą właśnie to"). `p` = szansa wylosowanego tokenu;
    None, gdy wylosowano „brak zdarzenia" (takich kroków nie pokazujemy na sali).
    """
    wiek_lat: float
    token: int
    kandydaci: list[tuple[int, float]]
    p: float | None = None


def ostrzezenia(historia: Sequence[Zdarzenie], wiek_teraz: float) -> list[str]:
    """Uczciwe uwagi o zakresie danych treningowych (do wyświetlenia nad prognozą)."""
    uwagi = []
    if wiek_teraz < MIN_WIEK_DANYCH:
        uwagi.append(f"Wiek {wiek_teraz:g} lat: w UK Biobank nie ma dzieci ani młodych dorosłych "
                     "(zapraszano osoby 40–69 lat; w chwili rekrutacji mieli 37–73 lata). "
                     "Prognoza to ekstrapolacja, a choroby z dzieciństwa "
                     "model zna tylko z relacji dorosłych.")
    if wiek_teraz > MAKS_WIEK_DANYCH:
        uwagi.append(f"Wiek {wiek_teraz:g} lat: dane treningowe kończą się na {MAKS_WIEK_DANYCH:g} r.ż.")
    return uwagi


def przygotuj_wejscie(historia: Sequence[Zdarzenie], plec: str, wiek_teraz: float
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sekwencja dla modelu: płeć (wiek 0), „brak zdarzenia" co 5 lat jak w treningu, choroby,
    a na końcu „brak zdarzenia" w wieku_teraz, żeby model wiedział, ile lat ma pacjent dziś."""
    if plec not in ("K", "M"):
        raise ValueError("plec musi być „K” albo „M”")
    for z in historia:
        if z.wiek_lat > wiek_teraz:
            raise ValueError(f"Zdarzenie w wieku {z.wiek_lat:g} jest po wieku „teraz” ({wiek_teraz:g}).")
    co_ile = tokeny.CO_ILE_LAT_BRAK_ZDARZENIA
    kontrolne = [Zdarzenie(tokeny.BRAK_ZDARZENIA, float(w))
                 for w in np.arange(co_ile, wiek_teraz, co_ile)]
    wpisy = sorted([*historia, *kontrolne], key=lambda z: z.wiek_lat)
    wpisy = [Zdarzenie(tokeny.KOBIETA if plec == "K" else tokeny.MEZCZYZNA, 0.0), *wpisy,
             Zdarzenie(tokeny.BRAK_ZDARZENIA, float(wiek_teraz))]
    idx = torch.tensor([[z.token for z in wpisy]], dtype=torch.long)
    wiek = torch.tensor([[z.wiek_lat * tokeny.DNI_W_ROKU for z in wpisy]], dtype=torch.float32)
    return idx, wiek


def _dozwolone_logity(logity: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Maskuje tokeny techniczne i już obecne choroby (no_repeat jak w Delphi.generate)."""
    l = logity.clone()
    l[..., list(tokeny.IGNOROWANE)] = -torch.inf
    juz = idx.clone()
    juz[juz == tokeny.BRAK_ZDARZENIA] = tokeny.PADDING
    return l.scatter(-1, juz, -torch.inf)


@torch.no_grad()
def ranking(model, historia: Sequence[Zdarzenie], plec: str, wiek_teraz: float,
            top: int = 10) -> pd.DataFrame:
    """Najbardziej prawdopodobne NASTĘPNE zdarzenia (choroby i zgon) dla pacjenta „dziś".

    Kolumny:
    - `p_nastepna` – prawdopodobieństwo WARUNKOWE: „pod warunkiem, że następnym wpisem będzie choroba
      albo zgon, z jaką szansą będzie to właśnie ta". Softmax liczony bez „brak zdarzenia"
      (ten token ma zwykle ok. 1/3 całej masy i jest wyłączony), więc kolumna sumuje się do 1 po wszystkich
      chorobach i zgonie, a nie mówi, czy cokolwiek wydarzy się w ogóle.
    - `ryzyko_1_rok`, `ryzyko_5_lat` – 1 − exp(−tempo · t) przy tempie zamrożonym na „dziś"
      (5 lat to ekstrapolacja stałego tempa, nie losowanie Monte Carlo jak w `krzywe_ryzyka`).
    """
    idx, wiek = przygotuj_wejscie(historia, plec, wiek_teraz)
    urz = next(model.parameters()).device
    logity = model(idx.to(urz), wiek.to(urz))[0][0, -1].cpu()
    l = _dozwolone_logity(logity[None], idx)[0]
    l[tokeny.BRAK_ZDARZENIA] = -torch.inf
    p = torch.softmax(l, 0)
    tempo_dzien = torch.exp(l)
    kolejnosc = torch.argsort(l, descending=True)[:top]
    return pd.DataFrame({
        "token": kolejnosc.numpy(),
        "p_nastepna": p[kolejnosc].numpy(),
        "ryzyko_1_rok": (1 - torch.exp(-tempo_dzien[kolejnosc] * tokeny.DNI_W_ROKU)).numpy(),
        "ryzyko_5_lat": (1 - torch.exp(-tempo_dzien[kolejnosc] * tokeny.DNI_W_ROKU * 5)).numpy(),
    })


@torch.no_grad()
def generuj(model, historia: Sequence[Zdarzenie], plec: str, wiek_teraz: float,
            do_wieku: float = MAKS_WIEK_DANYCH, maks_krokow: int = 80, ziarno: int | None = 0,
            top_kandydatow: int = 5) -> list[Krok]:
    """Losuje dalszy ciąg życia token po tokenie (ta sama reguła co Delphi.generate autorów).

    ziarno=None używa globalnego generatora torch (tak robi Delphi.generate; test zgodności).
    Kroki „brak zdarzenia" też są zwracane (pokazują upływ czasu); do wyświetlania można je pominąć.
    """
    gen = None if ziarno is None else torch.Generator().manual_seed(ziarno)
    idx, wiek = przygotuj_wejscie(historia, plec, wiek_teraz)
    urz = next(model.parameters()).device
    kroki: list[Krok] = []
    for _ in range(maks_krokow):
        logity = model(idx.to(urz), wiek.to(urz))[0][:, -1, :].cpu()
        l = _dozwolone_logity(logity, idx)
        u = torch.rand(l.shape, generator=gen)
        czasy = torch.clamp(-torch.exp(-l) * u.log(), min=0, max=365 * 80)
        dt, nastepny = czasy.min(1)
        nowy_wiek = wiek[0, -1] + dt[0]
        if nowy_wiek > do_wieku * tokeny.DNI_W_ROKU:
            break
        # Kandydaci jak w `ranking`: bez „brak zdarzenia". Nie zużywa losowości (parzystość z generate autorów).
        l_zd = l[0].clone()
        l_zd[tokeny.BRAK_ZDARZENIA] = -torch.inf
        p = torch.softmax(l_zd, 0)
        top = torch.topk(p, top_kandydatow)
        wylosowany = int(nastepny[0])
        kroki.append(Krok(float(nowy_wiek) / tokeny.DNI_W_ROKU, wylosowany,
                          [(int(t), float(v)) for v, t in zip(top.values, top.indices)],
                          None if wylosowany == tokeny.BRAK_ZDARZENIA else float(p[wylosowany])))
        idx = torch.cat([idx, nastepny[:, None]], 1)
        wiek = torch.cat([wiek, nowy_wiek.reshape(1, 1)], 1)
        if int(nastepny[0]) == tokeny.SMIERC:
            break
    return kroki


@torch.no_grad()
def krzywe_ryzyka(model, historia: Sequence[Zdarzenie], plec: str, wiek_teraz: float,
                  cele: Sequence[int], horyzont_lat: int = 10, n_probek: int = 200,
                  ziarno: int = 0, maks_krokow: int = 64) -> pd.DataFrame:
    """Monte Carlo: losujemy n_probek dalszych żyć (Delphi.generate autorów) i liczymy,
    jaki odsetek z nich ma daną chorobę do końca każdego roku. Zgon konkuruje z chorobami.

    Wiersz = lata od „dziś" (0..horyzont), kolumna = token celu. Cel już obecny w historii → NaN
    (model nigdy nie losuje drugi raz tej samej choroby, więc 0 byłoby mylące).
    """
    torch.manual_seed(ziarno)
    idx, wiek = przygotuj_wejscie(historia, plec, wiek_teraz)
    urz = next(model.parameters()).device
    koniec = (wiek_teraz + horyzont_lat) * tokeny.DNI_W_ROKU
    gi, ga, _ = model.generate(idx.repeat(n_probek, 1).to(urz), wiek.repeat(n_probek, 1).to(urz),
                               max_new_tokens=maks_krokow, max_age=koniec,
                               termination_tokens=[tokeny.SMIERC])
    gi, ga = gi[:, idx.shape[1]:].cpu(), ga[:, idx.shape[1]:].cpu()
    lata = np.arange(horyzont_lat + 1)
    obecne = {z.token for z in historia}
    wynik = {"lata": lata}
    for cel in cele:
        if cel in obecne:
            wynik[cel] = np.full(len(lata), np.nan)
            continue
        trafienia = gi == cel
        wiek_zd = torch.where(trafienia, ga, torch.full_like(ga, torch.inf)).min(1).values
        lata_do = ((wiek_zd / tokeny.DNI_W_ROKU) - wiek_teraz).numpy()
        wynik[cel] = np.array([(lata_do <= t).mean() for t in lata])
    return pd.DataFrame(wynik)


def ryzyko_stale_tempo(model, historia: Sequence[Zdarzenie], plec: str, wiek_teraz: float,
                       cele: Sequence[int], horyzont_lat: int = 10) -> pd.DataFrame:
    """Szybkie przybliżenie bez losowania: 1 − exp(−tempo · t), tempo zamrożone na „dziś"."""
    idx, wiek = przygotuj_wejscie(historia, plec, wiek_teraz)
    urz = next(model.parameters()).device
    with torch.no_grad():
        l = _dozwolone_logity(model(idx.to(urz), wiek.to(urz))[0][:, -1, :].cpu(), idx)[0]
    lata = np.arange(horyzont_lat + 1)
    obecne = {z.token for z in historia}
    wynik = {"lata": lata}
    for cel in cele:
        tempo = float(torch.exp(l[cel])) * tokeny.DNI_W_ROKU
        wynik[cel] = np.full(len(lata), np.nan) if cel in obecne else 1 - np.exp(-tempo * lata)
    return pd.DataFrame(wynik)
