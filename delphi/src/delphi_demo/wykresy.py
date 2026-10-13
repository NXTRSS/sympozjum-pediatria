"""Wykresy dla sali: oś życia pacjenta, ranking, krzywe ryzyka, krzywa uczenia, ślad generowania.

Funkcje nie znają modelu: dostają gotowe tabele i nazwy (funkcja `nazwa(token) -> str`).
Każda zwraca Figure (notebook robi plt.show(); testy tylko sprawdzają, że rysunek powstał).
"""
from __future__ import annotations

from typing import Callable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from delphi_demo import tokeny
from delphi_demo.tokeny import Zdarzenie

Nazwa = Callable[[int], str]
ATRYBUCJA = ("Model: Delphi (Shmatko i in., Nature 2025), kod MIT; trenowany przez nas "
             "na danych SYNTETYCZNYCH autorów, nie na UK Biobank")
ATRYBUCJA_STYL = dict(fontsize=8, color="#555")   # czytelne z projektora (wcześniej 6 pt szarym)


def _pl(x: float, cyfry: int = 1) -> str:
    """Liczba z przecinkiem dziesiętnym (po polsku)."""
    return f"{x:.{cyfry}f}".replace(".", ",")


def os_zycia(historia: Sequence[Zdarzenie], nazwa: Nazwa, kolor: Callable[[int], str] | None = None,
             wygenerowane: Sequence[Zdarzenie] = (), tytul: str = "Historia pacjenta"):
    """Każde zdarzenie w osobnym wierszu (czytelne także przy 30 zdarzeniach): x = wiek, podpis obok.
    Zdarzenia wygenerowane przez model rysujemy pustymi kółkami, pod przerywaną linią."""
    wszystkie = [(z, False) for z in historia] + [(z, True) for z in wygenerowane]
    wszystkie = [(z, g) for z, g in wszystkie if z.token not in (tokeny.BRAK_ZDARZENIA, tokeny.PADDING)]
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(wszystkie) + 1.2))
    for i, (z, gen) in enumerate(wszystkie):
        k = kolor(z.token) if kolor else "C0"
        ax.scatter(z.wiek_lat, -i, s=60, color="white" if gen else k, edgecolor=k, linewidth=2, zorder=3)
        ax.text(z.wiek_lat + 1.2, -i, f"{nazwa(z.token)}  ({z.wiek_lat:.0f} l.)", va="center", fontsize=8,
                style="italic" if gen else "normal")
    if wygenerowane:
        ax.axhline(-(len(wszystkie) - len([z for z in wygenerowane if z.token not in
                                            (tokeny.BRAK_ZDARZENIA, tokeny.PADDING)])) + 0.5,
                   ls="--", color="grey")
    ax.set_yticks([]); ax.set_xlim(-1, 125); ax.set_ylim(-len(wszystkie), 1)
    ax.set_xticks(range(0, 90, 10)); ax.set_xlabel("wiek [lata]"); ax.set_title(tytul)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def ranking_slupki(df: pd.DataFrame, tytul: str = "Co model przewiduje jako następne?"):
    """Słupki z df po slownik.opisz(): kolumny „nazwa" i „ryzyko_1_rok"."""
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1))
    ax.barh(df["nazwa"][::-1], 100 * df["ryzyko_1_rok"][::-1], color="C1")
    for y, v in enumerate(100 * df["ryzyko_1_rok"][::-1]):
        ax.text(v, y, f" {_pl(v)}%", va="center", fontsize=9)
    ax.set_xlabel("ryzyko w ciągu roku [%]"); ax.set_title(tytul)
    fig.text(0.01, 0.01, ATRYBUCJA, **ATRYBUCJA_STYL)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def krzywe(df: pd.DataFrame, nazwa: Nazwa, df_porownanie: pd.DataFrame | None = None,
           etykiety: tuple[str, str] = ("pacjent", "porównanie"), tytul: str = "Ryzyko w kolejnych latach"):
    """Krzywe skumulowanego ryzyka; opcjonalnie drugi pacjent linią przerywaną (ten sam kolor)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, kol in enumerate(c for c in df.columns if c != "lata"):
        if df[kol].isna().all():
            continue
        ax.plot(df["lata"], 100 * df[kol], color=f"C{i}", lw=2, label=f"{nazwa(int(kol))} ({etykiety[0]})")
        if df_porownanie is not None and not df_porownanie[kol].isna().all():
            ax.plot(df_porownanie["lata"], 100 * df_porownanie[kol], color=f"C{i}", lw=2, ls="--",
                    label=f"{nazwa(int(kol))} ({etykiety[1]})")
    ax.set_xlabel("lata od dziś"); ax.set_ylabel("skumulowane ryzyko [%]"); ax.set_title(tytul)
    ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, ATRYBUCJA, **ATRYBUCJA_STYL)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def krzywa_uczenia(ewal: pd.DataFrame, czas_min: float | None = None):
    """Wynik parsuj_log(): kolumny krok, trening, walidacja."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ewal["krok"], ewal["trening"], label="trening")
    ax.plot(ewal["krok"], ewal["walidacja"], label="walidacja (pacjenci odłożeni na bok)")
    najlepszy = int(ewal.loc[ewal["walidacja"].idxmin(), "krok"])
    wzrost = float(ewal["walidacja"].iloc[-1] - ewal["walidacja"].min())
    ax.axvline(najlepszy, ls=":", color="grey",
               label=f"najlepszy model (krok {najlepszy})" + ("; dalej model zapamiętuje" if wzrost > 0.1 else ""))
    ax.set_yscale("log"); ax.set_xlabel("krok treningu"); ax.set_ylabel("błąd (loss)")
    ax.set_title("Uczenie Delphi" + (f" — {czas_min:.0f} min na M1 Pro" if czas_min else ""))
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def slad_generowania(kroki, nazwa: Nazwa, ile: int = 8) -> str:
    """Tekstowy ślad jak w demo z nazwami chorób: krok, wiek, wybrany token (z jego p), kandydaci z p.

    p liczone spośród chorób i zgonu (bez „brak zdarzenia"), tak jak `p_nastepna` w rankingu."""
    def krotko(t: int) -> str:
        n = nazwa(t)
        return n if len(n) <= 38 else n[:37] + "…"

    linie = []
    widoczne = [k for k in kroki if k.token != tokeny.BRAK_ZDARZENIA][:ile]
    for n, k in enumerate(widoczne, 1):
        kandydaci = ", ".join([f"{krotko(t)} {_pl(100 * p)}%" for t, p in k.kandydaci
                               if t != tokeny.BRAK_ZDARZENIA][:3])
        p_wyl = getattr(k, "p", None)
        if p_wyl is not None:
            p_txt = "<0,1%" if 100 * p_wyl < 0.05 else f"{_pl(100 * p_wyl)}%"
        wybrany = krotko(k.token) + (f" ({p_txt})" if p_wyl is not None else "")
        linie.append(f"{n:>2}. wiek {_pl(k.wiek_lat):>5} → {wybrany:<46} (kandydaci: {kandydaci})")
    return "\n".join(linie)
