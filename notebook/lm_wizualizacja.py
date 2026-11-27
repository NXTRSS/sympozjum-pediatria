"""Wykresy i czytelne wypisywanie dla notatnika model_jezyka_choroby (nie logika ML)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def wykres_krzywy_bledu(historia, *, tytul: str | None = None) -> None:
    tytul = tytul or "Model uczy się: błąd maleje z każdą epoką"
    plt.figure(figsize=(7, 4))
    plt.plot(historia.history["loss"], color="#2563eb")
    plt.xlabel("Epoka")
    plt.ylabel("Błąd (loss)")
    plt.title(tytul)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def wykres_topk_prefiks(prefiks: str, probs: np.ndarray, id2znak: dict, top_k: int = 12) -> None:
    """Histogram P(następna litera | prefiks) — slajd inferencji o10."""
    top_idx = np.argsort(probs)[-top_k:][::-1]
    labels = [repr(id2znak[k]) for k in top_idx]
    vals = probs[top_idx]
    plt.figure(figsize=(7, 3.5))
    plt.bar(range(len(vals)), vals, color="#7c3aed")
    plt.xticks(range(len(vals)), labels, rotation=45, ha="right")
    plt.ylabel("Prawdopodobieństwo")
    plt.title(f"Top-{top_k} liter po prefiksie {prefiks!r}")
    plt.tight_layout()
    plt.show()


def pokaz_kroki(kroki, id2znak) -> None:
    for krok in kroki:
        p, w = krok["rozklad"], krok["wybor"]
        top5 = np.argsort(p)[::-1][:5]
        opis = "  ".join(f"{id2znak[i]!r}:{p[i]:.0%}" for i in top5)
        print(
            f"  kontekst: {krok['kontekst']:<28} → wylosowano {id2znak[w]!r:<5} "
            f"(p={p[w]:.0%})   top5: {opis}"
        )


def _slupki_dla_histogramu(
    rozklad, id2znak: dict, wielkosc_slownika: int
) -> tuple[list[str], np.ndarray]:
    """Litery i wysokości słupków (bez indeksu 0) — długości muszą się zgadzać."""
    probs = np.asarray(rozklad, dtype=np.float64).ravel()
    litery = [id2znak[i] for i in range(1, wielkosc_slownika)]
    bar_heights = probs[1 : 1 + len(litery)]
    if len(bar_heights) < len(litery):
        bar_heights = probs[1:]
    n = min(len(litery), len(bar_heights))
    return litery[:n], bar_heights[:n]


def histogramy_krokow(
    kroki, id2znak, wielkosc_slownika: int, kolumny: int = 4
) -> None:
    wiersze = int(np.ceil(len(kroki) / kolumny))
    fig, axes = plt.subplots(
        wiersze, kolumny, figsize=(4.6 * kolumny, 2.9 * wiersze), squeeze=False
    )
    for ax, krok in zip(axes.ravel(), kroki):
        w = int(krok["wybor"])
        litery, p = _slupki_dla_histogramu(krok["rozklad"], id2znak, wielkosc_slownika)
        kolory = ["#3A1C7D"] * len(p)
        if w > 0:
            wi = w - 1
            if 0 <= wi < len(kolory):
                kolory[wi] = "#FFAF7B"
        ax.bar(litery, p, color=kolory)
        ax.set_ylim(0, 1)
        ax.set_title(f"{krok['kontekst']}  →  {id2znak[w]!r}", fontsize=11)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(kroki) :]:
        ax.axis("off")
    fig.suptitle(
        "P(następna litera | kontekst) — rozkład ostrzeje, gdy kontekst rośnie", y=1.0
    )
    fig.tight_layout()
    plt.show()


def figura_histogramy_krokow(
    kroki,
    id2znak: dict,
    wielkosc_slownika: int,
    kolumny: int = 4,
):
    """Jedna plansza — wszystkie kroki (do Gradio). Wywołujący zamyka figurę po użyciu."""
    if not kroki:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        return fig
    wiersze = int(np.ceil(len(kroki) / kolumny))
    fig, axes = plt.subplots(
        wiersze, kolumny, figsize=(4.2 * kolumny, 2.6 * wiersze), squeeze=False
    )
    for ax, krok in zip(axes.ravel(), kroki):
        w = int(krok["wybor"])
        litery, bar_heights = _slupki_dla_histogramu(krok["rozklad"], id2znak, wielkosc_slownika)
        kolory = ["#3A1C7D"] * len(bar_heights)
        if w > 0:
            wi = w - 1
            if 0 <= wi < len(kolory):
                kolory[wi] = "#FFAF7B"
        ax.bar(litery, bar_heights, color=kolory)
        ax.set_ylim(0, 1)
        kontekst = krok.get("kontekst", krok.get("prompt", ""))
        ax.set_title(f"{kontekst}  →  {id2znak[w]!r}", fontsize=9)
        ax.tick_params(axis="x", labelsize=6)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(kroki) :]:
        ax.axis("off")
    fig.suptitle(
        "P(następna litera | kontekst) — jak w notatniku (histogramy_krokow)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def kroki_do_histogramu(kroki_demo: list[dict]) -> list[dict]:
    """Konwersja kroków z generuj() do formatu histogramy_krokow."""
    out = []
    for k in kroki_demo:
        out.append(
            {
                "kontekst": k["prompt"],
                "rozklad": np.asarray(k["rozklad"], dtype=np.float64),
                "wybor": int(k["idx"]),
            }
        )
    return out