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


def histogramy_krokow(
    kroki, id2znak, wielkosc_slownika: int, kolumny: int = 4
) -> None:
    litery = [id2znak[i] for i in range(1, wielkosc_slownika)]
    wiersze = int(np.ceil(len(kroki) / kolumny))
    fig, axes = plt.subplots(
        wiersze, kolumny, figsize=(4.6 * kolumny, 2.9 * wiersze), squeeze=False
    )
    for ax, krok in zip(axes.ravel(), kroki):
        p, w = krok["rozklad"][1:], krok["wybor"]
        kolory = ["#3A1C7D"] * len(p)
        if w > 0:
            kolory[w - 1] = "#FFAF7B"
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