#!/usr/bin/env python3
"""Rano 28.11: to samo co „Uruchom wszystko” w notebooku, bez klikania komórek.

    uv run python notebook/uruchom_demo.py

Trening na CPU, ziarno 42, potem aplikacja Gradio z publicznym linkiem gradio.live.
Sprawdzenie, że dwa przebiegi dają te same liczby:

    uv run python notebook/uruchom_demo.py --sprawdz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_model import TEMPERATURA, ZIARNO, przebieg, uruchom_aplikacje  # noqa: E402


def _porownaj(a, b) -> None:
    import numpy as np

    if a["loss"] != b["loss"] or a["val_loss"] != b["val_loss"]:
        raise SystemExit(f"Krzywe się różnią:\n{a['loss']}\n{b['loss']}")
    for i, (x, y) in enumerate(zip(a["wagi"], b["wagi"])):
        if not np.allclose(x, y):
            raise SystemExit(f"Wagi warstwy {i} się różnią")
    if a["pierwsza"] != b["pierwsza"] or a["dziesiec"] != b["dziesiec"]:
        raise SystemExit(f"Nazwy się różnią:\n{a['pierwsza']} {a['dziesiec']}\n{b['pierwsza']} {b['dziesiec']}")


def sprawdz() -> None:
    print("Przebieg 1…", flush=True)
    a = przebieg()
    print("Przebieg 2…", flush=True)
    b = przebieg()
    _porownaj(a, b)
    print(f"loss końcowe: {a['loss'][-1]:.4f}   val: {a['val_loss'][-1]:.4f}")
    print(f"pierwsza nazwa: {a['pierwsza']}")
    print("dziesięć:", ", ".join(a["dziesiec"]))
    print("Dwa przebiegi zgodne.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo mini-modelu na nazwach chorób")
    parser.add_argument("--sprawdz", action="store_true", help="dwa treningi i porównanie, bez Gradio")
    args = parser.parse_args()
    if args.sprawdz:
        sprawdz()
        return
    wynik = przebieg()
    print(f"\nWczytano {wynik['nazw']} nazw z: {wynik['zrodlo']}")
    print(f"Loss po {len(wynik['loss'])} epokach: {wynik['loss'][-1]:.4f} (walidacja {wynik['val_loss'][-1]:.4f})")
    print(f"Przykładowa nazwa (ziarno {ZIARNO}, temperatura {TEMPERATURA}): {wynik['pierwsza']}")
    uruchom_aplikacje(
        wynik["model"], wynik["znak2id"], wynik["id2znak"], wynik["znane"],
        temperatura=TEMPERATURA,
    )


if __name__ == "__main__":
    main()
