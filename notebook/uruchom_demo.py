#!/usr/bin/env python3
"""Rano 28.11: to samo co „Uruchom wszystko” w notebooku, bez klikania komórek.

    uv run python notebook/uruchom_demo.py

Cache w ``notebook/.model_cache/`` — drugie uruchomienie bez treningu (chyba że ``--trenuj``).

    uv run python notebook/uruchom_demo.py --sprawdz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_model import TEMPERATURA, ZIARNO, przebieg, uruchom_aplikacje  # noqa: E402
from model_cache import wyczysc as wyczysc_cache  # noqa: E402


def _porownaj(a, b) -> None:
    import numpy as np

    if a["loss"] != b["loss"]:
        raise SystemExit(f"Krzywe loss się różnią:\n{a['loss']}\n{b['loss']}")
    for i, (x, y) in enumerate(zip(a["wagi"], b["wagi"])):
        if not np.allclose(x, y):
            raise SystemExit(f"Wagi warstwy {i} się różnią")
    if a["pierwsza"] != b["pierwsza"] or a["dziesiec"] != b["dziesiec"]:
        raise SystemExit(f"Nazwy się różnią:\n{a['pierwsza']} {a['dziesiec']}\n{b['pierwsza']} {b['dziesiec']}")


def sprawdz() -> None:
    print("Świeży trening…", flush=True)
    wyczysc_cache()
    swiezy = przebieg(force_train=True, use_cache=True)
    print("Wczytanie z cache…", flush=True)
    z_cache = przebieg(force_train=False, use_cache=True)
    _porownaj(swiezy, z_cache)
    print("Ponowny trening (wymuszony)…", flush=True)
    wymuszony = przebieg(force_train=True, use_cache=True)
    _porownaj(swiezy, wymuszony)
    print(f"loss końcowe: {swiezy['loss'][-1]:.4f}")
    print(f"pierwsza nazwa: {swiezy['pierwsza']}")
    print("dziesięć:", ", ".join(swiezy["dziesiec"]))
    print("Cache i świeży trening zgodne.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo mini-modelu na nazwach chorób")
    parser.add_argument("--sprawdz", action="store_true", help="trening, cache i zgodność wag/nazw")
    parser.add_argument("--trenuj", action="store_true", help="wymuś trening od zera (nadpisuje cache)")
    parser.add_argument("--bez-linku", action="store_true", help="Gradio tylko lokalnie (share=False)")
    parser.add_argument("--bez-limitu", action="store_true", help="wyłącza limit 20 zapytań/min na klienta")
    parser.add_argument("--bez-monitora", action="store_true", help="nie otwiera okna Terminala z monitor.py")
    parser.add_argument(
        "--zbior",
        choices=("choroby", "leki", "dinozaury"),
        default="choroby",
        help="zbiór nazw (jak ZBIOR w notebooku)",
    )
    args = parser.parse_args()
    if args.sprawdz:
        sprawdz()
        return
    wynik = przebieg(force_train=args.trenuj, zbior=args.zbior)
    print(f"\nWczytano {wynik['nazw']} nazw z: {wynik['zrodlo']}")
    print(f"Loss po {len(wynik['loss'])} epokach: {wynik['loss'][-1]:.4f}")
    print(f"Przykładowa nazwa (ziarno {ZIARNO}, temperatura {TEMPERATURA}): {wynik['pierwsza']}")
    uruchom_aplikacje(
        wynik["model"],
        wynik["tokenizer"],
        wynik["znane"],
        temperatura=TEMPERATURA,
        zbior=args.zbior,
        share=not args.bez_linku,
        limit_klienta=not args.bez_limitu,
        uruchom_monitor=not args.bez_monitora,
    )


if __name__ == "__main__":
    main()