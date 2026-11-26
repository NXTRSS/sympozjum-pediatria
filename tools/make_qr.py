#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["segno==1.6.6"]
# ///
"""Kody QR do slajdów n05 i n15 (SVG w pliku, bez sieci w czasie pokazu).

Uruchomienie z katalogu repozytorium:

    uv run tools/make_qr.py

Potem: python3 assemble.py
"""
from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

import segno

ROOT = Path(__file__).resolve().parents[1]
DARK = "#0A0A14"
LIGHT = "#ffffff"
ERROR = "q"  # korekcja Q
QUIET = 4  # moduły strefy ciszy

# (plik, id, adres w kodzie, podpis pod kodem, bok SVG w px)
CODES = [
    (
        ROOT / "slides" / "n15_materialy.html",
        "qr-materialy",
        "https://github.com/NXTRSS/sympozjum-pediatria",
        "github.com/NXTRSS/\nsympozjum-pediatria",
        248,
    ),
    (
        ROOT / "slides" / "n05_qr.html",
        "qr-colab",
        "https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/notebook/model_jezyka_choroby.ipynb",
        "colab.research.google.com/\n…/model_jezyka_choroby.ipynb",
        200,
    ),
    (
        ROOT / "slides" / "n05_qr.html",
        "qr-gradio",
        "https://github.com/NXTRSS/sympozjum-pediatria#demo-na-zywo",
        "github.com/NXTRSS/\nsympozjum-pediatria\n#demo-na-zywo",
        188,
    ),
]


def svg_kodu(url: str, px: int) -> str:
    kod = segno.make(url, error=ERROR)
    modulow, wysokosc = kod.symbol_size(scale=1, border=QUIET)
    svg = kod.svg_inline(scale=1, border=QUIET, dark=DARK, light=LIGHT)
    svg = re.sub(r"\swidth=\"[^\"]*\"", f' width="{px}"', svg, count=1)
    svg = re.sub(r"\sheight=\"[^\"]*\"", f' height="{px}"', svg, count=1)
    if "viewBox" not in svg:
        svg = svg.replace("<svg", f'<svg viewBox="0 0 {modulow} {wysokosc}"', 1)
    svg = svg.replace(
        "<svg ",
        f'<svg role="img" aria-label="{escape(url)}" ',
        1,
    )
    return svg


def wnetrze(url: str, podpis: str, px: int) -> str:
    linie = "<br>".join(escape(linia) for linia in podpis.split("\n"))
    return (
        f"{svg_kodu(url, px)}\n"
        f'<div class="mono" style="color: {DARK}; font-size: 12px; line-height: 1.25; '
        f'text-align: center; font-weight: 500; max-width: 100%;">{linie}</div>'
    )


def podmien(path: Path, key: str, inner: str) -> None:
    tekst = path.read_text(encoding="utf-8")
    start = f"<!-- qr:{key} -->"
    end = f"<!-- /qr:{key} -->"
    i = tekst.find(start)
    j = tekst.find(end)
    if i < 0 or j < 0 or j < i:
        sys.exit(f"Brak znaczników {start} … {end} w {path}")
    nowy = tekst[: i + len(start)] + "\n" + inner + "\n" + tekst[j:]
    if nowy != tekst:
        path.write_text(nowy, encoding="utf-8")


def main() -> None:
    for path, key, url, podpis, px in CODES:
        podmien(path, key, wnetrze(url, podpis, px))
        print(f"{key}: {url}")


if __name__ == "__main__":
    main()
