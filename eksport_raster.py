# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright>=1.44", "python-pptx>=0.6.23", "img2pdf>=0.5", "pillow>=10"]
# ///
"""Eksport prezentacja.html do formatów dla laptopa DIL (Windows, tylko PDF/PPTX).

Każdy slajd jest fotografowany w 1920×1080 przez zainstalowanego Google Chrome
(Playwright, kanał "chrome"; nic nie jest pobierane poza pakietami Pythona).
Z tych zrzutów powstają:

  eksport/prezentacja.pdf   — strony-obrazy, wygląd identyczny w każdej przeglądarce PDF
  eksport/prezentacja.pptx  — slajdy-obrazy 16:9 + notatki prelegenta z decku
  eksport/slajdy/*.png      — same zrzuty

Dlaczego obrazy: tekst z gradientem (background-clip: text) w PDF z druku Chrome
różne przeglądarki PDF rysują różnie (ramki albo zasłaniający prostokąt).
Animacja „Krok po kroku” (slajd z #inference-anim) jest rozpisana na osobne
slajdy, po jednym na krok generowania. Elementy .build są w stanie końcowym.

Uruchomienie (najpierw złóż prezentację, jeśli zmieniały się slides/):
    python3 assemble.py
    uv run eksport_raster.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import img2pdf
from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).parent
DECK = ROOT / "prezentacja.html"
OUT = ROOT / "eksport"
PNG = OUT / "slajdy"
W, H = 1920, 1080

# Na zrzutach chowamy elementy interfejsu, które na obrazie nie działają:
# wskaźnik trybu (.no-print), banery step-editora, przyciski animacji.
UKRYJ_CSS = """
.no-print, .__se-banner, #__step-editor-toast { display: none !important; }
#inference-anim button { visibility: hidden !important; }
*, *::before, *::after { transition: none !important; animation-play-state: paused !important; caret-color: transparent !important; }
"""


def notatki(html: str) -> list[str]:
    bloki = re.findall(r'<script type="application/json" id="speaker-notes">(.*?)</script>', html, re.S)
    for b in reversed(bloki):  # pierwszy blok to wzmianka w komentarzu JS decku
        try:
            d = json.loads(b)
        except json.JSONDecodeError:
            continue
        if isinstance(d, list):
            return [str(x) for x in d]
    sys.exit("Nie znalazłem notatek prelegenta (speaker-notes) w prezentacja.html")


def zrob_zrzuty() -> list[tuple[Path, int, str | None]]:
    """Zwraca listę (plik, indeks slajdu w decku, podpis kroku albo None)."""
    if PNG.exists():
        shutil.rmtree(PNG)
    PNG.mkdir(parents=True)
    wynik = []
    url = DECK.resolve().as_uri()
    with sync_playwright() as p:
        przegl = p.chromium.launch(channel="chrome", headless=True)
        strona = przegl.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        strona.goto(url + "#1")
        strona.wait_for_function("document.querySelector('deck-stage') && document.querySelector('deck-stage').length > 0")
        strona.evaluate("document.fonts.ready")
        strona.add_style_tag(content=UKRYJ_CSS)
        # pasek nawigacji decku (‹ 1/29 › Reset) siedzi w shadow DOM <deck-stage>
        strona.evaluate("""() => {
            const s = document.createElement('style');
            s.textContent = '.overlay, .tapzones { display: none !important; }';
            document.querySelector('deck-stage').shadowRoot.appendChild(s);
        }""")
        n = strona.evaluate("document.querySelector('deck-stage').length")
        for i in range(n):
            strona.evaluate(f"document.querySelector('deck-stage').goTo({i})")
            strona.wait_for_timeout(250)
            aktywny = strona.locator("deck-stage > section[data-deck-active]")
            ma_animacje = aktywny.locator("#inference-anim").count() > 0
            if not ma_animacje:
                plik = PNG / f"{i + 1:02d}.png"
                strona.wait_for_timeout(900)  # nakładka z numerem slajdu znika
                strona.screenshot(path=str(plik))
                wynik.append((plik, i, None))
                continue
            # GENERATION_STEPS jest w domknięciu skryptu decku, więc idziemy krokami,
            # aż animacja sama wyłączy przycisk „Następny krok” (ostatni krok).
            strona.evaluate("inferenceRestart()")
            klatki = []
            for k in range(100):
                strona.evaluate("inferenceStep()")
                strona.wait_for_timeout(900 if k == 0 else 150)
                plik = PNG / f"{i + 1:02d}_krok{k + 1:02d}.png"
                strona.screenshot(path=str(plik))
                klatki.append(plik)
                if strona.evaluate("document.getElementById('btn-step').disabled"):
                    break
            else:
                sys.exit("Animacja „Krok po kroku” nie skończyła się po 100 krokach")
            for k, plik in enumerate(klatki):
                wynik.append((plik, i, f"Krok {k + 1}/{len(klatki)}"))
            strona.evaluate("inferenceRestart()")
        przegl.close()
    return wynik


def zbuduj_pdf(zrzuty):
    # 1920×1080 px przy 96 dpi = 1440×810 pt, tyle samo co druk z Chrome
    uklad = img2pdf.get_fixed_dpi_layout_fun((96, 96))
    (OUT / "prezentacja.pdf").write_bytes(img2pdf.convert([str(z[0]) for z in zrzuty], layout_fun=uklad))


def zbuduj_pptx(zrzuty, notes):
    prez = Presentation()
    prez.slide_width, prez.slide_height = Emu(12192000), Emu(6858000)  # 16:9, 13,333 × 7,5 cala
    pusty = prez.slide_layouts[6]
    for plik, i, krok in zrzuty:
        s = prez.slides.add_slide(pusty)
        s.shapes.add_picture(str(plik), 0, 0, width=prez.slide_width, height=prez.slide_height)
        tekst = notes[i] if i < len(notes) else ""
        if krok and not krok.startswith("Krok 1/"):
            tekst = f"{krok} — animacja „Krok po kroku”, ciąg dalszy (pełna notatka przy kroku 1)."
        elif krok:
            tekst = f"{krok}. {tekst}"
        s.notes_slide.notes_text_frame.text = tekst
    prez.save(OUT / "prezentacja.pptx")


def main():
    if not DECK.exists():
        sys.exit("Brak prezentacja.html — najpierw: python3 assemble.py")
    notes = notatki(DECK.read_text(encoding="utf-8"))
    zrzuty = zrob_zrzuty()
    slajdow = len({z[1] for z in zrzuty})
    if slajdow != len(notes):
        sys.exit(f"Liczba slajdów ({slajdow}) różna od liczby notatek ({len(notes)})")
    zbuduj_pdf(zrzuty)
    zbuduj_pptx(zrzuty, notes)
    print(f"OK: {slajdow} slajdów decku → {len(zrzuty)} stron/slajdów w {OUT.relative_to(ROOT)}/ "
          f"(prezentacja.pdf, prezentacja.pptx, slajdy/*.png)")


if __name__ == "__main__":
    main()
