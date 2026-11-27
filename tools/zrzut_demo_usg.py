#!/usr/bin/env python3
"""Zrzut aplikacji usg/app/demo_usg.py po analizie Nr 1 w trybie z pamięci."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "zrzuty"


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit("uv run --with playwright") from e

    OUT.mkdir(parents=True, exist_ok=True)
    python = ROOT / "usg" / ".venv" / "bin" / "python"
    proc = subprocess.Popen(
        [str(python), "app/demo_usg.py", "--bez-przegladarki", "--port", "7861"],
        cwd=ROOT / "usg",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1200})
            for _ in range(120):
                try:
                    page.goto("http://127.0.0.1:7861", wait_until="domcontentloaded", timeout=2000)
                    if page.locator("text=Znajdź brzuch").count():
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise SystemExit("Gradio nie wystartowało w czasie")

            page.locator("label").filter(has_text="Nr 1").first.click()
            page.get_by_role("button", name="Znajdź brzuch i zmierz").click()
            page.locator(".k-cyan, .karta").first.wait_for(state="visible", timeout=60000)
            time.sleep(1.5)

            page.screenshot(path=str(OUT / "usg-06-aplikacja.png"), full_page=True)
            pary = page.locator("#wyniki-pary")
            if pary.count():
                pary.first.screenshot(path=str(OUT / "usg-07-nnunet.png"))
            else:
                page.locator("#prawa").first.screenshot(path=str(OUT / "usg-07-nnunet.png"))
            page.get_by_role("tab", name="FUVAI — klatka po klatce").click()
            page.get_by_role("button", name="Załaduj klasyfikację klatek").click()
            page.locator("#fuvai-obraz img").first.wait_for(state="visible", timeout=120000)
            page.locator(".tabela-fuvai").first.wait_for(state="visible", timeout=30000)
            page.get_by_text("brzuch", exact=False).first.click()
            time.sleep(0.5)
            page.screenshot(path=str(OUT / "usg-08-fuvai-klatki.png"), full_page=True)
            browser.close()
        print(OUT / "usg-06-aplikacja.png")
        print(OUT / "usg-07-nnunet.png")
        print(OUT / "usg-08-fuvai-klatki.png")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()