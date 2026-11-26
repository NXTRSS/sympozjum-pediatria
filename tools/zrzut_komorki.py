#!/usr/bin/env python3
"""Zrzuty komórek notebooka (kod + wyjścia) do PNG — styl Jupyter, białe tło."""
from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

from nbconvert.filters import ansi2html
from nbformat import read
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name


def _komorka_tekst(cell) -> str:
    if isinstance(cell.get("source"), list):
        return "".join(cell["source"])
    return cell.get("source") or ""


def _wyjscia_html(outputs) -> str:
    czesci: list[str] = []
    for out in outputs or []:
        ot = out.get("output_type")
        if ot == "stream":
            tekst = out.get("text", "")
            if isinstance(tekst, list):
                tekst = "".join(tekst)
            czesci.append(
                f'<pre class="stream">{html.escape(tekst.rstrip())}</pre>'
            )
        elif ot in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                tp = data["text/plain"]
                if isinstance(tp, list):
                    tp = "".join(tp)
                czesci.append(
                    f'<pre class="text-plain">{html.escape(tp.rstrip())}</pre>'
                )
            if "image/png" in data:
                raw = data["image/png"]
                if isinstance(raw, str):
                    b64 = raw
                else:
                    b64 = base64.b64encode(raw).decode("ascii")
                czesci.append(
                    f'<img class="plot" src="data:image/png;base64,{b64}" alt="">'
                )
        elif ot == "error":
            tb = out.get("traceback", [])
            if isinstance(tb, list):
                tb = "".join(tb)
            czesci.append(ansi2html(tb))
    return "\n".join(czesci)


def html_komorki(cell) -> str:
    zrodlo = _komorka_tekst(cell).strip()
    fmt = HtmlFormatter(style=get_style_by_name("default"), noclasses=False)
    kod = highlight(zrodlo, PythonLexer(), fmt) if zrodlo else ""
    css = fmt.get_style_defs(".highlight")
    out = _wyjscia_html(cell.get("outputs"))
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
{css}
body {{ margin: 0; padding: 24px 28px; background: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.highlight {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 14px; font-size: 13px; line-height: 1.45; overflow-x: auto; }}
.stream, .text-plain {{ background: #fafafa; border-left: 4px solid #ddd; margin: 14px 0 0; padding: 10px 14px; font-size: 13px; line-height: 1.4; white-space: pre-wrap; }}
.plot {{ display: block; max-width: 100%; margin: 16px auto 0; border-radius: 4px; }}
</style></head><body>
<div class="highlight">{kod}</div>
{out}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("indeksy", nargs="+", type=int, help="numery komórek (0-based)")
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise SystemExit("playwright nie jest zainstalowany (uv run --with playwright)") from e

    nb = read(args.notebook, as_version=4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapa: dict[int, str] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        for idx in args.indeksy:
            if idx >= len(nb.cells) or nb.cells[idx].cell_type != "code":
                raise SystemExit(f"Komórka {idx} nie istnieje lub nie jest code")
            page.set_content(html_komorki(nb.cells[idx]), wait_until="networkidle")
            h = page.evaluate("() => Math.min(document.body.scrollHeight + 40, 1600)")
            page.set_viewport_size({"width": 1100, "height": max(h, 200)})
            nazwa = f"{args.prefix}{idx:02d}.png"
            sciezka = args.out_dir / nazwa
            page.screenshot(path=str(sciezka), full_page=True)
            mapa[idx] = str(sciezka)
            print(sciezka)
        browser.close()

    (args.out_dir / f"{args.prefix}mapa.json").write_text(
        json.dumps(mapa, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()