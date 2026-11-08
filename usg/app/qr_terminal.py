"""Kod QR w terminalu (ASCII) — link gradio.live po starcie demo USG."""
from __future__ import annotations


def linie_qr(url: str, *, border: int = 2) -> list[str]:
    import segno

    q = segno.make(url.strip(), error="m", micro=False)
    out: list[str] = []
    for row in q.matrix:
        out.append("".join("██" if cell else "  " for cell in row))
    return out


def wypisz_qr_terminal(url: str, *, naglowek: str = "QR (zeskanuj telefonem):") -> None:
    if not url or not url.strip():
        return
    url = url.strip()
    print(naglowek, flush=True)
    print(url, flush=True)
    for linia in linie_qr(url):
        print(linia, flush=True)
    print(flush=True)