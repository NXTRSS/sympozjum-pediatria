"""Wspólna logika indeksów bloków .build[data-step] jak step-editor.js w decku."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"

# Scalenia Kamila (localStorage step-editor:<pathname>) — źródło prawdy po zapisie do slides/.
KAMIL_SCALENIA: dict[str, int] = {
    "N01 Title::3": 1,
    "N01 Title::4": 1,
    "N02 Zagadka Alex::3": 2,
    "N02 Zagadka Alex::5": 3,
    "04 Section LM::2": 0,
    "N05 QR Interakcja::3": 1,
    "N05 QR Interakcja::4": 2,
    "12 Section Scale::2": 0,
    "13 Scale::3": 1,
    "N07c Kiedy dedykowany::6": 4,
    "N11 Section BabyNet::2": 0,
}

RX_LABEL = re.compile(r'\bdata-label\s*=\s*"([^"]+)"')
RX_BUILD_OPEN = re.compile(
    r"<([a-zA-Z][\w-]*)\b([^>]*?)\bclass\s*=\s*\"([^\"]*\bbuild\b[^\"]*)\"([^>]*?)>",
    re.IGNORECASE,
)
RX_DATA_STEP = re.compile(r'\bdata-step\s*=\s*"(\d+)"')


def _label_from_section(html: str) -> str | None:
    m = RX_LABEL.search(html)
    return m.group(1) if m else None


def indeksy_blokow(section_html: str) -> list[tuple[int, int, int, str]]:
    """Lista (start, end, idx, data-step) dla każdego otwierającego tagu build+data-step."""
    out: list[tuple[int, int, int, str]] = []
    idx = 0
    for m in RX_BUILD_OPEN.finditer(section_html):
        attrs = m.group(2) + " " + m.group(4)
        sm = RX_DATA_STEP.search(attrs)
        if not sm:
            continue
        out.append((m.start(), m.end(), idx, sm.group(1)))
        idx += 1
    return out


def krok_bloku(label: str, idx: int, section_html: str) -> int | None:
    for _s, _e, i, step in indeksy_blokow(section_html):
        if i == idx:
            return int(step)
    key = f"{label}::{idx}"
    if key in KAMIL_SCALENIA:
        return KAMIL_SCALENIA[key]
    return None


def zastosuj_w_sekcji(section_html: str, overrides: dict[str, int] | None = None) -> tuple[str, int]:
    """Jak applyOverridesToHTML, ale dla jednego pliku slajdu (jedna sekcja)."""
    overrides = overrides if overrides is not None else KAMIL_SCALENIA
    label = _label_from_section(section_html)
    if not label:
        return section_html, 0

    patched = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal patched
        attrs = m.group(2) + " " + m.group(4)
        if not RX_DATA_STEP.search(attrs):
            return m.group(0)
        blk_idx = repl.i  # type: ignore[attr-defined]
        repl.i += 1  # type: ignore[attr-defined]
        key = f"{label}::{blk_idx}"
        if key not in overrides:
            return m.group(0)
        new_step = str(overrides[key])
        patched += 1
        return RX_DATA_STEP.sub(f'data-step="{new_step}"', m.group(0), count=1)

    repl.i = 0  # type: ignore[attr-defined]
    return RX_BUILD_OPEN.sub(repl, section_html), patched


def pliki_dotkniete() -> list[Path]:
    touched: set[Path] = set()
    for key in KAMIL_SCALENIA:
        label = key.split("::", 1)[0]
        for path in SLIDES.glob("*.html"):
            txt = path.read_text(encoding="utf-8")
            if _label_from_section(txt) == label:
                touched.add(path)
                break
    return sorted(touched)


def nanos_na_slajdy(overrides: dict[str, int] | None = None) -> dict[str, int]:
    """Zapisuje slides/*.html; zwraca {ścieżka: liczba_podmian}."""
    overrides = overrides if overrides is not None else KAMIL_SCALENIA
    by_label: dict[str, list[str]] = {}
    for key in overrides:
        lbl, _ = key.split("::", 1)
        by_label.setdefault(lbl, []).append(key)

    stats: dict[str, int] = {}
    for path in SLIDES.glob("*.html"):
        raw = path.read_text(encoding="utf-8")
        label = _label_from_section(raw)
        if not label or label not in by_label:
            continue
        new_html, n = zastosuj_w_sekcji(raw, overrides)
        if n:
            path.write_text(new_html, encoding="utf-8")
            stats[str(path.relative_to(ROOT))] = n
    return stats


def weryfikuj(overrides: dict[str, int] | None = None) -> list[str]:
    """Puste = OK; inaczej lista błędów."""
    overrides = overrides if overrides is not None else KAMIL_SCALENIA
    errors: list[str] = []
    for key, want in overrides.items():
        label, idx_s = key.split("::", 1)
        idx = int(idx_s)
        found = False
        for path in SLIDES.glob("*.html"):
            txt = path.read_text(encoding="utf-8")
            if _label_from_section(txt) != label:
                continue
            found = True
            got = krok_bloku(label, idx, txt)
            if got is None:
                errors.append(f"{key}: brak bloku #{idx} w {path.name}")
            elif got != want:
                errors.append(f"{key}: oczekiwano {want}, jest {got} w {path.name}")
            break
        if not found:
            errors.append(f"{key}: nie znaleziono slajdu data-label={label!r}")
    return errors


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "apply":
        print(json.dumps(nanos_na_slajdy(), indent=2, ensure_ascii=False))
    else:
        bad = weryfikuj()
        if bad:
            print("\n".join(bad), file=sys.stderr)
            sys.exit(1)
        print(f"OK — {len(KAMIL_SCALENIA)} wpisów zgodnych ze slides/")