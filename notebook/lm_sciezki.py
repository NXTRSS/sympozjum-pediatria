"""Ścieżki repo dla notatnika LM (lokalnie + Colab `/content/sympozjum-pediatria`)."""
from __future__ import annotations

import sys
from pathlib import Path

COLAB_ROOT = Path("/content/sympozjum-pediatria")
_NOTEBOOK = Path(__file__).resolve().parent
_KORZEN = _NOTEBOOK.parent


def korzen_sympozjum() -> Path:
    if (COLAB_ROOT / "data" / "choroby.csv").is_file():
        return COLAB_ROOT
    for baza in (Path.cwd(), *Path.cwd().parents, _KORZEN):
        if (baza / "data" / "choroby.csv").is_file():
            return baza
    return _KORZEN


def dopisz_pythonpath() -> Path:
    """Katalog notebook/ i korzeń repo na sys.path — import demo_model, lm_zbiory."""
    root = korzen_sympozjum()
    for katalog in (_NOTEBOOK, root):
        s = str(katalog.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def sciezka_csv(nazwa_pliku: str) -> Path | None:
    root = korzen_sympozjum()
    for kandydat in (
        root / "data" / nazwa_pliku,
        Path("data") / nazwa_pliku,
        Path("..") / "data" / nazwa_pliku,
        COLAB_ROOT / "data" / nazwa_pliku,
        _KORZEN / "data" / nazwa_pliku,
    ):
        if kandydat.is_file():
            return kandydat.resolve()
    return None