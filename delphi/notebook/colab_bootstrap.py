"""Środowisko Colab dla delphi_2m.ipynb — bez efektu lokalnie."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/NXTRSS/sympozjum-pediatria.git"

DEPS = ("torch>=2.3", "numpy<2", "pandas>=2.2", "matplotlib>=3.8", "ipywidgets>=8")


def _zainstaluj_zaleznosci_colab() -> None:
    """Colab ma często Python 3.12/3.13 — `pip install -e .` pada na requires-python == 3.11.*."""
    brakuje = [
        p
        for p in ("torch", "pandas", "matplotlib", "numpy")
        if importlib.util.find_spec(p) is None
    ]
    if brakuje:
        print(f"Colab Delphi: doinstalowuję {', '.join(brakuje)}…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *DEPS],
        )
    else:
        print("Colab Delphi: PyTorch i pandas są już w kernelu — pomijam pip.")


def uruchom() -> Path:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return Path.cwd().resolve()
    root = Path("/content/sympozjum-pediatria")
    if not (root / "delphi").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(root)], check=True)
    delphi = root / "delphi"
    _zainstaluj_zaleznosci_colab()
    os.chdir(delphi)
    src = delphi / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    print(f"Colab: Delphi — kod z {src} (Python {sys.version_info.major}.{sys.version_info.minor}).")
    return delphi