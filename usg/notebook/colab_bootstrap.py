"""Środowisko Colab dla usg_masa_plodu.ipynb (lokalnie bez efektu)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = "https://github.com/NXTRSS/sympozjum-pediatria.git"


def uruchom() -> Path:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return Path.cwd().resolve()
    root = Path("/content/sympozjum-pediatria")
    if not (root / "usg").is_dir():
        import subprocess

        subprocess.run(["git", "clone", "--depth", "1", REPO, str(root)], check=True)
    usg = root / "usg"
    os.chdir(usg)
    src = usg / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from usg_masa.colab_przygotowanie import uruchom_jesli_potrzeba

    uruchom_jesli_potrzeba(usg)
    print(f"Colab USG: katalog roboczy {usg}")
    return usg