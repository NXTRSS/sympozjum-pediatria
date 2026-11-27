"""Środowisko Colab dla delphi_2m.ipynb — bez efektu lokalnie."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/NXTRSS/sympozjum-pediatria.git"


def uruchom() -> Path:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return Path.cwd().resolve()
    root = Path("/content/sympozjum-pediatria")
    if not (root / "delphi").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(root)], check=True)
    delphi = root / "delphi"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(delphi)], cwd=str(delphi))
    os.chdir(delphi)
    src = delphi / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    print("Colab: Delphi — repo + pakiet delphi-demo gotowe.")
    return delphi