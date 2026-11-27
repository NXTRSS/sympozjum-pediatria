"""Środowisko Colab dla model_jezyka_choroby.ipynb — bez efektu lokalnie."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/NXTRSS/sympozjum-pediatria.git"
ROOT = Path("/content/sympozjum-pediatria")


def uruchom() -> Path:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return Path.cwd().resolve()
    nb = ROOT / "notebook"
    if not (nb / "lm_utils.py").is_file():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(ROOT)], check=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "tensorflow>=2.16,<2.21",
            "pandas>=2.0",
            "numpy",
            "matplotlib>=3.8",
        ],
    )
    os.chdir(nb)
    s = str(nb)
    if s not in sys.path:
        sys.path.insert(0, s)
    print("Colab: repo + TensorFlow; katalog roboczy: notebook/")
    return nb