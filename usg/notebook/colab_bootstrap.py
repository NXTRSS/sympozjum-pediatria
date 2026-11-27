"""Środowisko Colab dla usg_masa_plodu.ipynb. Wagi i nagrania nie są w repo — tylko publiczne źródła."""
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
    if not (root / "usg").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(root)], check=True)
    usg = root / "usg"
    os.chdir(usg)
    print("Colab USG: instalacja zależności (PyTorch, nnU-Net) — pierwszy raz może potrwać kilka minut…")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(usg)], cwd=str(usg))
    except subprocess.CalledProcessError:
        os.environ["USG_COLAB_SKIP_HEAVY"] = "1"
        print(
            "Colab: nie udało się zainstalować pełnego środowiska USG (Python/nnU-Net). "
            "Komórki z modelami zostaną pominięte. Pełne demo: lokalnie w usg/."
        )
        return usg
    src = usg / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("USG_COLAB", "1")
    try:
        from usg_masa import dane  # noqa: WPS433

        ma_nagrania = dane.KATALOG_DANYCH.exists() and any(
            dane.KATALOG_DANYCH.rglob("*.mha")
        )
        if not ma_nagrania:
            os.environ["USG_COLAB_SKIP_HEAVY"] = "1"
            print(
                "Colab USG: w repo nie ma pamiec/* (nagrania ACOUSLIC-AI i wyników demo). "
                "Pobierz nagrania z Zenodo 12697994 (CC BY-NC-SA) i uruchom lokalnie "
                "usg/scripts/przygotuj_demo.py — tu pokażemy tylko lekkie komórki."
            )
        else:
            print("Colab USG: środowisko gotowe (tryb z pamięci wymaga wcześniej policzonych wyników).")
    except Exception as exc:  # noqa: BLE001
        os.environ["USG_COLAB_SKIP_HEAVY"] = "1"
        print(f"Colab USG: import usg_masa się nie udał ({exc}). Pomijamy ciężkie komórki.")
    return usg