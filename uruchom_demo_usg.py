#!/usr/bin/env python3
"""Demko USG z katalogu głównego repo — uruchamia środowisko w usg/ (PyTorch, nnU-Net).

    uv run python uruchom_demo_usg.py
    uv run python uruchom_demo_usg.py --bez-przegladarki
    uv run python uruchom_demo_usg.py --bez-linku
    uv run python uruchom_demo_usg.py --port 7862   # gdy 7860/7861 zajęte (np. generator chorób)

To samo co:
    cd usg && uv run python app/demo_usg.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

USG = Path(__file__).resolve().parent / "usg"
SKRYPT = USG / "app" / "demo_usg.py"


def main() -> None:
    if not SKRYPT.is_file():
        raise SystemExit(f"Brak {SKRYPT}")
    env = os.environ.copy()
    # USG ≠ generator chorób: bez jawnego portu nie dziedzicz 7860 z GRADIO_SERVER_PORT.
    if "GRADIO_SERVER_PORT" not in env and "--port" not in sys.argv[1:]:
        env.setdefault("GRADIO_SERVER_PORT", "7861")
    cmd = ["uv", "run", "python", "app/demo_usg.py", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=USG, env=env))


if __name__ == "__main__":
    main()