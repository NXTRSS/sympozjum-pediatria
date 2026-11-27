"""Gradio demo: brak flagowania i katalogu .gradio w repozytorium."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebook"))

from demo_model import ustaw_gradio_bez_flagowania  # noqa: E402


def test_ustaw_gradio_bez_flagowania():
    ustaw_gradio_bez_flagowania()
    assert os.environ["GRADIO_FLAGGING_MODE"] == "never"


def test_gitignore_pomija_katalog_gradio():
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert ".gradio" in gitignore