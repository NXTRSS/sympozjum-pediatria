"""Skrypt uruchom_demo.py i notebook dają ten sam model przy ziarnie 42."""
import json
import os
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
NB = NOTEBOOK_DIR / "model_jezyka_choroby.ipynb"
sys.path.insert(0, str(NOTEBOOK_DIR))

from demo_model import przebieg  # noqa: E402


def _z_notebooka(tmp: Path) -> dict:
    os.environ["DEMO_BEZ_GRADIO"] = "1"
    os.environ["DEMO_WYNIK"] = str(tmp / "wynik.json")
    nb = nbformat.read(NB, as_version=4)
    nb.cells = [c for c in nb.cells if "demo.launch" not in "".join(c.source)]
    nb.cells.append(nbformat.v4.new_code_cell(
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['DEMO_WYNIK']).write_text(json.dumps({\n"
        "    'loss': [float(x) for x in historia.history['loss']],\n"
        "    'val_loss': [float(x) for x in historia.history['val_loss']],\n"
        "    'wagi': [w.numpy().tolist() for w in model.weights],\n"
        "    'pierwsza': pierwsza_nazwa,\n"
        "    'dziesiec': lista_nazw,\n"
        "}))\n"
    ))
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
    return json.loads((tmp / "wynik.json").read_text())


def test_dwa_przebiegi_skryptu_sa_zgodne():
    a = przebieg()
    b = przebieg()
    assert a["loss"] == b["loss"]
    assert a["val_loss"] == b["val_loss"]
    for x, y in zip(a["wagi"], b["wagi"]):
        assert np.allclose(x, y)
    assert a["pierwsza"] == b["pierwsza"]
    assert a["dziesiec"] == b["dziesiec"]


def test_notebook_zgodny_ze_skryptem(tmp_path):
    skrypt = przebieg()
    nb = _z_notebooka(tmp_path)
    assert nb["loss"] == pytest.approx(skrypt["loss"])
    assert nb["val_loss"] == pytest.approx(skrypt["val_loss"])
    for x, y in zip(nb["wagi"], skrypt["wagi"]):
        assert np.allclose(np.array(x), y)
    assert nb["pierwsza"] == skrypt["pierwsza"]
    assert nb["dziesiec"] == skrypt["dziesiec"]
