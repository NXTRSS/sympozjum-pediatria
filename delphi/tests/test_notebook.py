import importlib.util
import json
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

from delphi_demo import pamiec, trening

NB = Path(__file__).resolve().parents[1] / "notebook" / "delphi_2m.ipynb"
SKRYPT = Path(__file__).resolve().parents[1] / "scripts" / "zbuduj_notebook.py"


def test_notebook_zgodny_ze_skryptem():
    """Chroni przed ręczną edycją w Jupyterze: notebook = KOMORKI z scripts/zbuduj_notebook.py."""
    spec = importlib.util.spec_from_file_location("zbuduj_notebook", SKRYPT)
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)
    nb = json.loads(NB.read_text())
    w_notebooku = [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]
    w_skrypcie = [("markdown" if r == "md" else "code", t) for r, t in modul.KOMORKI]
    assert w_notebooku == w_skrypcie, "notebook różni się od skryptu: uv run python scripts/zbuduj_notebook.py"


def test_notebook_zapisany_bez_wyjsc():
    nb = json.loads(NB.read_text())
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    tekst = "".join("".join(c["source"]) for c in nb["cells"])
    assert "Nie jest wyrobem medycznym" in tekst and 'TRYB = "z_pamieci"' in tekst


@pytest.mark.wolny
def test_notebook_z_pamieci_bez_internetu():
    if not trening.SCIEZKA_MODELU.exists() or not pamiec.jest("krzywe_cukrzyca"):
        pytest.fail("najpierw: uv run python scripts/przygotuj_demo.py")
    nb = nbformat.read(NB, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "import socket, subprocess\n"
        "import matplotlib.pyplot  # buduje cache fontów (subprocess) ZANIM zablokujemy subprocess\n"
        "import socket, subprocess\n"
        "def _blokada(*a, **k): raise RuntimeError('sieć zablokowana w teście')\n"
        "socket.socket = _blokada\n"
        "socket.create_connection = _blokada\n"
        "socket.getaddrinfo = _blokada\n"
        "subprocess.run = _blokada\n"
        "subprocess.Popen = _blokada"))
    ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NB.parent)}})
    wyjscia = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                        for o in c.get("outputs", []))
    assert "mln parametrów" in wyjscia and "tryb z_pamieci" in wyjscia and "Dane syntetyczne: 7144 osób" in wyjscia and " 1. wiek " in wyjscia
    assert "UWAGA:" not in wyjscia
    assert "kroków" in wyjscia and "Brak logu treningu" not in wyjscia   # komórka 6 poszła prawdziwą gałęzią
