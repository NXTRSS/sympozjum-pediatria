from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NB = Path(__file__).resolve().parents[1] / "notebook" / "usg_masa_plodu.ipynb"
PAMIEC = Path(__file__).resolve().parents[1] / "pamiec"


@pytest.mark.wolny
def test_notebook_z_pamieci_bez_internetu():
    if not any(PAMIEC.glob("*.npz")):
        pytest.skip("najpierw: uv run python scripts/przygotuj_demo.py")
    nb = nbformat.read(NB, as_version=4)
    # Programy blokujemy błędem OSError (jak „nie ma takiego programu”): numpy przy imporcie
    # próbuje uruchomić `lscpu` i sam łapie OSError; git clone i tak się wtedy wywróci.
    nb.cells.insert(0, nbformat.v4.new_code_cell(
        "import socket, subprocess\n"
        "def _blokada(*a, **k): raise RuntimeError('sieć zablokowana w teście')\n"
        "def _bez_programow(*a, **k): raise OSError('uruchamianie programów zablokowane w teście')\n"
        "socket.socket = _blokada\n"            # odcina HTTP (requests, remotezip)
        "subprocess.run = _bez_programow"))      # odcina git clone
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(NB.parent)}})
    tekst = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                      for o in c.get("outputs", []))
    assert "Przeanalizowano" in tekst and "Masa W DNIU PORODU wg modelu Sano" in tekst
    assert "wcześniej, pełnym nnU-Netem" in tekst   # tylko w trybie z_pamieci: notebook nie liczył na żywo
    assert "FUVAI przeanalizował" in tekst and "(wcześniej)." in tekst   # FUVAI też z pamięci
    assert "Model FUVAI wczytany" not in tekst                           # i nie wczytywał modelu
    assert "(wynik policzony wcześniej)" in tekst      # model Sano też z pamięci
    assert "Model Sano wczytany" not in tekst          # i nie budował sieci
