"""Skrypt uruchom_demo.py, cache i notebook dają ten sam model przy ziarnie 42."""
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
from model_cache import wyczysc as wyczysc_cache  # noqa: E402


def _z_notebooka(tmp: Path) -> dict:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["DEMO_WYNIK"] = str(tmp / "wynik.json")
    nb = nbformat.read(NB, as_version=4)
    nb.cells.append(nbformat.v4.new_code_cell(
        "import json, os\n"
        "from pathlib import Path\n"
        "Path(os.environ['DEMO_WYNIK']).write_text(json.dumps({\n"
        "    'loss': [float(x) for x in historia.history['loss']],\n"
        "    'wagi': [w.numpy().tolist() for w in model.weights],\n"
        "    'pierwsza': pierwsza_nazwa,\n"
        "    'dziesiec': lista_nazw,\n"
        "}))\n"
    ))
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(NOTEBOOK_DIR)}}).execute()
    return json.loads((tmp / "wynik.json").read_text())


def _porownaj_wyniki(a: dict, b: dict, *, porownaj_nazwy: bool = True) -> None:
    assert a["loss"] == b["loss"]
    for x, y in zip(a["wagi"], b["wagi"]):
        assert np.allclose(np.asarray(x), np.asarray(y))
    if not porownaj_nazwy:
        return
    assert a["pierwsza"] == b["pierwsza"]
    assert a["dziesiec"] == b["dziesiec"]


def test_dwa_przebiegi_skryptu_sa_zgodne():
    wyczysc_cache()
    a = przebieg(force_train=True)
    b = przebieg(force_train=True)
    _porownaj_wyniki(a, b)


def test_notebook_jest_deterministyczny(tmp_path):
    a = _z_notebooka(tmp_path)
    b = _z_notebooka(tmp_path)
    _porownaj_wyniki(a, b, porownaj_nazwy=False)


def test_cache_identyczny_ze_swiezym_treningiem():
    wyczysc_cache()
    swiezy = przebieg(force_train=True)
    z_cache = przebieg()
    assert z_cache.get("z_cache") is True
    _porownaj_wyniki(swiezy, z_cache)


def test_cache_uniewazniony_po_zmianie_ziarna():
    wyczysc_cache()
    przebieg(force_train=True, ziarno=42)
    inny = przebieg(force_train=False, ziarno=99, use_cache=True)
    assert not inny.get("z_cache")


def test_cache_uniewazniony_po_zmianie_danych(monkeypatch):
    wyczysc_cache()
    przebieg(force_train=True)
    import demo_model as dm

    real = dm.hash_danych

    def zmieniony(sciezka):
        return real(sciezka) + "x"

    monkeypatch.setattr(dm, "hash_danych", zmieniony)
    inny = przebieg(force_train=False, use_cache=True)
    assert not inny.get("z_cache")


def test_cache_uniewazniony_po_zmianie_epok(monkeypatch):
    wyczysc_cache()
    przebieg(force_train=True)
    import demo_model as dm

    monkeypatch.setattr(dm, "LICZBA_EPOK", 11)
    inny = przebieg(force_train=False, use_cache=True)
    assert not inny.get("z_cache")


def test_skrypt_zgodny_z_notebookiem(tmp_path):
    """Te same hiperparametry i ziarno treningu → te same wagi co notatnik (generowanie bez stałego ziarna)."""
    wyczysc_cache()
    skrypt = przebieg(force_train=True)
    notatnik = _z_notebooka(tmp_path)
    _porownaj_wyniki(skrypt, notatnik, porownaj_nazwy=False)