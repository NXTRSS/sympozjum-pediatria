"""Tekst pierwszej komórki (jedyne miejsce do edycji szablonu bootstrap).

Użyte w: model_jezyka_choroby.ipynb, delphi/scripts/zbuduj_notebook.py, usg_masa_plodu.ipynb
"""

PIERWSZA_KOMORKA = '''# Środowisko — po git pull lokalnie i w Colab (ten sam URL co na slajdach)
import importlib.util
import sys
import types
import urllib.request
from pathlib import Path

KTORY = "{ktory}"

def _mod_colab_sciezka():
    for b in [Path.cwd(), *Path.cwd().parents[:12]]:
        p = b / "notebook" / "colab_sciezka.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location("colab_sciezka", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    mod = types.ModuleType("colab_sciezka")
    u = "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/notebook/colab_sciezka.py"
    exec(urllib.request.urlopen(u, timeout=90).read().decode(), mod.__dict__)
    sys.modules["colab_sciezka"] = mod
    return mod

_mod_colab_sciezka().bootstrap(KTORY)
'''


def tekst(ktory: str) -> str:
    assert ktory in ("lm", "delphi", "usg")
    return PIERWSZA_KOMORKA.format(ktory=ktory)