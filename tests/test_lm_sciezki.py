from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    import importlib.util
    import sys

    nb = ROOT / "notebook"
    sys.path.insert(0, str(nb))
    spec = importlib.util.spec_from_file_location("lm_sciezki", nb / "lm_sciezki.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_korzen_i_csv_choroby():
    mod = _mod()
    assert mod.korzen_sympozjum() == ROOT
    p = mod.sciezka_csv("choroby.csv")
    assert p is not None and p.name == "choroby.csv"
    assert (ROOT / "data" / "choroby.csv").samefile(p)


def test_dinozaury_w_repo():
    mod = _mod()
    assert mod.sciezka_csv("dinozaury.csv") is not None