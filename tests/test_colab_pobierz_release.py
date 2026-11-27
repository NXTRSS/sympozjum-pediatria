import json
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
CPR = ROOT / "notebook" / "colab_pobierz_release.py"


def _mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("colab_pobierz_release", CPR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_lokalny(tmp_path):
    mod = _mod()
    man = {"delphi_pamiec": {"url": "http://x/a.tar.gz", "sha256": "0" * 64}}
    p = tmp_path / "colab_release_manifest.json"
    p.write_text(json.dumps(man))
    assert mod._wczytaj_manifest(p) == man