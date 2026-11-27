"""Mapa scalenia kroków (Kamil) jest zapisana w slides/*.html."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.step_kroki import KAMIL_SCALENIA, weryfikuj  # noqa: E402


def test_kamil_scalenia_w_zrodlach_slajdow():
    errors = weryfikuj(KAMIL_SCALENIA)
    assert not errors, "\n".join(errors)