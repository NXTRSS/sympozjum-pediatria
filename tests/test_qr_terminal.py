import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebook"))

from qr_terminal import linie_qr  # noqa: E402


def test_linie_qr_nie_puste():
    lines = linie_qr("https://example.test/demo")
    assert len(lines) > 10
    assert all("██" in row or "  " in row for row in lines)
    assert len(lines[0]) == len(lines[-1])