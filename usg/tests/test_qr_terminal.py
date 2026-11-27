import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from qr_terminal import linie_qr  # noqa: E402


def test_qr_niepusty():
    linie = linie_qr("http://127.0.0.1:7860")
    assert len(linie) > 10
    assert any("██" in row for row in linie)