"""Port HTTP demo USG — bez uruchamiania Gradio."""
import importlib.util
import socket
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "demo_usg.py"


@pytest.fixture
def demo_usg_mod():
    spec = importlib.util.spec_from_file_location("demo_usg_port", APP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_domyslny_port_to_7861(demo_usg_mod, monkeypatch):
    monkeypatch.delenv("GRADIO_SERVER_PORT", raising=False)
    assert demo_usg_mod._port_domyslny_z_env() == 7861


def test_port_z_env(demo_usg_mod, monkeypatch):
    monkeypatch.setenv("GRADIO_SERVER_PORT", "7877")
    assert demo_usg_mod._port_domyslny_z_env() == 7877


def test_wolny_port_od_zajety_pierwszy(demo_usg_mod):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blok:
        blok.bind(("127.0.0.1", 0))
        zajety = blok.getsockname()[1]
        blok.listen(1)
        wolny = demo_usg_mod._wolny_port_od(zajety)
        assert wolny > zajety