import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Testy sieciowe i wolne uruchamiamy tylko na żądanie: USG_PELNE=1 uv run pytest."""
    if os.environ.get("USG_PELNE") == "1":
        return
    pomin = pytest.mark.skip(reason="ustaw USG_PELNE=1, aby uruchomić")
    for item in items:
        if "siec" in item.keywords or "wolny" in item.keywords:
            item.add_marker(pomin)
