import os

import pytest
import torch

from delphi_demo import tokeny, zasoby


def pytest_collection_modifyitems(config, items):
    """Testy sieciowe i wolne uruchamiamy tylko na żądanie: DELPHI_PELNE=1 uv run pytest."""
    if os.environ.get("DELPHI_PELNE") == "1":
        return
    pomin = pytest.mark.skip(reason="ustaw DELPHI_PELNE=1, aby uruchomić")
    for item in items:
        if "siec" in item.keywords or "wolny" in item.keywords:
            item.add_marker(pomin)


@pytest.fixture(scope="session")
def maly_model():
    """Prawdziwa klasa Delphi autorów, ale malutka i z losowymi wagami: testy w milisekundach."""
    m = zasoby.modul_modelu()
    torch.manual_seed(0)
    cfg = m.DelphiConfig(block_size=96, vocab_size=tokeny.LICZBA_TOKENOW, n_layer=1, n_head=2,
                         n_embd=16, dropout=0.0, t_min=0.1, bias=False, mask_ties=True,
                         ignore_tokens=list(tokeny.IGNOROWANE))
    return m.Delphi(cfg).eval()
