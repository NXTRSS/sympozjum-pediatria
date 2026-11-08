import hashlib

import pytest

from usg_masa import zasoby


def test_sha256_pliku(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    assert zasoby.sha256_pliku(p) == hashlib.sha256(b"abc").hexdigest()


def test_pobierz_plik_pomija_gdy_sha_sie_zgadza(tmp_path, monkeypatch):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    monkeypatch.setattr(zasoby.requests, "get", lambda *a, **k: pytest.fail("nie powinno pobierać"))
    assert zasoby.pobierz_plik("http://x", p, hashlib.sha256(b"abc").hexdigest()) == p


def test_pobierz_plik_zly_sha_usuwa_i_zglasza(tmp_path, monkeypatch):
    class Odp:
        def raise_for_status(self): ...
        def iter_content(self, chunk_size): yield b"zle"
        def __enter__(self): return self
        def __exit__(self, *a): ...
    monkeypatch.setattr(zasoby.requests, "get", lambda *a, **k: Odp())
    cel = tmp_path / "w.pth"
    with pytest.raises(ValueError, match="sha256"):
        zasoby.pobierz_plik("http://x", cel, "0" * 64)
    assert not cel.exists()


def test_nieznane_repo():
    with pytest.raises(KeyError):
        zasoby.sklonuj("NieMaTakiego")


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwe_wagi_nnunet():
    folder = zasoby.katalog_modelu_nnunet()
    wagi = folder / "fold_0" / "checkpoint_final.pth"
    assert wagi.stat().st_size == 268_346_966
    assert (folder / "plans.json").exists()


def test_fuvai_zarejestrowane():
    url, commit = zasoby.REPOZYTORIA["FUVAI"]
    assert url == "https://github.com/SanoScience/FUVAI"
    assert commit == "9ae74297a90effd43d049a7d29f0326c1a59713c"
    assert len(zasoby.WAGI_FUVAI_SHA256) == 64


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwe_wagi_fuvai():
    wagi = zasoby.wagi_fuvai()
    assert wagi == zasoby.KATALOG_ZEWN / "FUVAI_wagi" / "fuvai_weights.pt"
    assert wagi.stat().st_size == 276_766_249
    assert zasoby.sha256_pliku(wagi) == zasoby.WAGI_FUVAI_SHA256


def test_tabattention_zarejestrowane():
    url, commit = zasoby.REPOZYTORIA["TabAttention"]
    assert url == "https://github.com/SanoScience/Tab-Attention"
    assert commit == "1a57920b782de9d47adfa87540102ecb34c37d70"
    assert zasoby.WAGI_SANO_URL == "https://www.dropbox.com/s/cdswqlhew638ebd/TabAttention.pt?dl=1"
    assert zasoby.WAGI_SANO_SHA256 == "935afc86237dd8afb4a2cb48884dc837c2db3d19651ed2d1e8d1d7ca8a28a69b"


@pytest.mark.siec
@pytest.mark.wolny
def test_prawdziwe_wagi_sano():
    wagi = zasoby.wagi_sano()
    assert wagi == zasoby.KATALOG_ZEWN / "TabAttention_wagi" / "TabAttention.pt"
    assert wagi.stat().st_size == 132_673_545
    assert zasoby.sha256_pliku(wagi) == zasoby.WAGI_SANO_SHA256
