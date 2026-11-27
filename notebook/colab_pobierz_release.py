"""Pobierz i rozpakuj paczki z GitHub Release (szybki start Colab, bez treningu)."""
from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO_RAW_MANIFEST = (
    "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/notebook/colab_release_manifest.json"
)


def _wczytaj_manifest(lokalny: Path | None) -> dict | None:
    sciezki = [lokalny] if lokalny and lokalny.is_file() else []
    for baza in (Path.cwd(), *Path.cwd().parents[:8]):
        p = baza / "notebook" / "colab_release_manifest.json"
        if p.is_file():
            sciezki.append(p)
    for p in sciezki:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    if os.environ.get("COLAB_RELEASE_OFF"):
        return None
    try:
        with urllib.request.urlopen(REPO_RAW_MANIFEST, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except OSError:
        return None


def _pobierz(url: str, sha256: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    nazwa = url.rsplit("/", 1)[-1]
    cel = folder / nazwa
    if cel.is_file() and hashlib.sha256(cel.read_bytes()).hexdigest() == sha256:
        return cel
    print(f"Pobieram {nazwa} ({url})…")
    tmp = folder / f".{nazwa}.pobieranie"
    urllib.request.urlretrieve(url, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 paczki {nazwa} się nie zgadza (oczekiwano {sha256[:12]}…)")
    tmp.replace(cel)
    return cel


def _rozpakuj(archiwum: Path, korzen: Path) -> None:
    with tarfile.open(archiwum, "r:gz") as tar:
        for member in tar.getmembers():
            cel = korzen / member.name
            if member.isdir():
                continue
            cel.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src:
                cel.write_bytes(src.read())


def pobierz_delphi_pamiec(katalog_delphi: Path, manifest: dict | None = None) -> bool:
    manifest = manifest or _wczytaj_manifest(None)
    if not manifest or "delphi_pamiec" not in manifest:
        return False
    meta = manifest["delphi_pamiec"]
    korzen = katalog_delphi.parent if katalog_delphi.name == "delphi" else katalog_delphi
    if (katalog_delphi / "pamiec" / "delphi_syntetyczny.pt").is_file():
        return True
    with tempfile.TemporaryDirectory() as tmp:
        arch = _pobierz(meta["url"], meta["sha256"], Path(tmp))
        _rozpakuj(arch, korzen)
    print("Colab Delphi: pobrano checkpoint i wyniki z GitHub Release (bez treningu).")
    return True


def pobierz_usg_startowy(katalog_usg: Path, manifest: dict | None = None) -> bool:
    manifest = manifest or _wczytaj_manifest(None)
    if not manifest or "usg_startowy" not in manifest:
        return False
    meta = manifest["usg_startowy"]
    korzen = katalog_usg.parent if katalog_usg.name == "usg" else katalog_usg
    nazwa = "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000"
    if (katalog_usg / "pamiec" / f"{nazwa}.npz").is_file():
        mha = katalog_usg / "dane" / "colab" / f"{nazwa}.mha"
        if mha.is_file():
            os.environ["USG_COLAB_STARTOWY"] = str(mha.resolve())
        return True
    with tempfile.TemporaryDirectory() as tmp:
        arch = _pobierz(meta["url"], meta["sha256"], Path(tmp))
        _rozpakuj(arch, korzen)
    mha = katalog_usg / "dane" / "colab" / f"{nazwa}.mha"
    if mha.is_file():
        os.environ["USG_COLAB_STARTOWY"] = str(mha.resolve())
    print("Colab USG: pobrano wycinek i wyniki z GitHub Release (bez wag sieci).")
    return True