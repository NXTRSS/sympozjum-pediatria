#!/usr/bin/env python3
"""Spakuj artefakty Colab do GitHub Release (nie commituj do gita).

delphi-pamiec-colab.tar.gz   — checkpoint (dane syntetyczne) + wyniki trybu z_pamieci
usg-startowy-colab.tar.gz    — tylko lokalnie, z --usg (zawiera wycinek nagrania; nie publikujemy)

Po pakowaniu: gh release create <tag> dist/colab-release/delphi-pamiec-colab.tar.gz
Następnie zaktualizuj notebook/colab_release_manifest.json (sha256 robi ten skrypt).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "notebook" / "colab_release_manifest.json"
DEFAULT_TAG = "colab-demo-2026-11-27"


def _sha256(plik: Path) -> str:
    h = hashlib.sha256()
    with plik.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tar(cel: Path, czlonowie: list[tuple[Path, str]]) -> None:
    cel.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cel, "w:gz") as tar:
        for sciezka, arcname in czlonowie:
            if not sciezka.exists():
                raise FileNotFoundError(sciezka)
            tar.add(sciezka, arcname=arcname)


def spakuj_delphi(out: Path) -> Path:
    archiwum = out / "delphi-pamiec-colab.tar.gz"
    model = ROOT / "delphi" / "pamiec" / "delphi_syntetyczny.pt"
    wyniki = ROOT / "delphi" / "pamiec" / "wyniki"
    if not model.is_file():
        sys.exit("Brak delphi/pamiec/delphi_syntetyczny.pt — uruchom najpierw przygotuj_demo.py")
    czlonowie: list[tuple[Path, str]] = [(model, "delphi/pamiec/delphi_syntetyczny.pt")]
    if wyniki.is_dir():
        for p in sorted(wyniki.iterdir()):
            if p.is_file():
                czlonowie.append((p, f"delphi/pamiec/wyniki/{p.name}"))
    _tar(archiwum, czlonowie)
    return archiwum


def spakuj_usg(out: Path) -> Path:
    archiwum = out / "usg-startowy-colab.tar.gz"
    pamiec = ROOT / "usg" / "pamiec"
    nazwa = "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000"
    mha = (
        ROOT
        / "usg"
        / "zewnetrzne"
        / "ACOUSLIC-AI-baseline"
        / "test"
        / "input"
        / "images"
        / "stacked-fetal-ultrasound"
        / f"{nazwa}.mha"
    )
    if not pamiec.is_dir() or not any(pamiec.glob(f"{nazwa}*")):
        sys.exit("Brak usg/pamiec dla wycinka startowego — uruchom przygotuj_demo.py")
    if not mha.is_file():
        sys.exit(f"Brak pliku startowego .mha: {mha}")
    czlonowie: list[tuple[Path, str]] = [
        (mha, f"usg/dane/colab/{nazwa}.mha"),
    ]
    for p in sorted(pamiec.iterdir()):
        if p.is_file() and p.name.startswith(nazwa):
            czlonowie.append((p, f"usg/pamiec/{p.name}"))
    _tar(archiwum, czlonowie)
    return archiwum


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "colab-release")
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--usg", action="store_true", help="spakuj też paczkę USG (tylko do użytku lokalnego)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    delphi_tar = spakuj_delphi(args.out)
    usg_tar = spakuj_usg(args.out) if args.usg else None
    base = f"https://github.com/NXTRSS/sympozjum-pediatria/releases/download/{args.tag}/"

    manifest = {
        "tag": args.tag,
        "base_url": base,
        "delphi_pamiec": {
            "plik": delphi_tar.name,
            "url": base + delphi_tar.name,
            "sha256": _sha256(delphi_tar),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Zapisano {delphi_tar} ({delphi_tar.stat().st_size // 1024} KiB)")
    if usg_tar:
        print(f"Zapisano {usg_tar} ({usg_tar.stat().st_size // 1024} KiB) — nie publikować")
    print(f"Manifest: {MANIFEST}")
    print(f"\nPublikacja:\n  gh release create {args.tag} --title 'Colab demo' {delphi_tar}")


if __name__ == "__main__":
    main()