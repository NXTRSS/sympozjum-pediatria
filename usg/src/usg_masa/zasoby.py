"""Zasoby zewnętrzne: kod BabyNet++, FUVAI, Tab-Attention i baseline ACOUSLIC-AI oraz wagi
nnU-Net, FUVAI i modelu Sano (3D ResNet-18 z pracy TabAttention).

Nic z tego nie trafia do repo: BabyNet++, FUVAI i Tab-Attention nie mają licencji (wszelkie prawa
zastrzeżone), a wagi ważą 268 MB (nnU-Net), 277 MB (FUVAI) i 133 MB (model Sano).
Wszystko ląduje w usg/zewnetrzne/ (w .gitignore).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import requests

KATALOG_ZEWN = Path(__file__).resolve().parents[2] / "zewnetrzne"

REPOZYTORIA = {
    "BabyNetPlusPlus": ("https://github.com/SanoScience/BabyNetPlusPlus",
                        "39c11428befce9ca1011e0e488d8f77b77bcb801"),
    "FUVAI": ("https://github.com/SanoScience/FUVAI",
              "9ae74297a90effd43d049a7d29f0326c1a59713c"),
    "ACOUSLIC-AI-baseline": ("https://github.com/DIAGNijmegen/ACOUSLIC-AI-baseline",
                             "eccdc9b8c48f77c7c4b30a2c99bd323869e6124c"),
    # repo nazywa się „Tab-Attention”; klucz (= katalog w zewnetrzne/) bez myślnika, jak TabAttention_wagi
    "TabAttention": ("https://github.com/SanoScience/Tab-Attention",
                     "1a57920b782de9d47adfa87540102ecb34c37d70"),
}

_MODEL_WZGL = ("resources/nnUNet_results/Dataset300_ACOptimalSuboptimal/"
               "nnUNetTrainer__nnUNetPlans__2d")
WAGI_NNUNET_URL = ("https://media.githubusercontent.com/media/DIAGNijmegen/ACOUSLIC-AI-baseline/"
                   f"{REPOZYTORIA['ACOUSLIC-AI-baseline'][1]}/{_MODEL_WZGL}/fold_0/checkpoint_final.pth")
WAGI_NNUNET_SHA256 = "50aa1cee62b97638c8bd4de44fea15b0e47d5f4373b220206fbd05a3275fa1a4"

# Wagi FUVAI: README autorów linkuje Google Drive (plik fuvai_weights.pt, id poniżej).
# To czysty state_dict — wczytujemy go wyłącznie z weights_only=True.
WAGI_FUVAI_URL = ("https://drive.usercontent.google.com/download"
                  "?id=1lTmOmiiLMtELhd0HKjVsg_MoWEZ3oovN&export=download&confirm=t")
WAGI_FUVAI_SHA256 = "8187b2b2e348ed391fab37a67600e3f8c0a248aa71792accec08c7a9e71e0a10"


def sha256_pliku(sciezka: Path) -> str:
    h = hashlib.sha256()
    with open(sciezka, "rb") as f:
        for kawalek in iter(lambda: f.read(1 << 20), b""):
            h.update(kawalek)
    return h.hexdigest()


def pobierz_plik(url: str, cel: Path, sha256: str | None = None) -> Path:
    """Pobiera url do cel. Pomija, jeśli plik już jest i sha256 się zgadza."""
    cel = Path(cel)
    if cel.exists() and (sha256 is None or sha256_pliku(cel) == sha256):
        return cel
    cel.parent.mkdir(parents=True, exist_ok=True)
    tymczasowy = cel.with_suffix(cel.suffix + ".czesc")
    with requests.get(url, stream=True, timeout=60) as odp:
        odp.raise_for_status()
        with open(tymczasowy, "wb") as f:
            for kawalek in odp.iter_content(chunk_size=1 << 20):
                f.write(kawalek)
    if sha256 is not None and sha256_pliku(tymczasowy) != sha256:
        tymczasowy.unlink()
        raise ValueError(f"Niezgodna suma sha256 pliku pobranego z {url}")
    tymczasowy.replace(cel)
    return cel


def sklonuj(nazwa: str) -> Path:
    """Klonuje repo w przypiętym commicie (bez plików LFS). Zwraca ścieżkę."""
    url, commit = REPOZYTORIA[nazwa]
    cel = KATALOG_ZEWN / nazwa
    if not (cel / ".git").exists():
        cel.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(["git", "clone", "-q", url, str(cel)], check=True, env=env)
        subprocess.run(["git", "-C", str(cel), "checkout", "-q", commit], check=True, env=env)
    return cel


def katalog_modelu_nnunet() -> Path:
    """Folder modelu nnU-Net baseline'u z prawdziwymi wagami (podmienia wskaźnik LFS)."""
    folder = sklonuj("ACOUSLIC-AI-baseline") / _MODEL_WZGL
    pobierz_plik(WAGI_NNUNET_URL, folder / "fold_0" / "checkpoint_final.pth", WAGI_NNUNET_SHA256)
    return folder


def wagi_fuvai() -> Path:
    """Wagi FUVAI (277 MB) w usg/zewnetrzne/FUVAI_wagi/; pobiera, gdy ich nie ma lub sha256 się nie zgadza."""
    return pobierz_plik(WAGI_FUVAI_URL, KATALOG_ZEWN / "FUVAI_wagi" / "fuvai_weights.pt", WAGI_FUVAI_SHA256)


# To czysty state_dict 3D ResNet-18 (model bazowy obrazowy) — wczytujemy z weights_only=True.
WAGI_SANO_URL = "https://www.dropbox.com/s/cdswqlhew638ebd/TabAttention.pt?dl=1"
WAGI_SANO_SHA256 = "935afc86237dd8afb4a2cb48884dc837c2db3d19651ed2d1e8d1d7ca8a28a69b"


def wagi_sano() -> Path:
    """Wagi modelu Sano (133 MB) w usg/zewnetrzne/TabAttention_wagi/; pobiera, gdy ich nie ma lub sha256 się nie zgadza."""
    return pobierz_plik(WAGI_SANO_URL, KATALOG_ZEWN / "TabAttention_wagi" / "TabAttention.pt", WAGI_SANO_SHA256)
