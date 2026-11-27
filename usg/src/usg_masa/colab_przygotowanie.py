"""Google Colab: zależności bez `pip install -e` (Python 3.12/3.13) + pierwsze przygotowanie demo.

Wagi, klonowane repozytoria i wyniki w `pamiec/` nie są w GitHubie — skrypt `przygotuj_demo.py`
pobiera je z publicznych źródeł. Domyślnie tylko 128-klatkowy wycinek (bez pełnego Zenodo).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path


def _pobierz_z_release(katalog_usg: Path) -> bool:
    korzen = katalog_usg.parent
    plik = korzen / "notebook" / "colab_pobierz_release.py"
    if not plik.is_file():
        try:
            with urllib.request.urlopen(
                "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/"
                "notebook/colab_pobierz_release.py",
                timeout=60,
            ) as resp:
                kod = resp.read().decode()
            mod = types.ModuleType("colab_pobierz_release")
            exec(kod, mod.__dict__)  # noqa: S102
            return bool(mod.pobierz_usg_startowy(katalog_usg))
        except OSError as exc:
            print(f"Colab USG: Release niedostępny ({exc}) — fallback do pełnego przygotowania.")
            return False
    spec = importlib.util.spec_from_file_location("colab_pobierz_release", plik)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return bool(mod.pobierz_usg_startowy(katalog_usg))

PARY = (
    ("torch", "torch>=2.2"),
    ("cv2", "opencv-python-headless>=4.8"),
    ("SimpleITK", "SimpleITK>=2.3"),
    ("remotezip", "remotezip>=0.12"),
    ("einops", "einops>=0.7"),
    ("pandas", "pandas>=2.0"),
    ("matplotlib", "matplotlib>=3.8"),
    ("ipywidgets", "ipywidgets>=8"),
    ("nnunetv2", "nnunetv2==2.2.1"),
)


def _zainstaluj_zaleznosci() -> None:
    brakuje = [pip for mod, pip in PARY if importlib.util.find_spec(mod) is None]
    if not brakuje:
        print("Colab USG: PyTorch, nnU-Net i SimpleITK są już w kernelu — pomijam pip.")
        return
    print(f"Colab USG: doinstalowuję {', '.join(brakuje)}…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *brakuje],
    )


def _ustaw_nnunet(pamiec: Path) -> None:
    for zmienna in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(zmienna, str(pamiec.resolve()))
    os.environ.setdefault("nnUNet_n_proc_DA", "12")


def _ma_wyniki_wycinka(katalog_usg: Path) -> bool:
    from usg_masa import dane, segmentacja as seg

    pamiec = katalog_usg / "pamiec"
    try:
        start = dane.pobierz_przypadek_startowy()
    except Exception:
        return False
    if not start.is_file():
        return False
    nazwa = start.name.removesuffix(".mha")
    pliki = (
        pamiec / f"{nazwa}.npz",
        pamiec / f"{nazwa}_fuvai.npz",
        pamiec / f"{nazwa}_sano.json",
    )
    return all(p.is_file() for p in pliki)


def uruchom_jesli_potrzeba(katalog_usg: Path) -> None:
    katalog_usg = Path(katalog_usg).resolve()
    src = katalog_usg / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("USG_COLAB", "1")
    _ustaw_nnunet(katalog_usg / "pamiec")
    _zainstaluj_zaleznosci()

    if _ma_wyniki_wycinka(katalog_usg):
        print("Colab USG: wyniki dla wycinka startowego są w pamiec/ — pomijam przygotowanie.")
        return

    if os.environ.get("USG_COLAB_SKIP_RELEASE", "").lower() not in ("1", "true"):
        try:
            if _pobierz_z_release(katalog_usg) and _ma_wyniki_wycinka(katalog_usg):
                return
        except Exception as exc:  # noqa: BLE001
            print(f"Colab USG: Release nieudane ({exc}) — fallback do przygotuj_demo.")

    ile = int(os.environ.get("USG_COLAB_ILE_ZENODO", "0"))
    co_ktora = int(os.environ.get("USG_COLAB_CO_KTORA", "2"))
    print(
        f"Colab USG: pobieram wagi i liczę wyniki "
        f"(nagrania z Zenodo: {ile}; pierwszy raz ok. 15–25 min z GPU, głównie pobieranie ~700 MB)."
    )
    skrypt = katalog_usg / "scripts" / "przygotuj_demo.py"
    env = {**os.environ, "PYTHONPATH": str(src)}
    subprocess.check_call(
        [
            sys.executable,
            str(skrypt),
            "--ile-zenodo",
            str(ile),
            "--co-ktora",
            str(co_ktora),
        ],
        cwd=str(katalog_usg),
        env=env,
    )
    print("Colab USG: gotowe — ustaw NR_NAGRANIA=0 dla wycinka albo 1+ po pobraniu z Zenodo.")