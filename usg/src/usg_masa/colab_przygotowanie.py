"""Google Colab: zależności bez `pip install -e` i przygotowanie demo.

Jeśli w GitHub Release jest paczka z gotowymi wynikami, bierzemy ją (szybko).
Bez niej uruchamiamy `przygotuj_demo.py`: kod autorów, wagi (~700 MB) i wycinek
nagrania pobierają się w Colabie, pierwsze uruchomienie trwa ok. 15–25 min.
Wyłączenie liczenia (tylko komunikat): `USG_COLAB_BEZ_PRZYGOTOWANIA=1`.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

# Ten sam case co w dane.pobierz_przypadek_startowy / paczka usg-startowy-colab.tar.gz
WYCINEK_NAZWA = "04a04f2e-840b-47f8-a907-abe7aeab3f41_1_0000"

PARY_LEKKIE = (
    ("pandas", "pandas>=2.0"),
    ("matplotlib", "matplotlib>=3.8"),
    ("ipywidgets", "ipywidgets>=8"),
    ("SimpleITK", "SimpleITK>=2.3"),
    ("numpy", "numpy<2"),
)

PARY_CIEZKIE = (
    ("torch", "torch>=2.2"),
    ("cv2", "opencv-python-headless>=4.8"),
    ("remotezip", "remotezip>=0.12"),
    ("einops", "einops>=0.7"),
    ("nnunetv2", "nnunetv2==2.2.1"),
)


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
            print(f"Colab USG: nie udało się pobrać Release ({exc}).")
            return False
    spec = importlib.util.spec_from_file_location("colab_pobierz_release", plik)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return bool(mod.pobierz_usg_startowy(katalog_usg))


def _zainstaluj(pipy: tuple[tuple[str, str], ...]) -> None:
    brakuje = [pip for mod, pip in pipy if importlib.util.find_spec(mod) is None]
    if not brakuje:
        return
    print(f"Colab USG: doinstalowuję {', '.join(brakuje)}…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *brakuje])


def _ustaw_nnunet(pamiec: Path) -> None:
    for zmienna in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ.setdefault(zmienna, str(pamiec.resolve()))
    os.environ.setdefault("nnUNet_n_proc_DA", "12")


def _gotowy_wycinek(katalog_usg: Path) -> bool:
    """Pamiec + .mha z Release — bez klonowania baseline."""
    pamiec = katalog_usg / "pamiec"
    mha_colab = katalog_usg / "dane" / "colab" / f"{WYCINEK_NAZWA}.mha"
    if mha_colab.is_file():
        os.environ.setdefault("USG_COLAB_STARTOWY", str(mha_colab.resolve()))
    wymagane = (
        pamiec / f"{WYCINEK_NAZWA}.npz",
        pamiec / f"{WYCINEK_NAZWA}_fuvai.npz",
        pamiec / f"{WYCINEK_NAZWA}_sano.json",
    )
    if not all(p.is_file() for p in wymagane):
        return False
    if not os.environ.get("USG_COLAB_STARTOWY"):
        try:
            from usg_masa import dane

            start = dane.pobierz_przypadek_startowy()
            if start.is_file():
                os.environ.setdefault("USG_COLAB_STARTOWY", str(start.resolve()))
        except Exception:
            return False
    return Path(os.environ["USG_COLAB_STARTOWY"]).is_file()


def uruchom_jesli_potrzeba(katalog_usg: Path) -> None:
    katalog_usg = Path(katalog_usg).resolve()
    src = katalog_usg / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    os.environ.setdefault("USG_COLAB", "1")
    _ustaw_nnunet(katalog_usg / "pamiec")

    if _gotowy_wycinek(katalog_usg):
        _zainstaluj(PARY_LEKKIE)
        print("Colab USG: wycinek i wyniki gotowe (tryb z_pamieci, NR_NAGRANIA=0).")
        return

    if os.environ.get("USG_COLAB_SKIP_RELEASE", "").lower() not in ("1", "true"):
        _pobierz_z_release(katalog_usg)

    if _gotowy_wycinek(katalog_usg):
        _zainstaluj(PARY_LEKKIE)
        print("Colab USG: wycinek i wyniki gotowe (tryb z_pamieci, NR_NAGRANIA=0).")
        return

    if os.environ.get("USG_COLAB_BEZ_PRZYGOTOWANIA", "").lower() in ("1", "true"):
        raise RuntimeError(
            "Colab USG: brak gotowych wyników, a przygotowanie jest wyłączone "
            "(USG_COLAB_BEZ_PRZYGOTOWANIA=1). Usuń tę zmienną i uruchom pierwszą komórkę ponownie."
        )

    _zainstaluj(PARY_LEKKIE + PARY_CIEZKIE)
    ile = int(os.environ.get("USG_COLAB_ILE_ZENODO", "0"))
    co_ktora = int(os.environ.get("USG_COLAB_CO_KTORA", "2"))
    print(
        "Colab USG: pierwsze uruchomienie — pobieram kod autorów, wagi (~700 MB) "
        f"i wycinek nagrania, potem liczę wyniki (Zenodo: {ile}). "
        "To potrwa ok. 15–25 min; kolejne komórki ruszą od razu po zakończeniu."
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
    print("Colab USG: gotowe — NR_NAGRANIA=0 dla wycinka.")