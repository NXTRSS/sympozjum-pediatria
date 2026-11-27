"""Jedna logika ścieżek: git pull lokalnie + Google Colab (bez osobnego .ipynb).

Colab otwiera tylko plik z GitHuba — reszta repozytorium nie jest na dysku.
Pierwsza komórka notebooka woła bootstrap() z tego modułu (lokalnie: plik z repo;
w Colab bez klonu: pobranie tego pliku z raw main, potem git clone całości).
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

REPO_GIT = "https://github.com/NXTRSS/sympozjum-pediatria.git"
REPO_RAW_SCIEZKA = (
    "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/notebook/colab_sciezka.py"
)
COLAB_ROOT = Path("/content/sympozjum-pediatria")


def w_colabie() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def korzen_repo_lokalnie() -> Path | None:
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / ".git").is_dir() and (base / "notebook" / "lm_utils.py").is_file():
            return base
        if (base / "pyproject.toml").is_file() and (base / "notebook" / "lm_utils.py").is_file():
            return base
    return None


def _dopisz_sys_path(*sciezki: Path) -> None:
    for p in sciezki:
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def upewnij_srodowisko(marker_wzgledem_root: str, folder_notebooka: str) -> Path:
    """marker np. notebook/lm_utils.py; folder_notebooka np. notebook lub delphi/notebook."""
    if w_colabie():
        mark = COLAB_ROOT / marker_wzgledem_root
        if not mark.is_file():
            subprocess.run(["git", "clone", "--depth", "1", REPO_GIT, str(COLAB_ROOT)], check=True)
        nb = COLAB_ROOT / folder_notebooka
        os.chdir(nb)
        _dopisz_sys_path(nb)
        if folder_notebooka.startswith("delphi/"):
            _dopisz_sys_path(COLAB_ROOT / "delphi" / "src")
        if folder_notebooka.startswith("usg/"):
            _dopisz_sys_path(COLAB_ROOT / "usg" / "src")
        return nb

    root = korzen_repo_lokalnie()
    if root is not None:
        nb = root / folder_notebooka
        _dopisz_sys_path(nb)
        if (root / "delphi" / "src").is_dir():
            _dopisz_sys_path(root / "delphi" / "src")
        if (root / "usg" / "src").is_dir():
            _dopisz_sys_path(root / "usg" / "src")
        return nb if nb.is_dir() else Path.cwd()

    # Jupyter odpalony z podfolderu (np. samo delphi/notebook) bez widocznego .git
    nb = Path.cwd()
    if (nb / "colab_bootstrap.py").is_file():
        _dopisz_sys_path(nb)
        src = nb.parent / "src"
        if src.is_dir():
            _dopisz_sys_path(src)
        return nb
    return Path.cwd()


def zaladuj_ten_modul():
    """Colab przed klonem: szukaj colab_sciezka.py na dysku, inaczej raw z GitHuba."""
    if "colab_sciezka" in sys.modules:
        return sys.modules["colab_sciezka"]
    for base in (Path.cwd(), *Path.cwd().parents):
        plik = base / "notebook" / "colab_sciezka.py"
        if plik.is_file():
            import importlib.util

            spec = importlib.util.spec_from_file_location("colab_sciezka", plik)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            sys.modules["colab_sciezka"] = mod
            return mod
    mod = types.ModuleType("colab_sciezka")
    with urllib.request.urlopen(REPO_RAW_SCIEZKA, timeout=90) as resp:
        kod = resp.read().decode()
    exec(kod, mod.__dict__)  # noqa: S102
    sys.modules["colab_sciezka"] = mod
    return mod


def _lokalizuj_colab_bootstrap(folder_notebooka: str) -> Path:
    for base in (korzen_repo_lokalnie(), COLAB_ROOT if w_colabie() else None):
        if base is None:
            continue
        p = base / folder_notebooka / "colab_bootstrap.py"
        if p.is_file():
            return p
    for entry in sys.path:
        p = Path(entry) / "colab_bootstrap.py"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Brak colab_bootstrap.py (szukano w {folder_notebooka}). "
        "W Colabie: otwórz notebook ponownie z GitHub (branch main), nie trzymaj starej kopii z cache."
    )


def _importuj_i_uruchom_colab_bootstrap(folder_notebooka: str) -> None:
    import importlib.util

    plik = _lokalizuj_colab_bootstrap(folder_notebooka)
    spec = importlib.util.spec_from_file_location("colab_bootstrap", plik)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.uruchom()


def bootstrap(ktory: str) -> None:
    """ktory: 'lm' | 'delphi' | 'usg' — pierwsza komórka każdego notebooka."""
    konfig = {
        "lm": ("notebook/lm_utils.py", "notebook"),
        "delphi": ("delphi/pyproject.toml", "delphi/notebook"),
        "usg": ("usg/pyproject.toml", "usg/notebook"),
    }[ktory]
    marker, folder = konfig
    upewnij_srodowisko(marker, folder)
    _importuj_i_uruchom_colab_bootstrap(folder)


def uruchom_notebook_lm() -> None:
    bootstrap("lm")


def uruchom_notebook_delphi() -> None:
    bootstrap("delphi")


def uruchom_notebook_usg() -> None:
    bootstrap("usg")