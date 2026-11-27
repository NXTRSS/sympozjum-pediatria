"""Jednorazowe przygotowanie Delphi w Google Colab (poza gitem: zewnetrzne/, pamiec/).

Notebook potrzebuje klonu gerstung-lab/Delphi oraz wytrenowanego checkpointu z danych syntetycznych.
Lokalnie robi to scripts/przygotuj_demo.py; w Colabie woła bootstrap przez colab_sciezka.py.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from delphi_demo import pamiec, trening, zasoby


def uruchom_jesli_potrzeba(katalog_delphi: Path) -> None:
    """Klon autorów; gdy brak modelu — pełne przygotowanie (trening + lekkie wyniki do trybu z_pamieci)."""
    katalog_delphi = Path(katalog_delphi).resolve()
    src = katalog_delphi / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if not (zasoby.katalog_kodu() / "model.py").exists():
        print("Colab Delphi: klonuję gerstung-lab/Delphi (kod MIT + dane syntetyczne)…")
        zasoby.sklonuj()
        print("Colab Delphi: klon gotowy.")
    else:
        print("Colab Delphi: kod autorów już jest.")

    pamiec.zsynchronizuj_wzorce()

    if trening.SCIEZKA_MODELU.exists():
        print(f"Colab Delphi: model {trening.SCIEZKA_MODELU.name} jest — pomijam trening.")
        return

    import torch

    urzadzenie = "cuda" if torch.cuda.is_available() else "cpu"
    kroki = int(os.environ.get("DELPHI_COLAB_KROKI", "2500"))
    n_probek = int(os.environ.get("DELPHI_COLAB_N_PROBEK", "200"))
    if urzadzenie == "cpu":
        print(
            "Colab Delphi: brak GPU (Runtime → Zmień typ środowiska wykonawczego na T4). "
            "Trening na CPU może potrwać ponad godzinę."
        )
    print(
        f"Colab Delphi: brak wytrenowanego modelu — przygotowanie "
        f"({kroki} kroków na {urzadzenie}, N={n_probek}; pierwszy raz zwykle 15–25 min na GPU)."
    )
    script = katalog_delphi / "scripts" / "przygotuj_demo.py"
    env = {**os.environ, "PYTHONPATH": str(src)}
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--urzadzenie",
            urzadzenie,
            "--kroki",
            str(kroki),
            "--n-probek",
            str(n_probek),
        ],
        cwd=str(katalog_delphi),
        env=env,
    )
    print("Colab Delphi: przygotowanie zakończone — możesz uruchomić resztę notebooka.")