"""Trening Delphi skryptem autorów (train.py) na ich danych syntetycznych + wczytywanie modelu.

Autorzy nie publikują wag Delphi-2M (UK Biobank, dostęp kontrolowany, upload ID 7318).
Dlatego przed prelekcją trenujemy ten sam model (2,24 mln parametrów) na danych syntetycznych z repo.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from delphi_demo import zasoby

CONFIG_DEMO = "config/train_delphi_demo.py"
SCIEZKA_MODELU = zasoby.KATALOG_PAMIECI / "delphi_syntetyczny.pt"
KATALOG_TRENINGU = zasoby.KATALOG_PAMIECI / "trening"
KOMUNIKAT_BRAK_MODELU = (
    "Brak wytrenowanego modelu {sciezka}. Przed prelekcją uruchom (z internetem), ok. 20 min (M1 Pro): "
    "cd delphi && uv run python scripts/przygotuj_demo.py"
)

_ITER = re.compile(r"^iter (\d+): loss ([\d.]+), time ([\d.]+)ms", re.MULTILINE)
_STEP = re.compile(r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)", re.MULTILINE)


def polecenie_treningu(katalog_wyj: Path, kroki: int = 5000, urzadzenie: str = "mps",
                       co_ile_ewaluacja: int = 250, partie_ewaluacji: int = 25,
                       co_ile_log: int = 25) -> list[str]:
    """Polecenie dla train.py autorów; uruchamiać z cwd = katalog kodu autorów."""
    rozgrzewka = min(500, max(1, kroki // 10))         # config: warmup_iters = 500 przy 5000 krokach
    return [sys.executable, "-u", "train.py", CONFIG_DEMO,
            f"--device={urzadzenie}", f"--out_dir={Path(katalog_wyj).resolve()}",
            f"--max_iters={kroki}", f"--lr_decay_iters={kroki}", f"--warmup_iters={rozgrzewka}",
            f"--eval_interval={co_ile_ewaluacja}", f"--eval_iters={partie_ewaluacji}",
            f"--log_interval={co_ile_log}"]


def trenuj(katalog_wyj: Path = KATALOG_TRENINGU, **kwargs) -> Path:
    """Uruchamia trening, zapisuje trening.log i czas.json w katalog_wyj. Zwraca ścieżkę ckpt.pt."""
    katalog_wyj = Path(katalog_wyj)
    katalog_wyj.mkdir(parents=True, exist_ok=True)
    polecenie = polecenie_treningu(katalog_wyj, **kwargs)
    start = time.time()
    with open(katalog_wyj / "trening.log", "w") as log:
        subprocess.run(polecenie, cwd=zasoby.sprawdz_kod(), stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    czas = time.time() - start
    (katalog_wyj / "czas.json").write_text(json.dumps({"czas_s": round(czas, 1), "polecenie": polecenie[2:]}))
    return katalog_wyj / "ckpt.pt"


def parsuj_log(tekst: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Log train.py → (kroki: iter, loss, ms) oraz (ewaluacje: krok, trening, walidacja)."""
    kroki = pd.DataFrame([(int(a), float(b), float(c)) for a, b, c in _ITER.findall(tekst)],
                         columns=["iter", "loss", "ms"])
    ewal = pd.DataFrame([(int(a), float(b), float(c)) for a, b, c in _STEP.findall(tekst)],
                        columns=["krok", "trening", "walidacja"])
    return kroki, ewal


def odchudz_checkpoint(ckpt: Path, cel: Path = SCIEZKA_MODELU) -> Path:
    """Zostawia tylko wagi i konfigurację (bez stanu optymalizatora): ~27 MB → ~9 MB."""
    dane = torch.load(ckpt, map_location="cpu", weights_only=False)
    cel = Path(cel)
    cel.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": dane["model"], "model_args": dane["model_args"],
                "iter_num": int(dane["iter_num"]), "best_val_loss": float(dane["best_val_loss"]),
                "zrodlo": "trening na danych syntetycznych gerstung-lab/Delphi (NIE UK Biobank)"}, cel)
    return cel


def zaladuj_model(sciezka: Path = SCIEZKA_MODELU, urzadzenie: str = "cpu"):
    """Wczytuje model Delphi (klasa autorów) w trybie eval."""
    sciezka = Path(sciezka)
    if not sciezka.exists():
        raise FileNotFoundError(KOMUNIKAT_BRAK_MODELU.format(sciezka=sciezka))
    m = zasoby.modul_modelu()
    dane = torch.load(sciezka, map_location=urzadzenie, weights_only=True)
    model = m.Delphi(m.DelphiConfig(**dane["model_args"]))
    model.load_state_dict(dane["model"])
    return model.to(urzadzenie).eval()


def liczba_parametrow(model) -> int:
    return sum(p.numel() for p in model.parameters())
