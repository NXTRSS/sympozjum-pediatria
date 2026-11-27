"""Pomocniki notatnika LM (odpowiednik utils.py z zajęć ALK) — seed, wypisywanie, wykresy kroków."""
from __future__ import annotations

import os
import random
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def ensure_package(nazwa: str, *, pip_name: str | None = None) -> None:
    """Colab: doinstaluj pakiet. Lokalnie/uv: tylko import."""
    pip_name = pip_name or nazwa
    try:
        __import__(nazwa)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def reset_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def display_generated_name(
    prompt: str,
    known_names: set[str] | None = None,
    *,
    start: str = "%",
    stop: str = "!",
) -> None:
    """Pretty-print: czy nazwa jest w zbiorze treningowym (jak display_generated_city)."""
    if prompt.startswith(start):
        nazwa = prompt[len(start) :]
    else:
        nazwa = prompt
    if nazwa.endswith(stop):
        nazwa = nazwa[: -len(stop)]
    known_names = known_names or set()
    if nazwa in known_names:
        print(f'⚠️  „{nazwa}” — już była w zbiorze treningowym (model ją odtworzył)')
    else:
        print(f'✨ „{nazwa}” — nie ma w zbiorze (model coś wymyślił)')


def visualize_probabilities(
    last_output: np.ndarray,
    krok: int,
    tokenizer,
    prompt: str,
    *,
    chosen_idx: int | None = None,
    top_n: int = 12,
    show: bool = True,
) -> None:
    """Histogram P(następna litera | kontekst) — jeden krok generacji."""
    probs = np.asarray(last_output, dtype=np.float64)
    if probs.ndim != 1:
        probs = probs.ravel()
    probs[0] = 0
    if probs.sum() > 0:
        probs /= probs.sum()
    idx = np.argsort(probs)[-top_n:][::-1]
    index_word = getattr(tokenizer, "index_word", None) or {}
    labels = []
    for i in idx:
        ch = index_word.get(int(i), "?")
        labels.append(repr(ch))
    fig, ax = plt.subplots(figsize=(7, 2.4))
    colors = ["#FFAF7B" if int(i) == chosen_idx else "#3A1C7D" for i in idx]
    ax.bar(range(len(idx)), probs[idx], color=colors)
    ax.set_xticks(range(len(idx)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, min(1.05, max(0.05, probs[idx].max() * 1.2)))
    ax.set_title(f"krok {krok}: kontekst {prompt!r}", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if show:
        plt.show()
    else:
        plt.draw()