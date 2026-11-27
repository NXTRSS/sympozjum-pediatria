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
    probabilities,
    step,
    tokenizer,
    prompt: str,
) -> None:
    """Wykres słupkowy jak w notatniku dinozaurów (wszystkie znaki ze słownika)."""
    index_word = getattr(tokenizer, "index_word", None) or {}
    characters = [char for idx, char in sorted(index_word.items()) if idx > 0]
    probs = np.asarray(probabilities, dtype=np.float64).ravel()
    if len(characters) + 1 == len(probs):
        bar_heights = probs[1:]
    elif len(characters) == len(probs) - 1:
        bar_heights = probs[1:]
    else:
        bar_heights = probs[1 : 1 + len(characters)]
    plt.figure(figsize=(10, 6))
    plt.bar(characters, bar_heights, color="skyblue")
    plt.title(f"Rozkład prawdopodobieństwa po kroku {step}, obecny wynik: {prompt}")
    plt.xlabel("Znaki")
    plt.ylabel("Prawdopodobieństwo")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis="y")