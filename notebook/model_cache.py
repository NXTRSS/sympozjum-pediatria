"""Lokalny cache wytrenowanego modelu (nie commitujemy — patrz .gitignore)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

WERSJA_CACHE = 1
KATALOG_CACHE = Path(__file__).resolve().parent / ".model_cache"
PLIK_MANIFEST = "manifest.json"
PLIK_WAG = "model.weights.h5"
PLIK_TOKENIZER = "tokenizer.json"
PLIK_ZNANE = "znane.json"


def _sha256_pliku(sciezka: Path) -> str:
    h = hashlib.sha256()
    with sciezka.open("rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def hash_kodu_demo() -> str:
    sciezka = Path(__file__).resolve().parent / "demo_model.py"
    return _sha256_pliku(sciezka)


def hash_danych(sciezka_danych: str | Path) -> str:
    p = Path(sciezka_danych)
    if p.is_file():
        return _sha256_pliku(p)
    return hashlib.sha256(sciezka_danych.encode()).hexdigest()


def sciezka_cache() -> Path:
    return KATALOG_CACHE


def _manifest_aktualny(oczekiwany: dict[str, Any]) -> bool:
    manifest_path = KATALOG_CACHE / PLIK_MANIFEST
    if not manifest_path.is_file():
        return False
    required = (
        PLIK_WAG,
        PLIK_TOKENIZER,
        PLIK_ZNANE,
    )
    for nazwa in required:
        if not (KATALOG_CACHE / nazwa).is_file():
            return False
    zapisany = json.loads(manifest_path.read_text(encoding="utf-8"))
    if zapisany.get("wersja_cache") != WERSJA_CACHE:
        return False
    klucze = (
        "ziarno",
        "liczba_epok",
        "learning_rate",
        "temperatura",
        "embedding_dim",
        "gru_units",
        "dropout",
        "batch_size",
        "hash_danych",
        "hash_kodu",
    )
    return all(zapisany.get(k) == oczekiwany.get(k) for k in klucze)


def zapisz(
    oczekiwany: dict[str, Any],
    model,
    tokenizer,
    znane: set[str],
    wynik: dict[str, Any],
) -> None:
    KATALOG_CACHE.mkdir(parents=True, exist_ok=True)
    model.save_weights(KATALOG_CACHE / PLIK_WAG)
    tok = {
        "word_index": tokenizer.word_index,
        "index_word": {str(k): v for k, v in tokenizer.index_word.items()},
    }
    (KATALOG_CACHE / PLIK_TOKENIZER).write_text(json.dumps(tok), encoding="utf-8")
    (KATALOG_CACHE / PLIK_ZNANE).write_text(
        json.dumps(sorted(znane), ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "wersja_cache": WERSJA_CACHE,
        **oczekiwany,
        "loss": wynik["loss"],
        "val_loss": wynik.get("val_loss", []),
        "pierwsza": wynik["pierwsza"],
        "dziesiec": wynik["dziesiec"],
        "zrodlo": wynik["zrodlo"],
        "nazw": wynik["nazw"],
    }
    (KATALOG_CACHE / PLIK_MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def wczytaj(
    oczekiwany: dict[str, Any],
    buduj_model,
    przywroc_tokenizer,
) -> dict[str, Any] | None:
    if not _manifest_aktualny(oczekiwany):
        return None
    manifest = json.loads((KATALOG_CACHE / PLIK_MANIFEST).read_text(encoding="utf-8"))
    tokenizer = przywroc_tokenizer(
        json.loads((KATALOG_CACHE / PLIK_TOKENIZER).read_text(encoding="utf-8"))
    )
    import tensorflow as tf

    vocab_size = len(tokenizer.word_index) + 1
    model = buduj_model(vocab_size)
    model(tf.constant([[1]], dtype=tf.int32))
    model.load_weights(KATALOG_CACHE / PLIK_WAG)
    znane = set(json.loads((KATALOG_CACHE / PLIK_ZNANE).read_text(encoding="utf-8")))
    wagi = [w.numpy().copy() for w in model.weights]
    return {
        "zrodlo": manifest["zrodlo"],
        "nazw": manifest["nazw"],
        "model": model,
        "tokenizer": tokenizer,
        "znane": znane,
        "loss": manifest["loss"],
        "val_loss": manifest.get("val_loss", []),
        "wagi": wagi,
        "pierwsza": manifest["pierwsza"],
        "dziesiec": manifest["dziesiec"],
        "z_cache": True,
    }


def wyczysc() -> None:
    if KATALOG_CACHE.is_dir():
        for p in KATALOG_CACHE.iterdir():
            p.unlink()