"""Logika mini-modelu dla skryptu uruchom_demo.py i testów deterministyczności.

Notatnik model_jezyka_choroby.ipynb ma ten sam przebieg napisany krok po kroku w komórkach
(bez importu z tego modułu). Skrypt sali woła funkcje stąd — z walidacją 10% w trenuj()."""
from __future__ import annotations

import os
import random

# Zanim ktokolwiek zaimportuje TensorFlow: CPU i operacje deterministyczne.
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pandas as pd

REPO = "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/data"
DANE = {
    "choroby": f"{REPO}/choroby.csv",
    "leki": f"{REPO}/leki.csv",
    "dinozaury": "https://raw.githubusercontent.com/junosuarez/dinosaurs/master/dinosaurs.csv",
}
LICZBA_EPOK = 12
TEMPO_UCZENIA = 0.003
TEMPERATURA = 0.4
ZIARNO = 42
KORZEN = Path(__file__).resolve().parents[1]


def ustaw_ziarna(ziarno: int = ZIARNO) -> None:
    """Python, NumPy, TensorFlow i — gdy jest zainstalowany — PyTorch. Tylko CPU."""
    random.seed(ziarno)
    np.random.seed(ziarno)
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    tf.random.set_seed(ziarno)
    tf.keras.utils.set_random_seed(ziarno)
    tf.config.experimental.enable_op_determinism()
    try:
        import torch

        torch.manual_seed(ziarno)
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Ten model jest na TensorFlow. Torch ziarno ustawiamy tylko, gdy pakiet jest.
        pass


def wczytaj_nazwy(zbior: str = "choroby", ziarno: int = ZIARNO, dane: dict | None = None):
    """Nazwy z lokalnego pliku (albo z adresu), bez duplikatów, przetasowane ziarnem."""
    dane = DANE if dane is None else dane
    zrodlo = dane[zbior]
    plik = os.path.basename(zrodlo)
    for kandydat in (
        Path(plik),
        Path("data") / plik,
        Path("..") / "data" / plik,
        KORZEN / "data" / plik,
    ):
        if kandydat.exists():
            zrodlo = str(kandydat)
            break
    df = pd.read_csv(zrodlo, header=None, names=["nazwa"], sep="\t")
    nazwy = df["nazwa"].dropna().astype(str).str.strip().str.lower().drop_duplicates().tolist()
    rng = np.random.default_rng(ziarno)
    nazwy = [nazwy[i] for i in rng.permutation(len(nazwy))]
    return nazwy, zrodlo


def przygotuj_dane(nazwy: list[str]):
    """% na początku, ! na końcu. X przewiduje Y przesunięte o jeden znak."""
    sekwencje = ["%" + n + "!" for n in nazwy]
    znaki = sorted(set("".join(sekwencje)))
    znak2id = {z: i + 1 for i, z in enumerate(znaki)}
    id2znak = {i: z for z, i in znak2id.items()}
    maks_dl = max(len(s) for s in sekwencje)
    X = np.zeros((len(sekwencje), maks_dl - 1), dtype="int32")
    Y = np.zeros((len(sekwencje), maks_dl - 1), dtype="int32")
    for i, s in enumerate(sekwencje):
        ids = [znak2id[z] for z in s]
        X[i, : len(ids) - 1] = ids[:-1]
        Y[i, : len(ids) - 1] = ids[1:]
    return X, Y, znak2id, id2znak, znaki


def zbuduj_model(wielkosc_slownika: int, tempo: float = TEMPO_UCZENIA, ziarno: int = ZIARNO):
    """Embedding(64) → GRU(128, dropout 0.2) → softmax. Ziarna inicjalizacji są jawne."""
    import tensorflow as tf
    from tensorflow import keras

    ustaw_ziarna(ziarno)
    model = keras.Sequential([
        keras.layers.Embedding(
            wielkosc_slownika, 64, mask_zero=True,
            embeddings_initializer=keras.initializers.GlorotUniform(seed=ziarno),
        ),
        keras.layers.GRU(
            128, return_sequences=True, dropout=0.2, seed=ziarno,
            kernel_initializer=keras.initializers.GlorotUniform(seed=ziarno),
            recurrent_initializer=keras.initializers.Orthogonal(seed=ziarno + 1),
            bias_initializer=keras.initializers.Zeros(),
        ),
        keras.layers.Dense(
            wielkosc_slownika, activation="softmax",
            kernel_initializer=keras.initializers.GlorotUniform(seed=ziarno + 2),
            bias_initializer=keras.initializers.Zeros(),
        ),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=tempo),
    )
    # tf jest użyty pośrednio przez keras; trzymamy import, żeby ziarno TF było ustawione.
    del tf
    return model


def trenuj(model, X, Y, epoki: int = LICZBA_EPOK):
    """Ostatnie 10% wierszy (po tasowaniu) to sprawdzian. Kolejność epok jest z ziarna TF."""
    return model.fit(X, Y, epochs=epoki, validation_split=0.1, shuffle=True, verbose=2)


def generuj(model, znak2id, id2znak, temperatura: float = TEMPERATURA, maks_znakow: int = 40):
    """Jedna nazwa. Losowanie idzie globalnym NumPy — po ustaw_ziarna jest powtarzalne."""
    kontekst, kroki = "%", []
    for _ in range(maks_znakow):
        ids = np.array([[znak2id[z] for z in kontekst]])
        rozklad = model(ids, training=False).numpy()[0, -1].astype("float64")
        rozklad = np.log(rozklad + 1e-9) / temperatura
        rozklad = np.exp(rozklad) / np.exp(rozklad).sum()
        rozklad[0] = 0
        rozklad /= rozklad.sum()
        wybor = int(np.random.choice(len(rozklad), p=rozklad))
        kroki.append({"kontekst": kontekst, "rozklad": rozklad, "wybor": wybor})
        if id2znak[wybor] == "!":
            break
        kontekst += id2znak[wybor]
    return kontekst[1:], kroki


def przebieg(ziarno: int = ZIARNO, epoki: int = LICZBA_EPOK, temperatura: float = TEMPERATURA):
    """Ta sama kolejność co notebook: dane → trening → jedna nazwa → dziesięć nazw."""
    nazwy, zrodlo = wczytaj_nazwy(ziarno=ziarno)
    X, Y, znak2id, id2znak, znaki = przygotuj_dane(nazwy)
    model = zbuduj_model(len(znaki) + 1, tempo=TEMPO_UCZENIA, ziarno=ziarno)
    historia = trenuj(model, X, Y, epoki=epoki)
    pierwsza, _ = generuj(model, znak2id, id2znak, temperatura=temperatura)
    dziesiec = [generuj(model, znak2id, id2znak, temperatura=temperatura)[0] for _ in range(10)]
    return {
        "zrodlo": zrodlo,
        "nazw": len(nazwy),
        "model": model,
        "znak2id": znak2id,
        "id2znak": id2znak,
        "znane": set(nazwy),
        "loss": [float(x) for x in historia.history["loss"]],
        "val_loss": [float(x) for x in historia.history["val_loss"]],
        "wagi": [w.numpy().copy() for w in model.weights],
        "pierwsza": pierwsza,
        "dziesiec": dziesiec,
    }


def uruchom_aplikacje(model, znak2id, id2znak, znane: set, temperatura: float = TEMPERATURA,
                      zbior: str = "choroby", share: bool = True):
    """Aplikacja Gradio z limitem kolejki. Publiczny link nie otwiera dysku Maca."""
    import time
    from collections import defaultdict, deque

    import gradio as gr

    maks_nazw = 6
    maks_znakow = 32
    temp_min, temp_max = 0.2, 1.5
    okno_s = 60.0
    limit_sesji = 8
    limit_ip = 20
    licznik = {"ok": 0, "odrzut": 0}
    okna: dict[str, deque] = defaultdict(deque)

    def wpusc(klucz: str, limit: int) -> bool:
        teraz = time.monotonic()
        kolejka = okna[klucz]
        while kolejka and teraz - kolejka[0] > okno_s:
            kolejka.popleft()
        if len(kolejka) >= limit:
            return False
        kolejka.append(teraz)
        return True

    def app_generuj(ile, temp, request: gr.Request):
        klient = getattr(request, "client", None)
        ip = getattr(klient, "host", None) or "brak-ip"
        sesja = getattr(request, "session_hash", None) or ip
        if not wpusc("s:" + str(sesja), limit_sesji) or not wpusc("i:" + str(ip), limit_ip):
            licznik["odrzut"] += 1
            print(f"[demo] odrzut  ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
            return "Za dużo próśb z tej przeglądarki. Odczekaj minutę i spróbuj jeszcze raz."
        try:
            ile_i = int(ile)
            temp_f = float(temp)
        except (TypeError, ValueError):
            licznik["odrzut"] += 1
            return "Nieprawidłowe ustawienia."
        if not 1 <= ile_i <= maks_nazw or not temp_min <= temp_f <= temp_max:
            licznik["odrzut"] += 1
            print(f"[demo] odrzut  ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
            return f"Dozwolone: 1–{maks_nazw} nazw, temperatura {temp_min}–{temp_max}."
        wyniki = []
        for _ in range(ile_i):
            nazwa, _ = generuj(model, znak2id, id2znak, temperatura=temp_f, maks_znakow=maks_znakow)
            znacznik = "  (istnieje w zbiorze)" if nazwa in znane else ""
            wyniki.append(f"• {nazwa}{znacznik}")
        licznik["ok"] += 1
        print(f"[demo] ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
        return "\n".join(wyniki)

    demo = gr.Interface(
        fn=app_generuj,
        inputs=[
            gr.Slider(1, maks_nazw, value=3, step=1, label="Ile nazw wygenerować?"),
            gr.Slider(temp_min, temp_max, value=temperatura, step=0.1, label="Kreatywność (temperatura)"),
        ],
        outputs=gr.Textbox(label="Nowe nazwy — właśnie wymyślone przez model", lines=10),
        title=f"🩺 Generator: {zbior}",
        description="Mały model językowy (GRU) wytrenowany na żywo podczas prelekcji. "
                    "Ten sam mechanizm co w ChatGPT — tylko ok. 10 milionów razy mniejszy (ok. 80 tys. parametrów).",
    )
    demo.queue(default_concurrency_limit=3, max_size=50, api_open=False)
    print("\nUruchamiam aplikację. Publiczny link gradio.live pojawi się poniżej.", flush=True)
    print("Ctrl+C = link natychmiast przestaje działać.", flush=True)
    print("Link otwiera tylko tę aplikację, nie dysk ani resztę Maca.\n", flush=True)
    demo.launch(
        share=share,
        prevent_thread_lock=True,
        footer_links=[],
        max_file_size="1kb",
        app_kwargs={"docs_url": None, "redoc_url": None, "openapi_url": None},
    )
    link = getattr(demo, "share_url", None) or ""
    print("\n" + "=" * 64)
    print("LINK DLA SALI:", link or "(brak — uruchomiono bez share)")
    print("Ctrl+C = link natychmiast przestaje działać.")
    print("=" * 64 + "\n", flush=True)
    demo.block_thread()
