"""Logika mini-modelu dla skryptu uruchom_demo.py i testów (ten sam przebieg co notatnik sali)."""
from __future__ import annotations

import os
import random

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import numpy as np
import pandas as pd

from model_cache import hash_danych, hash_kodu_demo, wczytaj as wczytaj_cache, zapisz as zapisz_cache

REPO = "https://raw.githubusercontent.com/NXTRSS/sympozjum-pediatria/main/data"
DANE = {
    "choroby": f"{REPO}/choroby.csv",
    "leki": f"{REPO}/leki.csv",
    "dinozaury": "https://raw.githubusercontent.com/junosuarez/dinosaurs/master/dinosaurs.csv",
}
LICZBA_EPOK = 12
TEMPO_UCZENIA = 0.003
TEMPERATURA = 0.4
EMBEDDING_DIM = 64
GRU_UNITS = 128
DROPOUT = 0.2
BATCH_SIZE = 64
ZIARNO = 42
KORZEN = Path(__file__).resolve().parents[1]


def ustaw_ziarna(ziarno: int = ZIARNO) -> None:
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


def wczytaj_nazwy(zbior: str = "choroby", ziarno: int = ZIARNO, dane: dict | None = None):
    dane = DANE if dane is None else dane
    zrodlo = dane[zbior]
    plik = os.path.basename(zrodlo)
    sciezka_lokalna: Path | None = None
    for kandydat in (
        Path(plik),
        Path("data") / plik,
        Path("..") / "data" / plik,
        KORZEN / "data" / plik,
    ):
        if kandydat.exists():
            zrodlo = str(kandydat)
            sciezka_lokalna = kandydat.resolve()
            break
    df = pd.read_csv(zrodlo, header=None, names=["nazwa"], sep="\t")
    nazwy = df["nazwa"].dropna().astype(str).str.strip().str.lower().drop_duplicates().tolist()
    rng = np.random.default_rng(ziarno)
    nazwy = [nazwy[i] for i in rng.permutation(len(nazwy))]
    return nazwy, zrodlo, sciezka_lokalna


def _zbuduj_klase_language_model():
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Embedding, GRU

    class LanguageModel(Model):
        def __init__(self, vocab_size: int):
            super().__init__()
            self.embedding = Embedding(vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True)
            self.rnn = GRU(GRU_UNITS, return_sequences=True, return_state=True, dropout=DROPOUT)
            self.fc = Dense(vocab_size, activation="softmax")

        def call(self, inputs, training=False, hidden_state=None, return_state=False):
            x = self.embedding(inputs, training=training)
            seq_out, last_hidden_state = self.rnn(x, training=training, initial_state=hidden_state)
            x = self.fc(seq_out)
            if return_state:
                return x, last_hidden_state
            return x

    return LanguageModel


def zbuduj_model(vocab_size: int, ziarno: int = ZIARNO):
    import tensorflow as tf

    ustaw_ziarna(ziarno)
    LanguageModel = _zbuduj_klase_language_model()
    return LanguageModel(vocab_size)


def przygotuj_tokenizer_i_macierze(nazwy: list[str]):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    name_list = ["%" + n + "!" for n in nazwy]
    tokenizer = Tokenizer(filters="", lower=True, char_level=True)
    tokenizer.fit_on_texts(name_list)
    x_teksty = [n[:-1] for n in name_list]
    y_teksty = [n[1:] for n in name_list]
    X = pad_sequences(tokenizer.texts_to_sequences(x_teksty), padding="post")
    Y = pad_sequences(tokenizer.texts_to_sequences(y_teksty), padding="post")
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, X, Y, vocab_size, name_list


def przywroc_tokenizer(dane: dict):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(filters="", lower=True, char_level=True)
    tokenizer.word_index = {str(k) if not isinstance(k, str) else k: v for k, v in dane["word_index"].items()}
    tokenizer.index_word = {int(k): v for k, v in dane["index_word"].items()}
    return tokenizer


def trenuj(model, X, Y, epoki: int = LICZBA_EPOK):
    import tensorflow.keras as keras

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=TEMPO_UCZENIA),
    )
    return model.fit(X, Y, epochs=epoki, batch_size=BATCH_SIZE, verbose=2)


def generuj(
    model,
    tokenizer,
    temperatura: float = TEMPERATURA,
    maks_krokow: int = 30,
    prompt: str = "%",
):
    """Generowanie jak w notatniku (hidden state + temperatura)."""
    import tensorflow as tf

    prompt = prompt.lower()
    encoded = tf.convert_to_tensor(tokenizer.texts_to_sequences([prompt]))
    model_out, hidden_state = model(encoded, return_state=True)
    last_output = model_out[0, -1, :].numpy()

    for _ in range(maks_krokow):
        last_output = np.array(last_output, dtype=np.float64)
        last_output[0] = 0.0
        s = last_output.sum()
        if s > 0:
            last_output /= s
        log_p = np.log(last_output + 1e-9) / max(temperatura, 1e-6)
        last_output = np.exp(log_p)
        last_output[0] = 0.0
        last_output /= last_output.sum()
        next_char_idx = int(np.random.choice(np.arange(len(last_output)), p=last_output))
        next_char = tokenizer.index_word[next_char_idx]
        prompt += next_char
        if next_char == "!":
            break
        encoded = tf.convert_to_tensor(tokenizer.texts_to_sequences([next_char]))
        model_out, hidden_state = model(encoded, hidden_state=hidden_state, return_state=True)
        last_output = model_out[0, -1, :].numpy()
    return prompt[1:-1], prompt


def _manifest(ziarno: int, epoki: int, temperatura: float, zrodlo: str) -> dict:
    return {
        "ziarno": ziarno,
        "liczba_epok": epoki,
        "learning_rate": TEMPO_UCZENIA,
        "temperatura": temperatura,
        "embedding_dim": EMBEDDING_DIM,
        "gru_units": GRU_UNITS,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "hash_danych": hash_danych(zrodlo),
        "hash_kodu": hash_kodu_demo(),
    }


def _wynik_z_modelu(
    model,
    tokenizer,
    nazwy: list[str],
    zrodlo: str,
    historia,
    temperatura: float,
    ziarno: int,
    *,
    z_cache: bool = False,
) -> dict:
    ustaw_ziarna(ziarno)
    pierwsza, _ = generuj(model, tokenizer, temperatura=temperatura)
    ustaw_ziarna(ziarno)
    dziesiec = [
        generuj(model, tokenizer, temperatura=temperatura)[0]
        for _ in range(10)
    ]
    return {
        "zrodlo": zrodlo,
        "nazw": len(nazwy),
        "model": model,
        "tokenizer": tokenizer,
        "znane": set(nazwy),
        "loss": [float(x) for x in historia.history["loss"]],
        "val_loss": [],
        "wagi": [w.numpy().copy() for w in model.weights],
        "pierwsza": pierwsza,
        "dziesiec": dziesiec,
        "z_cache": z_cache,
    }


def _przebieg_trenuj(
    ziarno: int,
    epoki: int,
    temperatura: float,
) -> dict:
    nazwy, zrodlo, _ = wczytaj_nazwy(ziarno=ziarno)
    ustaw_ziarna(ziarno)
    tokenizer, X, Y, vocab_size, _ = przygotuj_tokenizer_i_macierze(nazwy)
    model = zbuduj_model(vocab_size, ziarno=ziarno)
    model(X[:1])
    historia = trenuj(model, X, Y, epoki=epoki)
    wynik = _wynik_z_modelu(model, tokenizer, nazwy, zrodlo, historia, temperatura, ziarno)
    manifest = _manifest(ziarno, epoki, temperatura, zrodlo)
    zapisz_cache(manifest, model, tokenizer, wynik["znane"], wynik)
    return wynik


def przebieg(
    ziarno: int | None = None,
    epoki: int | None = None,
    temperatura: float | None = None,
    *,
    force_train: bool = False,
    use_cache: bool = True,
) -> dict:
    ziarno = ZIARNO if ziarno is None else ziarno
    epoki = LICZBA_EPOK if epoki is None else epoki
    temperatura = TEMPERATURA if temperatura is None else temperatura
    _, zrodlo, _ = wczytaj_nazwy(ziarno=ziarno)
    manifest = _manifest(ziarno, epoki, temperatura, zrodlo)
    if use_cache and not force_train:
        z_cache = wczytaj_cache(manifest, zbuduj_model, przywroc_tokenizer)
        if z_cache is not None:
            print("Wczytano zapisany model (bez treningu)", flush=True)
            return z_cache
    print("Trenuję…", flush=True)
    return _przebieg_trenuj(ziarno, epoki, temperatura)


def uruchom_aplikacje(
    model,
    tokenizer,
    znane: set,
    temperatura: float = TEMPERATURA,
    zbior: str = "choroby",
    share: bool = True,
):
    import time
    from collections import defaultdict, deque

    import gradio as gr

    maks_nazw = 6
    maks_krokow = 30
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
            nazwa, _ = generuj(model, tokenizer, temperatura=temp_f, maks_krokow=maks_krokow)
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
    if share:
        print("\nUruchamiam aplikację. Publiczny link gradio.live pojawi się poniżej.", flush=True)
    else:
        print("\nUruchamiam aplikację lokalnie (--bez-linku).", flush=True)
    print("Ctrl+C = serwer się zatrzymuje.\n", flush=True)
    wynik_launch = demo.launch(
        share=share,
        prevent_thread_lock=True,
        footer_links=[],
        max_file_size="1kb",
        app_kwargs={"docs_url": None, "redoc_url": None, "openapi_url": None},
    )
    lokalny = ""
    if isinstance(wynik_launch, tuple) and len(wynik_launch) >= 2:
        lokalny = wynik_launch[1] or ""
    if not lokalny:
        lokalny = getattr(demo, "local_url", None) or getattr(demo, "local_url", "") or ""
    link = getattr(demo, "share_url", None) or ""
    if isinstance(wynik_launch, tuple) and len(wynik_launch) >= 3:
        link = wynik_launch[2] or link
    print("\n" + "=" * 64)
    if share and not link:
        print(
            "Nie udało się utworzyć publicznego linku gradio.live "
            "(tunel zablokowany, np. przez antywirusa firmowego). "
            f"Aplikacja działa lokalnie: {lokalny or '(adres w oknie Gradio)'}. "
            "Sala: użyj QR do Colaba."
        )
    elif share and link:
        print("LINK DLA SALI:", link)
    else:
        print(f"Aplikacja lokalna: {lokalny or '(adres w oknie Gradio)'}. Sala: QR do Colaba / ekran.")
    print("Ctrl+C = serwer się zatrzymuje.")
    print("=" * 64 + "\n", flush=True)
    demo.block_thread()