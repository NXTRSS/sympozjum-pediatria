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


def ustaw_gradio_bez_flagowania() -> None:
    """Jak ``flagging_mode=\"never\"`` w ``gr.Interface`` — bez przycisku Flag i katalogu flagged."""
    os.environ["GRADIO_FLAGGING_MODE"] = "never"


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
    *,
    zbierz_kroki: bool = False,
):
    """Generowanie jak w notatniku (hidden state + temperatura)."""
    import tensorflow as tf

    prompt = prompt.lower()
    kroki: list[dict] = []
    encoded = tf.convert_to_tensor(tokenizer.texts_to_sequences([prompt]))
    model_out, hidden_state = model(encoded, return_state=True)
    last_output = model_out[0, -1, :].numpy()

    for krok_nr in range(maks_krokow):
        last_output = np.array(last_output, dtype=np.float64)
        last_output[0] = 0.0
        s = last_output.sum()
        if s > 0:
            last_output = last_output / s
        rozklad_wiz = last_output.copy()
        log_p = np.log(last_output + 1e-9) / max(temperatura, 1e-6)
        rozklad_wybor = np.exp(log_p)
        rozklad_wybor[0] = 0.0
        rozklad_wybor = rozklad_wybor / rozklad_wybor.sum()
        next_char_idx = int(np.random.choice(np.arange(len(rozklad_wybor)), p=rozklad_wybor))
        next_char = tokenizer.index_word[next_char_idx]
        if zbierz_kroki:
            kroki.append(
                {
                    "krok": krok_nr,
                    "prompt": prompt,
                    "znak": next_char,
                    "idx": next_char_idx,
                    "rozklad": rozklad_wiz,
                }
            )
        prompt += next_char
        if next_char == "!":
            break
        encoded = tf.convert_to_tensor(tokenizer.texts_to_sequences([next_char]))
        model_out, hidden_state = model(encoded, hidden_state=hidden_state, return_state=True)
        last_output = model_out[0, -1, :].numpy()
    nazwa = prompt[1:-1]
    if zbierz_kroki:
        return nazwa, prompt, kroki
    return nazwa, prompt


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


def _zapisz_stan_demo(glebokosc: int, odrzucone_kolejka: int) -> None:
    import json

    from zapytania_log import KATALOG_LOGI

    KATALOG_LOGI.mkdir(parents=True, exist_ok=True)
    path = KATALOG_LOGI / "stan_demo.json"
    path.write_text(
        json.dumps({"glebokosc_kolejki": glebokosc, "odrzucone_kolejka": odrzucone_kolejka}),
        encoding="utf-8",
    )


def _otworz_monitor_w_terminalu() -> None:
    import platform
    import subprocess

    repo = KORZEN.resolve()
    polecenie = f"cd {repo} && uv run python notebook/monitor.py"
    if platform.system() != "Darwin":
        print(f"Monitor (ręcznie): {polecenie}", flush=True)
        return
    skrypt = f'tell application "Terminal" to do script "{polecenie}"'
    try:
        subprocess.run(["osascript", "-e", skrypt], check=True, timeout=15)
        print("Uruchomiono monitor w nowym oknie Terminala.", flush=True)
    except Exception:
        print(
            "Nie udało się otworzyć Terminala (osascript). Uruchom monitor ręcznie:\n"
            f"  {polecenie}",
            flush=True,
        )


def uruchom_aplikacje(
    model,
    tokenizer,
    znane: set,
    temperatura: float = TEMPERATURA,
    zbior: str = "choroby",
    share: bool = True,
    *,
    limit_klienta: bool = True,
    uruchom_monitor: bool = True,
):
    import threading
    import time

    ustaw_gradio_bez_flagowania()
    import gradio as gr

    from ochrona_klienta import LimitKlienta
    from zapytania_log import skrot_klienta, zapisz_zdarzenie

    maks_nazw = 6
    maks_krokow = 30
    temp_min, temp_max = 0.2, 1.5
    max_kolejki = 50
    licznik = {"ok": 0, "odrzut": 0, "odrzucone_kolejka": 0}
    limiter = LimitKlienta(limit=20) if limit_klienta else None
    w_trakcie = {"n": 0}
    blokada = threading.Lock()

    from lm_utils import figura_prawdopodobienstw
    import matplotlib.pyplot as plt

    opis = (
        "Mały model językowy (GRU) wytrenowany na żywo podczas prelekcji. "
        "Ten sam mechanizm co w ChatGPT — tylko ok. 10 milionów razy mniejszy (ok. 80 tys. parametrów)."
    )

    def _kroki_do_json(kroki: list[dict]) -> list[dict]:
        return [
            {
                "krok": int(k["krok"]),
                "prompt": k["prompt"],
                "znak": k["znak"],
                "idx": int(k["idx"]),
                "rozklad": np.asarray(k["rozklad"], dtype=np.float64).tolist(),
            }
            for k in kroki
        ]

    def _wykres_kroku(stan: list[dict], nr: float | int):
        plt.close("all")
        if not stan:
            return None
        i = int(nr)
        i = max(0, min(i, len(stan) - 1))
        k = stan[i]
        probs = np.asarray(k["rozklad"], dtype=np.float64)
        return figura_prawdopodobienstw(probs, k["krok"], tokenizer, k["prompt"])

    def _suwak_krokow(liczba_krokow: int):
        """Gradio Slider wymaga maximum > minimum (nie można 0/0)."""
        if liczba_krokow <= 0:
            return gr.update(minimum=0, maximum=1, value=0, interactive=False)
        gora = max(liczba_krokow - 1, 1)
        return gr.update(minimum=0, maximum=gora, value=0, interactive=True)

    def _pusta_odp(msg: str):
        return msg, [], _suwak_krokow(0), None

    def app_generuj(ile, temp, request: gr.Request):
        start = time.perf_counter()
        klient_obj = getattr(request, "client", None)
        ip = getattr(klient_obj, "host", None) or "brak-ip"
        sesja = getattr(request, "session_hash", None) or ip
        skrot = skrot_klienta(str(ip), str(sesja))

        with blokada:
            if w_trakcie["n"] >= max_kolejki:
                licznik["odrzucone_kolejka"] += 1
                _zapisz_stan_demo(w_trakcie["n"], licznik["odrzucone_kolejka"])
                zapisz_zdarzenie(
                    klient=skrot,
                    ile_nazw=None,
                    temperatura=None,
                    czas_odp_ms=(time.perf_counter() - start) * 1000,
                    wynik=None,
                    odrzucone=True,
                    powod="kolejka",
                )
                return _pusta_odp("Serwer jest przeciążony — spróbuj za chwilę.")
            w_trakcie["n"] += 1
            _zapisz_stan_demo(w_trakcie["n"], licznik["odrzucone_kolejka"])

        try:
            if limiter is not None and not limiter.wpusc(skrot, time.monotonic()):
                licznik["odrzut"] += 1
                msg = "Za dużo próśb (max 20/min) — odczekaj minutę."
                zapisz_zdarzenie(
                    klient=skrot,
                    ile_nazw=int(ile) if str(ile).isdigit() else None,
                    temperatura=float(temp) if str(temp).replace(".", "", 1).isdigit() else None,
                    czas_odp_ms=(time.perf_counter() - start) * 1000,
                    wynik=msg,
                    odrzucone=True,
                    powod="limit_klienta",
                )
                print(f"[demo] odrzut limit  ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
                return _pusta_odp(msg)
            try:
                ile_i = int(ile)
                temp_f = float(temp)
            except (TypeError, ValueError):
                licznik["odrzut"] += 1
                msg = "Nieprawidłowe ustawienia."
                zapisz_zdarzenie(
                    klient=skrot,
                    ile_nazw=None,
                    temperatura=None,
                    czas_odp_ms=(time.perf_counter() - start) * 1000,
                    wynik=msg,
                    odrzucone=True,
                    powod="walidacja",
                )
                return _pusta_odp(msg)
            if not 1 <= ile_i <= maks_nazw or not temp_min <= temp_f <= temp_max:
                licznik["odrzut"] += 1
                msg = f"Dozwolone: 1–{maks_nazw} nazw, temperatura {temp_min}–{temp_max}."
                zapisz_zdarzenie(
                    klient=skrot,
                    ile_nazw=ile_i,
                    temperatura=temp_f,
                    czas_odp_ms=(time.perf_counter() - start) * 1000,
                    wynik=msg,
                    odrzucone=True,
                    powod="walidacja",
                )
                print(f"[demo] odrzut  ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
                return _pusta_odp(msg)
            wyniki = []
            kroki_json: list[dict] = []
            for numer in range(ile_i):
                if numer == 0:
                    nazwa, pelny, kroki = generuj(
                        model,
                        tokenizer,
                        temperatura=temp_f,
                        maks_krokow=maks_krokow,
                        zbierz_kroki=True,
                    )
                    kroki_json = _kroki_do_json(kroki)
                    _ = pelny
                else:
                    nazwa, _ = generuj(model, tokenizer, temperatura=temp_f, maks_krokow=maks_krokow)
                znacznik = "  (istnieje w zbiorze)" if nazwa in znane else ""
                wyniki.append(f"• {nazwa}{znacznik}")
            wynik_txt = "\n".join(wyniki)
            licznik["ok"] += 1
            zapisz_zdarzenie(
                klient=skrot,
                ile_nazw=ile_i,
                temperatura=temp_f,
                czas_odp_ms=(time.perf_counter() - start) * 1000,
                wynik=wynik_txt,
                odrzucone=False,
            )
            print(f"[demo] ok={licznik['ok']} odrzut={licznik['odrzut']}", flush=True)
            fig = _wykres_kroku(kroki_json, 0) if kroki_json else None
            return wynik_txt, kroki_json, _suwak_krokow(len(kroki_json)), fig
        finally:
            with blokada:
                w_trakcie["n"] = max(0, w_trakcie["n"] - 1)
                _zapisz_stan_demo(w_trakcie["n"], licznik["odrzucone_kolejka"])

    with gr.Blocks(title=f"🩺 Generator: {zbior}") as demo:
        gr.Markdown(f"## 🩺 Generator: {zbior}\n\n{opis}")
        ile = gr.Slider(1, maks_nazw, value=3, step=1, label="Ile nazw wygenerować?")
        temp_sl = gr.Slider(
            temp_min, temp_max, value=temperatura, step=0.1, label="Kreatywność (temperatura)"
        )
        btn = gr.Button("Generuj", variant="primary")
        out = gr.Textbox(label="Nowe nazwy — właśnie wymyślone przez model", lines=10)
        stan_krokow = gr.State([])
        with gr.Accordion("Pokaż prawdopodobieństwa liter", open=False):
            gr.Markdown(
                "Histogram jak w notatniku (`visualize_probabilities`) dla **pierwszej** wygenerowanej nazwy. "
                f"Maksymalnie {maks_krokow} kroków."
            )
            krok_sl = gr.Slider(
                0, 1, step=1, value=0, label="Krok generowania", interactive=False
            )
            plot = gr.Plot(label="P(raw)")
        btn.click(
            app_generuj,
            inputs=[ile, temp_sl],
            outputs=[out, stan_krokow, krok_sl, plot],
        )
        krok_sl.change(_wykres_kroku, inputs=[stan_krokow, krok_sl], outputs=plot)
    demo.queue(default_concurrency_limit=3, max_size=max_kolejki, api_open=False)
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
    url_qr = (link or lokalny or "").strip()
    if url_qr and url_qr.startswith("http"):
        from qr_terminal import wypisz_qr_terminal

        wypisz_qr_terminal(url_qr)
    print("Ctrl+C = serwer się zatrzymuje.")
    print("=" * 64 + "\n", flush=True)
    if uruchom_monitor:
        _otworz_monitor_w_terminalu()
    demo.block_thread()