"""Auto-optymalizacja generatora nazw chorób pod pokaz NA ŻYWO.

Cel: nazwy jak najbardziej „chorobowe" (i nowe), przy jak NAJKRÓTSZYM treningu,
bo model uczy się na sali. Dlatego każdy pomiar ma czas treningu na CPU
ograniczonym do 2 wątków (≈ darmowy Colab bez GPU).

Etapy:
  przeglad  — siatka konfiguracji × punkty kontrolne epok × temperatury,
              metryki automatyczne + czas  → wyniki_auto.json
  slepa     — dla wskazanych finalistów: nazwy wymieszane z prawdziwymi
              (kontrola), zapis do pliku do ślepej oceny 1–5 → slepa_proba.tsv
              (+ klucz w slepa_klucz.json — NIE czytać przed oceną)

Uruchomienie:
  uv run python eksperymenty/autooptymalizacja.py przeglad
  uv run python eksperymenty/autooptymalizacja.py slepa
"""
import json
import os
import sys
import time

# CPU, 2 wątki — przed importem TF-owych operacji
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import strojenie as S  # noqa: E402  (dane, metryki, podział train/val)

TU = os.path.dirname(__file__)
WYNIKI = os.path.join(TU, "wyniki_auto.json")


def zbuduj(cfg, seed):
    keras.utils.set_random_seed(seed)
    warstwy = [keras.layers.Embedding(S.V, cfg["emb"], mask_zero=True)]
    for _ in range(cfg.get("warstwy", 1)):
        warstwy.append(keras.layers.GRU(cfg["units"], return_sequences=True,
                                        dropout=cfg["dropout"]))
    warstwy.append(keras.layers.Dense(S.V, activation="softmax"))
    model = keras.Sequential(warstwy)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(cfg["lr"]))
    return model


def generuj(model, ile, temperatura, maks=45, rng=None):
    """Wsadowo, wywołanie model(...) zamiast predict (szybciej)."""
    rng = rng or np.random.default_rng(0)
    ctx = np.full((ile, 1), S.znak2id["%"], dtype="int32")
    gotowe = np.zeros(ile, dtype=bool)
    wynik = [""] * ile
    for _ in range(maks):
        p = model(ctx, training=False)[:, -1, :].numpy().astype("float64")
        p = np.exp(np.log(p + 1e-9) / temperatura)
        p[:, 0] = 0
        p /= p.sum(axis=1, keepdims=True)
        c = p.cumsum(axis=1)
        wyb = (c < rng.random((ile, 1))).sum(axis=1).clip(0, S.V - 1).astype("int32")
        for i, w in enumerate(wyb):
            if gotowe[i]:
                continue
            z = S.id2znak.get(int(w), "")
            if z == "!":
                gotowe[i] = True
            else:
                wynik[i] += z
        if gotowe.all():
            break
        ctx = np.concatenate([ctx, wyb[:, None]], axis=1)
    return [w.strip() for w in wynik if w.strip()]


def metryki(nazwy):
    dokl = [n for n in nazwy if n in S.ZNANE]
    dyst = [S.najblizszy_dystans(n) for n in nazwy]
    lp = [S.logprob_ngram(n) for n in nazwy]
    return {
        "nowe_%": round(100 * (1 - len(dokl) / len(nazwy)), 1),
        "swieze_%": round(100 * sum(d > 0.2 for d in dyst) / len(nazwy), 1),
        "wiarygodnosc_%": round(100 * float(np.mean(lp)) / S.BAZA_LOGPROB, 1),
        "konc_lac_%": round(100 * sum(n.endswith(S.KONCOWKI) for n in nazwy) / len(nazwy), 1),
        "sr_dlugosc": round(float(np.mean([len(n) for n in nazwy])), 1),
        "unikalne_%": round(100 * len(set(nazwy)) / len(nazwy), 1),
        "przyklady": nazwy[:8],
    }


def trenuj_z_pomiarem(cfg, punkty, temperatury, seed=7, ile=200):
    model = zbuduj(cfg, seed)
    out, czas, poprz = [], 0.0, 0
    for punkt in punkty:
        t0 = time.perf_counter()
        h = model.fit(S.Xtr, S.Ytr, batch_size=cfg["batch"], epochs=punkt - poprz,
                      verbose=0, shuffle=True)
        czas += time.perf_counter() - t0
        poprz = punkt
        val = float(model.evaluate(S.Xva, S.Yva, batch_size=256, verbose=0))
        for t in temperatury:
            m = metryki(generuj(model, ile, t, rng=np.random.default_rng(seed)))
            m.update({**cfg, "epoki": punkt, "temperatura": t, "czas_s": round(czas, 1),
                      "loss": round(float(h.history["loss"][-1]), 3),
                      "val_loss": round(val, 3), "seed": seed})
            out.append(m)
            print(f"{nazwa_cfg(cfg):<28} ep{punkt:>3} T{t} | {czas:6.1f}s | val {val:.3f} | "
                  f"nowe {m['nowe_%']:5}% świeże {m['swieze_%']:5}% wiar {m['wiarygodnosc_%']:6}% "
                  f"końc {m['konc_lac_%']:5}% | {', '.join(m['przyklady'][:3])}", flush=True)
    return out, model


def nazwa_cfg(c):
    return (f"u{c['units']}x{c.get('warstwy', 1)} e{c['emb']} d{c['dropout']} "
            f"lr{c['lr']} b{c['batch']}")


# ── siatka: nacisk na SZYBKĄ zbieżność ──────────────────────────────────────
KONFIGI = [
    # punkt odniesienia (obecny notebook: Adam 1e-3, batch 32)
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 1e-3, "batch": 32},
    # wyższy learning rate
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 32},
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 1e-2, "batch": 32},
    # większy batch (szybsze epoki) + wyższy lr
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 64},
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 1e-2, "batch": 128},
    # mniejszy model
    {"units": 64, "emb": 32, "dropout": 0.2, "lr": 1e-2, "batch": 32},
    {"units": 96, "emb": 48, "dropout": 0.2, "lr": 5e-3, "batch": 32},
    # większy model, krótko
    {"units": 256, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 64},
    {"units": 192, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 32},
    # dwie warstwy
    {"units": 128, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 32, "warstwy": 2},
    # bez dropoutu, szybko
    {"units": 128, "emb": 64, "dropout": 0.0, "lr": 3e-3, "batch": 32},
]
PUNKTY = [3, 5, 8, 12, 20, 30]
TEMPERATURY = [0.5, 0.65, 0.8]


def przeglad():
    wyniki = []
    for cfg in KONFIGI:
        w, _ = trenuj_z_pomiarem(cfg, PUNKTY, TEMPERATURY)
        wyniki += w
        with open(WYNIKI, "w") as f:  # zapis po każdej konfiguracji
            json.dump({"baza_logprob": S.BAZA_LOGPROB, "n_nazw": len(S.nazwy),
                       "wyniki": wyniki}, f, ensure_ascii=False, indent=1)
    print(f"\nZapisano {len(wyniki)} pomiarów → {WYNIKI}")


# ── ślepa ocena ─────────────────────────────────────────────────────────────
def slepa(finalisci, ile_na_kandydata=25, ile_kontroli=25, seed=11):
    """finalisci: lista (cfg, epoki, temperatura). Zapisuje przemieszaną próbę."""
    rng = np.random.default_rng(seed)
    pula = []
    for k, (cfg, ep, t) in enumerate(finalisci):
        model = zbuduj(cfg, 7)
        model.fit(S.Xtr, S.Ytr, batch_size=cfg["batch"], epochs=ep, verbose=0)
        nazwy = generuj(model, ile_na_kandydata * 2, t, rng=np.random.default_rng(100 + k))
        nazwy = [n for n in dict.fromkeys(nazwy)][:ile_na_kandydata]
        pula += [(n, f"K{k}") for n in nazwy]
        print(f"K{k}: {nazwa_cfg(cfg)} ep{ep} T{t} → {len(nazwy)} nazw", flush=True)
    kontrola = list(rng.choice(S.val_nazwy, size=ile_kontroli, replace=False))
    pula += [(n, "PRAWDZIWA") for n in kontrola]
    kolejnosc = rng.permutation(len(pula))
    with open(os.path.join(TU, "slepa_proba.tsv"), "w") as f:
        for i, j in enumerate(kolejnosc):
            f.write(f"{i}\t{pula[j][0]}\n")
    with open(os.path.join(TU, "slepa_klucz.json"), "w") as f:
        json.dump({"finalisci": [{"id": f"K{k}", **c, "epoki": e, "temperatura": t}
                                 for k, (c, e, t) in enumerate(finalisci)],
                   "klucz": {str(i): pula[j][1] for i, j in enumerate(kolejnosc)}},
                  f, ensure_ascii=False, indent=1)
    print(f"Ślepa próba: {len(pula)} nazw → slepa_proba.tsv")


# ── faza 2: zawężenie wokół frontu Pareto, 3 ziarna ─────────────────────────
KONFIGI_2 = [
    ({"units": 128, "emb": 64, "dropout": 0.0, "lr": 3e-3, "batch": 32}, [8, 10, 12, 16]),
    ({"units": 128, "emb": 64, "dropout": 0.2, "lr": 3e-3, "batch": 32}, [8, 10, 12, 16]),
    ({"units": 128, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 32}, [8, 10, 12, 16]),
    ({"units": 128, "emb": 64, "dropout": 0.3, "lr": 3e-3, "batch": 64}, [8, 12, 16, 20]),
    ({"units": 96, "emb": 48, "dropout": 0.2, "lr": 5e-3, "batch": 32}, [6, 8, 10, 12]),
    ({"units": 128, "emb": 64, "dropout": 0.2, "lr": 5e-3, "batch": 64}, [6, 8, 10, 12]),
    # punkt odniesienia = obecny notebook
    ({"units": 128, "emb": 64, "dropout": 0.3, "lr": 1e-3, "batch": 32}, [30]),
]
TEMPERATURY_2 = [0.4, 0.5, 0.6]


def faza2():
    plik = os.path.join(TU, "wyniki_auto_faza2.json")
    wyniki = []
    for seed in (7, 21, 99):
        for cfg, punkty in KONFIGI_2:
            w, _ = trenuj_z_pomiarem(cfg, punkty, TEMPERATURY_2, seed=seed)
            wyniki += w
            with open(plik, "w") as f:
                json.dump({"baza_logprob": S.BAZA_LOGPROB, "n_nazw": len(S.nazwy),
                           "wyniki": wyniki}, f, ensure_ascii=False, indent=1)
    print(f"\nZapisano {len(wyniki)} pomiarów → {plik}")


if __name__ == "__main__":
    etap = sys.argv[1] if len(sys.argv) > 1 else "przeglad"
    if etap == "przeglad":
        przeglad()
    elif etap == "faza2":
        faza2()
    elif etap == "slepa":
        with open(os.path.join(TU, "finalisci.json")) as f:
            fin = [(d["cfg"], d["epoki"], d["temperatura"]) for d in json.load(f)]
        slepa(fin)
