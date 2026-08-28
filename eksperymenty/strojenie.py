"""Strojenie hiperparametrów: model ma WYMYŚLAĆ nazwy chorób, a nie odtwarzać zbiór.

Mierzymy jednocześnie dwie rzeczy, które ciągną w przeciwne strony:
  • NOWOŚĆ    — ile nazw nie występuje w zbiorze treningowym (i nie jest jego prawie-kopią)
  • WIARYGODNOŚĆ — czy nazwa nadal brzmi po łacinie jak jednostka chorobowa

Uruchomienie:  uv run python eksperymenty/strojenie.py
"""
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

SEED = 7
DANE = os.path.join(os.path.dirname(__file__), "..", "data", "choroby.csv")
WYNIKI = os.path.join(os.path.dirname(__file__), "wyniki.json")

# ── dane ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DANE, header=None, names=["nazwa"], sep="\t")
nazwy = df["nazwa"].dropna().astype(str).str.strip().str.lower().drop_duplicates().tolist()

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(nazwy))
n_val = int(0.1 * len(nazwy))
val_nazwy = [nazwy[i] for i in perm[:n_val]]
train_nazwy = [nazwy[i] for i in perm[n_val:]]

znaki = sorted(set("".join("%" + n + "!" for n in nazwy)))
znak2id = {z: i + 1 for i, z in enumerate(znaki)}
id2znak = {i: z for z, i in znak2id.items()}
V = len(znaki) + 1
MAKS = max(len(n) for n in nazwy) + 2

TRAIN_SET = set(train_nazwy)
ZNANE = set(nazwy)


def macierze(lista):
    X = np.zeros((len(lista), MAKS - 1), dtype="int32")
    Y = np.zeros((len(lista), MAKS - 1), dtype="int32")
    for i, n in enumerate(lista):
        ids = [znak2id[z] for z in "%" + n + "!"]
        X[i, : len(ids) - 1] = ids[:-1]
        Y[i, : len(ids) - 1] = ids[1:]
    return X, Y


Xtr, Ytr = macierze(train_nazwy)
Xva, Yva = macierze(val_nazwy)

# ── metryki wiarygodności ─────────────────────────────────────────────────
# Model 4-gramowy znaków zbudowany na CAŁYM korpusie: mierzy, czy nazwa jest
# zbudowana z prawdopodobnych łacińskich zbitek. Kopiowanie łapiemy osobno.
N = 4
gram_licznik, kontekst_licznik = Counter(), Counter()
for n in nazwy:
    s = "^" * (N - 1) + n + "$"
    for i in range(N - 1, len(s)):
        gram_licznik[s[i - N + 1 : i + 1]] += 1
        kontekst_licznik[s[i - N + 1 : i]] += 1
ALFABET = len(set("".join(nazwy))) + 1


def logprob_ngram(nazwa, k=0.1):
    s = "^" * (N - 1) + nazwa + "$"
    lp = 0.0
    for i in range(N - 1, len(s)):
        g, c = s[i - N + 1 : i + 1], s[i - N + 1 : i]
        lp += np.log((gram_licznik[g] + k) / (kontekst_licznik[c] + k * ALFABET))
    return lp / max(1, len(s) - N + 1)


# Punkt odniesienia: jak wypadają PRAWDZIWE nazwy ze zbioru walidacyjnego
BAZA_LOGPROB = float(np.mean([logprob_ngram(n) for n in val_nazwy]))

KONCOWKI = ("itis", "osis", "oma", "ia", "us", "um", "ae", "is", "as", "icus",
            "alis", "aris", "osa", "iae", "ica", "ans", "ens")


def lewenshtein(a, b, limit=99):
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    poprzedni = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        biezacy = [i]
        for j, cb in enumerate(b, 1):
            biezacy.append(min(poprzedni[j] + 1, biezacy[j - 1] + 1,
                               poprzedni[j - 1] + (ca != cb)))
        poprzedni = biezacy
    return poprzedni[-1]


def najblizszy_dystans(nazwa):
    """Znormalizowana odległość edycyjna do najbliższej nazwy treningowej."""
    najlepszy = 99
    for t in train_nazwy:
        if abs(len(t) - len(nazwa)) > 3 or t[0] != nazwa[0]:
            continue
        d = lewenshtein(nazwa, t)
        if d < najlepszy:
            najlepszy = d
            if d == 0:
                break
    return najlepszy / max(len(nazwa), 1)


# ── generowanie wsadowe (szybkie) ─────────────────────────────────────────
def generuj_wsad(model, ile=120, temperatura=1.0, maks=45):
    ctx = np.full((ile, 1), znak2id["%"], dtype="int32")
    gotowe = np.zeros(ile, dtype=bool)
    wynik = [""] * ile
    for _ in range(maks):
        p = model.predict(ctx, verbose=0)[:, -1, :].astype("float64")
        p = np.log(p + 1e-9) / temperatura
        p = np.exp(p)
        p[:, 0] = 0
        p /= p.sum(axis=1, keepdims=True)
        wybory = np.array([rng.choice(V, p=row) for row in p], dtype="int32")
        for i, w in enumerate(wybory):
            if gotowe[i]:
                continue
            z = id2znak[w]
            if z == "!":
                gotowe[i] = True
            else:
                wynik[i] += z
        if gotowe.all():
            break
        ctx = np.concatenate([ctx, wybory[:, None]], axis=1)
    return [w for w in wynik if w]


def ocena(model, temperatura, ile=120):
    prob = generuj_wsad(model, ile=ile, temperatura=temperatura)
    if not prob:
        return None
    dokladne = [n for n in prob if n in ZNANE]
    dystanse = [najblizszy_dystans(n) for n in prob]
    prawie = [n for n, d in zip(prob, dystanse) if 0 < d <= 0.2]
    lp = [logprob_ngram(n) for n in prob]
    return {
        "temperatura": temperatura,
        "nowe_%": round(100 * (1 - len(dokladne) / len(prob)), 1),
        "prawie_kopie_%": round(100 * len(prawie) / len(prob), 1),
        "swieze_%": round(100 * sum(1 for d in dystanse if d > 0.2) / len(prob), 1),
        "logprob": round(float(np.mean(lp)), 3),
        "wiarygodnosc_%": round(100 * float(np.mean(lp)) / BAZA_LOGPROB, 1),
        "konc_lac_%": round(100 * sum(n.endswith(KONCOWKI) for n in prob) / len(prob), 1),
        "sr_dlugosc": round(float(np.mean([len(n) for n in prob])), 1),
        "przyklady": prob[:6],
    }


def trenuj_i_oceniaj(konfigi, punkty, temperatury, ile=120, wyniki_plik=WYNIKI, seed=SEED):
    """Trenuje każdą konfigurację i ocenia generowanie w zadanych punktach/temperaturach."""
    wyniki = []
    for cfg in konfigi:
        keras.utils.set_random_seed(seed)
        model = keras.Sequential([
            keras.layers.Embedding(V, cfg["emb"], mask_zero=True),
            keras.layers.GRU(cfg["units"], return_sequences=True, dropout=cfg["dropout"]),
            keras.layers.Dense(V, activation="softmax"),
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        poprzednia = 0
        for punkt in punkty:
            h = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                          epochs=punkt - poprzednia, verbose=0)
            poprzednia = punkt
            val = float(h.history["val_loss"][-1])
            tr = float(h.history["loss"][-1])
            for t in temperatury:
                o = ocena(model, t, ile=ile)
                o.update({**cfg, "epoki": punkt, "loss": round(tr, 3),
                          "val_loss": round(val, 3), "seed": seed})
                wyniki.append(o)
                print(f"u{cfg['units']} d{cfg['dropout']} ep{punkt:>3} T{t} | "
                      f"val={val:.3f} | nowe {o['nowe_%']:>5}% | świeże {o['swieze_%']:>5}% | "
                      f"wiarygodność {o['wiarygodnosc_%']:>5}% | końc {o['konc_lac_%']:>5}% | "
                      f"{', '.join(o['przyklady'][:3])}", flush=True)
    with open(wyniki_plik, "w") as f:
        json.dump({"baza_logprob": BAZA_LOGPROB, "wyniki": wyniki}, f,
                  ensure_ascii=False, indent=1)
    print(f"\nZapisano {len(wyniki)} pomiarów → {wyniki_plik}")
    return wyniki


# ── eksperyment ───────────────────────────────────────────────────────────
KONFIGI = [
    {"units": 128, "emb": 64, "dropout": 0.0},
    {"units": 128, "emb": 64, "dropout": 0.3},
    {"units": 256, "emb": 128, "dropout": 0.0},
    {"units": 256, "emb": 128, "dropout": 0.3},
]
PUNKTY = [10, 20, 30, 50, 80]
TEMPERATURY = [0.8, 1.0, 1.2]

if __name__ == "__main__":
    trenuj_i_oceniaj(KONFIGI, PUNKTY, TEMPERATURY)
