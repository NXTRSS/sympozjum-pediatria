#!/usr/bin/env python3
"""Wybór śladu generowania do slajdu o10 (GENERATION_STEPS).

Trening jak notebook / demo_model (ziarno 42, T=0.4). Próbki: losowanie numpy
przy indeksie `PROG_N = 10_000 + n` (n = numer próby, odtwarzalne).

    uv run python eksperymenty/wybor_sladu.py
    uv run python eksperymenty/wybor_sladu.py --indeks 10442 --zapisz-slajd
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

KORZEN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KORZEN / "notebook"))

from demo_model import (  # noqa: E402
    TEMPERATURA,
    ZIARNO,
    generuj,
    przygotuj_dane,
    trenuj,
    ustaw_ziarna,
    wczytaj_nazwy,
    zbuduj_model,
)

PROG_N = 10_000
MIN_DL = 12
MAX_DL = 22
MIN_P = 0.10
LAC_WZOR = re.compile(r"^[a-z][a-z \-']*$")
KONCOWKI = (
    "itis", "osis", "oma", "pathia", "algia", "emia", "uria", "iasis",
    "ismus", "ectasis", "plasia", "penia", "philia", "malacia",
)
# Slajd o10: próba n=553 (np.random.seed(10553)) → „pertus panalismus” (wybór ręczny z listy kandydatów).
INDEKS_WYBRANY = 553


def pokaz(z: str) -> str:
    return "␣" if z == " " else z


def eksporter_krokow(kroki: list, id2znak: dict) -> list[dict]:
    """Ten sam format co komórka „ŚLAD GENEROWANIA → JSON” w notebooku."""
    eksport = []
    for krok in kroki:
        p, w = krok["rozklad"], krok["wybor"]
        top5 = np.argsort(p)[::-1][:5]
        probs = [[pokaz(id2znak[int(i)]), round(float(p[i]), 2)] for i in top5]
        if w not in top5:
            probs.append([pokaz(id2znak[w]), round(float(p[w]), 2)])
        probs.append(["…", round(max(0.0, 1 - sum(x for _, x in probs)), 2)])
        eksport.append({
            "context": "".join(pokaz(z) for z in krok["kontekst"]),
            "probs": probs,
            "chosen": pokaz(id2znak[w]),
            "chosen_p": round(float(p[w]), 2),
            "top": pokaz(id2znak[int(top5[0])]),
            "desc": "",
        })
    return eksport


def min_chosen_p(kroki: list) -> float:
    return min(float(k["rozklad"][k["wybor"]]) for k in kroki)


def blisko_znanej(nazwa: str, znane: set[str], prog: float = 0.88) -> bool:
    n = nazwa.strip().lower()
    for z in znane:
        if n == z:
            return True
        if n in z or z in n:
            return True
        if SequenceMatcher(None, n, z).ratio() >= prog:
            return True
    return False


def brzmi_lacinski(nazwa: str) -> bool:
    n = nazwa.strip().lower()
    if not LAC_WZOR.match(n) or len(n) < MIN_DL:
        return False
    vowels = set("aeiou")
    if sum(1 for c in n if c in vowels) < 3:
        return False
    if re.search(r"[^aeiou]{5,}", n.replace(" ", "")):
        return False
    if any(n.endswith(s) for s in KONCOWKI):
        return True
    if " " in n and len(n.split()) == 2:
        return True
    return n.endswith(("ae", "is", "us", "um", "es", "a"))


def dopisz_opisy(eksport: list[dict], nazwa: str) -> None:
    for i, k in enumerate(eksport):
        # opisy bez zmiany liczb — tylko słowa
        total = len(eksport)
        ch, tp, cp = k["chosen"], k["top"], k["chosen_p"]
        los = ch != tp
        if i == 0:
            k["desc"] = (
                f"🎲 Start: faworyt \"{tp}\" ({int(k['probs'][0][1]*100)}%), "
                f"wylosowano \"{ch}\" ({int(cp*100)}%)."
            )
        elif i == total - 1:
            k["desc"] = (
                f"{'🎲 ' if los else ''}Token STOP (!) — {int(cp*100)}%. "
                f'Wynik: \"{nazwa}\".'
            )
        elif los:
            k["desc"] = f"🎲 Zamiast \"{tp}\" wypada \"{ch}\" ({int(cp*100)}%)."
        else:
            k["desc"] = f"\"{ch}\" ({int(cp*100)}%) — zgodnie z faworytem."


def pasuje(nazwa: str, kroki: list, znane: set[str]) -> tuple[bool, str]:
    if nazwa in znane:
        return False, "w zbiorze"
    if not MIN_DL <= len(nazwa) <= MAX_DL:
        return False, "długość"
    if blisko_znanej(nazwa, znane):
        return False, "podobna"
    if not brzmi_lacinski(nazwa):
        return False, "brzmi"
    mp = min_chosen_p(kroki)
    if mp < MIN_P:
        return False, f"p_min={mp:.3f}"
    return True, f"p_min={mp:.2f}"


def ocena(nazwa: str, kroki: list) -> float:
    s = 0.0
    if "infant" in nazwa or "neonat" in nazwa or "paed" in nazwa or "puer" in nazwa:
        s += 3
    if "syndrom" in nazwa or "morbus" in nazwa:
        s += 2
    if any(nazwa.endswith(x) for x in KONCOWKI):
        s += 1.5
    if " " in nazwa:
        s += 1
    losy = sum(
        1 for k in kroki
        if int(k["wybor"]) != int(np.argmax(k["rozklad"]))
    )
    s += min(losy, 4) * 0.4
    s -= abs(len(kroki) - 14) * 0.15
    return s


def probuj(model, znak2id, id2znak, znane: set[str], n: int):
    np.random.seed(PROG_N + n)
    nazwa, kroki = generuj(model, znak2id, id2znak, temperatura=TEMPERATURA)
    ok, powod = pasuje(nazwa, kroki, znane)
    return {
        "n": n,
        "prog": PROG_N + n,
        "nazwa": nazwa,
        "kroki": kroki,
        "ok": ok,
        "powod": powod,
        "ocena": ocena(nazwa, kroki) if ok else -1,
        "p_min": min_chosen_p(kroki) if kroki else 0,
        "krokow": len(kroki),
    }


def krok_do_js(eksport: list[dict]) -> str:
    def fmt_prob(p):
        return f"[{json.dumps(p[0], ensure_ascii=False)}, {p[1]}]"

    linie = ["["]
    for i, k in enumerate(eksport):
        probs = ",\n      ".join(fmt_prob(p) for p in k["probs"])
        linie.append(f"  {{\n    context: {json.dumps(k['context'])},\n    probs: [{probs}],\n    chosen: {json.dumps(k['chosen'])},\n    chosen_p: {k['chosen_p']}, top: {json.dumps(k['top'])},\n    desc: {json.dumps(k['desc'], ensure_ascii=False)}\n  }}{',' if i < len(eksport) - 1 else ''}")
    linie.append("]")
    return "\n".join(linie)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proby", type=int, default=2500)
    parser.add_argument("--indeks", type=int, default=None, help="numer próby n (prog=10000+n)")
    parser.add_argument("--zapisz-slajd", action="store_true")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    ustaw_ziarna(ZIARNO)
    nazwy, _ = wczytaj_nazwy(ziarno=ZIARNO)
    znane = set(nazwy)
    X, Y, znak2id, id2znak, znaki = przygotuj_dane(nazwy)
    model = zbuduj_model(len(znaki) + 1, ziarno=ZIARNO)
    print(f"Trening: ziarno {ZIARNO}, T={TEMPERATURA}, 12 epok…", flush=True)
    trenuj(model, X, Y, epoki=12)

    if args.indeks is not None:
        r = probuj(model, znak2id, id2znak, znane, args.indeks)
        print(json.dumps({k: v for k, v in r.items() if k != "kroki"}, ensure_ascii=False, indent=2))
        if args.zapisz_slajd:
            eksport = eksporter_krokow(r["kroki"], id2znak)
            dopisz_opisy(eksport, r["nazwa"])
            _zapisz_js(eksport, r["nazwa"], args.indeks)
        return

    kandydaci = []
    for n in range(args.proby):
        r = probuj(model, znak2id, id2znak, znane, n)
        if r["ok"]:
            kandydaci.append(r)
        if n and n % 500 == 0:
            print(f"  …{n} prób, kandydatów: {len(kandydaci)}", flush=True)

    kandydaci.sort(key=lambda x: (-x["ocena"], -x["p_min"], -len(x["nazwa"])))
    top = kandydaci[: args.top]
    print(f"\nKandydatów spełniających filtr: {len(kandydaci)} / {args.proby}\n")
    for r in top:
        print(
            f"  n={r['n']:5d} prog={r['prog']}  „{r['nazwa']}\"  "
            f"p_min={r['p_min']:.2f}  kroków={r['krokow']}  ocena={r['ocena']:.1f}"
        )
    if not top:
        sys.exit("Brak kandydatów — zwiększ --proby lub poluzuj filtr.")
    best = top[0]
    print(f"\nWybór: „{best['nazwa']}\" (n={best['n']}, prog={best['prog']})\n")
    out = KORZEN / "eksperymenty" / "wybor_sladu_wynik.json"
    out.write_text(
        json.dumps(
            {
                "wybrany": {k: v for k, v in best.items() if k != "kroki"},
                "inni": [{k: v for k, v in r.items() if k != "kroki"} for r in top[1:5]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Zapisano {out}")
    if args.zapisz_slajd:
        eksport = eksporter_krokow(best["kroki"], id2znak)
        dopisz_opisy(eksport, best["nazwa"])
        _zapisz_js(eksport, best["nazwa"], best["n"])


def _zapisz_js(eksport: list[dict], nazwa: str, indeks: int) -> None:
    tablica = krok_do_js(eksport)
    naglowek = f"""// GENERATION_STEPS dla slajdu „Krok po kroku" (o10_inference.html).
// Model: ziarno treningu {ZIARNO}, TEMPERATURA={TEMPERATURA}, próba n={indeks} (np.random.seed({PROG_N + indeks})).
// Nazwa: „{nazwa}" — prawdopodobieństwa wyłącznie z modelu (eksport notebookowy).
// ␣ = spacja. „…" = suma pozostałych liter.
"""
    sciezka = KORZEN / "slides" / "inference_steps.js"
    sciezka.write_text(naglowek + tablica + "\n", encoding="utf-8")
    print(f"OK: {sciezka}")


if __name__ == "__main__":
    main()