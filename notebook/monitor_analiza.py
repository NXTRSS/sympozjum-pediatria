"""Analiza logów zapytań — testowalne bez Gradio i rich."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _parse_czas(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def wczytaj_jsonl(sciezka: Path) -> list[dict]:
    if not sciezka.is_file():
        return []
    out = []
    for linia in sciezka.read_text(encoding="utf-8").splitlines():
        if linia.strip():
            out.append(json.loads(linia))
    return out


def statystyki(zdarzenia: list[dict], teraz: datetime | None = None) -> dict:
    teraz = teraz or datetime.now(timezone.utc)
    ok = [z for z in zdarzenia if not z.get("odrzucone")]
    odrzucone = [z for z in zdarzenia if z.get("odrzucone")]

    def w_oknie(sek: float) -> list[dict]:
        granica = teraz.timestamp() - sek
        return [
            z for z in zdarzenia
            if _parse_czas(z["czas"]).timestamp() >= granica
        ]

    w10 = w_oknie(10)
    w60 = w_oknie(60)
    r10 = len(w10) * 6.0
    r60 = len(w60)

    per_klient_60: dict[str, int] = defaultdict(int)
    for z in w60:
        per_klient_60[z["klient"]] += 1
    top5 = Counter(per_klient_60).most_common(5)

    czasy = [z["czas_odp_ms"] for z in ok if z.get("czas_odp_ms") is not None]
    sredni = sum(czasy) / len(czasy) if czasy else 0.0
    p95 = 0.0
    if czasy:
        czasy_s = sorted(czasy)
        idx = max(0, int(round(0.95 * (len(czasy_s) - 1))))
        p95 = czasy_s[idx]

    ostatnie = sorted(zdarzenia, key=lambda z: z["czas"])[-10:]

    return {
        "lacznie": len(zdarzenia),
        "r10": r10,
        "r60": r60,
        "unikalni_60": len(per_klient_60),
        "top5_60": top5,
        "odrzucone": len(odrzucone),
        "sredni_ms": sredni,
        "p95_ms": p95,
        "ostatnie": ostatnie,
        "max_klient_60": max(per_klient_60.values()) if per_klient_60 else 0,
    }


def alarmy(stat: dict, *, max_kolejki: int = 50, glebokosc_kolejki: int = 0) -> list[str]:
    komunikaty = []
    if stat["r60"] > 60:
        komunikaty.append("MOŻLIWY DDOS: >60 zapytań/min łącznie")
    if stat["max_klient_60"] > 20:
        komunikaty.append("MOŻLIWY DDOS: >20 zapytań/min z jednego klienta")
    if glebokosc_kolejki >= max(1, max_kolejki - 5):
        komunikaty.append("MOŻLIWY DDOS: kolejka bliska max_size")
    return komunikaty