"""Logi zapytań Gradio (bez danych osobowych — tylko skrót hash klienta)."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

KATALOG_LOGI = Path(__file__).resolve().parent / "logi"


def skrot_klienta(ip: str, sesja: str) -> str:
    surowe = f"{ip}|{sesja}"
    return hashlib.sha256(surowe.encode("utf-8")).hexdigest()[:16]


def plik_dnia(dzien: date | None = None) -> Path:
    dzien = dzien or date.today()
    KATALOG_LOGI.mkdir(parents=True, exist_ok=True)
    return KATALOG_LOGI / f"zapytania_{dzien:%Y-%m-%d}.jsonl"


def zapisz_zdarzenie(
    *,
    klient: str,
    ile_nazw: int | None = None,
    temperatura: float | None = None,
    czas_odp_ms: float | None = None,
    wynik: str | None = None,
    odrzucone: bool = False,
    powod: str | None = None,
    czas: datetime | None = None,
) -> dict:
    """Jeden wpis JSONL; zwraca słownik zapisanego zdarzenia."""
    czas = czas or datetime.now(timezone.utc)
    wpis = {
        "czas": czas.isoformat(),
        "klient": klient,
        "ile_nazw": ile_nazw,
        "temperatura": temperatura,
        "czas_odp_ms": czas_odp_ms,
        "wynik": (wynik[:240] + "…") if wynik and len(wynik) > 240 else wynik,
        "odrzucone": odrzucone,
        "powod": powod,
    }
    with plik_dnia(czas.date()).open("a", encoding="utf-8") as f:
        f.write(json.dumps(wpis, ensure_ascii=False) + "\n")
    return wpis