"""Logi zapytań, limity i alarmy monitora demo Gradio."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook"
sys.path.insert(0, str(NOTEBOOK))

from monitor_analiza import alarmy, statystyki, wczytaj_jsonl  # noqa: E402
from ochrona_klienta import LimitKlienta  # noqa: E402
from zapytania_log import KATALOG_LOGI, skrot_klienta, zapisz_zdarzenie  # noqa: E402


@pytest.fixture(autouse=True)
def _katalog_logow(monkeypatch, tmp_path):
    logi = tmp_path / "logi"
    logi.mkdir()
    monkeypatch.setattr("zapytania_log.KATALOG_LOGI", logi)
    monkeypatch.setattr("monitor_analiza.KATALOG_LOGI", logi, raising=False)
    return logi


def test_zapisz_zdarzenie_jsonl(_katalog_logow):
    wpis = zapisz_zdarzenie(
        klient="abc123",
        ile_nazw=3,
        temperatura=0.4,
        czas_odp_ms=120.5,
        wynik="• test",
        odrzucone=False,
    )
    pliki = list(_katalog_logow.glob("zapytania_*.jsonl"))
    assert len(pliki) == 1
    zdarzenia = wczytaj_jsonl(pliki[0])
    assert len(zdarzenia) == 1
    assert zdarzenia[0]["klient"] == "abc123"
    assert zdarzenia[0]["ile_nazw"] == 3
    assert wpis["temperatura"] == 0.4


def test_skrot_klienta_jest_haszem():
    a = skrot_klienta("1.2.3.4", "sesja-x")
    b = skrot_klienta("1.2.3.4", "sesja-x")
    c = skrot_klienta("1.2.3.4", "sesja-y")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_statystyki_stawka_na_minute():
    teraz = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    zdarzenia = []
    for i in range(30):
        zdarzenia.append(
            zapisz_zdarzenie(
                klient=f"k{i % 3}",
                odrzucone=False,
                czas_odp_ms=10.0,
                czas=teraz - timedelta(seconds=i),
            )
        )
    stat = statystyki(zdarzenia, teraz)
    assert stat["r60"] == 30
    assert stat["unikalni_60"] == 3


def test_alarm_ponad_60_na_minute():
    teraz = datetime.now(timezone.utc)
    zdarzenia = [
        {
            "czas": (teraz - timedelta(seconds=s)).isoformat(),
            "klient": "a",
            "odrzucone": False,
        }
        for s in range(65)
    ]
    stat = statystyki(zdarzenia, teraz)
    alarms = alarmy(stat)
    assert any("60 zapytań/min" in a for a in alarms)


def test_alarm_klient_ponad_20_na_minute():
    teraz = datetime.now(timezone.utc)
    zdarzenia = [
        {
            "czas": (teraz - timedelta(seconds=s)).isoformat(),
            "klient": "ten_sam",
            "odrzucone": False,
        }
        for s in range(25)
    ]
    stat = statystyki(zdarzenia, teraz)
    alarms = alarmy(stat)
    assert any("jednego klienta" in a for a in alarms)


def test_alarm_kolejka_blisko_max():
    stat = {"r60": 0, "max_klient_60": 0}
    alarms = alarmy(stat, glebokosc_kolejki=48, max_kolejki=50)
    assert any("kolejka" in a for a in alarms)


def test_limit_klienta_20_na_minute():
    lim = LimitKlienta(limit=20)
    t0 = time.monotonic()
    for _ in range(20):
        assert lim.wpusc("klient-a", t0)
    assert not lim.wpusc("klient-a", t0)
    assert lim.wpusc("klient-a", t0 + 61.0)