#!/usr/bin/env python3
"""Podgląd na żywo zapytań do demo Gradio (czytaj logi JSONL)."""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitor_analiza import alarmy, statystyki, wczytaj_jsonl  # noqa: E402
from zapytania_log import KATALOG_LOGI, plik_dnia  # noqa: E402


def _tabela(stat: dict, alarm_list: list[str]) -> str:
    from rich.table import Table

    tbl = Table(title="Demo chorób — monitoring zapytań", expand=True)
    tbl.add_column("Metryka", style="cyan")
    tbl.add_column("Wartość", style="white")
    tbl.add_row("Łącznie (dziś)", str(stat["lacznie"]))
    tbl.add_row("Zapytania/min (10 s)", f"{stat['r10']:.1f}")
    tbl.add_row("Zapytania/min (60 s)", str(stat["r60"]))
    tbl.add_row("Unikalni klienci (60 s)", str(stat["unikalni_60"]))
    tbl.add_row("Odrzucone (dziś)", str(stat["odrzucone"]))
    tbl.add_row("Średni czas odp. (ms)", f"{stat['sredni_ms']:.0f}")
    tbl.add_row("p95 czas odp. (ms)", f"{stat['p95_ms']:.0f}")
    if stat["top5_60"]:
        top = ", ".join(f"{k}:{n}" for k, n in stat["top5_60"])
        tbl.add_row("Top 5 klientów (60 s)", top)
    if alarm_list:
        for a in alarm_list:
            tbl.add_row("ALARM", f"[bold red]{a}[/]")
    tbl.add_row("Log", str(plik_dnia()))
    return tbl


def _ostatnie(stat: dict) -> str:
    from rich.panel import Panel

    linie = []
    for z in stat["ostatnie"]:
        flaga = "ODRZ" if z.get("odrzucone") else " OK "
        linie.append(
            f"{z['czas'][11:19]} {flaga} {z['klient'][:8]} "
            f"ile={z.get('ile_nazw')} temp={z.get('temperatura')} "
            f"{z.get('czas_odp_ms', 0):.0f}ms"
        )
    return Panel("\n".join(linie) or "(brak zdarzeń)", title="Ostatnie 10 zapytań")


def main() -> None:
    from rich.console import Console
    from rich.live import Live

    stan_path = KATALOG_LOGI / "stan_demo.json"

    def odczyt() -> tuple[dict, list[str]]:
        zdarzenia = wczytaj_jsonl(plik_dnia())
        teraz = datetime.now(timezone.utc)
        stat = statystyki(zdarzenia, teraz)
        gleb = 0
        if stan_path.is_file():
            import json

            stan = json.loads(stan_path.read_text(encoding="utf-8"))
            gleb = int(stan.get("glebokosc_kolejki", 0))
            stat["odrzucone_kolejka"] = int(stan.get("odrzucone_kolejka", 0))
        alarm_list = alarmy(stat, glebokosc_kolejki=gleb)
        return stat, alarm_list

    from rich.console import Group

    console = Console()
    dzwonilo = False
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            stat, alarm_list = odczyt()
            if alarm_list and not dzwonilo:
                console.bell()
                dzwonilo = True
            if not alarm_list:
                dzwonilo = False
            live.update(Group(_tabela(stat, alarm_list), _ostatnie(stat)))
            time.sleep(1)


if __name__ == "__main__":
    main()