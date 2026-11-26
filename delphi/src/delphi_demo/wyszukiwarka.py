"""Wyszukiwarka chorób po polskiej nazwie i panel do składania historii w notebooku.

Lekarz nie musi znać kodów ICD-10: wpisuje fragment nazwy („cukrz”, „zawal”, „niewydol serc”),
wybiera chorobę z listy, podaje wiek i dopisuje ją do historii. Historia zostaje zwykłym tekstem
w formacie parsera („E11 50, I10 55”), więc ręczne wpisywanie kodów nadal działa.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

from delphi_demo.slownik import NAZWY_PL, BladHistorii, Slownik, parsuj_historie

_KOD = re.compile(r"^[A-Z]\d{2}$")

# Potoczne nazwy, których nie ma w nazwach ICD-10 (dopisywane do przeszukiwanego tekstu; nie zmieniają wyświetlanej nazwy).
SYNONIMY = {
    "F32": "depresja", "F33": "depresja nawracająca", "F84": "autyzm spektrum autyzmu ASD",
    "F90": "ADHD nadpobudliwość", "K90": "celiakia", "E84": "mukowiscydoza",
    "E06": "Hashimoto", "D50": "anemia", "D64": "anemia", "M05": "RZS", "M06": "RZS",
    "M08": "MIZS młodzieńcze idiopatyczne", "G35": "SM", "K50": "Crohn", "K51": "WZJG",
    "K58": "IBS", "O24": "cukrzyca ciążowa", "A37": "koklusz", "A69": "borelioza Lyme",
    "G80": "MPD", "E05": "Graves Basedow", "F31": "ChAD choroba dwubiegunowa",
    "F50": "anoreksja bulimia", "E73": "laktoza", "F70": "niepełnosprawność intelektualna",
    "J45": "dychawica", "L20": "AZS", "J44": "przewlekła obturacyjna choroba płuc",
    "I48": "AF arytmia", "I10": "nadciśnienie", "E78": "cholesterol hipercholesterolemia",
    "P07": "wcześniak", "H66": "zapalenie ucha",
}


def bez_ogonkow(tekst: str) -> str:
    """„Zawał Serca” → „zawal serca” (małe litery, bez polskich znaków)."""
    t = unicodedata.normalize("NFKD", tekst.lower().replace("ł", "l"))
    return "".join(z for z in t if not unicodedata.combining(z))


def _indeks(sl: Slownik) -> list[tuple[str, str, str]]:
    """(kod, nazwa do pokazania, tekst do przeszukania) dla kodów chorób, które zna model."""
    if getattr(sl, "_indeks_wyszukiwarki", None) is None:
        wpisy = []
        for kod, token in sl.kod_do_tokenu.items():
            if not _KOD.match(kod):               # bez tokenów specjalnych: płeć, styl życia, zgon, CXX
                continue
            nazwa = sl.nazwa(token)
            # bez etykiet angielskich: „ospa” trafiała w „Hypospadias”; 6 kodów bez polskiej nazwy ma angielską w `nazwa`
            przeszukiwane = " ".join(filter(None, [kod, nazwa, sl.oficjalne.get(kod, ""), SYNONIMY.get(kod, "")]))
            wpisy.append((kod, nazwa, bez_ogonkow(przeszukiwane)))
        sl._indeks_wyszukiwarki = sorted(wpisy)
    return sl._indeks_wyszukiwarki


def szukaj(fraza: str, sl: Slownik, limit: int = 12) -> list[tuple[str, str]]:
    """Choroby pasujące do frazy → [(kod, nazwa)], najlepsze najpierw.

    Każde słowo frazy musi wystąpić (jako fragment) w kodzie, nazwie polskiej, oficjalnej albo angielskiej.
    Kolejność: dokładny kod, nazwa zaczynająca się od frazy, słowa zaczynające się od słów frazy, reszta.
    """
    slowa = bez_ogonkow(fraza).split()
    if not slowa:
        return []
    cala = " ".join(slowa)
    indeks = _indeks(sl)
    rdzenie = slowa
    if not any(all(s in tekst for s in slowa) for _, _, tekst in indeks):
        # nic dla pełnych słów → prosty rdzeń na polską odmianę: „otyłości” → „otyłość”, „serce” → „serca”
        # (tylko wtedy, bo rdzeń „zawa” z „zawal” łapałby też „zawarte”)
        rdzenie = [s[:-2] if len(s) >= 7 else s[:-1] if len(s) >= 5 else s for s in slowa]
    wyniki = []
    for kod, nazwa, tekst in indeks:
        if not all(r in tekst for r in rdzenie):
            continue
        n = bez_ogonkow(nazwa)
        if kod.lower() == cala:
            ocena = 0
        elif n.startswith(cala):
            ocena = 1
        elif all(re.search(r"(?<!\w)" + re.escape(r), n) for r in rdzenie):
            ocena = 2
        elif all(r in n for r in rdzenie):
            ocena = 3
        else:
            ocena = 4                                    # pasuje tylko nazwa oficjalna albo angielska
        popularna = 0 if kod in NAZWY_PL else 1          # przy remisie: częste choroby z krótką nazwą dla sali
        wyniki.append((ocena, popularna, len(nazwa), kod, nazwa))
    wyniki.sort()
    return [(kod, nazwa) for *_, kod, nazwa in wyniki[:limit]]


def _liczba(x: float) -> str:
    return f"{x:g}".replace(".", ",")


def dopisz(historia: str, kod: str, wiek: float) -> str:
    """„E11 50” + („I10”, 55.5) → „E11 50, I10 55,5”."""
    wpis = f"{kod} {_liczba(float(wiek))}"
    return f"{historia.strip().rstrip(',')}, {wpis}" if historia.strip() else wpis


def opis_historii(historia: str, sl: Slownik) -> str:
    """Historia rozpisana po polsku (HTML do widżetu); błąd parsera jako czytelna uwaga."""
    if not historia.strip():
        return "<i>Historia jest pusta: wyszukaj chorobę, podaj wiek i kliknij „Dodaj”.</i>"
    try:
        zdarzenia = parsuj_historie(historia, sl)
    except BladHistorii as blad:
        return f"<b>UWAGA:</b> {html.escape(str(blad))}"
    wiersze = "".join(
        f"<li>{_liczba(z.wiek_lat)} lat: <b>{html.escape(sl.nazwa(z.token))}</b> "
        f"<span style='opacity:.6'>({sl.kod(z.token)})</span></li>" for z in zdarzenia)
    return f"<ol style='margin:4px 0'>{wiersze}</ol>"


@dataclass
class Panel:
    szukaj: object
    wyniki: object
    wiek: object
    dodaj: object
    historia: object
    wyczysc: object
    opis: object
    plec: object
    wiek_teraz: object
    pokaz: object
    wyjscie: object
    widok: object


def panel(sl: Slownik, pokaz_prognoze: Callable[[str, str, int], object], historia: str = "",
          plec: str = "K", wiek_teraz: int = 60) -> Panel:
    """Widżety: wyszukiwarka → lista → wiek → „Dodaj”; pole historii; „Pokaż prognozę” woła pokaz_prognoze."""
    import ipywidgets as W
    from IPython.display import display

    szerokosc = W.Layout(width="100%")
    pole_szukaj = W.Text(placeholder="np. cukrzyca, zawal, astma, niewydol serc…",
                         description="Szukaj:", layout=szerokosc, continuous_update=True)
    lista = W.Select(options=[], rows=8, description="Choroba:", layout=szerokosc)
    pole_wiek = W.BoundedFloatText(value=50, min=0, max=110, step=1, description="W wieku:",
                                   layout=W.Layout(width="200px"))
    przycisk_dodaj = W.Button(description="Dodaj do historii", button_style="primary", icon="plus")
    pole_historia = W.Text(value=historia, description="Historia:", layout=szerokosc,
                           placeholder="możesz też wpisać kody: E11 50, I10 55")
    przycisk_wyczysc = W.Button(description="Wyczyść", icon="trash")
    opis = W.HTML(opis_historii(historia, sl))
    pole_plec = W.ToggleButtons(options=[("kobieta", "K"), ("mężczyzna", "M")], value=plec,
                                description="Płeć:")
    suwak_wiek = W.IntSlider(value=wiek_teraz, min=1, max=90, description="Wiek dziś:")
    przycisk_pokaz = W.Button(description="Pokaż prognozę", button_style="success", icon="line-chart")
    wyjscie = W.Output()

    def odswiez_liste(_=None):
        wyniki = szukaj(pole_szukaj.value, sl)
        lista.options = [(f"{kod} · {nazwa}", kod) for kod, nazwa in wyniki]
        lista.value = wyniki[0][0] if wyniki else None

    def po_zmianie_historii(_=None):
        opis.value = opis_historii(pole_historia.value, sl)

    def po_dodaj(_):
        if lista.value:
            pole_historia.value = dopisz(pole_historia.value, lista.value, pole_wiek.value)

    def po_wyczysc(_):
        pole_historia.value = ""

    def po_pokaz(_):
        wyjscie.clear_output()
        with wyjscie:
            wynik = pokaz_prognoze(pole_historia.value, pole_plec.value, suwak_wiek.value)
            if wynik is not None:
                display(wynik)

    pole_szukaj.observe(odswiez_liste, names="value")
    pole_historia.observe(po_zmianie_historii, names="value")
    przycisk_dodaj.on_click(po_dodaj)
    przycisk_wyczysc.on_click(po_wyczysc)
    przycisk_pokaz.on_click(po_pokaz)

    widok = W.VBox([
        W.HTML("<b>1. Znajdź chorobę po nazwie</b> (bez polskich znaków też działa)"),
        pole_szukaj, lista, W.HBox([pole_wiek, przycisk_dodaj]),
        W.HTML("<b>2. Historia pacjenta</b>"), W.HBox([pole_historia, przycisk_wyczysc]), opis,
        W.HTML("<b>3. Pacjent dziś</b>"), pole_plec, suwak_wiek, przycisk_pokaz, wyjscie,
    ], layout=W.Layout(max_width="900px"))
    return Panel(pole_szukaj, lista, pole_wiek, przycisk_dodaj, pole_historia, przycisk_wyczysc, opis,
                 pole_plec, suwak_wiek, przycisk_pokaz, wyjscie, widok)
