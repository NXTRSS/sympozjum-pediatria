"""Panel ipywidgets: generowanie życia token po tokenie (slajd „klamra")."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from delphi_demo import pamiec
from delphi_demo.slownik import Slownik
from delphi_demo.wyszukiwarka import dopisz, opis_historii, szukaj


@dataclass
class PanelGenerowania:
    szukaj: object
    wyniki: object
    wiek_choroby: object
    dodaj: object
    historia: object
    wyczysc: object
    opis: object
    plec: object
    wiek_teraz: object
    ziarno: object
    temperatura: object
    maks_krokow: object
    ile_sladu: object
    styl_info: object
    generuj: object
    status: object
    wyjscie: object
    widok: object


def _domyslne_dopasowanie(domyslny: dict | None, historia: str, plec: str, wiek_teraz: int,
                          ziarno: int, temperatura: float, maks_krokow: int) -> bool:
    if not domyslny:
        return False
    return (
        historia.strip() == domyslny["historia"].strip()
        and plec == domyslny["plec"]
        and wiek_teraz == domyslny["wiek_teraz"]
        and ziarno == 0
        and temperatura == 1.0
        and maks_krokow == 80
    )


def panel_generowania(
    sl: Slownik,
    pokaz: Callable[..., object],
    *,
    tryb: str = "z_pamieci",
    klucz_pamieci: str | None = None,
    domyslny: dict | None = None,
    historia: str = "",
    plec: str = "K",
    wiek_teraz: int = 60,
    ziarno: int = 0,
    temperatura: float = 1.0,
    maks_krokow: int = 80,
    ile_sladu: int = 12,
) -> PanelGenerowania:
    """Widżety + „Generuj". `pokaz(historia, plec, wiek_teraz, ziarno, temperatura, maks_krokow, ile_sladu)`."""
    import ipywidgets as W

    szerokosc = W.Layout(width="100%")
    zajete = {"tak": False}

    pole_szukaj = W.Text(
        placeholder="np. astma, cukrzyca, atopowe zapalenie skóry…",
        description="Szukaj:",
        layout=szerokosc,
        continuous_update=True,
    )
    lista = W.Select(options=[], rows=6, description="Choroba:", layout=szerokosc)
    pole_wiek_ch = W.BoundedFloatText(
        value=10, min=0, max=110, step=1, description="W wieku:",
        layout=W.Layout(width="200px"),
    )
    przycisk_dodaj = W.Button(description="Dodaj do historii", button_style="primary", icon="plus")
    pole_historia = W.Text(
        value=historia,
        description="Historia:",
        layout=szerokosc,
        placeholder="E11 50, I10 55 lub wybierz choroby powyżej",
    )
    przycisk_wyczysc = W.Button(description="Wyczyść historię", icon="trash")
    opis = W.HTML(opis_historii(historia, sl))

    pole_plec = W.ToggleButtons(options=[("kobieta", "K"), ("mężczyzna", "M")], value=plec, description="Płeć:")
    suwak_wiek = W.IntSlider(value=wiek_teraz, min=1, max=90, description="Wiek dziś:")

    pole_ziarno = W.IntText(value=ziarno, description="Ziarno losowania:", layout=W.Layout(width="220px"))
    suwak_temp = W.FloatSlider(
        value=temperatura, min=0.5, max=2.0, step=0.1, description="Temperatura:",
        readout_format=".1f",
    )
    suwak_krokow = W.IntSlider(value=maks_krokow, min=5, max=80, step=1, description="Maks. kroków:")
    suwak_sladu = W.IntSlider(value=ile_sladu, min=5, max=20, step=1, description="Wierszy śladu:")

    styl_info = W.HTML(
        "<i>Styl życia (BMI, palenie, alkohol): tokeny są w słowniku, "
        "ale model na danych syntetycznych ich nie widział — tu nie wpisujemy.</i>"
    )

    przycisk_generuj = W.Button(description="Generuj", button_style="success", icon="play")
    status = W.HTML("")

    wyjscie = W.Output()

    def odswiez_liste(_=None):
        wyniki = szukaj(pole_szukaj.value, sl)
        lista.options = [(f"{kod} · {nazwa}", kod) for kod, nazwa in wyniki]
        lista.value = wyniki[0][0] if wyniki else None

    def po_zmianie_historii(_=None):
        opis.value = opis_historii(pole_historia.value, sl)

    def po_dodaj(_):
        if lista.value:
            pole_historia.value = dopisz(pole_historia.value, lista.value, pole_wiek_ch.value)

    def po_wyczysc(_):
        pole_historia.value = ""

    def _parametry():
        return (
            pole_historia.value.strip(),
            pole_plec.value,
            int(suwak_wiek.value),
            int(pole_ziarno.value),
            float(suwak_temp.value),
            int(suwak_krokow.value),
            int(suwak_sladu.value),
        )

    def _wykonaj():
        h, pl, wiek, zi, temp, maks, ile = _parametry()
        if (
            tryb == "z_pamieci"
            and klucz_pamieci
            and _domyslne_dopasowanie(domyslny, h, pl, wiek, zi, temp, maks)
            and pamiec.jest(f"slad_{klucz_pamieci}")
        ):
            return pokaz(
                h, pl, wiek, zi, temp, maks, ile,
                kroki_z_pamieci=pamiec.wczytaj_kroki(klucz_pamieci),
            )
        return pokaz(h, pl, wiek, zi, temp, maks, ile)

    def po_generuj(_):
        if zajete["tak"]:
            status.value = "<span style='color:#c60'>Trwa liczenie — poczekaj na koniec.</span>"
            return
        zajete["tak"] = True
        przycisk_generuj.disabled = True
        przycisk_generuj.description = "Liczenie…"
        status.value = ""
        wyjscie.clear_output(wait=True)
        try:
            with wyjscie:
                import time

                start = time.time()
                _wykonaj()
                sek = time.time() - start
                status.value = f"<span style='opacity:.75'>Gotowe ({sek:.1f} s).</span>"
                if sek > 10:
                    status.value += " <span style='color:#c60'>Powyżej 10 s — zmniejsz „Maks. kroków”.</span>"
        finally:
            zajete["tak"] = False
            przycisk_generuj.disabled = False
            przycisk_generuj.description = "Generuj"

    def uruchom_domyslnie():
        """Run All w Jupyterze / nbconvert — jak dawna statyczna komórka."""
        po_generuj(None)

    pole_szukaj.observe(odswiez_liste, names="value")
    pole_historia.observe(po_zmianie_historii, names="value")
    przycisk_dodaj.on_click(po_dodaj)
    przycisk_wyczysc.on_click(po_wyczysc)
    przycisk_generuj.on_click(po_generuj)

    widok = W.VBox(
        [
            W.HTML("<b>Historia startowa</b> (choroby z przeszłości — wyszukiwarka jak wyżej)"),
            pole_szukaj,
            lista,
            W.HBox([pole_wiek_ch, przycisk_dodaj]),
            W.HBox([pole_historia, przycisk_wyczysc]),
            opis,
            W.HTML("<b>Pacjent „dziś” i losowanie</b>"),
            pole_plec,
            suwak_wiek,
            styl_info,
            W.HBox([pole_ziarno, suwak_temp]),
            W.HBox([suwak_krokow, suwak_sladu]),
            W.HBox([przycisk_generuj, status]),
            wyjscie,
        ],
        layout=W.Layout(max_width="920px"),
    )

    panel = PanelGenerowania(
        pole_szukaj, lista, pole_wiek_ch, przycisk_dodaj, pole_historia, przycisk_wyczysc, opis,
        pole_plec, suwak_wiek, pole_ziarno, suwak_temp, suwak_krokow, suwak_sladu, styl_info,
        przycisk_generuj, status, wyjscie, widok,
    )
    panel.uruchom_domyslnie = uruchom_domyslnie  # type: ignore[attr-defined]
    return panel