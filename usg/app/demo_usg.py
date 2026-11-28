"""Aplikacja pokazowa „Z nagrania USG do masy płodu” (Gradio, lokalnie, bez internetu).

Uruchomienie (środowisko uv w katalogu usg/, Python 3.11 + PyTorch):
    cd usg && uv sync && uv run python app/demo_usg.py
    cd usg && uv run python app/demo_usg.py --bez-przegladarki
    cd usg && uv run python app/demo_usg.py --bez-linku

Z katalogu głównego repozytorium (nie: uv run python usg/app/demo_usg.py — to bierze złe venv):
    uv run python uruchom_demo_usg.py
    uv run --directory usg python app/demo_usg.py

Port (domyślnie 7861 — generator chorób zwykle zajmuje 7860):
    GRADIO_SERVER_PORT=7862 uv run python uruchom_demo_usg.py
    uv run python uruchom_demo_usg.py --port 7862

Ten sam pipeline, liczby i zastrzeżenia co notebook (notebook/usg_masa_plodu.ipynb);
cała logika jest w usg_masa.aplikacja, tu jest tylko wygląd.
Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych.
"""
from __future__ import annotations

import os

# Zero ruchu sieciowego: bez statystyk Gradio i Hugging Face (muszą być przed importem gradio).
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Zanim załaduje się Gradio: USG na 7861, żeby nie walczyć z modelem chorób (7860).
if os.environ.get("GRADIO_SERVER_PORT", "").strip() in ("", "7860"):
    os.environ["GRADIO_SERVER_PORT"] = "7861"

import argparse  # noqa: E402
import html  # noqa: E402
import logging  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

_USG_KORZEN = Path(__file__).resolve().parents[1]
_SRC = _USG_KORZEN / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr  # noqa: E402

try:
    from usg_masa import aplikacja as A  # noqa: E402
    from usg_masa import dane, fuvai, model_sano  # noqa: E402
except ModuleNotFoundError:
    raise SystemExit(
        "Nie ma pakietu usg_masa w tym środowisku Pythona.\n"
        "USG ma osobny projekt w usg/ (PyTorch, Python 3.11). Uruchom np.:\n"
        "  cd usg && uv sync && uv run python app/demo_usg.py\n"
        "albo z katalogu głównego:\n"
        "  uv run python uruchom_demo_usg.py\n"
        "  uv run --directory usg python app/demo_usg.py\n"
        "Komenda „uv run python usg/app/demo_usg.py” z głównego katalogu używa złego venv (notebook/TF)."
    ) from None

KATALOG = Path(__file__).resolve().parent
STYL = KATALOG / "styl.css"
TYTUL = "Z nagrania USG do masy płodu"
ZASTRZEZENIE = "Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych."
TRYBY = [("z pamięci", A.Z_PAMIECI), ("na żywo", A.NA_ZYWO)]
KROK_ODTWARZANIA = 2          # klatek na jeden takt „▶ odtwórz”
TAKT_ODTWARZANIA_S = 0.12
BLAD_PLIKU = "Nie udało się wczytać pliku „{}” (oczekiwany format ACOUSLIC-AI: .mha, 3D)."
BLAD_NA_ZYWO = "Liczenie „na żywo” się nie udało. Przełącz na „z pamięci”."
BLAD_MODELI = "Nie udało się wczytać modeli do trybu <b>na żywo</b>. Przełącz na <b>z pamięci</b>."

# Szczegóły błędów idą do terminala (stderr), na ekran tylko krótki polski komunikat.
log = logging.getLogger("demo_usg")

# Gradio rysuje ciemny motyw, gdy w adresie jest ?__theme=dark — wymuszamy go zawsze.
JS_CIEMNY = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark');
    window.location.replace(u.href);
  }
}
"""

e = html.escape


# ── kawałki HTML ─────────────────────────────────────────────────────────────

def html_naglowek() -> str:
    return f"""
<header class="naglowek">
  <div>
    <div class="eyebrow">Sympozjum DIL · Wrocław · 28.11.2026</div>
    <h1>Z nagrania USG <span class="grad">do masy płodu</span></h1>
    <p class="lead">Model sam przegląda całe nagranie „na ślepo”, znajduje brzuch płodu, mierzy jego obwód (AC)
    i szacuje masę „teraz”. <b>Nie lepsze od Hadlocka</b> — ale w pełni automatyczne, na nagraniu nowicjusza.
    Polski model Sano (TabAttention, MICCAI 2023) szacuje z wideo masę <em>w dniu porodu</em> — uczony na badaniach tuż przed porodem; na naszych nagraniach (28–34 tc.) podaje szacunek masy urodzeniowej, nie aktualnej.</p>
  </div>
  <div class="zastrzezenie">{e(ZASTRZEZENIE)}</div>
</header>"""


def html_podpis(tekst: str, legenda: str = "", atrybucja: str = dane.ATRYBUCJA) -> str:
    """Podpis pod obrazem USG: co widać + skąd są dane (zawsze; ACOUSLIC-AI albo plik użytkownika)."""
    return (f'<div class="podpis"><span>{tekst}</span>{legenda}'
            f'<span class="atrybucja">{e(atrybucja)}</span></div>')


LEGENDA_NNUNET = ('<span class="leg"><i class="kreska cyan"></i>obrys nnU-Net</span>'
                  '<span class="leg"><i class="kreska bursztyn przerywana"></i>dopasowana elipsa → AC</span>')
LEGENDA_FUVAI = '<span class="leg"><i class="kreska fiolet"></i>obrys FUVAI</span>'


def html_pusta(numer: str, tytul: str, klasa: str, tekst: str) -> str:
    return (f'<div class="karta {klasa}"><div class="karta-eyebrow">{numer} · {tytul}</div>'
            f'<div class="pusto">{tekst}</div></div>')


def karta_nnunet(a: A.Analiza | None) -> str:
    tytul = "nnU-Net (ACOUSLIC-AI) · masa „teraz”"
    if a is None:
        return html_pusta("1", tytul, "k-cyan", "Kliknij „Znajdź brzuch i zmierz”.")
    o, W = a.pomiar, a.nnunet
    if W.najlepsza_klatka < 0 or o.ac_mm is None:
        return html_pusta("1", tytul, "k-cyan", e(o.komunikat))
    eksperci = " / ".join(f"{ac:.0f}" for ac in a.eksperci) if a.eksperci else "—"
    masa = f"ok. {A.gramy(o.masa_g)} g" if o.wiarygodne else "—"
    tydzien = f"ok. {A.po_polsku(o.tydzien)} tc." if o.wiarygodne else "—"
    gdzie = f"Klatka {W.najlepsza_klatka}"
    if a.przeciagniecie:
        gdzie += f" · przeciągnięcie głowicą nr {a.przeciagniecie} z 6"
        if a.eksperci:
            gdzie += (f" (ekspert zmierzył w nim {a.ekspert_przeciagniecia:.0f} mm)" if a.ekspert_przeciagniecia
                      else " (w tym przeciągnięciu ekspert nie mierzył)")
    skosna = a.plaszczyzna and a.plaszczyzna.startswith("sub")
    plaszczyzna = (f'<span class="pigulka {"p-ostrz" if skosna else "p-ok"}">płaszczyzna: {e(a.plaszczyzna)}</span>'
                   if a.plaszczyzna else "")
    uwaga = "" if o.wiarygodne else f'<div class="maly ostrz">{e(o.komunikat)}</div>'
    if skosna:
        uwaga += '<div class="maly ostrz">Przekrój skośny może zawyżać AC.</div>'
    zrodlo_ekspertow = " Eksperci: ACOUSLIC-AI." if a.eksperci else ""
    return f"""
<div class="karta k-cyan">
  <div class="karta-eyebrow">1 · {tytul}</div>
  <div class="siatka4">
    <div><div class="etyk">AC model</div><div class="liczba cyan">{o.ac_mm:.0f}<small> mm</small></div></div>
    <div><div class="etyk">AC eksperci</div><div class="liczba jasna">{eksperci}<small> mm</small></div></div>
    <div><div class="etyk">masa teraz</div><div class="liczba cyan">{masa}</div></div>
    <div><div class="etyk">wiek wg AC</div><div class="liczba jasna">{tydzien}</div></div>
  </div>
  <div class="pigulki">{plaszczyzna}</div>
  <div class="maly">{e(gdzie)}. Masa wzorem Campbella, wiek wzorem Hadlocka: z jednej miary (AC), więc to przybliżenie;
  wiek wg AC to nie datowanie ciąży.{zrodlo_ekspertow}</div>
  {uwaga}
</div>"""


def karta_fuvai(a: A.Analiza | None) -> str:
    tytul = "FUVAI (Sano + WUM) · polski, publiczne wagi"
    if a is None:
        return html_pusta("2", tytul, "k-fiolet", "Druga opinia: polski model, który sam rozpoznaje brzuch.")
    f = a.ocena_fuvai
    prog = fuvai.PROG_ARTYKULU
    if not f.znalazl:
        miernik = _miernik(f.p_max, prog, "najwyższe p(brzuch) w nagraniu")
        return f"""
<div class="karta k-fiolet">
  <div class="karta-eyebrow">2 · {tytul}</div>
  <div class="liczba fiolet">nie znalazł</div>
  {miernik}
  <div class="maly">Wśród klatek-kandydatek nie ma takiej, w której brzuch byłby dla modelu najbardziej
  prawdopodobny i dał pełny obrys. Nie podajemy AC ani masy.</div>
</div>"""
    masa = (f'<div class="liczba fiolet">ok. {A.gramy(f.masa_g)} g</div>' if f.masa_g is not None
            else '<div class="liczba wyciszona">masy nie podajemy</div>')
    rozklad = " · ".join(f"{k} {A.po_polsku(p, 2)}" for k, p in f.rozklad[:3])
    if f.pole_nnunet is None or f.klatka == a.nnunet.najlepsza_klatka:
        nnunet = ("nnU-Net wybrał tę samą klatkę." if f.klatka == a.nnunet.najlepsza_klatka else "")
    else:
        nnunet = ("W tej klatce nnU-Net też widzi brzuch." if f.pole_nnunet
                  else "W tej klatce nnU-Net nie widzi brzucha wcale.")
    if f.bez_masy:
        powod = ('<div class="maly ostrz">Masy nie podajemy: FUVAI nie jest pewny brzucha, a nnU-Net go tu nie widzi '
                 '(to może być np. główka).</div>')
    elif f.masa_g is None:   # np. AC poza zakresem fizjologicznym albo nieudany obrys
        powod = f'<div class="maly ostrz">{e(f.komunikat)}</div>'
    else:
        powod = ""
    ac = f'{f.ac_mm:.0f}<small> mm</small>' if f.ac_mm is not None else "—"
    return f"""
<div class="karta k-fiolet">
  <div class="karta-eyebrow">2 · {tytul}</div>
  <div class="siatka2">
    <div><div class="etyk">AC wg FUVAI (klatka {f.klatka})</div><div class="liczba fiolet">{ac}</div></div>
    <div><div class="etyk">masa teraz</div>{masa}</div>
  </div>
  {_miernik(f.p_brzuch, prog, "p(brzuch) w wybranej klatce")}
  <div class="maly">Co widzi w tej klatce: {e(rozklad)}. {e(nnunet)}</div>
  {powod}
</div>"""


def _miernik(p: float, prog: float, etyk: str) -> str:
    stan = "nad progiem" if p > prog else "poniżej progu"
    return f"""
<div class="miernik">
  <div class="miernik-opis"><span>{etyk}: <b>{A.po_polsku(p, 2)}</b></span>
  <span>próg z artykułu: {A.po_polsku(prog)} · <b class="{'ok' if p > prog else 'ostrz'}">{stan}</b></span></div>
  <div class="miernik-pasek"><div class="miernik-wartosc" style="width:{100 * p:.0f}%"></div>
  <div class="miernik-prog" style="left:{100 * prog:.0f}%"></div></div>
</div>"""


def karta_sano(a: A.Analiza | None) -> str:
    tytul = "Model Sano (TabAttention, MICCAI 2023) · masa w dniu porodu"
    if a is None:
        return html_pusta("3", tytul, "k-roz", "Polski model, który z samego obrazu szacuje masę w dniu porodu "
                          "(uczony na badaniach &lt; 24 h przed porodem).")
    S = a.sano
    if S.masa_g is None:
        tresc = '<div class="liczba wyciszona">brak wyniku</div><div class="maly">nnU-Net nie widzi w tym nagraniu brzucha, więc model Sano nie ma czego oglądać.</div>'
    else:
        gw = "*" if S.dopelniony else ""
        n_s = len(S.oceny_g)
        teraz = f"ok. {A.gramy(a.masa_teraz_g)} g" if a.masa_teraz_g else "—"
        przypis = (f'<div class="maly ostrz">* Za mało klatek brzucha na pełny segment (jest {S.n_klatek_brzucha}, '
                   f'potrzeba {A.KLATEK_NA_SEGMENT}): segment dopełniony zerami, wynik orientacyjny.</div>'
                   if S.dopelniony else "")
        zywo = ('<div class="maly">Na żywo nnU-Net ogląda co 4. klatkę, więc model Sano dostaje inne klatki brzucha '
                'i wynik różni się od slajdu (tam: tryb „z pamięci”).</div>' if a.tryb == A.NA_ZYWO else "")
        tresc = f"""
  <div class="siatka2">
    <div><div class="etyk">masa w dniu porodu (wg modelu)</div><div class="liczba roz duza">{A.gramy(S.masa_g)}{gw}<small> g</small></div></div>
    <div><div class="etyk">dla porównania: masa teraz (nnU-Net + AC)</div><div class="liczba cyan">{teraz}</div></div>
  </div>
  <ul class="zastrz">
    <li>Polski model (Sano Kraków + WUM): <b>3D ResNet-18</b> z pracy TabAttention (MICCAI 2023), <b>publiczny checkpoint</b>.</li>
    <li>Szacuje masę <b>w dniu porodu</b> z samego wideo — uczył się na badaniach <b>&lt; 24 h przed porodem</b>
    (ciąże donoszone, 38,3 ± 1,5 tc., masa ur. {A.gramy(A.MASA_UR_UCZACA).replace(" ", "&nbsp;")} ± {A.SD_UCZACA} g).</li>
    <li><b>Poza domeną:</b> uczył się na <b>&lt; 24 h przed porodem</b> (~{A.gramy(A.MASA_UR_UCZACA)} g średnio); na nagraniach <b>28–34 tc.</b> zwraca ~3,6–4,0 kg prawie niezależnie od płodu (np. Nr 3 z najmniejszym AC → najwyższy wynik). To ilustracja, <b>nie prognoza</b>.</li>
    <li>W pracy: wariant <b>obraz + 6 pomiarów lekarza</b> — błąd <b>170 g</b> (lepiej niż klinicyści <b>205 g</b>);
    pokazany tu model obrazowy: <b>289 g</b> — kierunek jest jasny: AI <b>wzmacnia</b> pomiary lekarza.</li>
    <li><b>Uwaga:</b> mas urodzeniowych tych dzieci nie znamy — nie ma tu walidacji na naszym materiale.</li>
  </ul>
  {przypis}{zywo}
  <div class="maly">Sano ogląda {model_sano.KLATKI} klatek z brzuchem, które wybrał nnU-Net ({n_s} {A.odmiana(n_s, 'segment', 'segmenty', 'segmentów')}).</div>"""
    return f"""
<div class="karta k-roz">
  <div class="karta-eyebrow">3 · {tytul}</div>
  {tresc}
</div>"""


def _czas_na_zywo(zrodlo: str | None, z_wczytaniem: bool) -> str:
    """„To nagranie (840 klatek): ok. 25 s.” — szacunek z pomiarów na M1 Pro (A.szacowany_czas_na_zywo_s)."""
    try:
        n = len(A.wczytaj_sweep(zrodlo).klatki) if zrodlo else 840
    except Exception:   # noqa: BLE001 — to tylko podpowiedź w pasku stanu
        return ""
    t = A.szacowany_czas_na_zywo_s(n, z_wczytaniem)
    return f"Liczenie tego nagrania ({n} {A.odmiana(n, 'klatka', 'klatki', 'klatek')}): ok. {max(round(t), 1)} s."


def html_status(a: A.Analiza | None, tekst: str | None = None) -> str:
    if tekst is not None:
        return f'<div class="status">{tekst}</div>'
    if a is None:
        return '<div class="status">Wybierz nagranie i kliknij przycisk.</div>'
    n = len(a.nnunet.indeksy)
    if a.tryb == A.Z_PAMIECI:
        return (f'<div class="status"><b>Z pamięci</b>: wyniki policzone wcześniej pełnym nnU-Netem '
                f'({n} {A.odmiana(n, "klatka", "klatki", "klatek")}, co {a.krok}. z {a.n_klatek}).</div>')
    c = a.czasy
    return (f'<div class="status"><b>Na żywo</b>: {A.po_polsku(c.get("razem", 0), 0)} s '
            f'(nnU-Net {A.po_polsku(c.get("nnunet", 0), 0)} s · FUVAI {A.po_polsku(c.get("fuvai", 0), 0)} s · '
            f'Sano {A.po_polsku(c.get("sano", 0))} s'
            + (f' · wczytanie modeli {A.po_polsku(c["wczytanie"])} s' if c.get("wczytanie", 0) > 0.5 else "")
            + f'); {n} {A.odmiana(n, "klatka", "klatki", "klatek")}, co {a.krok}. z {a.n_klatek}.</div>')


def html_zrodla() -> str:
    return """
<div class="zrodla">
<ol>
  <li><b>Nagranie</b>: 840 klatek = 6 przeciągnięć głowicą „na ślepo” (tania głowica, osoba bez doświadczenia w USG).
  Nikt nie zaznaczył, gdzie jest brzuch.</li>
  <li><b>nnU-Net</b> (rozwiązanie bazowe konkursu ACOUSLIC-AI, Radboudumc) zaznacza brzuch w każdej oglądanej klatce
  (z pamięci co 2., na żywo co 4.). Bierzemy klatkę z największym polem brzucha.</li>
  <li>Do obrysu dopasowujemy elipsę; jej obwód to <b>AC</b>. Z AC: masa „teraz” wzorem Campbella i Wilkina (1975)
  i wiek ciążowy odpowiadający AC wzorem Hadlocka (1984). Jedna miara to przybliżenie.</li>
  <li><b>FUVAI</b> (Płotka i in., <i>Physics in Medicine &amp; Biology</i> 2022; Sano + WUM) sam rozpoznaje głowę, brzuch,
  kość udową lub tło. Autorzy mierzą tylko przy pewności &gt; 0,9; na nagraniach „na ślepo” prawie się to nie zdarza.
  Wybór klatki to nasza adaptacja; gdy FUVAI nie jest pewny, a nnU-Net nie widzi brzucha, masy nie podajemy.</li>
  <li><b>Model Sano</b>: 3D ResNet-18 z pracy <i>TabAttention</i> (Grzeszczyk, Płotka i in., MICCAI 2023), 16 klatek brzucha
  → masa w dniu porodu. Uczony na badaniach &lt; 24 h przed porodem (ciąże donoszone), więc na tych nagraniach (ok. 28–34 tc.)
  wychodzi poza dane, na których się uczył. W pracy autorów sam obraz (ten model) mylił się średnio o 289 g, bardziej niż
  klinicyści z wzorem Hadlocka (205 g); najlepszy był model <b>obraz + pomiary lekarza</b> (pełny TabAttention: 170 g).
  AI nie zastępuje pomiarów lekarza, tylko je wzmacnia. Pokazujemy tu wyłącznie model bazowy obrazowy (289 g) — tylko jego
  checkpoint jest publiczny; wag pełnego modelu (170 g) nie ma publicznie.</li>
  <li><b>BabyNet / BabyNet++</b> (Płotka i in., MICCAI 2022, CIBM 2023): kod publiczny, ale wytrenowanych wag nie ma publicznie.</li>
  <li><b>Dane</b>: ACOUSLIC-AI (Radboudumc, Zenodo 12697994), licencja CC BY-NC-SA 4.0, tylko do użytku niekomercyjnego.
  Kod i wagi nnU-Netu: Apache-2.0. Kodu ani wag zespołu Sano nie rozpowszechniamy.</li>
</ol>
<p class="dzieki">Dziękujemy zespołom Sano i WUM za publicznie udostępnione prace, kod i wagi.</p>
<p class="maly">""" + e(ZASTRZEZENIE) + "</p></div>"


# ── logika zdarzeń ───────────────────────────────────────────────────────────

def _opis_klatki(sweep: dane.Sweep, i: int, opis: str) -> str:
    tekst = f"<b>Klatka {i}</b> z {len(sweep.klatki)}"
    nr = A.przeciagniecie(i, len(sweep.klatki))
    if nr:
        tekst += f" · przeciągnięcie głowicą nr {nr} z 6"
    if opis:
        tekst += f" · <span class='cyan'>{e(opis)}</span>"
    return tekst


def pokaz_klatke(zrodlo: str | None, i: int, analiza: A.Analiza | None):
    if not zrodlo:
        return None, html_podpis("Wybierz nagranie.")
    sweep = A.wczytaj_sweep(zrodlo)
    obraz, opis = A.obraz_klatki(sweep, int(i), analiza, atrybucja=A.wbudowane(zrodlo))
    return obraz, html_podpis(_opis_klatki(sweep, int(i), opis), atrybucja=A.atrybucja(zrodlo))


def _puste_wyniki():
    return (None, html_podpis("Tu pojawi się klatka wybrana przez model."), None,
            karta_nnunet(None), karta_fuvai(None), karta_sano(None), None, html_podpis("Klatka FUVAI."))


def _poczatek(zrodlo: str, sweep: dane.Sweep):
    """Wartości dla [zrodlo, analiza, suwak, podglad, podglad_podpis]: środkowa klatka, bez analizy."""
    srodek = len(sweep.klatki) // 2
    obraz, opis = A.obraz_klatki(sweep, srodek, None, atrybucja=A.wbudowane(zrodlo))
    suwak = gr.Slider(minimum=0, maximum=len(sweep.klatki) - 1, value=srodek, step=1)
    return (zrodlo, None, suwak, obraz, html_podpis(_opis_klatki(sweep, srodek, opis), atrybucja=A.atrybucja(zrodlo)))


def wybierz_nagranie(etykieta: str | None):
    """Zmiana nagrania: wczytaj, pokaż środkową klatkę, wyczyść wyniki."""
    if not etykieta:
        return (gr.skip(),) * 5 + _puste_wyniki()
    return _poczatek(etykieta, A.wczytaj_sweep(etykieta)) + _puste_wyniki()


def wgraj_plik(plik: str | None):
    """Własny plik .mha: sprawdzamy go, zanim cokolwiek pokażemy; błąd = krótki komunikat po polsku."""
    if not plik:
        return (gr.skip(),) * 6 + _puste_wyniki()
    try:
        sweep = A.sprawdz_plik(plik)
    except Exception:   # noqa: BLE001 — każdy błąd wczytania kończy się tym samym komunikatem
        log.exception("Nie wczytano wgranego pliku %s", plik)
        raise gr.Error(BLAD_PLIKU.format(Path(plik).name), title="Błąd", print_exception=False) from None
    return _poczatek(plik, sweep) + (gr.Radio(value=None),) + _puste_wyniki()


def analizuj_klik(zrodlo: str | None, tryb: str, progress=gr.Progress()):
    """Główny przycisk. Wyjścia: analiza, suwak, podgląd, podpis, wybrana, jej podpis, wykres, 3 karty, FUVAI, podpis, status."""
    if not zrodlo:
        raise gr.Error("Najpierw wybierz nagranie.", title="Uwaga", print_exception=False)
    try:
        a = A.analizuj(zrodlo, tryb, postep=lambda u, opis: progress(u, desc=opis))
    except A.BrakWynikowWPamieci:
        gr.Warning("Dla tego nagrania nie ma wyników policzonych wcześniej. Przełącz na „na żywo”.", title="Uwaga")
        return (None, gr.skip(), gr.skip(), gr.skip()) + _puste_wyniki() + (
            html_status(None, "Brak wyników w pamięci dla tego pliku. Przełącz tryb na <b>na żywo</b>."),)
    except Exception:   # noqa: BLE001 — na scenie żadnych tekstów z Pythona
        log.exception("Analiza %s w trybie %s nie powiodła się", zrodlo, tryb)
        komunikat = (BLAD_NA_ZYWO if tryb == A.NA_ZYWO
                     else "Nie udało się wczytać wyników z pamięci. Szczegóły są w terminalu.")
        raise gr.Error(komunikat, title="Błąd", print_exception=False) from None
    sweep = A.wczytaj_sweep(zrodlo)
    zrodlo_danych = A.atrybucja(zrodlo)
    k = a.nnunet.najlepsza_klatka
    i = k if k >= 0 else len(sweep.klatki) // 2
    podglad, opis = A.obraz_klatki(sweep, i, a, atrybucja=a.wbudowane)
    wybrana = A.obraz_wybranej(sweep, a)
    opis_wybranej = (f"<b>Klatka {k}</b>: brzuch wybrany przez nnU-Net"
                     + (" (FUVAI wybrał tę samą)" if a.fuvai.najlepsza_klatka == k else "") if k >= 0
                     else "nnU-Net nie znalazł brzucha w tym nagraniu.")
    fv = A.obraz_fuvai(sweep, a)
    opis_fuvai = (f"<b>Klatka {a.fuvai.najlepsza_klatka}</b>: wybór FUVAI" if fv is not None
                  else "FUVAI nie wybrał klatki.")
    legenda = LEGENDA_NNUNET + (LEGENDA_FUVAI if a.fuvai.najlepsza_klatka == k >= 0 else "")
    return (a, gr.Slider(value=i), podglad, html_podpis(_opis_klatki(sweep, i, opis), atrybucja=zrodlo_danych),
            wybrana, html_podpis(opis_wybranej, legenda, zrodlo_danych),
            A.wykres(a, i),
            karta_nnunet(a), karta_fuvai(a), karta_sano(a),
            fv, html_podpis(opis_fuvai, LEGENDA_FUVAI, zrodlo_danych), html_status(a))


def odswiez_wykres(analiza: A.Analiza | None, i: int):
    return A.wykres(analiza, int(i)) if analiza is not None else gr.skip()


CAVEAT_FUVAI = (
    '<div class="uwaga-fuvai">FUVAI był uczone na nagraniach szpitalnych w płaszczyznach standardowych (WUM, aparaty GE). '
    'Nagrania ACOUSLIC-AI powstają „na ślepo” tanią głowicą — to <b>inna domena</b>. '
    'Poniższy podgląd pokazuje, co model widzi w każdej klatce; to materiał edukacyjny, nie pomiar kliniczny.</div>'
)


def html_status_fuvai(
    klatki: fuvai.WynikKlatekFUVAI | None,
    tryb: str,
    tekst: str | None = None,
    *,
    n_klatek: int | None = None,
) -> str:
    if tekst:
        return f'<div class="status">{tekst}</div>'
    if klatki is None:
        return ('<div class="status">Zakładka FUVAI: wybierz nagranie u góry i kliknij '
                '<b>„Załaduj klasyfikację klatek”</b>.</div>')
    n = n_klatek if n_klatek is not None else len(klatki.indeksy)
    if tryb == A.Z_PAMIECI:
        zrodlo = "z pamięci (plik _fuvai_klatki.npz albo wynik rozszerzony z _fuvai.npz)"
    else:
        zrodlo = f"na żywo ({A.po_polsku(klatki.czas_s, 1)} s)"
    bez_masek = ""
    if klatki.mapy_prob is None:
        bez_masek = (
            ' <span class="maly">Bez map segmentacji — obrys i ramka nie będą widoczne '
            "(uruchom przygotuj_demo.py albo tryb na żywo).</span>"
        )
    liczba_sekcji = len(fuvai.sekcje_argmax(klatki, n))
    return (
        f'<div class="status"><b>FUVAI sekcja po sekcji</b> · {zrodlo} · '
        f'{liczba_sekcji} {A.odmiana(liczba_sekcji, "sekcja", "sekcje", "sekcji")} '
        f"(ciągłe fragmenty z tą samą częścią ciała).{bez_masek}</div>"
    )


WYBOR_KLAS_FUVAI = [("Wszystkie sekcje", -1)] + [(f"Sekcje: {n}", i) for i, n in enumerate(fuvai.KLASY)]


def _puste_fuvai_wyjscia(srodek: int = 420, max_i: int = 839):
    return (
        None,
        html_status_fuvai(None, A.Z_PAMIECI),
        None,
        html_podpis("Załaduj klasyfikację FUVAI."),
        None,
        None,
        gr.Slider(minimum=0, maximum=max_i, value=srodek, step=1),
        "<p class=\"maly\">Tabela pojawi się po załadowaniu.</p>",
        -1,
        gr.Radio(choices=WYBOR_KLAS_FUVAI, value=-1),
    )


def _model_fuvai_na_zywo(tryb: str):
    return A._model("fuvai") if tryb == A.NA_ZYWO else None


def zaladuj_fuvai_klatki(zrodlo: str | None, tryb: str, progress=gr.Progress()):
    if not zrodlo:
        raise gr.Error("Najpierw wybierz nagranie.", title="Uwaga", print_exception=False)
    try:
        klatki = A.wczytaj_fuvai_klatki(
            zrodlo, tryb, postep=lambda u, opis: progress(u, desc=opis), doladuj_mapy=True,
        )
    except A.BrakWynikowWPamieci:
        raise gr.Error("Brak wyników w pamięci. Uruchom scripts/przygotuj_demo.py albo przełącz na „na żywo”.",
                        title="Uwaga", print_exception=False) from None
    except Exception:   # noqa: BLE001
        log.exception("FUVAI klatki %s tryb %s", zrodlo, tryb)
        raise gr.Error(BLAD_NA_ZYWO if tryb == A.NA_ZYWO else "Nie udało się wczytać danych FUVAI.",
                        title="Błąd", print_exception=False) from None
    sweep = A.wczytaj_sweep(zrodlo)
    n = len(sweep.klatki)
    stats = fuvai.statystyki_klas(klatki, n)
    tabela = A.html_tabela_klas_fuvai(stats)
    i = len(sweep.klatki) // 2
    obraz, opis = A.obraz_fuvai_podglad(
        sweep,
        klatki,
        i,
        model=_model_fuvai_na_zywo(tryb),
        atrybucja_nagrania=A.wbudowane(zrodlo),
        ramka=True,
    )
    return (
        klatki,
        html_status_fuvai(klatki, tryb, n_klatek=n),
        obraz,
        html_podpis(f"{A.opis_sekcji_fuvai(klatki, n, i)} · {opis}", atrybucja=A.atrybucja(zrodlo)),
        A.wykres_prawdopodobienstw_fuvai(klatki, i),
        A.pasek_timeline_fuvai(klatki, n, i),
        gr.Slider(minimum=0, maximum=n - 1, value=i, step=1),
        tabela,
        -1,
        gr.Radio(choices=WYBOR_KLAS_FUVAI, value=-1),
    )


def pokaz_fuvai_klatke(
    zrodlo: str | None,
    i: int,
    klatki: fuvai.WynikKlatekFUVAI | None,
    tryb: str,
    ramka: bool,
    elipsa: bool,
):
    if not zrodlo or klatki is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()
    sweep = A.wczytaj_sweep(zrodlo)
    i = int(i)
    obraz, opis = A.obraz_fuvai_podglad(
        sweep,
        klatki,
        i,
        model=_model_fuvai_na_zywo(tryb),
        atrybucja_nagrania=A.wbudowane(zrodlo),
        ramka=ramka,
        elipsa_ac=elipsa,
    )
    podpis = f"{A.opis_sekcji_fuvai(klatki, len(sweep.klatki), i)} · {opis}"
    return (
        obraz,
        html_podpis(podpis, atrybucja=A.atrybucja(zrodlo)),
        A.wykres_prawdopodobienstw_fuvai(klatki, i),
        A.pasek_timeline_fuvai(klatki, len(sweep.klatki), i),
    )


def zmien_filtr_klasy_fuvai(
    zrodlo: str | None,
    filtr: int,
    klatki: fuvai.WynikKlatekFUVAI | None,
    tryb: str,
    ramka: bool,
    elipsa: bool,
):
    if not zrodlo or klatki is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    sweep = A.wczytaj_sweep(zrodlo)
    n = len(sweep.klatki)
    i = A.pierwsza_klatka_klasy(klatki, n, int(filtr))
    obraz, opis = A.obraz_fuvai_podglad(
        sweep, klatki, i, model=_model_fuvai_na_zywo(tryb), atrybucja_nagrania=A.wbudowane(zrodlo),
        ramka=ramka, elipsa_ac=elipsa,
    )
    return (
        int(filtr),
        i,
        obraz,
        html_podpis(f"{A.opis_sekcji_fuvai(klatki, n, i)} · {opis}", atrybucja=A.atrybucja(zrodlo)),
        A.wykres_prawdopodobienstw_fuvai(klatki, i),
        A.pasek_timeline_fuvai(klatki, n, i),
    )


def krok_klatki_fuvai(
    zrodlo: str | None,
    i: int,
    klatki: fuvai.WynikKlatekFUVAI | None,
    filtr: int,
    krok: int,
    tryb: str,
    ramka: bool,
    elipsa: bool,
):
    if not zrodlo or klatki is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    sweep = A.wczytaj_sweep(zrodlo)
    n = len(sweep.klatki)
    i2 = A.nastepna_klatka_w_filtrze(klatki, n, int(i), int(filtr), krok=krok)
    obraz, opis = A.obraz_fuvai_podglad(
        sweep, klatki, i2, model=_model_fuvai_na_zywo(tryb), atrybucja_nagrania=A.wbudowane(zrodlo),
        ramka=ramka, elipsa_ac=elipsa,
    )
    return (
        i2,
        obraz,
        html_podpis(f"{A.opis_sekcji_fuvai(klatki, n, i2)} · {opis}", atrybucja=A.atrybucja(zrodlo)),
        A.wykres_prawdopodobienstw_fuvai(klatki, i2),
        A.pasek_timeline_fuvai(klatki, n, i2),
    )


def krok_fuvai_poprzednia(
    zrodlo, i, klatki, filtr, tryb, ramka, elipsa,
):
    return krok_klatki_fuvai(zrodlo, i, klatki, filtr, -1, tryb, ramka, elipsa)


def krok_fuvai_nastepna(
    zrodlo, i, klatki, filtr, tryb, ramka, elipsa,
):
    return krok_klatki_fuvai(zrodlo, i, klatki, filtr, 1, tryb, ramka, elipsa)


def nastepna_klatka_fuvai(zrodlo: str | None, i: int, klatki: fuvai.WynikKlatekFUVAI | None, filtr: int):
    if not zrodlo or klatki is None:
        return gr.skip()
    n = len(A.wczytaj_sweep(zrodlo).klatki)
    return A.nastepna_klatka_w_filtrze(
        klatki, n, int(i), int(filtr if filtr is not None else -1), krok=1
    )


# Wczytywanie modeli w tle (po przełączeniu na „na żywo”); zegar w interfejsie sprawdza, czy się skończyło.
_ROZGRZEWANIE: dict[str, object] = {"watek": None, "blad": None, "czas": None}
_BLOKADA_ROZGRZEWANIA = threading.Lock()


def _rozgrzej() -> None:
    try:
        _ROZGRZEWANIE["czas"] = A.rozgrzej()
    except Exception as blad:   # noqa: BLE001
        log.exception("Wczytywanie modeli do trybu „na żywo” nie powiodło się")
        _ROZGRZEWANIE["blad"] = blad


def rozgrzej_w_tle() -> bool:
    """Startuje wczytywanie modeli, jeśli trzeba. True = wczytywanie trwa."""
    with _BLOKADA_ROZGRZEWANIA:
        watek = _ROZGRZEWANIE["watek"]
        if watek is not None and watek.is_alive():
            return True
        if A.modele_wczytane():
            return False
        _ROZGRZEWANIE.update(blad=None, czas=None)
        watek = threading.Thread(target=_rozgrzej, daemon=True, name="rozgrzewanie-modeli")
        _ROZGRZEWANIE["watek"] = watek
        watek.start()
        return True


def _status_modeli_gotowych(zrodlo: str | None) -> str:
    return html_status(None, f"Tryb <b>na żywo</b>: modele gotowe. {_czas_na_zywo(zrodlo, False)}")


def zmien_tryb(tryb: str, zrodlo: str | None = None):
    """Zwraca (status, zegar sprawdzający wczytywanie modeli)."""
    if tryb != A.NA_ZYWO:
        return (html_status(None, "Tryb <b>z pamięci</b>: wyniki policzone wcześniej, od razu."),
                gr.Timer(active=False))
    if rozgrzej_w_tle():
        return (html_status(None, "Tryb <b>na żywo</b>: wczytuję modele w tle… " + _czas_na_zywo(zrodlo, True)),
                gr.Timer(active=True))
    return _status_modeli_gotowych(zrodlo), gr.Timer(active=False)


def sprawdz_modele(tryb: str, zrodlo: str | None):
    """Takt zegara: gdy modele się wczytały (albo nie), podmień pasek stanu i wyłącz zegar."""
    watek = _ROZGRZEWANIE["watek"]
    if watek is not None and watek.is_alive():
        return gr.skip(), gr.skip()
    if tryb != A.NA_ZYWO:
        return gr.skip(), gr.Timer(active=False)
    if _ROZGRZEWANIE["blad"] is not None:
        return html_status(None, BLAD_MODELI), gr.Timer(active=False)
    return _status_modeli_gotowych(zrodlo), gr.Timer(active=False)


def przelacz_odtwarzanie(gra: bool):
    gra = not gra
    return gra, gr.Timer(active=gra), gr.Button(value="⏸ zatrzymaj" if gra else "▶ odtwórz")


def nastepna_klatka(zrodlo: str | None, i: int):
    if not zrodlo:
        return gr.skip()
    n = len(A.wczytaj_sweep(zrodlo).klatki)
    return (int(i) + KROK_ODTWARZANIA) % n


# ── interfejs ────────────────────────────────────────────────────────────────

def motyw() -> gr.themes.Base:
    """Ciemny motyw jak slajdy; czcionki systemowe (bez Google Fonts)."""
    systemowa = [gr.themes.Font("Helvetica Neue"), gr.themes.Font("Arial"), "sans-serif"]
    return gr.themes.Base(primary_hue="cyan", secondary_hue="orange", neutral_hue="slate",
                          text_size=gr.themes.sizes.text_lg, font=systemowa,
                          font_mono=[gr.themes.Font("Menlo"), "monospace"]).set(
        body_background_fill_dark="#0A0A14",
        body_text_color_dark="#EDEDF4",
        block_background_fill_dark="rgba(255,255,255,0.035)",
        block_border_color_dark="rgba(255,255,255,0.10)",
        block_label_background_fill_dark="rgba(255,255,255,0.06)",
        block_label_text_color_dark="#C8C8D8",
        block_title_text_color_dark="#C8C8D8",
        border_color_primary_dark="rgba(255,255,255,0.14)",
        input_background_fill_dark="rgba(255,255,255,0.06)",
        button_primary_background_fill_dark="linear-gradient(95deg, #FFAF7B 0%, #D76D77 100%)",
        button_primary_background_fill_hover_dark="linear-gradient(95deg, #FFC08F 0%, #E27E88 100%)",
        button_primary_text_color_dark="#0A0A14",
        button_secondary_background_fill_dark="rgba(255,255,255,0.08)",
        button_secondary_text_color_dark="#EDEDF4",
        slider_color_dark="#00BCDE",
        color_accent_soft_dark="rgba(0,188,222,0.18)",
        checkbox_label_background_fill_selected_dark="rgba(0,188,222,0.22)",
        checkbox_label_border_color_selected_dark="#00BCDE",
    )


def zbuduj_interfejs() -> gr.Blocks:
    nagrania = A.dostepne_nagrania()
    start = nagrania[0] if nagrania else None
    with gr.Blocks(title=TYTUL, analytics_enabled=False, fill_width=True,
                   delete_cache=(1800, 3600)) as demo:
        zrodlo = gr.State(None)
        analiza = gr.State(None)
        gra = gr.State(False)
        zegar = gr.Timer(TAKT_ODTWARZANIA_S, active=False)
        zegar_modeli = gr.Timer(1.0, active=False)

        gr.HTML(html_naglowek())
        with gr.Row(elem_id="sterowanie-wspolne", equal_height=True):
            wybor = gr.Radio(nagrania, value=None, label="Nagranie", show_label=False, container=False,
                             elem_id="wybor-nagrania")
            tryb = gr.Radio(TRYBY, value=A.Z_PAMIECI, label="Tryb", show_label=False, container=False, elem_id="tryb")
        fuvai_klatki = gr.State(None)
        filtr_klasy_f = gr.State(-1)
        gra_f = gr.State(False)
        zegar_f = gr.Timer(TAKT_ODTWARZANIA_S, active=False)
        with gr.Tabs(elem_id="zakladki"):
            with gr.TabItem("Masa płodu — nnU-Net + FUVAI", elem_id="tab-masa"):
                with gr.Row(equal_height=False, elem_id="glowny"):
                    with gr.Column(scale=5, min_width=640, elem_id="lewa"):
                        przycisk = gr.Button("Znajdź brzuch i zmierz", variant="primary", size="lg",
                                             min_width=280, elem_id="przycisk")
                        podglad = gr.Image(label="Nagranie USG", interactive=False, format="jpeg", height=320,
                                           buttons=["fullscreen"], elem_id="podglad", show_label=False)
                        podglad_podpis = gr.HTML(html_podpis("Wybierz nagranie."))
                        with gr.Row(elem_id="suwak-rzad", equal_height=True):
                            suwak = gr.Slider(0, 839, value=420, step=1, label="Klatka nagrania", scale=5,
                                              elem_id="suwak")
                            graj = gr.Button("▶ odtwórz", size="md", scale=1, min_width=130, elem_id="graj")
                        wykres = gr.Image(label="Pole brzucha w kolejnych klatkach", interactive=False, format="png",
                                          height=200, buttons=[], show_label=False, elem_id="wykres")
                        status = gr.HTML(html_status(None))
                    with gr.Column(scale=7, min_width=920, elem_id="prawa"):
                        with gr.Row(equal_height=False, elem_id="wyniki-pary"):
                            with gr.Column(scale=1, min_width=400, elem_id="panel-nnunet"):
                                gr.HTML('<div class="panel-tytul panel-cyan">nnU-Net · masa „teraz”</div>')
                                wybrana = gr.Image(label="Klatka nnU-Net", interactive=False, format="jpeg", height=200,
                                                   buttons=["fullscreen"], show_label=False, elem_id="wybrana")
                                wybrana_podpis = gr.HTML(html_podpis("Klatka nnU-Net z obrysem i elipsą (AC)."))
                                k_nnunet = gr.HTML(karta_nnunet(None))
                            with gr.Column(scale=1, min_width=400, elem_id="panel-fuvai"):
                                gr.HTML('<div class="panel-tytul panel-fiolet">FUVAI · polski, publiczne wagi</div>')
                                obraz_f = gr.Image(label="Klatka FUVAI", interactive=False, format="jpeg", height=200,
                                                   buttons=["fullscreen"], show_label=False, elem_id="obraz-fuvai")
                                podpis_f = gr.HTML(html_podpis("Klatka FUVAI z fioletowym obrysem."))
                                k_fuvai = gr.HTML(karta_fuvai(None))
                        k_sano = gr.HTML(karta_sano(None))
            with gr.TabItem("FUVAI — sekcja po sekcji", elem_id="tab-fuvai"):
                gr.HTML(CAVEAT_FUVAI)
                with gr.Row(elem_id="fuvai-gora"):
                    przycisk_f = gr.Button("Załaduj klasyfikację klatek", variant="secondary", size="lg",
                                           elem_id="przycisk-fuvai")
                    status_f = gr.HTML(html_status_fuvai(None, A.Z_PAMIECI))
                with gr.Row(equal_height=False, elem_id="fuvai-tresc"):
                    with gr.Column(scale=5, min_width=520):
                        fuvai_obraz = gr.Image(interactive=False, format="jpeg", height=300, buttons=["fullscreen"],
                                               elem_id="fuvai-obraz")
                        fuvai_podpis = gr.HTML(html_podpis("Załaduj klasyfikację FUVAI."))
                        with gr.Row(elem_id="suwak-fuvai-rzad"):
                            suwak_f = gr.Slider(0, 839, value=420, step=1, scale=5, elem_id="suwak-fuvai")
                            graj_f = gr.Button("▶ odtwórz", size="md", min_width=130, elem_id="graj-fuvai")
                        timeline_f = gr.Image(interactive=False, format="png", height=70, buttons=[],
                                              show_label=False, elem_id="timeline-fuvai")
                    with gr.Column(scale=4, min_width=420):
                        fuvai_tabela = gr.HTML(
                            "<p class=\"maly\">Podsumowanie klas — po załadowaniu.</p>",
                            elem_id="fuvai-tabela",
                        )
                        filtr_klasy_radio = gr.Radio(
                            WYBOR_KLAS_FUVAI,
                            value=-1,
                            label="Przeglądaj sekcje (filtrem klasy)",
                            elem_id="fuvai-filtr-radio",
                        )
                        with gr.Row(elem_id="fuvai-filtr-rzad"):
                            fuvai_poprzednia = gr.Button("◀ Poprzednia sekcja", size="sm")
                            fuvai_nastepna = gr.Button("Następna sekcja ▶", size="sm")
                        with gr.Row():
                            pokaz_ramke_f = gr.Checkbox(label="Ramka wokół struktury (bbox maski)", value=True)
                            pokaz_elipse_f = gr.Checkbox(
                                label="Elipsa dopasowana do maski (jak AC)", value=False
                            )
                        fuvai_wykres = gr.Image(interactive=False, format="png", height=220, buttons=[],
                                                show_label=False, elem_id="fuvai-wykres-p")
        with gr.Row():
            with gr.Accordion("Własny plik .mha (format ACOUSLIC-AI)", open=False, elem_id="wgraj"):
                plik = gr.File(file_types=[".mha"], type="filepath", show_label=False, height=90)
        with gr.Accordion("Jak to działa / Źródła", open=False, elem_id="zrodla"):
            gr.HTML(html_zrodla())

        wyniki = [wybrana, wybrana_podpis, wykres, k_nnunet, k_fuvai, k_sano, obraz_f, podpis_f]
        poczatek = [zrodlo, analiza, suwak, podglad, podglad_podpis]

        wyjscia_fuvai = [
            fuvai_klatki,
            status_f,
            fuvai_obraz,
            fuvai_podpis,
            fuvai_wykres,
            timeline_f,
            suwak_f,
            fuvai_tabela,
            filtr_klasy_f,
            filtr_klasy_radio,
        ]

        wybor.change(wybierz_nagranie, [wybor], poczatek + wyniki, queue=False).then(
            lambda: None, None, [fuvai_klatki], queue=False)
        plik.upload(wgraj_plik, [plik], poczatek + [wybor] + wyniki, show_progress="hidden").then(
            lambda: None, None, [fuvai_klatki], queue=False)
        suwak.change(pokaz_klatke, [zrodlo, suwak, analiza], [podglad, podglad_podpis],
                     trigger_mode="always_last", show_progress="hidden")
        suwak.release(odswiez_wykres, [analiza, suwak], [wykres], show_progress="hidden")
        tryb.change(zmien_tryb, [tryb, zrodlo], [status, zegar_modeli], queue=False)
        zegar_modeli.tick(sprawdz_modele, [tryb, zrodlo], [status, zegar_modeli], queue=False)
        graj.click(przelacz_odtwarzanie, [gra], [gra, zegar, graj], queue=False)
        zegar.tick(nastepna_klatka, [zrodlo, suwak], [suwak], show_progress="hidden")
        (przycisk_f.click(lambda: gr.Button(interactive=False, value="Ładuję…"), None, [przycisk_f], queue=False)
         .then(zaladuj_fuvai_klatki, [zrodlo, tryb], wyjscia_fuvai, show_progress="full")
         .then(lambda: gr.Button(interactive=True, value="Załaduj klasyfikację klatek"), None, [przycisk_f],
               queue=False))
        suwak_f.change(
            pokaz_fuvai_klatke,
            [zrodlo, suwak_f, fuvai_klatki, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            trigger_mode="always_last",
            show_progress="hidden",
        )
        pokaz_ramke_f.change(
            pokaz_fuvai_klatke,
            [zrodlo, suwak_f, fuvai_klatki, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            show_progress="hidden",
        )
        pokaz_elipse_f.change(
            pokaz_fuvai_klatke,
            [zrodlo, suwak_f, fuvai_klatki, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            show_progress="hidden",
        )
        filtr_klasy_radio.change(
            zmien_filtr_klasy_fuvai,
            [zrodlo, filtr_klasy_radio, fuvai_klatki, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [filtr_klasy_f, suwak_f, fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            show_progress="hidden",
        )
        fuvai_poprzednia.click(
            krok_fuvai_poprzednia,
            [zrodlo, suwak_f, fuvai_klatki, filtr_klasy_f, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [suwak_f, fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            show_progress="hidden",
        )
        fuvai_nastepna.click(
            krok_fuvai_nastepna,
            [zrodlo, suwak_f, fuvai_klatki, filtr_klasy_f, tryb, pokaz_ramke_f, pokaz_elipse_f],
            [suwak_f, fuvai_obraz, fuvai_podpis, fuvai_wykres, timeline_f],
            show_progress="hidden",
        )
        graj_f.click(przelacz_odtwarzanie, [gra_f], [gra_f, zegar_f, graj_f], queue=False)
        zegar_f.tick(
            nastepna_klatka_fuvai,
            [zrodlo, suwak_f, fuvai_klatki, filtr_klasy_f],
            [suwak_f],
            show_progress="hidden",
        )
        (przycisk.click(lambda: gr.Button(interactive=False, value="Liczę…"), None, [przycisk], queue=False)
         # Bez wspólnego limitu kolejki: „z pamięci” nie czeka za liczeniem „na żywo”;
         # dwa liczenia na żywo i tak idą po kolei (blokada MPS w usg_masa.aplikacja).
         .then(analizuj_klik, [zrodlo, tryb],
               [analiza, suwak, podglad, podglad_podpis] + wyniki + [status],
               show_progress="full", show_progress_on=[wybrana])
         .then(lambda: gr.Button(interactive=True, value="Znajdź brzuch i zmierz"), None, [przycisk], queue=False))
        if start:
            demo.load(lambda: start, None, [wybor], queue=False)
    return demo


# Generator chorób (notebook/demo_model.py) domyślnie startuje na 7860 — USG niech bierze 7861.
DOMYSLNY_PORT_HTTP = 7861


def _port_domyslny_z_env() -> int:
    raw = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    if not raw:
        return DOMYSLNY_PORT_HTTP
    p = int(raw)
    # Wspólna zmienna Gradio (często 7860) — USG niech nie wchodzi w kolizję z modelem chorób.
    if p == 7860:
        return DOMYSLNY_PORT_HTTP
    return p


def _wolny_port_od(pierwszy: int, prob: int = 32) -> int:
    """Pierwszy wolny port TCP na 127.0.0.1 od ``pierwszy`` (włącznie)."""
    for p in range(pierwszy, pierwszy + prob):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", p))
            except OSError:
                continue
            return p
    raise OSError(f"Brak wolnego portu w zakresie {pierwszy}–{pierwszy + prob - 1}")


def _port_serwera(z_cli: int | None, *, sztywny: bool) -> int:
    start = z_cli if z_cli is not None else _port_domyslny_z_env()
    if sztywny:
        return start
    return _wolny_port_od(start)


def _adresy_z_launch(wynik_launch, demo) -> tuple[str, str]:
    lokalny, link = "", ""
    if isinstance(wynik_launch, tuple):
        if len(wynik_launch) >= 2:
            lokalny = (wynik_launch[1] or "").strip()
        if len(wynik_launch) >= 3:
            link = (wynik_launch[2] or "").strip()
    if not lokalny:
        lokalny = (getattr(demo, "local_url", None) or "").strip()
    if not link:
        link = (getattr(demo, "share_url", None) or "").strip()
    return lokalny, link


def _otworz_zakladke(url: str) -> None:
    """Nowa zakładka — niezawodniejsze niż samo inbrowser Gradio."""
    import webbrowser

    def _go() -> None:
        try:
            webbrowser.open(url, new=2)
        except OSError:
            pass

    threading.Timer(1.0, _go).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=TYTUL)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="N",
        help="port HTTP (domyślnie: zmienna GRADIO_SERVER_PORT albo 7861; szukaj wolnego od N)",
    )
    parser.add_argument(
        "--port-sztywny",
        action="store_true",
        help="nie przechodź na kolejny port, gdy N jest zajęty (wtedy błąd startu)",
    )
    parser.add_argument("--bez-przegladarki", action="store_true", help="nie otwieraj przeglądarki")
    parser.add_argument(
        "--bez-linku",
        action="store_true",
        help="tylko lokalnie (share=False); domyślnie tworzy publiczny link gradio.live",
    )
    args = parser.parse_args()
    żądany = args.port if args.port is not None else _port_domyslny_z_env()
    port = _port_serwera(args.port, sztywny=args.port_sztywny)
    print(f"USG demo · port HTTP: {port} (żądany: {żądany})", flush=True)
    share = not args.bez_linku
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)   # bez wpisu o każdym zapytaniu do 127.0.0.1
    demo = zbuduj_interfejs()
    demo.queue(default_concurrency_limit=4)   # liczenie na żywo i tak idzie po jednym (blokada MPS)
    if not args.port_sztywny and port != żądany:
        print(f"Port {żądany} jest zajęty — uruchamiam USG na http://127.0.0.1:{port}", flush=True)
    if share:
        print("\nUruchamiam aplikację USG. Publiczny link gradio.live pojawi się poniżej.", flush=True)
    else:
        print(f"\nUruchamiam aplikację USG lokalnie (--bez-linku) na porcie {port}.", flush=True)
    print("Ctrl+C = serwer się zatrzymuje.\n", flush=True)
    # Gradio (http_server / Node) czyta GRADIO_SERVER_PORT — musi być zgodny z server_port.
    os.environ["GRADIO_SERVER_PORT"] = str(port)
    wynik_launch = demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=False,
        prevent_thread_lock=True,
        theme=motyw(),
        css_paths=[STYL],
        js=JS_CIEMNY,
        footer_links=[],
        ssr_mode=False,
        max_file_size="1gb",
        show_error=True,
    )
    lokalny, link = _adresy_z_launch(wynik_launch, demo)
    if not lokalny and port:
        lokalny = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 64)
    if share and not link:
        print(
            "Nie udało się utworzyć publicznego linku gradio.live "
            "(tunel zablokowany, np. przez antywirusa firmowego). "
            f"Aplikacja działa lokalnie: {lokalny or '(adres w oknie Gradio)'}."
        )
    elif share and link:
        print("LINK DLA SALI:", link)
    else:
        print(f"Aplikacja lokalna: {lokalny or '(adres w oknie Gradio)'}")
    url_qr = (link or lokalny or "").strip()
    if url_qr.startswith("http"):
        from qr_terminal import wypisz_qr_terminal

        wypisz_qr_terminal(url_qr)
    if not args.bez_przegladarki and lokalny.startswith("http"):
        _otworz_zakladke(lokalny)
    print("Ctrl+C = serwer się zatrzymuje.")
    print("=" * 64 + "\n", flush=True)
    demo.block_thread()


if __name__ == "__main__":
    main()
