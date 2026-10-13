"""Generuje notebook/delphi_2m.ipynb z komórek poniżej (jedno źródło prawdy, czysty diff w gicie).

uv run python scripts/zbuduj_notebook.py
Uwaga: ręczne poprawki w Jupyterze nadpisze ponowne uruchomienie; zmiany wprowadzaj tutaj.
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebook" / "delphi_2m.ipynb"

KOMORKI = [
    ("md", """# Delphi-2M: model językowy, którego „słowami" są diagnozy
### Sympozjum „Nowoczesne technologie w pediatrii" · Kamil Jędryczek

W pierwszym notebooku model uczył się, jaka **litera** będzie następna w nazwie choroby. Delphi-2M (Shmatko i in., *Nature* 2025) robi to samo, tylko jego tokenami są **diagnozy ICD-10**, a pozycją w zdaniu jest **wiek**.

> **Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych.**
>
> **Uczciwie:** prawdziwe wagi Delphi-2M (UK Biobank) są dostępne tylko po wniosku do UK Biobank. Tutaj ten sam model (2,24 mln parametrów, kod autorów) trenowaliśmy na **danych syntetycznych**, które opublikowali autorzy. Pokazujemy **mechanizm**. Liczby ryzyka **nie są** prawdziwymi ryzykami dla pacjentów."""),
    ("code", """# ── KONFIGURACJA ──────────────────────────────────────────────
TRYB = "z_pamieci"        # "z_pamieci" = krzywe policzone przed prelekcją; "na_zywo" = liczymy teraz
N_PROBEK_NA_ZYWO = 100    # ile „żyć" losujemy na żywo (100 ≈ 6 s, 200 ≈ 6–11 s na pacjenta; komórka liczy 2 pacjentów)

import json, time, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
from delphi_demo import dane, pamiec, prognoza, przyklady, trening, wykresy, zasoby
from delphi_demo.slownik import BladHistorii, parsuj_historie, wczytaj_slownik
from delphi_demo.tokeny import KOBIETA, MEZCZYZNA, Zdarzenie

def pl(x, cyfry=1):   # liczba z przecinkiem dziesiętnym (po polsku)
    return f"{x:.{cyfry}f}".replace(".", ",")

SL = wczytaj_slownik()
MODEL = trening.zaladuj_model()          # wytrenowany przed prelekcją na danych syntetycznych
print(f"Delphi: {pl(trening.liczba_parametrow(MODEL) / 1e6, 2)} mln parametrów, "
      f"słownik: {len(SL.etykiety)} tokenów (choroby ICD-10, płeć, styl życia, zgon)")"""),
    ("md", """## Od liter do diagnoz

| | nasz model z nazwami chorób | Delphi-2M |
|---|---|---|
| token | litera | kod ICD-10 (np. E11), płeć, BMI, palenie, alkohol |
| „zdanie" | nazwa choroby | historia zdrowia jednej osoby |
| pozycja | numer litery | **wiek** (w dniach) |
| koniec | znak „!" | token **zgonu** |

Poniżej jeden z ~7 tys. „pacjentów" z danych syntetycznych autorów. Nikt taki nie istnieje: wymyślił go wytrenowany Delphi-2M."""),
    ("code", """# ── PACJENT SYNTETYCZNY ───────────────────────────────────────
VAL = dane.wczytaj_bin(zasoby.katalog_danych() / "val.bin")
s = dane.statystyki(VAL)
print(f"Dane syntetyczne: {s['pacjentow']} osób, {s['zdarzen']} wpisów, "
      f"średnio {s['zdarzen_na_pacjenta']:.0f} na osobę (diagnozy i płeć)")
print(f"{pl(100 * s['odsetek_zdarzen_przed_18'])}% diagnoz w danych syntetycznych przypada na wiek poniżej 18 lat. "
      "W UK Biobank choroby z dzieciństwa pochodzą z wywiadu dorosłych.")
PID = dane.pacjenci_z(VAL, SL.kod_do_tokenu["J45"], przed_wiekiem=18, min_zdarzen=15)[0]
wykresy.os_zycia(dane.pacjent(VAL, PID), SL.nazwa, kolor=lambda t: SL.kolor[t],
                 tytul=f"Pacjent syntetyczny nr {PID} (wymyślony przez model autorów)")
plt.show()"""),
    ("md", """## Jak Delphi liczy „co dalej"?

Dla **każdej** z ~1250 chorób (i zgonu) model podaje **tempo**: jak szybko ta choroba „nadchodzi".
Wyobraźmy sobie ~1250 zegarów, które tykają naraz. Każdy ma inne tempo. Który zadzwoni pierwszy, ta choroba będzie następna.

- **ranking** = które zegary tykają najszybciej,
- **ryzyko w ciągu roku** = szansa, że dany zegar zadzwoni w ciągu 12 miesięcy,
- **losowanie życia** = puszczamy zegary, zapisujemy pierwszy dzwonek, dopisujemy go do historii i powtarzamy. To ta sama pętla co przy nazwach chorób.

Najpierw jednak model musiał się nauczyć. Trening (przed prelekcją) wyglądał tak. Patrzcie na dwie linie: błąd na pacjentach treningowych i na pacjentach odłożonych na bok. Dopóki obie spadają, model uczy się reguł. Gdy treningowa spada, a odłożona przestaje spadać albo rośnie, model zaczyna **zapamiętywać** 7 tys. pacjentów. To samo widzieliśmy przy nazwach chorób. Zapisujemy model z najlepszego kroku."""),
    ("code", """# ── JAK SIĘ UCZYŁ ─────────────────────────────────────────────
LOG = pamiec.KATALOG_WYNIKOW / "trening.log"
if LOG.exists():
    _, EWAL = trening.parsuj_log(LOG.read_text())
    czas = json.loads((pamiec.KATALOG_WYNIKOW / "czas_treningu.json").read_text())["czas_s"] / 60
    najlepszy = int(EWAL.loc[EWAL["walidacja"].idxmin(), "krok"])
    wzrost = float(EWAL["walidacja"].iloc[-1] - EWAL["walidacja"].min())
    print(f"{int(EWAL['krok'].iloc[-1])} kroków, {czas:.0f} min; najlepszy model z kroku {najlepszy}.")
    if wzrost > 0.1:
        print(f"Po kroku {najlepszy} błąd walidacyjny rośnie (+{pl(wzrost, 2)}), choć treningowy spada: model zaczął zapamiętywać.")
    elif najlepszy == int(EWAL["krok"].iloc[-1]):
        print("Błąd walidacyjny spadał do końca: model wciąż uczył się reguł i nie zaczął zapamiętywać. "
              "(Przy dłuższym treningu, np. 5000 kroków, widać, jak zaczyna.)")
    else:
        print("Błąd walidacyjny przestał spadać, ale jeszcze nie rośnie: to moment, w którym warto skończyć trening.")
    wykresy.krzywa_uczenia(EWAL, czas)
    plt.show()
else:
    print("Brak logu treningu: uruchom scripts/przygotuj_demo.py")"""),
    ("md", """## Twoja historia → prognoza

Wpisz historię jako **kod ICD-10 (3 znaki) i wiek**, np. `E11 50, I10 55` albo `E11 od 50 r.ż.`.
Model zna tylko rozdziały A–Q ICD-10: nie ma objawów (R), urazów (S–T) ani kodów Z.
Stylu życia (palenie, BMI, alkohol) nie da się wpisać: ma w słowniku swoje tokeny, ale w danych syntetycznych autorów nie występuje ani razu, więc nasz model go nie zna.

W tabeli **„szansa, że to następne”** liczymy tylko spośród chorób i zgonu (bez „nic się nie dzieje”), czyli odpowiada na pytanie „jeśli coś się wydarzy, to co?”; **ryzyko w 5 lat*** zakłada, że tempo z dziś nie zmieni się przez 5 lat."""),
    ("code", """# ── TWOJA HISTORIA → PROGNOZA ─────────────────────────────────
HISTORIA = "E11 50, I10 55"   # np. „J45 6, E66 30, I10 41"
PLEC = "K"                    # "K" albo "M"
WIEK_TERAZ = 60

def pokaz_prognoze(historia_tekst, plec, wiek_teraz, top=10):
    try:
        historia = parsuj_historie(historia_tekst, SL)
        df = SL.opisz(prognoza.ranking(MODEL, historia, plec, wiek_teraz, top=top))
    except (BladHistorii, ValueError) as blad:
        print("UWAGA:", blad)
        return None
    print("INFO: model trenowany na danych SYNTETYCZNYCH; liczby pokazują mechanizm, nie ryzyko kliniczne.")
    for uwaga in prognoza.ostrzezenia(historia, wiek_teraz):
        print("INFO:", uwaga)
    osoba = "Kobieta" if plec == "K" else "Mężczyzna"
    wykresy.ranking_slupki(df, f"{osoba}, {wiek_teraz} lat: co model przewiduje jako następne?")
    plt.show()
    tabela = pd.DataFrame({
        "kod": df["kod"], "choroba": df["nazwa"],
        "szansa, że to następne [%]": (100 * df["p_nastepna"]).map(pl),
        "ryzyko w ciągu roku [%]": (100 * df["ryzyko_1_rok"]).map(pl),
        "ryzyko w 5 lat [%]*": (100 * df["ryzyko_5_lat"]).map(pl),
    })
    tabela.index = range(1, len(tabela) + 1)
    return tabela

pokaz_prognoze(HISTORIA, PLEC, WIEK_TERAZ)"""),
    ("code", """# ── INTERAKTYWNIE ─────────────────────────────────────────────
import ipywidgets as W

W.interact_manual(
    pokaz_prognoze,
    historia_tekst=W.Text(value=HISTORIA, description="Historia:", layout=W.Layout(width="60%")),
    plec=W.ToggleButtons(options=[("kobieta", "K"), ("mężczyzna", "M")], value=PLEC),
    wiek_teraz=W.IntSlider(value=WIEK_TERAZ, min=1, max=90, description="Wiek dziś:"),
    top=W.fixed(10),
);"""),
    ("md", """## Co zmienia jedna diagnoza?

Model losuje wiele możliwych dalszych żyć tej samej osoby (100 na żywo, 1000 w wersji przygotowanej przed prelekcją) i liczy, w ilu z nich pojawił się zawał, niewydolność serca, udar, choroba nerek albo zgon. Linia ciągła: z cukrzycą. Przerywana: ta sama osoba bez cukrzycy."""),
    ("code", """# ── KRZYWE RYZYKA: Z CUKRZYCĄ I BEZ ───────────────────────────
A, B = przyklady.PRZYKLADY["cukrzyca"], przyklady.PRZYKLADY["bez_cukrzycy"]
CELE = [SL.kod_do_tokenu[k] for k in przyklady.CELE]

def krzywe_dla(nazwa, p, cele):   # → (tabela, skąd): z pamięci (policzone przed prelekcją) albo na żywo
    if TRYB == "z_pamieci" and pamiec.jest(f"krzywe_{nazwa}"):
        n = pamiec.wczytaj_json("krzywe_info").get("n_probek") if pamiec.jest("krzywe_info") else None
        return pamiec.wczytaj_tabele(f"krzywe_{nazwa}"), (f"z pamięci, N={n}" if n else "z pamięci")
    df = prognoza.krzywe_ryzyka(MODEL, parsuj_historie(p["historia"], SL), p["plec"],
                                p["wiek_teraz"], cele, n_probek=N_PROBEK_NA_ZYWO)
    skad = f"na żywo, N={N_PROBEK_NA_ZYWO}"
    if TRYB == "z_pamieci":
        skad += "; brak wyników w pamięci, uruchom scripts/przygotuj_demo.py"
    return df, skad

start = time.time()
(KA, ZA), (KB, ZB) = krzywe_dla("cukrzyca", A, CELE), krzywe_dla("bez_cukrzycy", B, CELE)
skad = ZA if ZA == ZB else f"{ZA} / {ZB}"
print(f"{A['opis']}\\nvs {B['opis']}\\n(tryb {TRYB}: {skad}; {pl(time.time() - start)} s)")
wykresy.krzywe(KA, SL.nazwa, KB, etykiety=("z cukrzycą", "bez cukrzycy"))
plt.show()"""),
    ("md", """## Klamra: życie generowane token po tokenie

Ta sama pętla co przy nazwach chorób: model patrzy na dotychczasową historię, podaje kandydatów z prawdopodobieństwami, losujemy jednego, dopisujemy i powtarzamy, aż do zgonu albo 80. roku życia.

W nawiasie: szansa wylosowanego zdarzenia; „kandydaci” to trzy najbardziej prawdopodobne. Obie liczby liczymy jak w rankingu: spośród chorób i zgonu. Wylosowane zdarzenie często nie jest w czołówce: każda z ~1250 chorób ma trochę szansy, a my **losujemy**, a nie wybieramy najbardziej prawdopodobnej. Trzech czołowych kandydatów to razem tylko ok. 10% szansy, a pozostałe ~90% rozkłada się na setki chorób po ułamku procenta. Los najczęściej trafia więc w ten „długi ogon” rzadkich chorób. Dlatego jedno wylosowane życie to przykład, a nie prognoza; prognozą są krzywe z 1000 losowań wyżej."""),
    ("code", """# ── ŻYCIE TOKEN PO TOKENIE ────────────────────────────────────
P = przyklady.PRZYKLADY["astma_od_dziecka"]
H = parsuj_historie(P["historia"], SL)
ZIARNO = 0   # zmień, żeby wylosować inne życie tej samej osoby

if TRYB == "z_pamieci" and ZIARNO == 0 and pamiec.jest("slad_astma_od_dziecka"):
    KROKI = pamiec.wczytaj_kroki("slad_astma_od_dziecka")
else:
    KROKI = prognoza.generuj(MODEL, H, P["plec"], P["wiek_teraz"], ziarno=ZIARNO)
print(P["opis"])
print(wykresy.slad_generowania(KROKI, SL.nazwa, ile=12))
GEN = [Zdarzenie(k.token, k.wiek_lat) for k in KROKI]
PLEC_P = KOBIETA if P["plec"] == "K" else MEZCZYZNA
wykresy.os_zycia([Zdarzenie(PLEC_P, 0.0), *H], SL.nazwa, kolor=lambda t: SL.kolor[t],
                 wygenerowane=GEN, tytul="Historia (pełne kółka) i dalszy ciąg wylosowany przez model (puste)")
plt.show()"""),
    ("md", """## Uczciwie: czego ten pokaz NIE pokazuje

- **Wagi:** prawdziwy Delphi-2M (UK Biobank) jest dostępny tylko po wniosku do UK Biobank (licencja wag CC BY-NC-ND 4.0, tylko niekomercyjnie). Nasz model widział wyłącznie dane syntetyczne autorów, ok. 7 tys. wymyślonych osób zamiast 400 tys. prawdziwych. Autorzy sami piszą, że model z danych syntetycznych służy tylko do testów kodu.
- **Artefakty danych:** osoba z zaledwie dwiema diagnozami w wieku 60 lat wygląda w danych na „zdrowszą niż przeciętna", więc model daje jej bardzo niskie ryzyko zgonu. To cecha danych (kto trafia do szpitala, ten ma wpisy), nie wiedza medyczna.
- **Dzieci:** UK Biobank zapraszał osoby 40–69-letnie (w chwili rekrutacji 37–73 lata). Choroby z dzieciństwa model zna tylko z wywiadu dorosłych. Pediatria czeka na swój Delphi.
- **Nazwy chorób:** polska ICD-10 (Centrum e-Zdrowia, Rejestr Systemów Kodowania, dane.gov.pl, CC BY 4.0); kilkadziesiąt skróciliśmy dla czytelności, 6 kodów bez odpowiednika w ICD-10-2008-PL zostaje po angielsku.
- **Wyniki z artykułu:** średnie AUC ok. 0,76 (UK Biobank), 0,67 w danych duńskich (1,9 mln osób, bez dostrajania).
- Źródło: Shmatko i in., „Learning the natural history of human disease with generative transformers", *Nature* 2025, doi:10.1038/s41586-025-09529-3 · kod: github.com/gerstung-lab/Delphi (MIT)."""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if r == "md" else nbf.v4.new_code_cell(t) for r, t in KOMORKI]
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"Zapisano {NB} ({len(nb.cells)} komórek)")


if __name__ == "__main__":
    main()
