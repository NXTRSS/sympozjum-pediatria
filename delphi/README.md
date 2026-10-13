# Delphi-2M na pokaz: historia chorób → prognoza

Podprojekt prelekcji „Nowoczesne technologie w pediatrii" (DIL, Wrocław, 28.11.2026).
Notebook `notebook/delphi_2m.ipynb` pokazuje, jak **Delphi** (Shmatko i in., *Nature* 2025) z historii chorób przewiduje następne zdarzenia i generuje dalsze życie token po tokenie. Używa tego samego mechanizmu co demo z nazwami chorób, tylko tokenami są diagnozy ICD-10.

> **Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych.**
> Model trenujemy sami na **danych syntetycznych** autorów, a nie na UK Biobank. Liczby ryzyka nie są prawdziwymi ryzykami.

## Uruchomienie

Wymagany jest zainstalowany `uv`; Python 3.11 uv dociągnie sam.

```bash
cd delphi
uv sync
uv run python scripts/przygotuj_demo.py      # z internetem; ok. 20 min na M1 Pro (trening 2500 kroków na MPS ~18 min + ok. 2 min krzywe ryzyka)
# albo --kroki 5000: trening ok. 36 min, z wykresem przeuczenia
uv run jupyter notebook notebook/delphi_2m.ipynb
```

Po przygotowaniu notebook działa **bez internetu**. Tryb `TRYB = "z_pamieci"` (domyślny) czyta krzywe ryzyka policzone wcześniej. `"na_zywo"` losuje je na sali (N=100: 3,6 s na pacjenta, a komórka notebooka licząca dwóch pacjentów 7,7 s; N=200: 6–11 s zależnie od obciążenia). Przed prelekcją wyłącz Wi-Fi i uruchom notebook od początku do końca, żeby upewnić się, że niczego nie brakuje.

Na nowym komputerze uruchom `przygotuj_demo.py` (albo chociaż `uv run python -c "import matplotlib.pyplot"`) przed wyjściem na salę: pierwszy import matplotlib buduje cache fontów (kilka–kilkanaście sekund).

Notebook generuje skrypt `scripts/zbuduj_notebook.py`, więc zmiany treści wprowadzaj w skrypcie.

## Testy

```bash
uv run pytest                  # szybkie (kilka sekund)
DELPHI_PELNE=1 uv run pytest   # + sieć, krótki trening, wydajność, wykonanie notebooka offline
```

Zmierzone: zestaw szybki to 71 testów PASS i 5 SKIP w ok. 3 s. Zestaw pełny to 76 testów PASS w ok. 46 s.

## Zmierzone czasy (MacBook Pro M1 Pro, 32 GB)

Pomiary robione, gdy w tle działały inne obliczenia, więc w spokojnej chwili bywa szybciej.

| Co | Gdzie | Czas |
|---|---|---|
| trening demo (2500 kroków, domyślnie) | MPS | ~18 min (17,9 min) |
| trening demo (5000 kroków, config autorów; wykres przeuczenia) | MPS | ~36 min (35,8 min) |
| ten sam trening 5000 kroków | CPU | ~1,5–2 h (1,1–1,5 s/krok) |
| `przygotuj_demo.py` po treningu (3 przykłady, N=1000 losowań; samo to można powtórzyć z `--tylko-wyniki`) | CPU | ok. 2 min 10 s (szczyt RAM ok. 10 GB) |
| ranking następnych zdarzeń | CPU | ~0,005 s |
| jedno życie token po tokenie | CPU | ~0,12 s |
| krzywe ryzyka, N = 100 / 200 / 500 losowań | CPU | 3,6 / 6–11 / 17–25 s |
| komórka notebooka z krzywymi (2 pacjentów × N=100), tryb `na_zywo` | CPU | 7,7 s |
| notebook od początku do końca, tryb `z_pamieci` | CPU | ~5 s |
| notebook od początku do końca, tryb `na_zywo` | CPU | ~13 s |

Wnioskowanie liczymy na CPU. Na MPS (torch 2.3.0) `Delphi.generate` się wywraca (`aten::isin` nie jest zaimplementowany), a pojedynczy ranking trwa ~2 s zamiast milisekund.

Przebieg treningu: przy 2500 krokach błąd walidacyjny najlepszy jest w kroku 2000 (12,17) i do końca prawie się nie zmienia (12,18), bez wyraźnego przeuczenia. Przy 5000 krokach najlepszy jest ten sam krok 2000 (12,19), a na końcu błąd rośnie do 13,08, więc przeuczenie widać na wykresie. Dlatego domyślne 2500 kroków daje praktycznie równie dobry model (walidacja 12,17 vs 12,19, różne przebiegi) w połowie czasu.

## Co trafia do gita, a co nie

W gicie jest kod: `src/`, `scripts/`, `tests/`, `pyproject.toml`, `uv.lock`, ten README oraz notebook **z wyczyszczonymi wyjściami**. Poza gitem (plik `delphi/.gitignore`) zostają:

- `zewnetrzne/` – klon kodu autorów razem z ich danymi syntetycznymi (`zewnetrzne/Delphi/data/ukb_simulated_data`), pobierany skryptem w przypiętym commicie;
- `pamiec/` – nasz checkpoint z danych syntetycznych, log treningu i policzone wyniki (krzywe, rankingi, ślady);
- `dane/` – polska ICD-10 (pobierana w przygotowaniu);
- `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.

Nigdy nie commitujemy wag, checkpointów ani danych. Notebook uruchamiany lokalnie ma wyjścia, więc przed commitem trzeba je wyczyścić (albo zbudować od nowa skryptem `scripts/zbuduj_notebook.py`).

## Licencje i źródła

| Zasób | Licencja | Jak używamy |
|---|---|---|
| Kod Delphi, `github.com/gerstung-lab/Delphi` @ `fb72166` | MIT | klonowany skryptem do `zewnetrzne/` (poza gitem) |
| Dane syntetyczne autorów (`data/ukb_simulated_data`, w klonie: `zewnetrzne/Delphi/data/`) | brak jawnej licencji danych | tylko lokalnie, nie redystrybuujemy |
| Wagi Delphi-2M (UK Biobank) | CC BY-NC-ND 4.0, dostęp kontrolowany przez UK Biobank (upload ID 7318) | **nie mamy** |
| Nasz checkpoint z danych syntetycznych | — | `pamiec/` (poza gitem) |
| Polskie nazwy ICD-10 (ICD-10-2008-PL) | CC BY 4.0, Centrum e-Zdrowia, dane.gov.pl zasób 55725 | pobierane w przygotowaniu do `dane/` (poza gitem) |

Polska ICD-10 ma 2050 kategorii 3-znakowych i pokrywa 1248 z 1254 kodów modelu. Bez polskiej nazwy zostają: A97, B98, C42, C86, G14, K64.

Artykuł: Shmatko A., Jung A.W., Gaurav K. i in., „Learning the natural history of human disease with generative transformers", *Nature* 2025, doi:10.1038/s41586-025-09529-3.

## Czego ten pokaz nie potrafi

- **Prawdziwego ryzyka.** Model widział ok. 7 tys. syntetycznych osób zamiast 400 tys. prawdziwych. Liczby z notebooka (ryzyka, porównania „z chorobą" i „bez") pokazują, **jak działa mechanizm**, a nie jak chorują ludzie. Wynik zależy od przebiegu treningu, a różnice poniżej ~1 punktu procentowego to szum losowania.
- **Dzieci.** UK Biobank zapraszał osoby 40–69-letnie (37–73 w chwili rekrutacji). Dzieciństwo model zna z wywiadu dorosłych: tylko ok. 1,2% diagnoz w danych walidacyjnych przypada na wiek poniżej 18 lat, a notebook ostrzega przy wieku < 40.
- **Objawów, urazów, kodów Z.** Słownik to 3-znakowe kody ICD-10 z rozdziałów A–Q.
- **Wyników z artykułu** (AUC 0,76 w UKB, 0,67 w Danii). Do ich odtworzenia potrzebne są dane UK Biobank i duńskie.
