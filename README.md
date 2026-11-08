# Trafny po 17 lekarzach, omylny w 83% przypadków

Materiały do prelekcji na sympozjum **„Nowoczesne technologie w pediatrii — od teorii do praktyki"** (Dolnośląska Izba Lekarska, Wrocław).
Kamil Jędryczek, ML Engineer.

Prelekcja pokazuje **od środka, jak działa model autoregresyjny** — budujemy na żywo mały model językowy, który wymyśla nowe łacińskie nazwy chorób, a potem tym samym mechanizmem tłumaczymy halucynacje AI, model Delphi-2M (Nature 2025) i predykcję masy urodzeniowej z wideo USG.

## 🚀 Uruchom model w przeglądarce (także na telefonie)

[![Otwórz w Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/notebook/model_jezyka_choroby.ipynb)

Nie trzeba nic instalować: kliknij przycisk *Open in Colab* (potrzebne konto Google), potem `Środowisko wykonawcze` → `Uruchom wszystko`. Dane pobiorą się same, trening trwa kilka minut. Na końcu notatnik uruchamia aplikację Gradio (liczba nazw + „kreatywność"/temperatura) z publicznym linkiem.

W notatniku można zmienić `ZBIOR` na `"leki"` albo `"dinozaury"` i wytrenować model na innym zbiorze nazw.

## 📂 Co tu jest

| Ścieżka | Zawartość |
|---|---|
| [`notebook/model_jezyka_choroby.ipynb`](notebook/model_jezyka_choroby.ipynb) | Mini model językowy (GRU, poziom liter): dane → trening → generowanie z histogramami rozkładów → aplikacja Gradio |
| [`data/choroby.csv`](data/choroby.csv) | 1292 łacińskie nazwy chorób |
| [`data/leki.csv`](data/leki.csv) | 9438 nazw handlowych leków dopuszczonych w Polsce |
| [`assemble.py`](assemble.py), [`slides/`](slides) | Źródła slajdów. `assemble.py` składa `prezentacja.html` (`←`/`→` nawigacja, druk do PDF) z plików `slides/*.html` i szablonu [`public-template/published-deck.html`](public-template/published-deck.html) — wszystko jest w repozytorium, nic nie trzeba dokładać |
| [`prezentacja.html`](prezentacja.html) | Złożona prezentacja (wynik `assemble.py`), gotowa do otwarcia w przeglądarce |
| [`usg/`](usg) | Notebook „USG → masa płodu" (nnU-Net z ACOUSLIC-AI, FUVAI, model bazowy z pracy TabAttention) — uruchamiany lokalnie, nie w Colabie |
| [`delphi/`](delphi) | Notebook Delphi-2M na danych syntetycznych autorów — uruchamiany lokalnie |
| [`docs/specs/`](docs/specs) | Projekt prelekcji: struktura, źródła, plan awaryjny |

## 💻 Uruchomienie lokalne

```bash
uv sync
uv run jupyter notebook notebook/model_jezyka_choroby.ipynb
```

Wymaga [uv](https://docs.astral.sh/uv/). Na Macu z Apple Silicon trening idzie po GPU (`tensorflow-metal`).

Złożenie prezentacji (nadpisuje `prezentacja.html`):

```bash
uv run python assemble.py
```

Szablon decku jest w repozytorium (`public-template/published-deck.html`) — to wcześniej opublikowana wersja tej samej prezentacji, z której bierzemy styl, skrypty i slajd końcowy; treść slajdów zawsze pochodzi z `slides/*.html`. Innego szablonu można użyć przez `SZABLON_DECK=/ścieżka/do/decku.html`.

## 📚 Źródła omawiane w prelekcji

- **Delphi-2M** — *Learning the natural history of human disease with generative transformers*, [Nature 2025](https://www.nature.com/articles/s41586-025-09529-3) · [kod](https://github.com/gerstung-lab/Delphi)
- **BabyNet** — *Residual Transformer Module for Birth Weight Prediction on Fetal Ultrasound Video*, MICCAI 2022 ([arXiv](https://arxiv.org/abs/2205.09382)) · [kod](https://github.com/SanoScience/BabyNet) · [BabyNet++](https://github.com/SanoScience/BabyNetPlusPlus)
- **TabAttention** — *Learning Attention Conditionally on Tabular Data*, MICCAI 2023 ([arXiv](https://arxiv.org/abs/2310.18129)) · [kod](https://github.com/SanoScience/Tab-Attention) — publiczny checkpoint zawiera model bazowy obrazowy (289 g / 8,5% błędu w pracy); wag pełnego modelu z pomiarami biometrycznymi (170 g / 5,0%) nie ma publicznie
- **FUVAI** — Płotka i in., *Physics in Medicine & Biology* 2022 · [kod i wagi](https://github.com/SanoScience/FUVAI)
- **ACOUSLIC-AI** — nagrania USG i rozwiązanie bazowe nnU-Net (Radboudumc), dane: [Zenodo 12697994](https://zenodo.org/records/12697994), **CC BY-NC-SA 4.0**, tylko niekomercyjnie
- **ChatGPT w pediatrii** — [JAMA Pediatrics 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10762631/) (83% błędnych diagnoz) vs [case tethered cord](https://www.today.com/health/mom-chatgpt-diagnosis-pain-rcna101843)
- **Halucynacje w transkrypcji wizyt** — [AP/Whisper 2024](https://www.healthcare-brew.com/stories/2024/11/18/openai-transcription-tool-whisper-hallucinations)
- **MAI-DxO** — [Microsoft 2025](https://microsoft.ai/news/the-path-to-medical-superintelligence/) · [arXiv](https://arxiv.org/pdf/2506.22405)

## ⚖️ Dane i licencje

- `choroby.csv` — nazwy łacińskie z infoboksów polskiej Wikipedii oraz tytułów artykułów łacińskiej Wikipedii (kategoria *Morbi*), oczyszczone ręcznie z pozycji niebędących chorobami (osoby, leki, patogeny, dziedziny medycyny itp.). Treści Wikimedia, licencja **CC BY-SA 4.0**.
- `leki.csv` — [Rejestr Produktów Leczniczych](https://dane.gov.pl/pl/dataset/397) (CeZ/URPL), otwarte dane publiczne. Wyłącznie produkty do stosowania u ludzi, bez członów dawkowych.
- `dinozaury` — zbiór [junosuarez/dinosaurs](https://github.com/junosuarez/dinosaurs).

Model generuje **nazwy nieistniejące**. Notebook sprawdza każdą wygenerowaną nazwę względem zbioru treningowego i oznacza, czy jest nowa. Nic tu nie jest narzędziem diagnostycznym ani poradą medyczną.

## 🔬 Dobór hiperparametrów

Pierwsza wersja modelu (256 jednostek, 120 epok) **odtwarzała nazwy ze zbioru** zamiast wymyślać nowe. Katalog [`eksperymenty/`](eksperymenty) zawiera pomiar, którym dobrano nastawy — mierzone są dwie przeciwstawne rzeczy:

- **świeżość** — odsetek nazw, których nie ma w zbiorze treningowym ani w odległości edycyjnej ≤ 0,2 od którejś z nich,
- **wiarygodność** — średnie prawdopodobieństwo nazwy pod modelem 4-gramowym znaków; 100% = tak samo „łacińska" jak prawdziwe nazwy, więcej = bardziej bełkotliwa.

| Nastawy | Świeże | Wiarygodność |
|---|---|---|
| 256 jedn., 80 epok, T=0.8 (przeuczony) | 10% | 100% |
| 256 jedn., 50 epok, T=0.8 | 46% | 116% |
| 128 jedn., dropout 0.3, 30 epok, T=0.5 (wersja pośrednia) | 93% | 115% |
| 128 jedn., 10 epok, T=1.2 (niedouczony) | 99% | 215% |

Tabela pochodzi z pomiaru na pierwotnej wersji zbioru (1465 nazw). Po oczyszczeniu zbioru do 1292 nazw wybrane nastawy sprawdzono ponownie na 3 ziarnach losowych (`final.py`, T=0.5): 98,7% nazw nowych, 91,3% świeżych, wiarygodność 121,5%, 71,5% z poprawną łacińską końcówką (wcześniej: 98,2% / 92,7% / 114,6% / 75%).

```bash
uv run python eksperymenty/strojenie.py             # przegląd: architektura × epoki × temperatura
uv run python eksperymenty/strojenie2_temperatura.py # doprecyzowanie niskich temperatur
uv run python eksperymenty/final.py                  # weryfikacja zwycięzcy na 3 ziarnach
```

### Auto-optymalizacja pod trening na żywo (październik 2026)

Model uczy się **na sali**, więc drugim kryterium obok jakości nazw był **czas treningu** (mierzony na 2 rdzeniach CPU, jak w darmowym Colabie). Ponad 400 pomiarów (architektura, tempo uczenia, batch, dropout, epoki, temperatura; 3 ziarna losowe) plus **ślepa ocena** 175 nazw w skali 1–5 przez model Claude Opus 5.5, z prawdziwymi nazwami jako kontrolą. Szczegóły: [`eksperymenty/AUTOOPTYMALIZACJA.md`](eksperymenty/AUTOOPTYMALIZACJA.md).

| Nastawy | Czas treningu | Wiarygodność | Nowe | Ocena ślepa (1–5) |
|---|---|---|---|---|
| 128 jedn., dropout 0.3, lr 0,001, 30 epok, T=0.5 (poprzednie) | 36 s | 121% | 98% | 2,60 |
| **128 jedn., dropout 0.2, lr 0,003, 12 epok, T=0.4** ← wybrane | **15 s** | **101%** | **98%** | **2,68** |
| prawdziwe nazwy (kontrola) | — | 100% | — | 4,96 |

```bash
uv run python eksperymenty/autooptymalizacja.py przeglad   # faza 1: szeroka siatka
uv run python eksperymenty/autooptymalizacja.py faza2      # faza 2: zawężenie, 3 ziarna
uv run python eksperymenty/autooptymalizacja.py slepa      # próba do ślepej oceny (finalisci.json)
```
