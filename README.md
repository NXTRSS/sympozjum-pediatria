# Trafny po 17 lekarzach, omylny w 83% przypadków

Materiały do prelekcji na sympozjum **„Nowoczesne technologie w pediatrii — od teorii do praktyki"** (Dolnośląska Izba Lekarska, Wrocław).
Kamil Jędryczek, ML Engineer.

Prelekcja pokazuje **od środka, jak działa model autoregresyjny** — budujemy na żywo mały model językowy, który wymyśla nowe łacińskie nazwy chorób, a potem tym samym mechanizmem tłumaczymy halucynacje AI, model Delphi-2M (Nature 2025) i predykcję masy urodzeniowej z wideo USG.

## 🚀 Uruchom model w przeglądarce (także na telefonie)

[![Otwórz w Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NXTRSS/sympozjum-pediatria/blob/main/notebook/model_jezyka_choroby.ipynb)

Nie trzeba nic instalować: kliknij badge, potem `Środowisko wykonawcze` → `Uruchom wszystko`. Dane pobiorą się same, trening trwa kilka minut.

W notatniku można zmienić `ZBIOR` na `"leki"` albo `"dinozaury"` i wytrenować model na innym zbiorze nazw.

## 📂 Co tu jest

| Ścieżka | Zawartość |
|---|---|
| [`notebook/model_jezyka_choroby.ipynb`](notebook/model_jezyka_choroby.ipynb) | Mini model językowy (GRU, poziom liter): dane → trening → generowanie z histogramami rozkładów → aplikacja Gradio |
| [`data/choroby.csv`](data/choroby.csv) | 1465 łacińskich nazw chorób |
| [`data/leki.csv`](data/leki.csv) | 9438 nazw handlowych leków dopuszczonych w Polsce |
| [`prezentacja.html`](prezentacja.html) | Slajdy (otwórz w przeglądarce; `←`/`→` nawigacja, druk do PDF) |
| [`docs/specs/`](docs/specs) | Projekt prelekcji: struktura, źródła, plan awaryjny |

## 💻 Uruchomienie lokalne

```bash
uv sync
uv run jupyter notebook notebook/model_jezyka_choroby.ipynb
```

Wymaga [uv](https://docs.astral.sh/uv/). Na Macu z Apple Silicon trening idzie po GPU (`tensorflow-metal`).

## 📚 Źródła omawiane w prelekcji

- **Delphi-2M** — *Learning the natural history of human disease with generative transformers*, [Nature 2025](https://www.nature.com/articles/s41586-025-09529-3) · [kod](https://github.com/gerstung-lab/Delphi)
- **BabyNet** — *Residual Transformer Module for Birth Weight Prediction on Fetal Ultrasound Video*, MICCAI 2022 ([arXiv](https://arxiv.org/abs/2205.09382)) · [kod](https://github.com/SanoScience/BabyNet) · [BabyNet++](https://github.com/SanoScience/BabyNetPlusPlus)
- **ChatGPT w pediatrii** — [JAMA Pediatrics 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10762631/) (83% błędnych diagnoz) vs [case tethered cord](https://www.today.com/health/mom-chatgpt-diagnosis-pain-rcna101843)
- **Halucynacje w transkrypcji wizyt** — [AP/Whisper 2024](https://www.healthcare-brew.com/stories/2024/11/18/openai-transcription-tool-whisper-hallucinations)
- **MAI-DxO** — [Microsoft 2025](https://microsoft.ai/news/the-path-to-medical-superintelligence/) · [arXiv](https://arxiv.org/pdf/2506.22405)

## ⚖️ Dane i licencje

- `choroby.csv` — nazwy łacińskie z infoboksów polskiej Wikipedii oraz tytułów artykułów łacińskiej Wikipedii (kategoria *Morbi*). Treści Wikimedia, licencja **CC BY-SA 4.0**.
- `leki.csv` — [Rejestr Produktów Leczniczych](https://dane.gov.pl/pl/dataset/397) (CeZ/URPL), otwarte dane publiczne. Wyłącznie produkty do stosowania u ludzi, bez członów dawkowych.
- `dinozaury` — zbiór [junosuarez/dinosaurs](https://github.com/junosuarez/dinosaurs).

Model generuje **nazwy nieistniejące**. Notebook sprawdza każdą wygenerowaną nazwę względem zbioru treningowego i oznacza, czy jest nowa. Nic tu nie jest narzędziem diagnostycznym ani poradą medyczną.
