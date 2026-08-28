# Prelekcja: Sympozjum „Nowoczesne technologie w pediatrii — od teorii do praktyki" (Wrocław)

**Data spisania:** 2026-08-28 · **Prelegent:** Kamil Jędryczek (ML Engineer) · **Czas:** ~45 min · **Publiczność:** lekarze (pediatrzy), bez przygotowania technicznego

## Teza

Żeby mądrze korzystać z AI (i mądrze jej nie ufać), trzeba raz zobaczyć mechanizm autoregresji.
Halucynacje, Delphi-2M i BabyNet to wnioski z tego jednego mechanizmu.

## Struktura (hybryda: hak z „mit vs mechanizm", timing demo z „od dinozaura do diagnozy")

| Czas | Sekcja | Treść |
|---|---|---|
| 0–5' | Zagadka | Case Alex (17 lekarzy → ChatGPT → tethered cord, Today.com 2023) vs JAMA Pediatrics 2024 (83% błędnych diagnoz GPT-3.5). Tło: Gallup 25%, 40 mln pytających o zdrowie, 79% rodziców. **Trening modelu startuje w tle na 1. slajdzie.** |
| 5–20' | Demo | Mini-LM na łacińskich nazwach chorób (Colab, Keras GRU char-level). Rozkład P(następna litera\|kontekst), sampling, interaktywny slajd krok-po-kroku. Przełącznik zbiorów: choroby / leki / dinozaury / miasta. QR #1: Gradio (gradio.live) na telefony sali. QR #2: Colab dla chętnych. |
| 20–28' | Rozwiązanie zagadki | Model zawsze płynnie generuje następny token i nie wie, kiedy nie wie. Geniusz i halucynacja = ten sam proces. Whisper/Nabla wymyślający leki w transkrypcjach (AP 2024). |
| 28–35' | Skalowanie | Delphi-2M (Nature 2025): GPT-2 o 2,2 mln parametrów, tokeny = ICD-10; walidacja 1,9 mln Duńczyków; uczciwie: brak dzieci w danych (UK Biobank 40–70 lat). Rzut oka: MAI-DxO (85,5% vs ~20%). |
| 35–41' | Apel: BabyNet | Hadlock ~7,5–10% błędu, tylko ~66% oszacowań w ±10%, najgorzej przy makrosomii → decyzje o indukcji/cesarce. BabyNet/BabyNet++ (Sano Kraków, MICCAI 2022, CIBM 2023): wideo USG 3 płaszczyzn <24h przed porodem, MAPE 5,1%; open source; FNP MAB + Horizon 2020 (857533). Kontekst <24h = walidacja na znanym ground truth; praca AJOG MFM 2023 rozszerza na 16–38 tż. Wyścig: Samsung×Sonio (FDA), GE SonoLyst, Chiny MICCAI 2025 (MAE 166 g). |
| 41–45' | Zamknięcie | QR z materiałami, Q&A. |

Sekcje „Skalowanie" (MAI-DxO) i częściowo „Whisper" są zbudowane jako cięte w locie przy obsuwie.

## Demo — architektura

- **Jeden notebook Colab (PL)**, ~6–8 komórek, Keras + GRU char-level (rdzeń z `Model_Języka_dinozaury.ipynb`, maksymalnie uproszczony). Ostatnia komórka: **Gradio** (dropdown zbioru, suwak temperatury, „Generuj", opcjonalnie wykres rozkładu) → publiczny link `*.gradio.live` (72h) → QR #1.
- **Wagi 4 zbiorów pretrenowane, wczytywane z GitHuba** — live trenujemy tylko choroby (start w tle od 1. slajdu); przełączanie zbiorów natychmiastowe.
- Zbiory: `choroby.csv` (łacińskie nazwy, ICD-10 PL lub równoważne), `leki.csv` (Rejestr Produktów Leczniczych URPL), `dinozaury` (junosuarez/dinosaurs), `miasta` (z zajęć ALK).

## Prezentacja HTML

- Baza: szata graficzna i infrastruktura z `~/Documents/ALK/prezentacje/Model_Jezyka.html` („cosmic aurora": Inter/JetBrains Mono, palette --grad-*, karty/chipy/buildy, web component `<deck-stage>` 1920×1080 z nawigacją, speaker notes, print-to-PDF).
- Kopiowane slajdy (blok mechanizmu): Section LM, LM Definition, Data, X and Y, RNN Cell, **Inference Animation (interaktywne demo krok-po-kroku)**, Sampling, Section Scale, Scale, Base Model, Thanks.
- Nowe slajdy: tytuł, zagadka (Alex, JAMA), skala zjawiska, QR, rozwiązanie zagadki, Whisper, Delphi-2M ×2, MAI-DxO, sekcja+problem Hadlocka, BabyNet, apel, materiały.
- Slajd „Inference Animation": tablica `GENERATION_STEPS` do podmiany na nazwę choroby wygenerowaną przez Kamila z notebooka (TODO po pierwszym uruchomieniu notebooka).
- Budowa: fragmenty w `slides/` + `assemble.py` → `prezentacja.html`.

## Plan awaryjny

- Wi-Fi ↓ → hotspot z telefonu prelegenta (sala łączy się własnym LTE).
- Colab/Gradio ↓ → nagrany screencast (2 min) + slajd z wygenerowanymi nazwami.
- Trening nie zbiegnie → checkpoint wag „po pełnym treningu".

## Deliverables

1. `prezentacja.html` (deck-stage, aurora) — ten projekt
2. Notebook Colab + repo GitHub (dane + wagi)
3. Aplikacja Gradio (komórka notebooka)
4. Strona/slajd z linkami i QR

## Czego świadomie NIE robimy

Transformer od zera (GRU wystarcza; 1 slajd „to samo, tylko większe"), własny hosting (gradio.live), trening na telefonach uczestników (Colab QR #2 = opcja dla chętnych).

## Otwarte TODO

- [x] Dane: `data/choroby.csv` (1465, Wikimedia CC BY-SA), `data/leki.csv` (9438, URPL)
- [x] Repo publiczne: https://github.com/NXTRSS/sympozjum-pediatria — dane ładowane w Colabie z raw URL (zweryfikowane)
- [ ] Nazwa choroby do slajdu demo (Kamil wygeneruje z notebooka)
- [ ] Obrazki QR (repo/Colab + gradio.live w dniu prelekcji) w miejsce placeholderów w `slides/n05_qr.html` i `slides/n15_materialy.html`
- [ ] Decyzja: czy przywrócić slajdy Transformer/Attention GIF z ALK (dostępne w oryginale)
- [ ] Wagi pretrenowane 4 zbiorów → GitHub
- [ ] Screencast awaryjny
