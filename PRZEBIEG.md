# PRZEBIEG — sympozjum 28.11.2026 · DIL Wrocław · 10:45–11:45

Checklist na druk. **Strona PPTX** ≈ numer slajdu w decku + 14 (wyjątek: slajd 9 „Inferencja” = **str. 9–23**, 15 kroków). Plik: `eksport/prezentacja_zapas.pptx`, **62 strony**, 48 slajdów HTML.

**ZAPAS** — pomiń, jeśli demo działa.

## 1. Dzień przed (27.11)

- [ ] `python3 assemble.py --zapas && uv run eksport_raster.py --zapas`
- [ ] Sprawdzić `eksport/prezentacja_zapas.pptx` w PowerPoincie (notatki prelegenta)
- [ ] `.pptx` + `.pdf` (_zapas) na dwa pendrive exFAT; wysłać organizatorom
- [ ] Mac: `python3 assemble.py && uv run eksport_raster.py` (wersja na żywo)

## 2. Rano 28.11

- [ ] `caffeinate -dis`, zasilanie, Wi‑Fi
- [ ] `uv run python notebook/uruchom_demo.py` — **tylko skrypt**, nie Gradio z notebooka (drugi start: trening z cache, szybciej)
- [ ] `uv run python notebook/uruchom_demo.py --bez-linku` gdy antywirus blokuje tunel Gradio (CrowdStrike / frpc)
- [ ] Link `gradio.live` **albo** plan B: QR Colab (**str. 26**) + laptop DIL / podgląd lokalny na ekranie
- [ ] Test QR (**str. 26**) + laptop DIL

## 3. Slajdy i czas (~52 min + ~8 min Q&A)

| Str. PPTX | Deck | Co | min |
|----------:|-----:|----|----:|
| 1–8 | 1–8 | Wstęp, mechanizm LM | 11 |
| **9–23** | 9 | Inferencja — klikaj powoli | 8 |
| 24–25 | 10–11 | Losowanie / temperatura | 3 |
| **26** | 12 | **QR — demo nazw** | **3** |
| **27–34** | 13–20 | **ZAPAS LM** z01–z08 | 0 |
| 35–43 | 21–29 | Skala, zagadka, Whisper, LM w medycynie | 14 |
| 44–45 | 30–31 | Delphi | 4 |
| **46–51** | 32–37 | **ZAPAS Delphi** zd01–zd06 | 0 |
| 52–55 | 35–38 | BabyNet, Hadlock, slajd demo USG | 9 |
| **53–59** | 39–45 | **ZAPAS USG** zu01–zu07 | 0 |
| 60 | 46 | Model Sano (slajd) | 4 |
| 61 | 47 | Materiały | 2 |
| **62** | 48 | Dziękuję + Q&A | 8 |

**Kotwice:** QR **26** · ZAPAS LM **27–34** · ZAPAS Delphi **46–51** · ZAPAS USG **53–59** · Sano **60** · koniec **62**.

## 4. Plan A

- [ ] Chrome: `prezentacja.html`
- [ ] Delphi: notebook, `TRYB = "z_pamieci"`
- [ ] USG: `uv run python usg/app/demo_usg.py` → :7860, Nr 1, z pamięci
- [ ] Jedno liczenie naraz (MPS/GPU)

## 5. Protokół awaryjny

- [ ] Demo zalane → **Ctrl+C** → link martwy → slajdy ZAPAS LM
- [ ] Brak `gradio.live` (tunel zablokowany) → **QR do Colaba** (`model_jezyka_choroby.ipynb`) lub `--bez-linku` i pokaz lokalny na rzutniku
- [ ] Fallback: `prezentacja_zapas.pptx` / `.pdf` na DIL

→ [`URUCHOMIENIE.md`](URUCHOMIENIE.md)