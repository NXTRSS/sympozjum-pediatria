# Auto-optymalizacja generatora nazw chorób

**Cel:** nazwy, które brzmią jak najbardziej jak prawdziwe łacińskie nazwy chorób, a jednocześnie są nowe, przy **możliwie najkrótszym treningu**, bo model uczy się na żywo podczas prelekcji.

**Dane:** `data/choroby.csv` po oczyszczeniu: 1292 nazwy (usunięto osoby, leki, patogeny, dziedziny medycyny i miejsca). Podział: 90% trening, 10% walidacja, losowo (ziarno 7).

**Sprzęt pomiaru czasu:** CPU ograniczone do 2 wątków, bez GPU (`autooptymalizacja.py` ustawia to na starcie). To przybliża darmowego Colaba bez akceleratora. Rdzeń M1 Pro jest szybszy od vCPU Colaba, więc czasy w Colabie na CPU będą pewnie ~1,5–2× dłuższe. Na T4 krótsze.

## Miary

| Miara | Co mierzy | Lepiej |
|---|---|---|
| `nowe_%` | odsetek nazw, których nie ma w zbiorze | ↑ |
| `swieze_%` | nowe **i** nie są prawie-kopią istniejącej (odległość edycyjna > 20% długości) | ↑ |
| `wiarygodnosc_%` | średnie log-prawdopodobieństwo pod 4-gramowym modelem znaków z korpusu, względem prawdziwych nazw z walidacji (100% = brzmi jak prawdziwe) | ↓ (bliżej 100) |
| `konc_lac_%` | odsetek z typową łacińską końcówką (-itis, -osis, -oma, -ia, -us…) | ↑ |
| `unikalne_%` | czy generator się nie powtarza | ↑ |
| `val_loss` | błąd na nazwach niewidzianych w treningu | ↓ |
| `czas_s` | czas samego treningu | ↓ |
| **ocena ślepa (Opus 5.5)** | 1–5 „czy brzmi jak prawdziwa łacińska nazwa choroby", oceniane bez wiedzy o konfiguracji, z prawdziwymi nazwami jako kontrolą | ↑ |

## Faza 1: szeroki przegląd (11 konfiguracji × 6 punktów epok × 3 temperatury, 1 ziarno)

Pliki: `wyniki_auto.json`, `log_auto.txt`.

Wnioski:

1. **Ustawienia notebooka sprzed optymalizacji (Adam lr=0,001, 30 epok) uczyły się za wolno.** Przy lr=0,003 model po **12 epokach** daje nazwy co najmniej tak dobre jak stary po 30, w czasie 14–20 s zamiast 47 s.
2. **lr=0,01 przy batch 32 jest za agresywny:** val_loss rośnie (2,17 po 30 epokach), nazwy robią się bełkotliwe.
3. **Duże modele (192, 256 jednostek) po 30 epokach zapamiętują zbiór:** tylko 73–81% nazw nowych, val_loss ~2,0. Są też najwolniejsze (60 s).
4. **Dwie warstwy GRU dają najniższy val_loss (1,68–1,70),** ale są 2× wolniejsze, a nazwy nie są wyraźnie lepsze.
5. **Bez dropoutu najlepiej przy 12 epokach (wiarygodność 108%),** ale po 30 epokach mocno się przeucza (val_loss 2,0). To ryzyko na żywo, jeśli ktoś zostawi trening na dłużej.
6. **Temperatura 0,5 wygrywa z 0,65 i 0,8** we wszystkich konfiguracjach: wiarygodność jest lepsza o 10–30 pkt, a odsetek nowych nazw ciągle ≥ 95%.

Front Pareto (czas treningu ↔ wiarygodność), przy warunku świeże ≥ 85% i nowe ≥ 95%:

| czas | wiarygodność | konfiguracja |
|---|---|---|
| 2,9 s | 160% | 64 jedn., lr 0,01, 3 epoki |
| 6,2 s | 132% | 64 jedn., lr 0,01, 8 epok |
| 8,1 s | 119% | 96 jedn., dropout 0,2, lr 0,005, 8 epok |
| 11,6 s | 117% | 96 jedn., dropout 0,2, lr 0,005, 12 epok |
| 13,9 s | 108% | 128 jedn., dropout 0, lr 0,003, 12 epok |
| *47,2 s* | *120%* | *dotychczasowy notebook (128, d0,3, lr 0,001, 30 epok)* |

## Faza 2: zawężenie (3 ziarna, epoki 6–20, T 0,4/0,5/0,6)

Pliki: `wyniki_auto_faza2.json` (225 pomiarów), `log_auto_faza2.txt`. Średnie z 3 ziaren (± odchylenie standardowe wiarygodności). Warunek: świeże ≥ 85%, nowe ≥ 95%.

| konfiguracja | epoki | T | czas | wiarygodność | świeże | nowe | końcówki | unikalne | val_loss |
|---|---|---|---|---|---|---|---|---|---|
| **128, d0,2, lr 0,003, b32** | **12** | **0,4** | **14,8 s** | **101,4 ± 2,1** | 91,0 | 97,8 | 73,8 | 92,0 | 1,765 |
| 128, d0,3, lr 0,003, b32 | 16 | 0,4 | 19,5 s | 103,2 ± 2,6 | 86,2 | 95,8 | 66,5 | 90,5 | 1,748 |
| 96, d0,2, lr 0,005, b32 | 12 | 0,4 | 10,8 s | 103,5 ± 1,0 | 86,7 | 96,7 | 69,3 | 91,0 | 1,771 |
| 128, d0,0, lr 0,003, b32 | 10 | 0,4 | 12,2 s | 104,9 ± 1,1 | 91,2 | 97,5 | 75,5 | 93,0 | 1,783 |
| 128, d0,2, lr 0,005, b64 | 10 | 0,4 | 10,9 s | 107,8 ± 1,4 | 92,0 | 98,7 | 80,2 | 96,8 | 1,808 |
| 96, d0,2, lr 0,005, b32 | 8 | 0,4 | 7,5 s | 110,4 ± 1,1 | 90,0 | 98,3 | 79,7 | 94,8 | 1,832 |
| 128, d0,2, lr 0,003, b32 | 12 | 0,5 | 14,8 s | 113,9 ± 2,3 | 92,8 | 98,2 | 76,5 | 96,7 | 1,765 |
| 96, d0,2, lr 0,005, b32 | 6 | 0,4 | 5,9 s | 122,8 ± 2,1 | 94,2 | 99,2 | 81,8 | 97,5 | 1,920 |
| *poprzedni notebook: 128, d0,3, lr 0,001* | *30* | *0,5* | *35,8 s* | *121,1 ± 4,0* | *92,5* | *98,3* | *71,8* | *98,7* | *1,779* |

Wnioski:

1. **T=0,4 poprawia wiarygodność o ~10–12 pkt względem 0,5** w każdej konfiguracji (różnica ≫ rozrzut między ziarnami). Kosztuje to trochę różnorodności: w próbie 200 nazw 92–95% jest unikalnych zamiast 97–99%. Przy 3 nazwach na kliknięcie powtórka zdarzy się rzadko.
2. **Optimum to 10–12 epok.** Po 16 epokach model zaczyna zapamiętywać: nowe spadają poniżej 95%, a pewne konfiguracje wypadają z frontu.
3. **Bez dropoutu jest szybko i dobrze, ale krucho.** W fazie 1 po 30 epokach val_loss rósł do 2,0. Dropout 0,2 daje podobną jakość z marginesem bezpieczeństwa na wypadek dłuższego treningu.

## Faza 3: ślepa ocena przez Opus 5.5

Sześciu finalistów, każdy trenowany od zera (ziarno 7), po 25 unikalnych nazw. Do tego 25 prawdziwych nazw ze zbioru walidacyjnego jako kontrola. Całość przemieszana (`slepa_proba.tsv`). Oceniałem wyłącznie listę nazw, bez klucza (`slepa_klucz.json`), w skali:

- **5:** nie do odróżnienia od prawdziwej łacińskiej nazwy choroby;
- **4:** bardzo wiarygodna, z drobną usterką;
- **3:** brzmi medycznie, ale wyraźnie zmyślona lub niezgrabna;
- **2:** medyczne strzępy z bełkotem;
- **1:** bełkot.

Oceny: `slepa_oceny_opus.tsv`.

| grupa | konfiguracja | czas | śr. ocena | ≥ 4 | ≤ 2 |
|---|---|---|---|---|---|
| K0 | **128, d0,2, lr 0,003, 12 ep., T 0,4** | 15 s | **2,68** | 16% | 48% |
| K1 | 128, d0,2, lr 0,003, 12 ep., T 0,5 | 15 s | 2,64 | 20% | 52% |
| K5 | poprzedni notebook (lr 0,001, 30 ep., T 0,5) | 36 s | 2,60 | 24% | 60% |
| K2 | 96, d0,2, lr 0,005, 8 ep., T 0,4 | 7,5 s | 2,44 | 4% | 52% |
| K3 | 128, d0,2, lr 0,005, b64, 10 ep., T 0,4 | 11 s | 2,40 | 12% | 56% |
| K4 | 96, d0,2, lr 0,005, 6 ep., T 0,4 | 6 s | 2,36 | 4% | 52% |
| kontrola | prawdziwe nazwy | — | **4,96** | 100% | 0% |

Wnioski:

1. **Ocena rozróżnia prawdę od generacji:** 4,96 wobec 2,4–2,7. To nie jest losowe przyznawanie punktów.
2. **12 epok przy lr 0,003 jest co najmniej tak dobre jak stare 30 epok,** przy 2,4× krótszym czasie.
3. **Najszybsze warianty (6–8 epok) są gorsze o ~0,25 pkt.** Różnica jest na granicy szumu (25 nazw, błąd standardowy ~0,2), ale spójna z miarą 4-gramową.
4. **T 0,4 i T 0,5 w ocenie ślepej to remis** (2,68 i 2,64). Rozstrzyga miara automatyczna (101% wobec 114%) i ona przesądziła o wyborze T 0,4.
5. **Miara 4-gramowa słabo koreluje z oceną ślepą między konfiguracjami.** Przykład: stary notebook miał słabą wiarygodność 4-gramową (121%), a przyzwoitą ocenę ślepą. 4-gram widzi zbitki liter, a nie to, czy nazwa jest „medyczna" jako całość. Stąd sens obu miar naraz.

Typowe błędy, które widać w ocenie ślepej (i których nie naprawi żaden hiperparametr):

- **Niezgodność rodzaju:** *carcinoma maligna* zamiast *malignum*, *syndroma maligna*. Model nie zna gramatyki łacińskiej, tylko częste zbitki.
- **Długie nazwy się rozpadają:** po 3. słowie pojawia się bełkot lub powtórzenie (*chronica chronica*).
- **Pseudo-eponimy** (*syndroma prageni*, *morbus antoeni*) brzmią wiarygodnie, bo prawdziwe eponimy są dla modelu ciągiem przypadkowych liter.
- **Najlepsze wygenerowane nazwy brzmią jak literówki prawdziwych** (*paryngis*, *hyperprasia*, *choligitis*, *postitis*, która akurat istnieje jako *posthitis*). To dobry materiał na scenę: „model nauczył się wzorców, nie faktów".

## Decyzja

**128 jednostek GRU, embedding 64, dropout 0,2, Adam lr 0,003, batch 32, 12 epok, temperatura 0,4.**

- Trening: **~15 s** na 2 rdzeniach CPU M1 Pro, wobec 36–47 s wcześniej. Cały notebook od zera, z importami i wykresami, wykonuje się w ~35 s.
- Jakość: najlepsza wiarygodność automatyczna (101%) i najlepsza średnia w ocenie ślepej (2,68), przy ~98% nazw nowych.
- Wpisane do `notebook/model_jezyka_choroby.ipynb` (komórki KONFIGURACJA i TRENING).

## Zastrzeżenia

- **Oceniający (Opus 5.5) zna łacinę medyczną lepiej niż przeciętny słuchacz,** ale nie jest lekarzem. Próba jest mała: 25 nazw na grupę.
- **Czasy w Colabie na CPU będą pewnie ~1,5–2× dłuższe** (wolniejsze vCPU), czyli ~25–30 s. Na T4 szybsze.
- **Nie trenować dłużej niż 12 epok:** przy 16+ model zaczyna odtwarzać zbiór.
- **Colab czyta dane z GitHuba,** więc oczyszczony zbiór (1292 nazwy) zadziała tam dopiero po `git push`.
