# HANDOVER — stan projektu na Mac24 (8.11.2026)

Prelekcja „Trafny po 17 lekarzach, omylny w 83% przypadków”, sympozjum DIL, Wrocław, **28.11.2026**.
Plik opisuje stan gałęzi integracyjnej przeniesionej na drugi komputer. Nie jest dokumentacją użytkową: ta jest w `README.md`, `delphi/README.md` i `usg/README.md`.

## 1. Repozytorium i gałąź

| | |
|---|---|
| Checkout | `~/Documents/sympozjum-pediatria` (świeży klon 8.11.2026, wcześniej na tym Macu nie było kopii) |
| Gałąź robocza | `reconcile/public-final-20261108-1618`, śledzi `origin/…` |
| HEAD | `43ef88f0aae93527f2a2e2271c25df2b151d1c41`, zgodny z oczekiwanym |
| `origin/main` | `48072ba`, czyli merge-base. Gałąź jest o 1 commit przed `main` (80 plików, +28,6 tys. / −1,4 tys. linii) |
| Scalenie do `main` | **nie wykonano**. Nie scalać i nie pushować bez decyzji Kamila |

Zawartość gałęzi: prezentacja (`slides/`, `assemble.py`, `public-template/`, `prezentacja.html`), notebook GRU (`notebook/`), Delphi (`delphi/`, w tym polska ICD-10 z atrybucją CeZ w `delphi/dane/icd10_pl.csv`), USG z aplikacją Gradio (`usg/`).
Celowo **nie ma** w repo: prywatnej korespondencji, danych pacjentów, wag, danych ACOUSLIC-AI/UKB, kodu autorów (klonowany skryptami do `*/zewnetrzne/`, poza gitem). Nie odtwarzać tego i nie importować starych gałęzi USG.

**Uwaga o gicie na tym Macu:** globalny `~/.gitconfig` przepisuje `https://github.com/` na SSH (`url.git@github.com:.insteadof`), a w konfiguracji SSH jako klucz prywatny wskazany jest `~/.ssh/id_ed25519.pub`. Przez to `git clone` i `git fetch` po SSH kończą się błędem. Klon zrobiono z pominięciem globalnej konfiguracji: `GIT_CONFIG_GLOBAL=/dev/null git …`. Konfiguracji użytkownika nie zmieniano.

## 2. Interpretacja naukowa (obowiązuje w slajdach, notebookach, aplikacji)

- USG to **demonstracja edukacyjna**: nie jest wyrobem medycznym ani modelem lepszym od Hadlocka.
- Model Sano szacuje **masę w dniu porodu z badania < 24 h przed porodem**. Nie prognozuje masy na tygodnie naprzód. Nagrania z 28–34 tc. leżą poza domeną treningową.
- Dostępny checkpoint to **obrazowy baseline 3D ResNet-18**, nie pełny TabAttention.
- Wyniki z publikacji, których nie wolno mieszać: baseline **289 g** (8,5%); pełny model obraz + pomiary **170 g** (5,0%, brak publicznych wag); klinicyści z Hadlockiem **205 g** (5,9%).
- Delphi działa na modelu trenowanym przez nas na **danych syntetycznych** autorów, więc liczby ryzyka nie są prawdziwymi ryzykami. W UK Biobank nie ma dzieci.
- Cytowania, atrybucje (ACOUSLIC-AI CC BY-NC-SA 4.0, CeZ CC BY 4.0, Wikimedia CC BY-SA 4.0) i ograniczenia zostają bez zmian.

## 3. Kontrole: historyczne vs wykonane teraz

### 3a. Wyniki historyczne (Mac 32 GB, 8.11.2026, przekazane przez Kamila; tu nie powtarzane)

- USG: 112 passed, 7 skipped, 14 deselected.
- Delphi: 29 passed, 6 failed, 56 errors, 5 deselected. Identycznie na bazowym `main`.
- Aplikacja USG importuje się i buduje Gradio Blocks.
- Trzy notebooki mają poprawny schemat i puste wyjścia.
- Starsze pomiary z README (M1 Pro: czasy treningu, inferencji, wyniki AC i Sano) też są historyczne.

### 3b. Wykonane na Mac24 (Apple M4 Pro, 24 GB, macOS 27, uv 0.7.6), 8.11.2026

| Kontrola | Wynik |
|---|---|
| Środowiska `delphi/` i `usg/` | `uv sync --frozen` OK (Python 3.11, tylko pakiety z PyPI, bez wag i danych). `uv.lock` bez zmian. Środowiska głównego (TensorFlow) **nie** instalowano |
| `cd usg && uv run pytest` (szybkie) | **112 passed, 21 skipped**, 41 s. 21 = 7 + 14 z wyniku historycznego: conftest *pomija* (skip) testy `siec`/`wolny`, więc bez `-m` są liczone jako skipped, a nie deselected |
| `cd delphi && uv run pytest` (szybkie) | **29 passed, 6 failed, 56 errors, 5 skipped**, 8 s. Wszystkie 62 problemy to ten sam `FileNotFoundError: Brak kodu Delphi w delphi/zewnetrzne/Delphi` (brak klonu autorów, który robi `przygotuj_demo.py`). Zgadza się z wynikiem historycznym. Testów nie osłabiano |
| `python3 assemble.py` (2×) | 29 slajdów. Wynik **bajtowo identyczny** z commitowanym `prezentacja.html` (sha256 `30f526e7…f835`), drzewo czyste |
| Notatki prelegenta | JSON `speaker-notes`: 29 wpisów, żaden nie jest pusty |
| JavaScript decku | 5 skryptów inline przechodzi `node --check` (Node 24). Wystąpienia `<script src…>` są tylko w komentarzach JS, deck nie ładuje niczego z zewnątrz |
| `.gitignore` | `usg/{dane,zewnetrzne,pamiec}` i `delphi/{dane/*,zewnetrzne,pamiec}` są ignorowane. Wyjątek `delphi/dane/icd10_pl.csv` jest śledzony celowo |

### 3c. Przygotowanie demo i pełne testy (Mac24, 8.11.2026, za zgodą Kamila)

| Kontrola | Wynik |
|---|---|
| `usg/scripts/przygotuj_demo.py` | OK, 13,8 min (pobrane wagi nnU-Net/FUVAI/Sano i nagrania). Wyniki zgodne z README i slajdami: AC nnU-Net 304/257/242/262 mm, FUVAI 242/273/209 mm/brak, Sano 3 551 / 3 855 / 3 898 / 4 000* g |
| `delphi/scripts/przygotuj_demo.py` | 1. próba: błąd TLS przy pobieraniu ICD-10 z `api.dane.gov.pl`. Łańcuch Certum jest prawdziwy, ale Python z uv nie ufa korzeniowi Certum. 2. próba z `SSL_CERT_FILE=<certifi z .venv>`: OK. Trening 2500 kroków na MPS **27,8 min** (wolniej niż ~18 min z README, w tle działały inne zadania), krzywe 3 przykładów ok. 2 min |
| `USG_PELNE=1 uv run pytest` | **133 passed**, 79 s |
| `uv run pytest` (Delphi, po przygotowaniu) | **91 passed, 5 skipped** (zgodnie z README) |
| `DELPHI_PELNE=1 uv run pytest` | **96 passed**, 40 s |
| Notebook USG, `z_pamieci` (kopia, nbconvert) | 6 komórek, 0 błędów, 6 wykresów, 13 s; liczby jak na slajdach |
| Notebook Delphi, `z_pamieci` (kopia, nbconvert) | 7 komórek, 0 błędów, 5 wykresów, 7,6 s |
| Logika aplikacji USG (`aplikacja.analizuj`, z pamięci) | 4 nagrania, każde < 1 s, masy Sano jak na slajdzie |

Uwagi:
- `przygotuj_demo.py` (Delphi) **przepisuje śledzony `delphi/dane/icd10_pl.csv` z końcami linii CRLF** (moduł `csv` domyślnie pisze `\r\n`). Treść jest identyczna (`git diff -w --ignore-cr-at-eol` pusty), ale plik pokazuje się jako zmieniony. Przywrócić: `git checkout -- delphi/dane/icd10_pl.csv`; trwała poprawka: `lineterminator="\n"` w generatorze.
- Oba skrypty przygotowania na tym Macu trzeba uruchamiać z `GIT_CONFIG_GLOBAL=/dev/null` (SSH), a Delphi dodatkowo z `SSL_CERT_FILE` (patrz `URUCHOMIENIE.md`).

### 3d. Nadal niewykonane

- klik-test aplikacji Gradio w przeglądarce i tryb „na żywo” (nnU-Net/FUVAI/Sano na MPS),
- środowisko główne (TensorFlow + metal) i trening GRU na tym Macu,
- otwarcie PPTX/PDF na Windows,
- nagrania mp4 dem.

`git-lfs` nie jest potrzebny: `usg/src/usg_masa/zasoby.py` klonuje z `GIT_LFS_SKIP_SMUDGE=1` i podmienia wskaźnik LFS na pobrane wagi. **Ale** oba `przygotuj_demo.py` robią `git clone https://github.com/…`, które globalna reguła insteadOf przepisze na SSH (pkt 1), więc na tym Macu uruchamiać je z `GIT_CONFIG_GLOBAL=/dev/null` albo najpierw naprawić klucz SSH.

## 4. Uruchamianie (z dokumentacji repo)

Pełna ściąga z komendami, w tym eksportem i przygotowaniem demo na tym Macu: [`URUCHOMIENIE.md`](URUCHOMIENIE.md).

```bash
# prezentacja (tylko biblioteka standardowa; nadpisuje prezentacja.html)
python3 assemble.py            # albo: uv run python assemble.py

# notebook GRU (instaluje TensorFlow; na Apple Silicon trening na GPU przez tensorflow-metal)
uv sync && uv run jupyter notebook notebook/model_jezyka_choroby.ipynb

# Delphi: przygotowanie z internetem, ok. 20 min na M1 Pro (klon autorów + trening + krzywe)
cd delphi && uv sync && uv run python scripts/przygotuj_demo.py
uv run jupyter notebook notebook/delphi_2m.ipynb
uv run pytest                       # szybkie
DELPHI_PELNE=1 uv run pytest        # pełne

# USG: przygotowanie pobiera ok. 680 MB wag i nagrania z Zenodo, ok. 15 min za pierwszym razem
cd usg && uv sync && uv run python scripts/przygotuj_demo.py
uv run jupyter notebook notebook/usg_masa_plodu.ipynb
uv run python app/demo_usg.py       # aplikacja, http://127.0.0.1:7860
uv run pytest                       # szybkie
USG_PELNE=1 uv run pytest           # pełne
```

Uwagi z README: nie uruchamiać dwóch procesów na MPS naraz. Inferencja Delphi i modelu Sano działa na CPU (ograniczenia MPS w torch 2.2/2.3). Na sali używać trybu `z_pamieci`.

## 5. NOWE ograniczenie miejsca (e-mail DIL, otrzymany 8.11.2026)

- Prezentacje odtwarza **wyłącznie laptop DIL z Windows**. Obsługiwane formaty: **ppt/pptx i PDF**. Filmy **mp4** tylko osobno, poza prezentacją.
- **Nie można podłączyć własnego laptopa**, a sprzętu Apple i Keynote nie obsługują.
- Trzeba mieć kopię na pendrivie lub SSD. Organizatorzy proszą o wcześniejsze przesłanie prezentacji.

Konsekwencje dla obecnego stanu:
- `prezentacja.html` nie jest formatem wymienionym jako obsługiwany. Wymagany jest PDF lub PPTX (+ ewentualnie mp4).
- Dema na żywo z Maca (trening GRU, Delphi, aplikacja USG na MPS) **nie zadziałają** na sali bez zgody na własny sprzęt. Najprostsza ścieżka to nagrania mp4. Ewentualnie Colab w przeglądarce laptopa DIL, ale to wymaga internetu i potwierdzenia.
- Slajd z kodem QR do Colaba (`n05_qr`) zależy od Wi-Fi dla uczestników. Do potwierdzenia.
- Decyzja o kierunku czeka na odpowiedź organizatora (pytania w wiadomości do organizatora).

### Eksport dla laptopa DIL (8.11.2026, Mac24)

- **Druk do PDF z Chrome** (`prezentacja.html` → PDF): 29 stron, 1440×810 pt. **Nie nadaje się** na salę. Tekst z gradientem (`.grad-text*`, 39 miejsc) różne przeglądarki PDF rysują różnie: poppler dodaje ramki, macOS Quartz/Podgląd zasłania część nagłówka prostokątem. Animacja „Krok po kroku” (slajd 9) wychodzi pusta.
- **Eksport rastrowy**: `eksport_raster.py` (uv, zależności w nagłówku skryptu: playwright + python-pptx + img2pdf; używa zainstalowanego Google Chrome, przeglądarki nie pobiera).
  ```bash
  python3 assemble.py && uv run eksport_raster.py   # ok. 1 min
  ```
  Wynik w `eksport/` (w `.gitignore`): `prezentacja.pdf` (43 strony-obrazy 1920×1080, 19 MB), `prezentacja.pptx` (43 slajdy 16:9 z notatkami prelegenta, 19 MB), `slajdy/*.png`. 29 slajdów decku daje 43 strony, bo animacja „Krok po kroku” jest rozpisana na 15 slajdów (notatka przy kroku 1). Elementy `.build` są w stanie końcowym (bez odsłaniania na klik). Pasek nawigacji, wskaźnik trybu i przyciski animacji są ukryte.
  - Sprawdzone: przegląd wizualny wszystkich 43 stron (poppler) i strony 1 w Quartz (gradient poprawny); PPTX ma 43 slajdy, każdy z jednym obrazem i niepustą notatką.
  - **Niesprawdzone**: otwarcie PPTX w PowerPoincie i PDF w Edge/Adobe na Windows.
- Kody QR na slajdach 11 i 28 (`#qr-gradio`, `#qr-colab`, `#qr-materialy`) to **zaślepki** w źródle. Trzeba je zastąpić prawdziwymi kodami, gdy zapadnie decyzja, dokąd mają prowadzić.

### Zmiany treści i narzędzia (8.11.2026, niecommitowane)

- Slajd 14 (`slides/o17_base_model.html`, notatka w `assemble.py`): zamiast „Base Model” jeden slajd o trzech etapach: pretrening (model bazowy) → SFT → RL/RLHF, z kartą-pomostem do rozwiązania zagadki. Puenta „nie wie, kiedy nie wie” zostaje na slajdzie 15.
- Wersja do edycji słów: Artifact https://claude.ai/artifact/7L4eVEovbndo7KQoTL5Qgx (prywatny), budowany przez `eksport_edycja.py`. Poprawki są w bazie Artifactu (kolekcja `poprawki`) do czasu naniesienia na `slides/`. 8.11: naniesione 2 poprawki (`n06_rozwiazanie.html`: „…specjalnego tokenu… token tekstu”; `n07_whisper.html`: „lekarzy klinicznych”) i usunięte z bazy.
- Usunięty slajd „Czy Polska powinna budować własne medyczne AI?” (`n10b_kornowska.html`, na prośbę Kamila; plik do odzyskania z gita). Notatka następnego slajdu („Zależy, o co pytamy”) poprawiona. Deck ma teraz **28 slajdów** (eksport: 42 strony).
- Delphi: wyszukiwarka chorób po polskiej nazwie w notebooku (`delphi/src/delphi_demo/wyszukiwarka.py`, panel ipywidgets zamiast pola z kodami; 12 nowych testów). Sprawdzone w żywym Jupyterze: „zawal” → I21, „astma” + wiek 6 → „Dodaj” → „Pokaż prognozę” daje wykres i tabelę. Delphi: 103 passed / 5 skipped (szybkie), 108 passed (pełne).
- Notatki z Obsidiana (przegląd tylko do odczytu): do ewentualnego dodania twierdzenie z Delphi (model trenowany wyłącznie na 400 tys. syntetycznych trajektorii: AUC 0,74, 3 pkt proc. mniej niż oryginał; Results „synthetic data”, Fig. 3f, sprawdzone w pełnym tekście PMC12589094). Nasz pokaz używa małej publicznej próbki autorów, a nie tego zbioru.

## 6. Pozostałe kontrole i drobne niespójności

1. Otwarcie `eksport/prezentacja.pptx` i `.pdf` na Windows (PowerPoint, Edge, Adobe); prawdziwe kody QR.
2. `slides/README.md` jest nieaktualny: wspomina `ALK/Model_Jezyka.html`, a szablon to już `public-template/published-deck.html`.
3. `HANDOFF_DELPHI.md` opisuje stan z 13.10 (worktree `sympozjum-pediatria-delphi`, plan USG `docs/superpowers/plans/2026-10-13-usg-masa-plodu.md`, którego nie ma w repo). To dokument historyczny.
4. `delphi/README.md` podaje „91 PASS i 5 SKIP” dla zestawu szybkiego. To dotyczy sytuacji z klonem autorów. Bez niego jest 29/6/56/5.
5. Badge Colab w `README.md` wskazuje notebook na `main`. Dopóki gałąź nie jest scalona, kod QR i badge prowadzą do starej wersji (warto porównać ten notebook między gałęziami).
6. Po decyzji o zgodzie na pobranie: `przygotuj_demo.py` (Delphi, USG), testy pełne, wykonanie notebooków, klik-test aplikacji, pomiar czasów na M4 Pro / 24 GB (szczyt RAM Delphi to ok. 10 GB).
