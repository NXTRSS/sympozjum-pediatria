# Ściąga: jak uruchomić wszystko

Wszystkie komendy w jednym miejscu. Szczegóły i zastrzeżenia są w `README.md`, `usg/README.md`, `delphi/README.md`, a stan projektu w `HANDOVER.md`.
Komendy zakładają repo w `~/Documents/sympozjum-pediatria` i zainstalowany [uv](https://docs.astral.sh/uv/).

## 1. Prezentacja

| Plik | Do czego |
|---|---|
| `prezentacja.html` | wersja główna, interaktywna: otwórz w Chrome, strzałki ←/→ |
| `eksport/prezentacja.pptx` | dla laptopa DIL (Windows): slajdy-obrazy + notatki prelegenta |
| `eksport/prezentacja.pdf` | to samo jako PDF, kopia zapasowa |

```bash
cd ~/Documents/sympozjum-pediatria
python3 assemble.py            # składa prezentacja.html ze slides/*.html
uv run eksport_raster.py       # robi eksport/prezentacja.pdf i .pptx (ok. 1 min, potrzebny Google Chrome)
open prezentacja.html
```

Po każdej zmianie w `slides/` uruchom obie komendy ponownie. Katalogu `eksport/` nie ma w gicie: pliki kopiuje się na pendrive.

### Poprawianie słów w przeglądarce (Artifact na claude.ai)

Wersja do edycji: https://claude.ai/artifact/7L4eVEovbndo7KQoTL5Qgx (prywatna, widzi ją tylko właściciel).

1. Otwórz link, kliknij **✏️ Edytuj** (lewy dolny róg), kliknij tekst na slajdzie i pisz. Zapis następuje po kliknięciu obok, a poprawione fragmenty mają niebieskie podkreślenie. **↺ Oryginał** przywraca tekst zaznaczonego fragmentu. Strzałki ←/→ przełączają slajdy, gdy nie piszesz.
2. Poprawki trafiają do bazy Artifactu (kolekcja `poprawki`), a nie do repo. Poproś Claude: „wczytaj poprawki z Artifactu i nanieś na slides”. Claude czyta je narzędziem ArtifactData, nanosi na `slides/*.html` i uruchamia `assemble.py` oraz eksport.
3. Po zmianach w repo odśwież Artifact (Claude publikuje ten sam plik):
   ```bash
   python3 assemble.py && python3 eksport_edycja.py
   ```

Notatek prelegenta nie da się tu edytować: są w `assemble.py` (lista `NOTES`).

## 2. Notebook „nazwy chorób” (GRU)

Bez instalacji: przycisk *Open in Colab* w `README.md`. Lokalnie (instaluje TensorFlow):

```bash
cd ~/Documents/sympozjum-pediatria
uv sync
uv run jupyter notebook notebook/model_jezyka_choroby.ipynb
```

## 3. USG: masa płodu

Jednorazowe przygotowanie, z internetem: pobiera ok. 680 MB wag i nagrania z Zenodo, za pierwszym razem ok. 15 min.

```bash
cd ~/Documents/sympozjum-pediatria/usg
uv sync
GIT_CONFIG_GLOBAL=/dev/null uv run python scripts/przygotuj_demo.py
```

`GIT_CONFIG_GLOBAL=/dev/null` jest potrzebne tylko na tym Macu: globalny `~/.gitconfig` przepisuje GitHub na SSH, a klucz SSH jest źle ustawiony. Po naprawie klucza można to pominąć.

Pokaz (działa bez internetu):

```bash
cd ~/Documents/sympozjum-pediatria/usg
uv run jupyter notebook notebook/usg_masa_plodu.ipynb     # TRYB = "z_pamieci", potem Kernel → Restart & Run All
uv run python app/demo_usg.py                             # aplikacja: http://127.0.0.1:7860
```

## 4. Delphi-2M

Jednorazowe przygotowanie, z internetem: klon kodu autorów + trening na ich danych syntetycznych, ok. 20 min na M1 Pro.

```bash
cd ~/Documents/sympozjum-pediatria/delphi
uv sync
GIT_CONFIG_GLOBAL=/dev/null SSL_CERT_FILE="$PWD/.venv/lib/python3.11/site-packages/certifi/cacert.pem" uv run python scripts/przygotuj_demo.py
```

`SSL_CERT_FILE` jest potrzebne, bo Python z uv nie ufa certyfikatowi Certum serwera `api.dane.gov.pl` (skąd skrypt pobiera polską ICD-10). Skrypt przepisuje przy tym `dane/icd10_pl.csv` z innymi końcami linii; treść jest ta sama, przywrócisz go przez `git checkout -- dane/icd10_pl.csv`.

Pokaz (działa bez internetu):

```bash
cd ~/Documents/sympozjum-pediatria/delphi
uv run jupyter notebook notebook/delphi_2m.ipynb          # TRYB = "z_pamieci", potem Run All
```

**Nie uruchamiaj dwóch przygotowań ani dwóch obliczeń „na żywo” naraz**, bo oba używają GPU (MPS).

## 5. Testy

```bash
cd ~/Documents/sympozjum-pediatria/usg    && uv run pytest              # szybkie
cd ~/Documents/sympozjum-pediatria/usg    && USG_PELNE=1 uv run pytest  # pełne (po przygotowaniu)
cd ~/Documents/sympozjum-pediatria/delphi && uv run pytest
cd ~/Documents/sympozjum-pediatria/delphi && DELPHI_PELNE=1 uv run pytest
```

Bez przygotowania Delphi (brak `delphi/zewnetrzne/Delphi`) szybkie testy dają 6 failed i 56 errors. To oczekiwane, nie błąd kodu.

## 6. Przed wyjazdem na sympozjum

1. `python3 assemble.py && uv run eksport_raster.py`, potem przegląd `eksport/prezentacja.pptx` (najlepiej w PowerPoincie na Windows).
2. Skopiuj na pendrive: `eksport/prezentacja.pptx`, `eksport/prezentacja.pdf`, `prezentacja.html` i nagrania demo (mp4).
3. Wyślij organizatorowi PDF lub PPTX wcześniej (prośba DIL).
