# Przekazanie: środowisko i notebook pokazowy dla Delphi-2M

**Worktree:** `~/Documents/sympozjum-pediatria-delphi`, gałąź `delphi-2m` (z `main` @ fdd1228)
**Data przekazania:** 2026-10-13
**Zlecający:** Kamil Jędryczek (ML Engineer, GitHub NXTRSS)

## Kontekst

Kamil przygotowuje ~45-minutową prelekcję dla lekarzy (głównie pediatrów, bez przygotowania technicznego) na sympozjum „Nowoczesne technologie w pediatrii — od teorii do praktyki" (Dolnośląska Izba Lekarska, Wrocław, **28.11.2026**). Repo prelekcji jest publiczne: `github.com/NXTRSS/sympozjum-pediatria`.

Oś prelekcji to mechanizm modelu autoregresyjnego. Najpierw na żywo trenuje się mały model znakowy (GRU), który generuje łacińskie nazwy chorób (`notebook/model_jezyka_choroby.ipynb`). Potem pada pytanie: co, jeśli tokenami nie są litery, tylko **diagnozy** pacjenta?

W decku są dwa slajdy o **Delphi-2M**, z tymi treściami:

- Shmatko i in., *Nature* 2025, „Learning the natural history of human disease with generative transformers" (EMBL-EBI / DKFZ, grupa Moritza Gerstunga).
- To GPT-2 z ok. **2,2 mln parametrów**. Tokeny to kody ICD-10 (pierwsze 3 znaki) plus płeć, styl życia i inne. Pozycja to **wiek** (kodowanie ciągłe), a jeden z tokenów oznacza **śmierć**.
- Trening na ok. 400 tys. uczestników UK Biobank, walidacja zewnętrzna na 1,93 mln Duńczyków. Ponad 1000 chorób, AUC ~0,76.
- Uczciwe zastrzeżenie: UK Biobank rekrutował osoby w wieku 40–70 lat, więc w danych nie ma dzieci.

## Prośba Kamila (dosłownie)

> „Otwórz nowy worktree i zrób hand over, żeby też mieć środowisko dla tego Delphi 2M i też z notebookiem, żeby pokazywać na żywo, jakieś dane, żeby wgrać na pokaz itd. — niech tam też to zostanie zaplanowane za pomocą writing-plans."

## Zadanie dla przejmującego

1. **Rozpoznanie (zweryfikuj u źródła, nie z pamięci):**
   - Oficjalny kod Delphi-2M (prawdopodobnie `github.com/gerstung-lab/Delphi`): licencja, framework, wersje, przypięty commit.
   - Czy są wytrenowane wagi i na czym (UK Biobank / dane syntetyczne)?
   - Czy repo zawiera **dane syntetyczne** lub ich generator? Jaki jest format danych wejściowych (sekwencja token + wiek)?
   - Jak liczy się predykcję (prawdopodobieństwo następnej choroby i czas do zdarzenia)?
   - Jakie są wymagania sprzętowe (CPU/MPS na M1 Pro)?
   - Prawdziwe dane UK Biobank wymagają wniosku o dostęp i **nie mogą** trafić do publicznego repo. Pokaz musi się opierać na danych syntetycznych albo na ręcznie ułożonych przykładowych „historiach pacjenta".
2. **Plan:** napisz plan wdrożenia skillem **superpowers:writing-plans**. Zapisz go jako `docs/superpowers/plans/2026-10-13-delphi-2m.md` w tym worktree.
   - Cel: podprojekt `delphi/` z własnym środowiskiem uv i z notebookiem `delphi/notebook/delphi_2m.ipynb`, którym Kamil pokaże na żywo, jak model z historii chorób przewiduje kolejne zdarzenia (ranking następnych diagnoz, krzywe ryzyka w czasie).
   - Najlepiej, żeby notebook pozwalał wpisać własną krótką historię (np. „E11 cukrzyca t.2 w wieku 50 lat, I10 nadciśnienie w 55") i od razu zobaczyć prognozę.
   - Notebook ma też wygenerować przykładową trajektorię życia token po tokenie. To klamra do demo z nazwami chorób: ten sam mechanizm generowania.
   - Plan ma zawierać tryb bez internetu (wszystko pobrane i policzone przed prelekcją).
   - Jeżeli wag nie ma, plan przewiduje szybki trening na danych syntetycznych z repo autorów: zmierz czas na M1 Pro i podaj go w planie.
3. **Nie wykonuj planu.** Oddaj ścieżkę do planu i podsumowanie ustaleń. Kamil zdecyduje o wykonaniu (domyślnie subagentami, równolegle tam, gdzie to możliwe).

## Ograniczenia globalne (jak w podprojekcie USG)

- Podprojekt w `delphi/`, z osobnym `pyproject.toml` zarządzanym przez **uv**. Wersję Pythona dobierz do kodu autorów.
- Kodu ani wag autorów nie kopiujemy do repo, jeśli licencja na to nie pozwala. Zamiast tego skrypt klonuje je w przypiętym commicie do `delphi/zewnetrzne/` (w `.gitignore`). Dane i wyniki pośrednie (`delphi/dane/`, `delphi/pamiec/`) też są poza gitem.
- Teksty dla ludzi (komentarze, markdown notebooka, komunikaty) piszemy **po polsku**, w stylu `notebook/model_jezyka_choroby.ipynb`: nagłówki komórek `# ── NAZWA ───`, proste słowa, polskie cudzysłowy „…". Odbiorcy to lekarze.
- Notebook zawiera zastrzeżenie: „Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych." Commitujemy go z wyczyszczonymi wyjściami.
- Uczciwość: jeżeli coś działa na danych syntetycznych lub na losowych wagach, notebook mówi to wprost.
- Bez `git push`. Commity tylko na gałęzi `delphi-2m`.
- Testy: pytest z markerami `siec` / `wolny`, uruchamiane flagą środowiskową, jak w planie USG.

## Wzorzec do naśladowania

Równoległy plan dla USG/BabyNet ma tę samą strukturę: rozpoznanie „Stan faktyczny" (zweryfikowane), „Global Constraints", „Review Focus", moduły w `src/<pakiet>/`, tryby `z_pamieci` / `na_zywo`. Plan leży w głównym repo:
`~/Documents/sympozjum-pediatria/docs/superpowers/plans/2026-10-13-usg-masa-plodu.md`
