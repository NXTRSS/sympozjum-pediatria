# USG: masa płodu z całego nagrania USG

Część prelekcji dla pediatrów (sympozjum DIL, Wrocław, 28.11.2026).

> Materiał edukacyjny. Nie jest wyrobem medycznym i nie służy do decyzji klinicznych.

## Co tu jest

Dwa pytania, trzy ścieżki:

1. **Czy to da się zrobić dziś, na publicznych narzędziach?** Tak, w części. Z nagrania USG brzucha ciężarnej (840 klatek = 6 przeciągnięć głowicą po 140 klatek) model nnU-Net z konkursu ACOUSLIC-AI sam znajduje najlepszą klatkę brzucha w całym nagraniu. Z maski liczymy obwód brzucha (AC), a z AC wzorem Campbell i Wilkin (1975) masę płodu „teraz”. Wzorem Hadlocka (1984) podajemy też wiek ciążowy odpowiadający temu AC (to nie jest datowanie ciąży). To jest ścieżka, która działa od początku do końca. Ale nnU-Net to model holenderski (Radboudumc).
2. **Opcja pośrednia: polski FUVAI** (Płotka i in., Sano + Warszawski Uniwersytet Medyczny, *Physics in Medicine & Biology* 2022, 67(4):045013). Jedna sieć rozpoznaje, co widać w klatce (głowa / brzuch / kość udowa / tło), i zaznacza tę strukturę. Uczony na nagraniach z 700 ciąż z WUM (aparaty GE), robionych „z ręki” w płaszczyznach standardowych; w artykule na 50 nagraniach mierzył porównywalnie z 5 doświadczonymi ultrasonografistami. **Autorzy opublikowali wagi**, więc model działa u nas od nagrania do AC i masy. Na nagraniach „na ślepo” działa jednak tylko częściowo, bo jest poza swoją domeną (szczegóły i zastrzeżenia: sekcja „FUVAI: co działa, a co nie” niżej).
3. **A masa w dniu porodu, czyli to, co interesuje neonatologa?** Pierwszym kandydatem był BabyNet++ (Płotka i in., CIBM 2023), ale jego wytrenowane wagi nie są publicznie dostępne. Notebook uruchamia więc **model bazowy Sano**: obrazowy 3D ResNet-18 z pracy TabAttention (Grzeszczyk, Płotka i in., MICCAI 2023), z **publicznego checkpointu** wskazanego w README autorów (to jedyny wariant, który da się dziś uruchomić — pełnego modelu z cechami klinicznymi nie ma publicznie). Model z 16 klatek brzucha szacuje masę w dniu porodu (z badania < 24 h przed porodem), w gramach; nie prognozuje masy na tygodnie do przodu. Uczono go wyłącznie na ciążach donoszonych, więc na naszych nagraniach (ok. 28–34 tc.) wychodzi poza dane, na których się uczył — i każdemu dziecku podaje liczbę typową dla noworodka (3,6–4,0 kg) bez cienia wątpliwości. Mas urodzeniowych tych dzieci nie znamy, więc nie ma tu walidacji. Szczegóły: sekcja „Model Sano: masa w dniu porodu” niżej.

Notebook: `notebook/usg_masa_plodu.ipynb`. Kod pomocniczy: `src/usg_masa/`. Skrypt przygotowujący wszystko do pokazu: `scripts/przygotuj_demo.py`.

### Jak to opowiadać

Nie udajemy narzędzia lepszego od Hadlocka (takiego dziś publicznie nie ma) i mówimy to pierwsi. Trzy mocne punkty:

1. **Automatyzacja na trudnych danych.** Model sam znajduje brzuch w 840 klatkach nagrania „na ślepo” (tania głowica, smartfon, nowicjusz w Sierra Leone); AC wychodzi kilka–kilkanaście mm od pomiaru ekspertów. Hadlock wymaga doświadczonego sonografisty.
2. **Najlepszy był model obraz + pomiary lekarza.** W pracy TabAttention sam obraz (3D ResNet-18 — to ten model, jedyny dostępny publicznie) mylił się bardziej niż klinicyści z Hadlockiem (289 g vs 205 g), a obraz + 6 pomiarów biometrycznych mniej (170 g; tych wag nie ma publicznie). AI nie zastępuje pomiarów lekarza, tylko je wzmacnia.
3. **Ten model nie wie, kiedy nie wie.** Uczony na dniu porodu, płodowi z 28. tc. podaje 3,6–4 kg bez wahania; ten sam mechanizm co halucynacje ChatGPT. To ekstrapolacja poza dane uczące tego konkretnego checkpointu, a nie ocena metody. Pytanie do każdego produktu AI: „na jakich danych to uczono?”.

Puenta do apelu: polskie modele lepsze od klinicystów (BabyNet++, pełny TabAttention) opisano w publikacjach, ale ich wytrenowanych wag nie ma publicznie. Archiwizujmy i walidujmy to, co sami sfinansowaliśmy.

## Uruchomienie

```bash
cd usg && uv sync && uv run python scripts/przygotuj_demo.py && uv run jupyter notebook notebook/usg_masa_plodu.ipynb
```

Skrypt `przygotuj_demo.py` klonuje w przypiętych commitach repozytoria BabyNet++, FUVAI, Tab-Attention i baseline'u ACOUSLIC-AI (z modelem nnU-Net), pobiera wagi nnU-Net (268 MB), FUVAI (277 MB) i modelu Sano (133 MB, link z README autorów; gdy plik już jest w `zewnetrzne/TabAttention_wagi/`, sprawdza tylko sha256), przykładowy wycinek i trzy pełne nagrania z Zenodo, liczy wyniki referencyjne nnU-Netu, FUVAI i modelu Sano i zapisuje je w `pamiec/` (poza gitem). Dane, wagi i klonowany kod autorów też są poza gitem (`dane/`, `zewnetrzne/`, `pamiec/`).

W notebooku jest przełącznik `TRYB`:

- `"z_pamieci"`: czyta wyniki policzone wcześniej, nie dotyka sieci. Bezpieczny wybór na sali bez internetu.
- `"na_zywo"`: liczy segmentację nnU-Netem i FUVAI oraz wynik modelu Sano na oczach publiczności (nnU-Net skróconą wersją, patrz czasy niżej).

Uwagi praktyczne:

- Pełne nagrania mają na końcach przeciągnięć całkiem czarne klatki. nnU-Net i FUVAI traktują je jako tło i nie wysyłają do sieci.
- **Nie uruchamiaj `przygotuj_demo.py`, gdy notebook liczy na żywo.** Dwa procesy na MPS naraz mogą zawiesić jądro notebooka.
- Jeśli maszyna na sali jest wolniejsza, w notebooku zmień `CO_KTORA_NA_ZYWO = 4` na `8` (co ósma klatka zamiast co czwartej, mniej więcej dwa razy szybciej).

## Przed pokazem

1. W Jupyterze: `Kernel` → `Restart & Run All`, w trybie `TRYB = "z_pamieci"`.
2. Jeśli któraś komórka wisi dłużej niż minutę: `Kernel` → `Restart`, potem znowu `Run All` (zdarzyło się nam, że jądro zawiesiło się bez powodu, a powtórka przeszła w kilka sekund).
3. Nie uruchamiaj w tym samym czasie `przygotuj_demo.py`.
4. Obejrzyj klatki wybrane przez FUVAI na nagraniach 1 i 2 (`0199616b`, `02d3a9bc`): nnU-Net nie widzi w nich brzucha, może to być np. główka.
5. Liczby modelu Sano na slajdzie (3 551 / 3 855 / 3 898 / 4 000* g) pochodzą z trybu `z_pamieci`. **Na żywo wychodzą inne**: nnU-Net ogląda wtedy co 4. klatkę zamiast co 2., więc model Sano dostaje inny zestaw klatek brzucha (tabela w sekcji „Model Sano: masa w dniu porodu”). Do porównania ze slajdem pokazuj `z_pamieci`.

Odrzucanie masek „dotykających krawędzi” (oba modele) sprawdza tylko krawędź obrazu, nie krawędź wiązki USG (wachlarza). Obrys ucięty przez brzeg wachlarza nie zostanie odrzucony.

## Aplikacja (UI)

Ładniejsza alternatywa dla notebooka na scenę: ten sam pipeline, te same liczby i zastrzeżenia, jeden ekran.

```bash
cd usg && uv run python app/demo_usg.py            # otwiera http://127.0.0.1:7860
uv run python app/demo_usg.py --bez-przegladarki   # sam serwer (Ctrl+C kończy)
```

- **Nagranie**: Nr 1 (`0199616b`), Nr 2 (`02d3a9bc`), Nr 3 (`02ee26a5`), Wycinek (`04a04f2e`, 128 klatek), jak na slajdach. Pod spodem można wgrać własny plik `.mha` w formacie ACOUSLIC-AI (dla niego działa tylko „na żywo”, chyba że jego wyniki są w `pamiec/`).
- **Tryb „z pamięci”** (domyślny): wyniki z `pamiec/`, od razu, bez modeli. Liczby jak na slajdzie (AC 304 mm, Sano 3 551 g dla Nr 1 itd.).
- **Tryb „na żywo”**: po przełączeniu modele wczytują się w tle (ok. 2–5 s; pasek stanu mówi, kiedy są gotowe albo że wczytanie się nie udało), potem przycisk liczy nnU-Net (skrócona ścieżka, co 4. klatka) z paskiem postępu, FUVAI i model Sano. Zmierzone na M1 Pro: ok. 25 s na nagranie 840-klatkowe razem z wczytaniem modeli (nnU-Net 16–21 s, FUVAI 5–7 s, Sano 0,5 s), wycinek 128 klatek ok. 5–10 s; pierwsze liczenie po starcie bywa wolniejsze. Wyniki różnią się od slajdu (inne klatki, patrz tabela w sekcji o modelu Sano). Dwa liczenia na żywo idą po kolei (blokada MPS); „z pamięci” nie czeka w tej kolejce.
- **Błędy**: na ekranie tylko krótki komunikat po polsku (zły plik: „Nie udało się wczytać pliku… (oczekiwany format ACOUSLIC-AI: .mha, 3D)”; awaria na żywo: „Przełącz na „z pamięci””), szczegóły w terminalu, w którym działa aplikacja. Wgrany plik jest podpisany „Dane: plik wgrany przez użytkownika”, nie jako ACOUSLIC-AI.
- **Suwak i „▶ odtwórz”** przewijają nagranie. Po analizie klatka wybrana przez nnU-Net ma obrys (cyjan) i elipsę (bursztyn), klatka FUVAI obrys fioletowy; po analizie na żywo obrys widać we wszystkich klatkach, w których nnU-Net znalazł brzuch.
- **Bez internetu**: wyłączone statystyki Gradio (`GRADIO_ANALYTICS_ENABLED=False`), czcionki systemowe, nic z CDN (sprawdzone w logu sieci przeglądarki: zero zapytań poza `127.0.0.1`). Wyniki „z pamięci” wymagają wcześniejszego `przygotuj_demo.py`.
- Logika: `src/usg_masa/aplikacja.py` (`analizuj(nagranie, tryb)`), wygląd: `app/demo_usg.py` + `app/styl.css`, testy: `tests/test_aplikacja.py`.

Na scenie:

1. Uruchom aplikację **przed prelekcją** i zostaw kartę otwartą (pierwsze wczytanie nagrania trwa ok. 1 s).
2. Pokazuj w trybie **„z pamięci”**: od razu i z liczbami ze slajdu. „Na żywo” tylko jako bonus (ok. 25 s); przełącz tryb kilka sekund przed kliknięciem, żeby modele zdążyły się wczytać.
3. **F11** (albo Ctrl+Cmd+F na Macu) = pełny ekran. Przy 1920×1080 cały ekran (sterowanie, obrazy, wykres i trzy karty) mieści się w powiększeniu **80%** (Cmd −); pod spodem zostają tylko zwinięte „Własny plik” i „Źródła”. Przy 80% zastrzeżenia mają ok. 16 px na ekranie (w CSS 20 px), etykiety ok. 13 px, atrybucja 12 px. Na projektorze o niższej rozdzielczości zmniejsz dalej albo przewijaj.
4. Nie uruchamiaj w tym czasie notebooka w trybie „na żywo” ani `przygotuj_demo.py` (dwa procesy na MPS).
5. Okienko błędu ma nagłówek „Error” (stały napis Gradio, nie da się go podmienić); sama treść jest po polsku.

## Testy

```bash
cd usg
uv run pytest                 # szybkie, bez sieci
USG_PELNE=1 uv run pytest     # pełne: pobiera wagi, liczy prawdziwy nnU-Net, odpala notebook
```

## Zmierzone czasy

MacBook M1 Pro, 32 GB, akcelerator MPS.

| Co | Czas |
|---|---|
| Pobranie jednego pełnego nagrania ACOUSLIC-AI (351 MB) | ok. 100 s |
| Odczyt spisu treści ZIP-a (bez pobierania całości) | ok. 1 do 6 s |
| Pobranie wag nnU-Net (268 MB) | jednorazowo |
| Pobranie wag FUVAI (277 MB, Google Drive autorów) | jednorazowo |
| Referencja offline (`przygotuj_demo.py`, pełny nnU-Net, co 2. klatka) | 93 do 104 s na nagranie 840-klatkowe |
| To samo dla 128-klatkowego wycinka | ok. 20 s |
| FUVAI offline (`przygotuj_demo.py`, co 2. klatka, 420 z 840) | 13 do 27 s na nagranie 840-klatkowe, zależnie od obciążenia komputera (wycinek: 3 do 6 s) |
| Cały `przygotuj_demo.py`, trzy nagrania już pobrane | ok. 6 min bez FUVAI; FUVAI dokłada ok. 1 min (pierwsze uruchomienie z pobieraniem: ok. 15 min) |
| **Tryb na żywo** (`segmentuj_sweep_szybko`, co 4. klatka, czyli 210 z 840) | ok. 25 do 35 s + 3 do 5 s na uruchomienie predyktora |
| Tryb na żywo, co 8. klatka (105 klatek) | mniej więcej połowa (oszacowanie, nie pomiar) |
| **FUVAI na żywo** (co 4. klatka, czyli 210 z 840; komórka notebooka od początku do końca) | ok. 8 s na nagranie 840-klatkowe (zmierzone 7 do 8,5 s, w tym ok. 0,6 s wczytanie modelu, wagi w pamięci podręcznej systemu); wycinek 128 klatek: ok. 2 do 3 s |
| **Model Sano offline** (`przygotuj_demo.py`, 1 segment na nagranie, CPU) | 0,3 do 0,4 s na nagranie + ok. 0,5 s budowa modelu |
| **Model Sano na żywo** (komórka notebooka od początku do końca, CPU) | 1,2 do 1,7 s, w tym ok. 0,5 do 0,6 s wczytanie modelu (z kontrolą sha256 wag) |
| Komórka modelu Sano w trybie `z_pamieci` | ok. 0,3 s |
| Notebook w trybie `z_pamieci`, od początku do końca | ok. 7 do 8 s, bez sieci |
| Notebook w trybie `na_zywo`, nagranie 840-klatkowe | ok. 35 do 41 s (wycinek: ok. 14 s) |
| **Aplikacja (UI), na żywo** (`app/demo_usg.py`, nagranie 840-klatkowe, od kliknięcia do wyników) | ok. 24 s z modelami wczytanymi w tle; 26 s, gdy kliknąć zaraz po przełączeniu trybu (wczytanie modeli ok. 2 s, wagi w pamięci podręcznej systemu) |
| Aplikacja (UI), z pamięci | ok. 2 s od kliknięcia (z odświeżeniem obrazów i wykresu) |

Model Sano liczy się na **CPU**, bo operacja `Conv3D` nie jest obsługiwana na MPS w torch 2.2. Model ma 33 147 976 parametrów i ogląda jeden segment 16×128×128, więc CPU w zupełności wystarcza.

### Sprawdzenie, czy liczby są sensowne

Obwód brzucha z pełnego nnU-Net i z FUVAI (offline, co 2. klatka) porównany z wartością podaną przez ekspertów w ACOUSLIC-AI dla tych samych nagrań:

| Nagranie | nnU-Net | FUVAI (klatka; p(brzuch)) | AC ekspertów |
|---|---|---|---|
| `0199616b` | 304 mm | 242 mm (klatka 464; 0,77) | 312,3 / 293,8 mm |
| `02d3a9bc` | 257 mm | 273 mm (klatka 282; 0,53, p(głowa) 0,38) | 257,2 mm |
| `02ee26a5` | 242 mm | 209 mm (klatka 50, ta sama co nnU-Net; 0,96) | 217,0 mm |
| `04a04f2e` (wycinek 128 klatek) | 262 mm | brak klatki brzucha (maks. p(brzuch) 0,17) | 271,6 / 249,9 mm |

Tryb na żywo (co 4. klatka): nnU-Net na nagraniu `0199616b` dał 308 mm. FUVAI na żywo: 242 mm / nie znaleziono / 204 mm (klatka 48; 0,85) i nie znaleziono na wycinku. **Wybór klatki FUVAI zależy od gęstości próbkowania**: co 2. klatka daje 242 / 273 / 209 mm, co 4. klatka 242 mm / nic / 204 mm. Na `02d3a9bc` przy co 4. klatce w żadnej z kandydatek brzuch nie jest dla FUVAI najbardziej prawdopodobną klasą, więc wyniku nie podajemy. (Wcześniejsza wersja reguły wybierała tu klatkę, którą sam FUVAI uznawał za głowę, p(głowa) 0,87, i przypadkiem trafiała w 257 mm.)

To są **cztery nagrania, nie walidacja.** Pokazują, że pipeline działa i daje liczby o właściwym rzędzie wielkości. Nie mówią nic o dokładności na innych nagraniach.

## FUVAI: co działa, a co nie

- **Działa:** kod autorów + publiczne wagi (sha256 się zgadza, wczytanie `strict`, 69 157 401 parametrów), na MPS ok. 8 s na całe 840-klatkowe nagranie co 4. klatkę (wycinek: ok. 2 do 3 s).
- **Poza domeną:** autorzy biorą do pomiaru klatki z p(klasa) > 0,9. Na naszych nagraniach „na ślepo” przy co 4. klatce p(brzuch) nigdy nie przekracza 0,9 (maks. 0,77 / 0,37 / 0,85, wycinek 0,17); przy co 2. klatce przekracza go jedna klatka w jednym nagraniu (`02ee26a5`, klatka 50, 0,96).
- **Wybór klatki to nasza adaptacja** (`usg_masa.fuvai.wybierz_klatke`): kandydaci = górne 10% p(brzuch) w danym nagraniu (w praktyce p od ok. 0,02 w górę), w których brzuch jest najbardziej prawdopodobną klasą (argmax), maska nie może dotykać krawędzi kadru; wygrywa najwyższa średnia z p(brzuch) i zgodności elipsy z maską. Brak takiej klatki = „nie znaleziono”.
- **Nie mamy pewności, że wybrana klatka to brzuch.** Na `0199616b` (klatka 464) i `02d3a9bc` (klatka 282) nnU-Net w klatce wybranej przez FUVAI nie widzi brzucha wcale; na `02d3a9bc` FUVAI waha się między brzuchem (0,53) a głową (0,38). Notebook drukuje rozkład klas wybranej klatki i to, czy nnU-Net widzi w niej brzuch.
- Każda klatka osobno (T = 1): sieć ma warstwę ConvLSTM, ale seria klatek pogarsza p(brzuch) na tych danych.
- **Masa z samego AC wzorem Campbella**, nie Hadlock III jak w artykule (HC, AC, FL): ACOUSLIC-AI zawiera tylko brzuch. Autorzy (§2.3.3) w ogóle nie podają masy, gdy brakuje którejkolwiek z płaszczyzn głowy, brzucha lub kości udowej, więc FUVAI w wersji z artykułu nie dałby na ACOUSLIC-AI żadnej masy. Notebook nie podaje masy także wtedy, gdy p(brzuch) wybranej klatki < 0,9, a nnU-Net nie widzi w niej brzucha (pokazuje tylko obrys i AC).
- Maska FUVAI często nie obejmuje dolnej krawędzi brzucha, więc AC bywa mniejsze niż u nnU-Netu.
- Gałąź segmentacji FUVAI ma jedno wyjście niezależne od klasy (n_class=1): maska to „struktura”, nie koniecznie brzuch. Stąd reguła argmax == brzuch.
- Błąd w kodzie autorów (`ConvLSTMCell.init_hidden` tworzy stan na CPU, na MPS kończy się błędem) omijamy owijką po naszej stronie; kodu autorów nie zmieniamy.

## Model Sano: masa w dniu porodu

**Co to jest.** Obrazowy 3D ResNet-18 z pracy *TabAttention: Learning Attention Conditionally on Tabular Data* (Grzeszczyk, Płotka i in., MICCAI 2023, arXiv 2310.18129; Sano Kraków + WUM). Dostaje 16 klatek nagrania brzucha (128×128 px) i zwraca masę **w dniu porodu** w gramach (uczony na badaniach < 24 h przed porodem), z samego obrazu. To nie jest prognoza masy na tygodnie do przodu. To wiersz „3D ResNet-18” z Tab. 1 pracy, a nie pełny TabAttention: modułu uwagi i cech klinicznych w tym checkpoincie nie ma (sprawdzone strukturą pliku wag i konfiguracjami kodu autorów), a cechy kliniczne nie wpływają na wynik. Kod: `src/usg_masa/model_sano.py`, testy: `tests/test_model_sano.py`.

**Skąd wagi.** Plik `TabAttention.pt` (132 673 545 B, sha256 `935afc86237dd8afb4a2cb48884dc837c2db3d19651ed2d1e8d1d7ca8a28a69b`) pochodzi z publicznego linku w README autorów: `https://www.dropbox.com/s/cdswqlhew638ebd/TabAttention.pt?dl=1`. Kod sieci: `SanoScience/Tab-Attention` @ `1a57920b782de9d47adfa87540102ecb34c37d70`, `src/models/resnet.py::r3d_18(msha=False, n_frames=16, input_size=(128, 128))`, `load_state_dict(strict=True)`, 33 147 976 parametrów. Wag pełnego TabAttention (u autorów 170 g / 5,0%) ani BabyNet++ nie opublikowano.

**Licencje.** Kodu autorów ani wag nie redystrybuujemy; repo autorów nie zawiera pliku LICENSE. Publiczna dostępność nie oznacza zgody na redystrybucję.

**Na czym się uczył.** 92 nagrania brzucha w płaszczyźnie standardowej z WUM (3 ośrodki, aparaty GE), badanie < 24 h przed porodem, klipy 5–10 s; ciąże donoszone, 38,3 ± 1,5 tc.; masa urodzeniowa 3 495 ± 507 g. U autorów: mMAE 289 g, mMAPE 8,5% (5-krotna walidacja krzyżowa; który fold to ten plik, nie wiadomo), czyli gorzej niż klinicyści z wzorem Hadlocka (205 g / 5,9%). Najlepszy był pełny TabAttention: obraz + 6 pomiarów biometrycznych, 170 g / 5,0%, lepiej niż klinicyści (jego wag nie ma).

**Preprocessing** (wg pracy §3 i `src/video_data_loader.py` autorów, tryb `val`; krok 3 — skala 0,2 mm/px — to **nasza interpretacja** pracy, loader autorów obrazu nie przeskalowuje):
1. Tylko klatki brzucha, w kolejności nagrania. Autorzy mieli je oznaczone ręcznie; u nas klatka jest „brzuchem”, gdy nnU-Net (wynik co k-tą klatkę) widzi w najbliższej przeanalizowanej klatce `(j // k)·k` pole brzucha > 0. Czarne klatki odpadają.
2. Segmenty jak `chunks(L, n=16, skip=2)` autorów: początek co 46 klatek listy, w segmencie co 3. klatka; niepełne segmenty pomijamy. Gdy nie ma żadnego pełnego: jeden segment dopełniony zerami i flaga `dopelniony`.
3. Piksel 0,28 → 0,2 mm (nasza interpretacja: praca §3 podaje piksel 0,2 mm, a loader autorów tego kroku nie ma; bez niego wynik przesuwa się o 350–800 g, kolumna „Bez zmiany piksela” niżej), potem `int(h/1680·128)` (cv2 INTER_LINEAR), wyśrodkowane dopełnienie zerami do 128×128.
4. Normalizacja segmentu: `(x − 0,5·max)/(0,5·max)`.
5. Wynik nagrania = średnia z segmentów. CPU.

**Wyniki** (klatki brzucha z pełnego nnU-Netu co 2. klatkę, tryb `z_pamieci`; „masa teraz” = AC z nnU-Netu + Campbell):

| Nagranie | AC nnU-Net → masa teraz (wiek wg AC) | Klatek brzucha | Segm. | **Model Sano: masa w dniu porodu** | Wszystkie niepuste klatki zamiast klatek brzucha | Bez zmiany piksela 0,28 → 0,2 mm | Na żywo (nnU-Net co 4. klatkę) |
|---|---|---|---|---|---|---|---|
| `0199616b` | 304 mm → 2 590 g (34,4 tc.) | 71 | 1 | **3 551 g** | 3 656 g (15 segm.) | 3 048 g | 3 703 g* (45 klatek) |
| `02d3a9bc` | 257 mm → 1 652 g (29,9 tc.) | 75 | 1 | **3 855 g** | 3 734 g (14 segm.) | 3 076 g | 3 539 g (49 klatek) |
| `02ee26a5` | 242 mm → 1 373 g (28,4 tc.) | 60 | 1 | **3 898 g** | 3 588 g (14 segm.) | 3 122 g | 4 021 g* (35 klatek) |
| `04a04f2e` (wycinek 128 klatek) | 262 mm → 1 734 g (30,3 tc.) | 32 | 1 | **4 000 g*** | 3 881 g (2 segm.) | 3 634 g | 4 032 g* (32 klatki) |

\* za mało klatek brzucha na pełny segment (potrzeba 46), segment dopełniony zerami: wynik orientacyjny.

**Jak to czytać.**
- Na nagraniach z 28.–34. tc. model wychodzi poza dane, na których się uczył (ekstrapolacja), i każdemu dziecku podaje liczbę typową dla noworodka (3,6–4,0 kg) bez cienia wątpliwości, choć wg AC te płody mają dziś ok. 1,4–2,6 kg. Wynik prawie nie zależy od tego, co model widzi: na wszystkich niepustych klatkach wychodzi 3,6–3,9 kg.
- Mas urodzeniowych dzieci z ACOUSLIC-AI nie znamy, więc **nie ma tu żadnej walidacji**. To też nie jest „prognoza na termin”: model uczono tylko na badaniach < 24 h przed porodem, więc nie ma skąd wiedzieć, ile urośnie płód z 30. tygodnia.
- **Wynik jest kruchy:** sama skala piksela przesuwa go o 350–800 g (bez przeskalowania: 3,0–3,6 kg), a próbkowanie nnU-Netu na żywo o 30–320 g. Nasz „segment” to klatki z kilku przeciągnięć „na ślepo” (np. klatki 50…615 na `0199616b`), a nie ciągły klip w płaszczyźnie standardowej jak u autorów.
- Model nie sygnalizuje, że jest poza swoimi danymi: zawsze zwraca liczbę w gramach. Na prelekcji to pomost do części o modelach językowych („model nie wie, kiedy nie wie”).

## Licencje i źródła

| Zasób | Co jest publiczne | Czego NIE ma | Licencja |
|---|---|---|---|
| **BabyNet** (MICCAI 2022), `SanoScience/BabyNet` @ `7922261975b9542ea440b68a06e50e12eb2f9c10` | kod PyTorch (`src/models/babynet.py`, `train_video.py`, `video_data_loader.py`) | **wag**, danych (225 nagrań, 75 pacjentek, prywatne) | **brak pliku LICENSE**, czyli domyślnie „wszelkie prawa zastrzeżone" |
| **BabyNet++** (CIBM 2023), `SanoScience/BabyNetPlusPlus` @ `39c11428befce9ca1011e0e488d8f77b77bcb801` | kod PyTorch: `src/models/resnet.py::r3d_18(msha=True, …)` + 6 cech klinicznych | **wag** (nie są publicznie dostępne), danych (582 nagrania, 194 pacjentki, prywatne) | **brak pliku LICENSE** |
| **Tab-Attention** (MICCAI 2023, model Sano), `SanoScience/Tab-Attention` @ `1a57920b782de9d47adfa87540102ecb34c37d70` | kod PyTorch (`src/models/resnet.py::r3d_18(msha=False, …)`) + **wagi** `TabAttention.pt` (Dropbox, link z README autorów; 132 673 545 B, sha256 `935afc86237dd8afb4a2cb48884dc837c2db3d19651ed2d1e8d1d7ca8a28a69b`). W środku jest model bazowy 3D ResNet-18, bez modułu TabAttention | wag pełnego TabAttention (z cechami klinicznymi), danych (92 nagrania z WUM, prywatne) | **brak pliku LICENSE**; wagi wskazane w publicznym README autorów; nie oznacza to licencji na redystrybucję. Kodu ani wag nie redystrybuujemy |
| Masa „przez całą ciążę” (Płotka i in., AJOG MFM 2023) | tylko artykuł | kodu, wag, danych | — |
| **FUVAI** (PMB 2022), `SanoScience/FUVAI` @ `9ae74297a90effd43d049a7d29f0326c1a59713c` | definicja sieci `models/fuvai.py::YNet` (bez kodu inferencji) + **wagi** `fuvai_weights.pt` na Google Drive autorów (link z README repo; 276 766 249 B, sha256 `8187b2b2e348ed391fab37a67600e3f8c0a248aa71792accec08c7a9e71e0a10`; czysty `state_dict`, wczytujemy `torch.load(..., weights_only=True)`) | danych (nagrania z WUM, prywatne), kodu inferencji i wyboru klatki | **brak pliku LICENSE** (kodu ani wag nie redystrybuujemy) |
| **Baseline ACOUSLIC-AI**, `DIAGNijmegen/ACOUSLIC-AI-baseline` @ `eccdc9b8c48f77c7c4b30a2c99bd323869e6124c` | nnU-Net 2D **z wagami** (Git LFS, 268 346 966 B, sha256 `50aa1cee62b97638c8bd4de44fea15b0e47d5f4373b220206fbd05a3275fa1a4`) + 1 przykładowy plik `.mha` (53 MB, zwykły plik w gicie, **128 klatek** — wycinek przypadku `04a04f2e-…` ze zbioru Zenodo) | — | kod i wagi: **Apache-2.0**; przykładowy plik to **dane ACOUSLIC-AI → CC BY-NC-SA 4.0** |
| **Dane ACOUSLIC-AI** (Zenodo 12697994, v1.1) | 300 przypadków; każdy to **jeden plik `.mha` z 840 klatkami = 6 przeciągnięć głowicy × 140 klatek** (3 poprzeczne + 3 strzałkowe), 744×562 px, 0,28 mm/px; maski brzucha (1 = płaszczyzna optymalna, 2 = suboptymalna); referencyjne AC w mm na sweep (`circumferences/fetal_abdominal_circumferences_per_sweep.csv`, kolumny `uuid, subject_id, sweep_1_ac_mm … sweep_6_ac_mm`). Ścieżki w ZIP-ie: `acouslic-ai-train-set/images/stacked_fetal_ultrasound/<uuid>.mha` (351 MB nieskompresowane, ~128 MB do pobrania), `acouslic-ai-train-set/masks/…`, `acouslic-ai-train-set/circumferences/…` | wieku ciążowego, HC/FL i masy urodzeniowej | **CC BY-NC-SA 4.0** |

**Kod BabyNet++, FUVAI i Tab-Attention nie jest redystrybuowany w tym repozytorium.** Ponieważ autorzy nie dołączyli licencji, skrypt `przygotuj_demo.py` klonuje go (raz) do `usg/zewnetrzne/` (katalog w `.gitignore`), w przypiętych commitach z tabeli. Wagi (FUVAI z Google Drive autorów do `usg/zewnetrzne/FUVAI_wagi/`, model Sano do `usg/zewnetrzne/TabAttention_wagi/`) i dane też nigdy nie trafiają do gita.

Dane ACOUSLIC-AI wolno używać wyłącznie niekomercyjnie. Każda wizualizacja klatek w notebooku i na slajdach nosi atrybucję: „Dane: ACOUSLIC-AI (Radboudumc, Zenodo 12697994), CC BY-NC-SA 4.0".

## Stan wag BabyNet++

**Wytrenowanych wag BabyNet++ nie ma publicznie** — repozytorium autorów zawiera sam kod. Notebook nie uruchamia BabyNet++ z losowymi wagami (moduł `usg_masa.babynet` i jego testy zostają jako zapis architektury). Nawet z wagami byłby to wynik poza domeną modelu: BabyNet++ uczono na standardowych nagraniach szpitalnych z trzech płaszczyzn, wykonanych mniej niż 24 godziny przed porodem. Publiczny kod nie zawiera też modułu DAFT opisanego w artykule.
