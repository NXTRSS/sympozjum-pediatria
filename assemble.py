#!/usr/bin/env python3
"""Składa prezentacja.html: infrastruktura + styl z szablonu decku,
slajdy = sekcje wspólne z szablonu (podziękowanie) + nowe (slides/*.html).

Szablon jest w repozytorium: `public-template/published-deck.html` — to opublikowana
wcześniej wersja tej samej prezentacji (daje styl, skrypty, pasek postępu i slajd
końcowy). Składanie działa więc bez żadnych plików z zewnątrz.
Zmienną `SZABLON_DECK` można wskazać inny szablon (np. własną wersję decku).

Slajdy z szablonu nie trafiają do wyniku poza sekcjami wymienionymi w ORDER jako
("orig", …) — treść bierzemy zawsze z slides/*.html.
"""
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parent
ORIG = Path(os.environ.get("SZABLON_DECK") or ROOT / "public-template" / "published-deck.html")
OUT = ROOT / "prezentacja.html"

# Kolejność slajdów: ("orig", data-label z ALK) lub ("new", plik w slides/)
ORDER = [
    ("new", "n01_title.html"),
    ("new", "n02_alex.html"),
    ("new", "n03_jama.html"),
    ("new", "n04_skala.html"),
    # o*.html = sekcje o mechanizmie modelu języka, przerobione na choroby
    ("new", "o04_section_lm.html"),
    ("new", "o05_lm_definition.html"),
    ("new", "o07_data.html"),
    ("new", "o08_x_and_y.html"),
    ("new", "o09_rnn_cell.html"),
    ("new", "o10_inference.html"),   # dane animacji: slides/inference_steps.js
    ("new", "o11_sampling.html"),
    ("new", "n05_qr.html"),
    ("new", "o12_section_scale.html"),
    ("new", "o13_scale.html"),
    ("new", "o17_base_model.html"),
    ("new", "n06_rozwiazanie.html"),
    ("new", "n07_whisper.html"),
    ("new", "n07b_ogolne_vs_medyczne.html"),
    ("new", "n07c_kiedy_dedykowany.html"),
    ("new", "n08_delphi_a.html"),
    ("new", "n09_delphi_b.html"),
    ("new", "n10_maidxo.html"),
    ("new", "n11_section_babynet.html"),
    ("new", "n12_hadlock.html"),
    ("new", "n13_babynet.html"),
    ("new", "n13b_demo_usg.html"),
    ("new", "n13c_model_sano.html"),
    ("new", "n14_apel.html"),
    ("new", "n15_materialy.html"),
    ("orig", "27 Thanks"),
]

NOTES = [
    "Slajd tytułowy. Przedstawiam się: Staff ML Engineer, wykładam ML w Akademii Leona Koźmińskiego, nie lekarz. Obietnica: za 45 minut zobaczą Państwo od środka, jak działa AI — i dlaczego jest jednocześnie genialna i omylna. UWAGA: przed wejściem na scenę odpalić trening w Colabie!",
    "Zagadka, część 1: Alex, 3 lata bólu (od 4. r.ż.), 17 lekarzy (nie tylko specjaliści — był też np. dentysta i ortodonta). Mama wpisuje objawy i opis MRI do ChatGPT — pada 'tethered cord syndrome'. Neurochirurg potwierdza. Historia prawdziwa (Today.com, 12.09.2023; mama i neurochirurg dziecięca opowiadały ją w podcaście NEJM AI Grand Rounds, 11.2024). Nie puenta, tylko połowa zagadki.",
    "Zagadka, część 2: JAMA Pediatrics 2024 — ten sam ChatGPT na 100 pediatrycznych case challenges: 83% błędów. Uczciwie: GPT-3.5. Pytanie do sali: jak obie historie mogą być prawdziwe? Odpowiedź znajdziemy metodą Państwa: diagnozą różnicową.",
    "Skala: to nie ciekawostka, to codzienność gabinetu. 25% dorosłych Amerykanów użyło narzędzia AI po informację lub poradę zdrowotną (West Health–Gallup, badanie 27.10–22.12.2025, próba 5660 osób, publ. 14.04.2026); ponad 40 mln osób DZIENNIE pyta ChatGPT o zdrowie i co 4. z ponad 800 mln tygodniowych użytkowników pyta o zdrowie co tydzień (raport OpenAI, I 2026 — dane własne firmy, globalnie).",
    "Przejście: żeby rozwiązać zagadkę, musimy zajrzeć pod maskę. Co to jest model języka?",
    "Definicja: rozkład prawdopodobieństwa nad sekwencją tokenów. Model pyta: co powinno być następne? Ta jedna idea to podstawa wszystkiego — od naszego mini-modelu po ChatGPT.",
    "Dane: 1292 łacińskie nazwy chorób + tokeny specjalne % (start) i ! (stop). Token = pojedynczy znak: 26 liter, spacja, myślnik, apostrof — razem ze specjalnymi 32 tokeny.",
    "X i Y: model uczy się przewidywać następny znak po każdym prefiksie. Jedno przejście przez sieć pokrywa całą nazwę. Spacja (␣) też jest tokenem — model sam uczy się, kiedy zacząć drugi człon nazwy.",
    "Komórka rekurencyjna: wchodzi aktualny znak + stan z poprzedniego kroku. Dlatego model 'pamięta' kontekst. Nie wchodzić głębiej w matematykę — sala nie musi.",
    "SERCE PRELEKCJI. (Slajd o komórce RNN wycięty — wystarczy zdanie: sieć czyta nazwę litera po literze i niesie w sobie „pamięć” o tym, co już przeczytała.) Interaktywne demo krok po kroku: kontekst → rozkład prawdopodobieństwa → wybór tokenu → powtórz. Klikać powoli. Ślad prawdziwy — nazwa 'pertus panalismus' z modelu w notebooku (próba n=553, seed losowania 10553). Pokazać dwa momenty: '%' — faworyt 'c' (24%), wylosowano 'p' (20%); '%pertus' — faworyt drugie 's' (50%), a wypada spacja (48%) i zaczyna się drugi człon. Pert- brzmi jak pertussis — przypadek dla sali pediatrycznej, a nazwy w zbiorze nie ma. Podkreślić: model NIGDY nie mówi 'nie wiem' — zawsze jest rozkład i zawsze coś wybierzemy.",
    "Wybór zachłanny (zawsze najbardziej prawdopodobny) vs losowy (zgodnie z prawdopodobieństwem) + temperatura. To wyjaśnia, czemu ChatGPT za każdym razem odpowiada inaczej — i czemu czasem 'odważniej' zmyśla.",
    "QR: lewy kod → sekcja „Demo na żywo” w README (link gradio.live wpisuje się rano 28.11); prawy kod → Colab z uproszczonym notebookiem (trening + lista nazw u siebie, bez aplikacji webowej). Salę kierujemy na lewy QR. Dać 2–3 minuty zabawy, poprosić o odczytanie nazw na głos.",
    "Przejście: od naszego modelu do prawdziwych LLM. Co się zmienia? Tylko skala.",
    "Ten sam cel — P(następny token | kontekst) — ok. 7 rzędów wielkości różnicy w liczbie parametrów. Nasz model: ok. 80 tys. parametrów (80 672), 32 tokeny, 1292 nazwy chorób. GPT-4: słownik publicznego tokenizera ok. 100 tys. kawałków słów; liczba parametrów szacunkowo ~10^12 (nieoficjalnie) — ok. 10 milionów razy więcej; uczony na bilionach tokenów z internetu.",
    "Ok. 1 min. Uczciwość wobec sali: to, co pokazałem (nasz model i 'skala'), to dopiero etap 1, czyli model bazowy. On tylko 'śni internet': na pytanie z quizu dopisuje kolejne pytania z quizu, bo tak wyglądają takie strony. Żeby powstał asystent typu ChatGPT, dochodzą jeszcze dwa etapy. SFT: dostrajanie na rozmowach pisanych przez ludzi (pytanie → wzorowa odpowiedź); model uczy się roli i formy, czyli naśladuje. RL / RLHF: model generuje wiele odpowiedzi, ludzie (RLHF) albo automatyczny sprawdzian (np. w matematyce i kodzie) je oceniają, a model wzmacnia lepiej ocenione. Analogie dla sali (biblioteka → staż → superwizja) są tylko ilustracją. Najważniejsza jest dolna karta, bo to pomost do rozwiązania zagadki na następnym slajdzie (puentę „nie wie, kiedy nie wie” zostawić tam): SFT i RL zmieniają, JAKI tekst jest najbardziej prawdopodobny, ale nie zmieniają mechanizmu. Odpowiedź to nadal losowanie następnego tokenu. RL może nagradzać 'nie wiem' i zmniejszać halucynacje, ale ich nie usuwa. Gdy ktoś zapyta: w raporcie technicznym GPT-4 (OpenAI 2023) kalibracja pewności modelu po dostrajaniu była gorsza niż modelu bazowego. Asystent bywa więc PEWNIEJSZY w tonie, a nie lepiej świadomy swojej niewiedzy. Nazwy modeli z dopiskiem -Instruct/-Chat to właśnie wersje po etapach 2–3.",
    "ROZWIĄZANIE ZAGADKI: oba przypadki to ten sam mechanizm. Trafia po 17 lekarzach, bo w danych są miliony opisów i nie ma silosów specjalizacji. Myli się w 83%, bo generuje najbardziej prawdopodobne, nie prawdziwe. Model nie wie, kiedy nie wie.",
    "Whisper/Nabla (AP, 26–27.10.2024): transkrypcja wizyt, ponad 30 tys. klinicystów, ok. 7 mln wizyt. Halucynacje także w ciszy. Przykład „hiperaktywowane antybiotyki” pochodzi z testu badaczy na samym Whisperze, nie z karty pacjenta Nabli. Nabla: oryginalnego nagrania domyślnie nie zostawiają. Wniosek: czujność należy się też narzędziom w tle gabinetu, nie tylko chatbotom.",
    "Praktyczne pytanie z sali brzmi zwykle: to czego używać? Nature Medicine 2026 (NYU): GPT-5.2, Gemini 3.1 Pro i Claude Opus 4.6 kontra OpenEvidence i UpToDate Expert AI. Ogólne wygrały na egzaminie, na kryteriach lekarzy i — co najważniejsze — na 100 prawdziwych pytaniach lekarzy ocenianych na ślepo (12/12 oceniających). Klamra do zagadki: 83% błędów z JAMA 2024 to był GPT-3.5; dziś ogólne modele mają 97% na egzaminie. Uczciwie: Matters Arising (wrzesień 2026) — egzamin mógł wyciec do treningu, HealthBench jest od OpenAI; autorzy podtrzymują wniosek na ślepej ocenie. To migawka rynku, nie wyrok na wieki. Inna miara niż NOHARM: tu oceniano odpowiedzi na pytania, tam zalecenia konsultacyjne — dlatego wyniki o bezpieczeństwie się nie wykluczają. Jeśli ktoś zapyta o odmowy UpToDate (19%) — przecież „nie wiem” to zaleta? Tak, ale tylko gdy idzie w parze z faktyczną niewiedzą. Tu odmawiał także na pytania, na które odpowiedź istniała i którą inne systemy dały poprawnie — to filtr zakresu, a nie skalibrowana niepewność.",
    "Odpowiedź na pytanie, które zostaje po poprzednim slajdzie (czy to koniec modeli medycznych?) — i JEDYNE miejsce puenty: do rozmowy skala, do pomiaru i predykcji z Państwa danych model dedykowany. NOHARM (1100 zadań z konsultacji, 10 specjalności) — ogólne modele miały ryzyko poważnej szkody do 24,6% przypadków; ponad 80% poważnych błędów to pominięcia; narzędzia kliniczne były bezpieczniejsze. Lang1 (NYU): po dotrenowaniu na zadaniach szpitala model ok. 1 mld parametrów bije do 70× większe ogólne; wstępnie widział 80 mld tokenów z dokumentacji (i tekst z internetu). Mistral Large 4 tylko wspomnieć: 6.10.2026, nieformalny test Maziyara Panahiego, 617/669, triage 238/240, zero poważnych pomyłek, statystyczny remis na szczycie; autor sam mówi, że test jest już nasycony — ważny jest wątek otwartych wag, nie liczba. Puenta prowadzi do Delphi-2M i BabyNet: to są właśnie modele dedykowane, uczone na danych, których ogólny chatbot nigdy nie widział. SLAJD DO CIĘCIA przy obsuwie — ale wtedy puentę powiedzieć ustnie na poprzednim slajdzie (Nature Medicine). Hospital-bench (Zavo, Can Zehebi, 12.11.2026): gdy model ma samodzielnie wykonywać pracę w systemach szpitala, najwięcej zadań zaliczonych w całości to 17 z 48 (Claude Opus 5.5); najwyższy średni wynik to 62,4% (95% CI 53,0–71,3%, GPT-6.2 Astra), w całości 14 z 48. Jednym z powtarzających się błędów było udostępnianie dokumentacji pacjenta, której polityka zabraniała wydać. Uczciwie: fikcyjny szpital w USA, dane syntetyczne, zadania administracyjne, nie medyczne, raport firmy bez recenzji.",
    "Delphi-2M (Nature 2025, Shmatko i in.): ten sam mechanizm autoregresji. Słownik ma 1270 tokenów: choroby ICD-10 (poziom 3), śmierć, płeć, styl życia (BMI, palenie, alkohol po trzy poziomy), token „brak zdarzenia” i techniczny padding. W tekście głównym 1258 „stanów” to diagnozy, śmierć i brak zdarzenia; płeć i styl życia są dodatkowym wejściem, którego model nie przewiduje. Zamiast numeru pozycji każdy token dostaje wiek w dniach od urodzenia (sinus i cosinus). Drugi wynik to czas do następnego zdarzenia: konkurujące rozkłady wykładnicze. Śmierć jest tokenem, który model przewiduje (jak chorobę). Token „brak zdarzenia” wstawiają średnio raz na 5 lat, żeby nie modelować wieloletnich pustych przerw. GPT-2 o 2,2 mln parametrów — ok. 30× więcej niż nasz model z sali (ok. 80 tys.), a wciąż maleńki przy ChatGPT.",
    "Delphi-2M wyniki: trening ok. 400 tys. osób UK Biobank, walidacja 1,93 mln Duńczyków, ponad 1000 chorób. Średnie AUC następnej diagnozy ok. 0,76 to walidacja wewnętrzna w UK. Wynik duński 0,67 to inny test, podłużny; ten sam test w UK dał 0,69. Najpewniej model przewiduje zgon (AUC 0,97) — nie mówić, że „najlepiej nowotwory i zawał”. Uczciwie: zaproszenia 40–69 lat (37–73 w chwili rekrutacji) — dzieci w danych nie ma; choroby dziecięce tylko z wywiadu dorosłych (w danych syntetycznych autorów 1,2% diagnoz przypada przed 18. r.ż.). Pediatria czeka na swój model. Po sesji: notebook delphi/ — wpisać historię kogoś z sali (np. E11 w 50 r.ż., I10 w 55) i pokazać ranking oraz krzywe ryzyka (model na danych syntetycznych — poziomy nierealne, kierunki sensowne).",
    "MAI-DxO (Microsoft 2025): panel wirtualnych lekarzy, 85,5% vs ~20% na 304 najtrudniejszych case NEJM. Zaznaczyć nierówność warunków (lekarze bez internetu i konsultacji). SLAJD DO CIĘCIA przy obsuwie.",
    "Zmiana tonu: sprawa osobista. Ile waży dziecko przed porodem? Tu wątek osobisty — wedle uznania.",
    "Hadlock: 4 ręczne pomiary, wzór z 1985, rozrzut: 1 SD ≈ 7,5%, czyli ok. ±15% (Hadlock 1985). ±15% to ±2 SD: w tym zakresie mieści się ok. 95% szacunków, czyli prawie każdy, a nie średni błąd. Przykład: dziecko 3500 g może dostać szacunek od ok. 2975 do 4025 g i to wciąż mieści się w normie metody; tylko ok. 62% oszacowań w ±10% (Chauhan 2005: 54 z 175 prac, łącznie 8895/14384). W badaniu Dittkrist 2022 (11 049 porodów, Hadlock IV, USG do 14 dni przed porodem) w ±10% mieści się ok. 70–72%; w najwyższym przedziale masy średnie niedoszacowanie ok. 12% ± 8% (to nie jest średnia dla każdej makrosomii powyżej 4000 g). A od tej liczby zależy indukcja i cesarka.",
    "BabyNet (Sano Kraków, MICCAI 2022; BabyNet++ 2023): model ogląda całe wideo USG z 3 płaszczyzn <24h przed porodem (wtedy znamy ground truth). BabyNet++ (194 ciąże, 582 nagrania, 4 ośrodki): średni błąd 5,1% vs 5,4% klinicystów — „porównywalny z ekspertami”. UWAGA: wersja z 2022 (BabyNet) przegrała z klinicystami 7,5% vs 6,3%, dopiero średnia model + lekarz dała 5,2% — nie mieszać tych liczb. BabyNet++ bierze też ręczne pomiary biometryczne, więc nie „eliminuje” operatora; jest mniej wrażliwy na drobne błędy pomiarów. Przy niskiej masie i makrosomii jego błąd też rośnie. Kod publiczny (bez licencji); wytrenowane wagi BabyNet++ nie są publicznie dostępne — mówić życzliwie, bez spekulacji dlaczego. Finansowanie: Horizon 2020 (857533), FNP, część NIH. Osobna praca 2023 (AJOG MFM): brzuch + wiek ciąży. W teście 16–38 tc. średnia różnica wobec 4 ekspertów ok. 2,6% (wobec ich szacunku Hadlockiem, nie wobec masy urodzeniowej); z rezydentami ok. 3,8%. Teraz: demo — z prawdziwego nagrania USG.",
    "PLAN CZASU: slajdy 25–26 razem ok. 3 min (+ opcjonalnie 30 s aplikacji). ZACZĄĆ OD ZDANIA: „Nie pokażę Państwu systemu lepszego od Hadlocka — takiego dziś publicznie nie ma. Pokażę, co da się zrobić z otwartych narzędzi, gdzie to się psuje i dlaczego polski model, który był lepszy od klinicystów, jest dziś niedostępny.” Napis przy tytule mówi to samo — mówimy to pierwsi. Schemat (ok. 30 s): nagranie z konkursu ACOUSLIC-AI — 840 klatek, 6 przeciągnięć tanią głowicą podłączoną do smartfona, robione „na ślepo” przez osoby bez doświadczenia w USG (Sierra Leone). Nikt nie zaznaczył brzucha — model sam go znajduje, mierzy obwód i z tej jednej miary liczy masę „teraz” (wzór Campbella). Uczciwie: masa z samego AC to przybliżenie — błąd AC o 10% to błąd masy ok. 30%; Hadlock z 4 pomiarów ma ok. ±15% i ok. 62% oszacowań w ±10%, ale wymaga doświadczonego sonografisty. Mocny punkt tego demo to AUTOMATYZACJA NA TRUDNYCH DANYCH, nie dokładność. Trzy modele (ok. 40 s): holenderski nnU-Net (baseline konkursu, Radboudumc) — działa od początku do końca, ok. 25–35 s na nagranie; polski FUVAI (Sano + WUM, Physics in Medicine & Biology 2022) — ma publiczne wagi, ale uczono go na nagraniach szpitalnych w płaszczyznach standardowych, więc na nagraniach „na ślepo” rzadko jest pewny (próg autorów 0,9) i brzuch wiarygodnie znalazł tylko na nagraniu nr 3 (209 mm vs 217 u ekspertów); „niepewne” = wybrał klatkę, w której nnU-Net brzucha nie widzi. Model Sano: masa w dniu porodu — następny slajd. Tabela (ok. 20 s): nnU-Net na 3 z 4 nagrań kilka–kilkanaście mm od ekspertów, na nr 3 ok. 25 mm za dużo (skośny przekrój). Cztery nagrania to nie walidacja. OPCJONALNIE 30 s APLIKACJI (tylko przy zapasie czasu): aplikacja usg/app/demo_usg.py w przeglądarce (port 7860), tryb „z pamięci” (bez sieci, natychmiast) — nagranie Nr 1, „Znajdź brzuch i zmierz”, pokazać tylko kartę nnU-Net i obrys brzucha na klatce; FUVAI i Sano pominąć. Dane: ACOUSLIC-AI, CC BY-NC-SA 4.0, tylko niekomercyjnie.",
    "Ok. 1,5 min. Polski model bazowy (obrazowy 3D ResNet-18) z pracy TabAttention (Grzeszczyk, Płotka i in., MICCAI 2023), Sano + WUM — publiczny checkpoint. NIE prognozuje masy na tygodnie do przodu: szacuje masę w dniu porodu z badania < 24 h przed porodem (92 nagrania WUM, ciąże donoszone 38,3 ± 1,5 tc., masa ur. 3 495 ± 507 g). Tabela: AC rośnie 242→304 mm, masa „teraz” wg Campbella ok. 1,4–2,6 kg przy 28–34 tc., a model Sano zwraca wartości typowe dla urodzenia (3,6–4,0 kg) — bo tego się uczył. To nie ocena metody jako złej; w swoich danych działa. Wycinek (4 000 g*): segment dopełniony, orientacyjnie. Karta prawa górna: tylko model obrazowy (289 g w pracy) — jedyny publiczny; pełny TabAttention obraz + 6 pomiarów: 170 g, lepiej niż klinicyści 205 g, wag nie ma publicznie. AI wzmacnia pomiary lekarza. Karta prawa dolna: na naszych nagraniach 28–34 tc. czytamy wynik jako szacunek masy urodzeniowej, nie „teraz”; mas urodzeniowych nie znamy. Dla dociekliwych: skalowanie 0,2 mm/px to nasza interpretacja (±350–800 g). Dane ACOUSLIC-AI CC BY-NC-SA, materiał edukacyjny. Pytanie: „na jakich danych to uczono?”. Podziękowania Sano + WUM za publiczne prace. Opcjonalnie jedno zdanie przed QR: archiwizujmy wagi polskich modeli (Zenodo), pilotaże DIL.",
    "Apel: oś czasu — to wyścig o AI w USG płodowym, nie „kradzież pomysłu”. Kraków publikuje BabyNet i udostępnia kod (2022). W 2024 Samsung kupuje Sonio (wykrywanie płaszczyzn i raportowanie), a GE HealthCare przejmuje SonoLyst od Intelligent Ultrasound (automatyczny wybór płaszczyzn i biometria w Volusonach). W 2025 grupa z Shenzhen University (MICCAI) podaje MAPE 5,1% ± 4,6% (166 g) na własnym zbiorze USG 3D. To ten sam rząd liczby co opublikowane 5,1% ± 0,6% BabyNet++, ale inny test i inne pacjentki; ich Hadlock na tych danych miał MAPE 5,0%. Uczciwie: Sonio i SonoLyst nie szacują masy wprost z wideo jak BabyNet — automatyzują płaszczyzny i pomiary. GE w 2024 kupiło dział klinicznego AI Intelligent Ultrasound, który napędza SonoLyst już obecny w Volusonach; Samsung domknął zakup Sonio 30.08.2024. BabyNet zrobił zespół Sano (nie ja). Puenta: nie czekajmy, aż ktoś sprzeda nam gotowy produkt na pomyśle, który powstał w Polsce i którego kod jest publiczny od 2022. Puenta USG (jedno zdanie): polskie modele lepsze od klinicystów — BabyNet++ i pełny TabAttention — opisano w publikacjach, ale ich wytrenowane wagi nie są publicznie dostępne; archiwizujmy (np. Zenodo) i walidujmy to, co sami sfinansowaliśmy. Konkret: pilotaże, Rejestr Lekarzy Innowatorów DIL.",
    "Materiały: jeden QR do repo — notebook, slajdy, źródła, zbiory danych. Wspomnieć dwa notebooki: USG → masa płodu (pokazany dziś) i Delphi-2M (po sesji mogę pokazać na laptopie) — oba chodzą lokalnie, nie w Colabie. Zachęcić do puszczenia notebooka w domu.",
    "Dziękuję. Pytania.",
]

# Slajdy wycięte ze względu na czas (szacunek krytyka: 48–52′ bez Q&A).
# Pliki zostają w slides/ razem z notatkami; przywrócenie = usunięcie nazwy z tego zbioru.
POMINIETE = {"o09_rnn_cell.html", "n10_maidxo.html", "n14_apel.html"}
assert len(NOTES) == len(ORDER), f"notes {len(NOTES)} != slides {len(ORDER)}"
ORDER, NOTES = map(list, zip(*[(o, n) for o, n in zip(ORDER, NOTES) if o[1] not in POMINIETE]))

def _wstaw(po_pliku, pary):
    """Dokłada slajdy zapasu zaraz po podanym pliku. Nic nie robi, gdy pliku nie ma w decku."""
    global ORDER, NOTES
    nowe_o, nowe_n = [], []
    for pozycja, notatka in zip(ORDER, NOTES):
        nowe_o.append(pozycja)
        nowe_n.append(notatka)
        if pozycja[1] == po_pliku:
            for plik, tekst in pary:
                nowe_o.append(("new", plik))
                nowe_n.append(tekst)
    ORDER, NOTES = nowe_o, nowe_n


# Wariant zapasowy (laptop sali bez notebooka na żywo): slajdy ze zrzutów.
# Zwykły deck ich nie dostaje. Na sali, gdy demo działa, te bloki się przeskakuje.
ZAPAS = "--zapas" in sys.argv
if ZAPAS:
    OUT = ROOT / "prezentacja_zapas.html"
    _wstaw("n05_qr.html", [
        ("z01_dane.html",
         "Plan B, zaraz po slajdzie z kodami QR — gdy notebooka nie da się odpalić na laptopie sali. Zapis komórki z danymi: 1292 łacińskie nazwy z pliku w repo, ziarno 42, więc kolejność jest stała. Odczytać dwa, trzy przykłady z listy. Jeśli demo nazw działa na żywo, pomiń z01–z08."),
        ("z02_tokeny.html",
         "Plan B. Tokenizacja: każda nazwa dostaje % na początku i ! na końcu. Słownik ma 32 tokeny — litery, spacja, myślnik, apostrof i te dwa znaki. Macierz treningowa: 1292 wiersze na 60 pozycji. To jest ten sam mechanizm, co na slajdach o X i Y, tylko z prawdziwego uruchomienia."),
        ("z03_model.html",
         "Plan B. Definicja modelu z notebooka: embedding, GRU ze 128 jednostkami i dropout 0,2, na końcu softmax. Kompilacja: entropia krzyżowa i Adam. Nie wchodzić w warstwy głębiej — pokazać, że cały model to kilkanaście linii."),
        ("z04_epoki.html",
         "Plan B. Dwanaście epok na CPU, około sekundy na epokę. Błąd treningu spada z epoki do epoki — model uczy się reguł składania nazw, nie tylko zapamiętywania wierszy. W notatniku widać sam wiersz loss (bez osobnego sprawdzianu)."),
        ("z05_krzywa.html",
         "Plan B. Wykres krzywej błędu z tej samej sesji. Jedna linia — cały zbiór idzie do treningu. Im niżej, tym trafniej następna litera."),
        ("z06_generowanie.html",
         "Plan B. Jedna nazwa, krok po kroku, z tego samego uruchomienia. Na starcie model waha się między c, p, s i a, a wypada d (2%). Im dłuższy kontekst, tym rozkład ostrzejszy — pod koniec dysplasia cordis znak stop ma 99%. Tej nazwy nie ma w zbiorze. Zaznaczyć moment, w którym jedna litera ma prawie 100%."),
        ("z07_nazwy.html",
         "Plan B. Dziesięć losowań z temperatury 0,4. Wszystkie dziesięć nazw jest nowych: morbus dufandei, tystiopathia pertus, syndroma poli i kolejne. Gdyby któraś była w zbiorze, przy niej byłby znacznik „istnieje”. Odczytać dwie na głos."),
        ("z08_gradio.html",
         "Plan B. Tak wygląda aplikacja dla sali: suwak „ile nazw” (tu 3) i „kreatywność” (tu 0,4). Po Submit: angioma — ta jest w zbiorze — oraz pertinia i coplasia, których w zbiorze nie ma. Na laptopie DIL aplikacji nie odpalamy; to zrzut z notebooka uruchomionego wcześniej. Koniec zapasu nazw chorób."),
    ])
    _wstaw("n09_delphi_b.html", [
        ("zd01_model.html",
         "Plan B · Delphi — zaraz po slajdzie z wynikami Delphi. Wczytanie modelu 2,24 mln parametrów i słownika 1270 tokenów z pamięci (TRYB z_pamieci). Jeśli notebook delphi_2m działa na żywo po sesji, pomiń zd01–zd06."),
        ("zd02_trajektoria.html",
         "Plan B · Delphi. Przykładowy pacjent syntetyczny: oś życia z diagnozami w czasie. Podkreślić: nikt taki nie istnieje, dane autorów."),
        ("zd03_krzywa.html",
         "Plan B · Delphi. Krzywa uczenia z logu treningu na danych syntetycznych — ten sam mechanizm co przy nazwach chorób."),
        ("zd04_ranking.html",
         "Plan B · Delphi. Historia E11 50, I10 55 → ranking następnych diagnoz i ryzyka. Liczby edukacyjne, nie kliniczne."),
        ("zd05_krzywe.html",
         "Plan B · Delphi. Porównanie krzywych ryzyka z cukrzycą i bez — z pamięci, N=1000."),
        ("zd06_slad.html",
         "Plan B · Delphi · koniec bloku. Ślad generowania życia token po tokenie (astma od dzieciństwa). Potem wróć do zwykłych slajdów."),
    ])
    _wstaw("n13b_demo_usg.html", [
        ("zu01_nagranie.html",
         "Plan B · USG — zaraz po slajdzie demo USG. Klatka z nagrania ACOUSLIC-AI (TRYB z_pamieci). Jeśli aplikacja lub notebook działa, pomiń zu01–zu07."),
        ("zu02_maska.html",
         "Plan B · USG. nnU-Net przegląda nagranie i wskazuje klatkę z brzuchem — obrys i elipsa AC."),
        ("zu03_pomiar.html",
         "Plan B · USG. Obwód brzucha, tydzień ciąży z Campbell, masa „teraz” — porównanie z ekspertami z CSV."),
        ("zu04_fuvai.html",
         "Plan B · USG. Polski FUVAI na tym samym nagraniu — kiedy jest pewny, kiedy nie."),
        ("zu05_sano.html",
         "Plan B · USG. Model Sano (checkpoint bazowy): masa w dniu porodu z nagrania <24 h przed porodem — tu ekstrapolacja poza dane uczące."),
        ("zu06_aplikacja.html",
         "Plan B · USG. Zrzut aplikacji demo_usg.py: nagranie Nr 1, tryb „z pamięci”, po „Znajdź brzuch i zmierz” (zakładka masa)."),
        ("zu08_fuvai_klatki.html",
         "Plan B · USG. Druga zakładka aplikacji: FUVAI klatka po klatce — paski klas, obrys struktury, oś czasu kolorów argmax."),
        ("zu07_nnunet.html",
         "Plan B · USG · koniec bloku. Ta sama sesja — dwa panele nnU-Net i FUVAI. Wróć do slajdu o modelu Sano lub materiałach."),
    ])
if not ORIG.exists():
    sys.exit(
        f"Brak szablonu decku: {ORIG}\n"
        "Normalnie leży on w repozytorium (public-template/published-deck.html).\n"
        "Inny szablon można wskazać zmienną środowiskową:\n"
        "  SZABLON_DECK=/sciezka/do/decku.html uv run python assemble.py"
    )

html = ORIG.read_text(encoding="utf-8")

# Szablon mógł przejść przez podgląd w przeglądarce, który dokleja własne skrypty i atrybuty — usuwamy je.
html = re.sub(r'<(style|script)[^>]*\sdata-omelette-injected(?:="[^"]*")?[^>]*>.*?</\1>', "", html, flags=re.S)
html = re.sub(
    r'<script[^>]*\ssrc="[^"]*(?:omelette|livereload|__preview|hot-update)[^"]*"[^>]*>\s*</script>', "", html
)
html = re.sub(r'\s(?:data-omelette-injected|data-om-id|data-src-ver|data-om-text-attrs)="[^"]*"', "", html)
# mapa źródeł z podglądu (instrumentacja edytora, nie źródło projektu)
html = re.sub(r'<script[^>]*\sid="__om_srcmap"[^>]*>.*?</script>', "", html, flags=re.S)

# Dokumentacja step-editor.js pokazuje w komentarzu JS dosłowne `</script>`, co kończy element
# <script> wcześniej (HTML nie zagląda w składnię JS): reszta komentarza zostaje niedomknięta.
# W linii komentarza (`*`) zapisujemy to tak, jak zapisuje się przykłady w kodzie: `<\/script>`.
html = re.sub(r"(?m)^\s*\*.*$", lambda m: m.group(0).replace("</script>", "<\\/script>"), html)


class Struktura(HTMLParser):
    """Pozycje znakowe *prawdziwych* tagów HTML w dokumencie.

    HTMLParser nie zagląda w treść `<script>`/`<style>` ani w komentarze, więc przykłady
    w rodzaju `<section …>` z dokumentacji JS decku nie są brane za początek slajdu — a to
    one psuły wyszukiwanie regexem (head ucinany przed prawdziwym `<script id="speaker-notes">`).
    """

    def __init__(self, tekst: str):
        super().__init__(convert_charrefs=True)
        self.offsety = [0]                              # offset początku każdej linii (parser liczy linie po \n)
        for linia in tekst.split("\n"):
            self.offsety.append(self.offsety[-1] + len(linia) + 1)
        self.zdarzenia = []                             # (poz, 'start'|'end', tag, atrybuty, koniec_tagu)
        self.feed(tekst)
        self.close()

    def _poz(self) -> int:
        lineno, offset = self.getpos()
        return self.offsety[lineno - 1] + offset

    def handle_starttag(self, tag, attrs):
        poz = self._poz()
        self.zdarzenia.append((poz, "start", tag, dict(attrs), poz + len(self.get_starttag_text() or "")))

    def handle_startendtag(self, tag, attrs):           # `<br/>` nie otwiera elementu
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        poz = self._poz()
        self.zdarzenia.append((poz, "end", tag, {}, poz))


struktura = Struktura(html)

# prawdziwy <deck-stage> (pierwszy strukturalny) i jego domknięcie
sceny = [z for z in struktura.zdarzenia if z[2] == "deck-stage"]
if not sceny or sceny[0][1] != "start":
    sys.exit(f"W szablonie {ORIG} nie ma elementu <deck-stage> — to nie wygląda na deck.")
scena_start = sceny[0][4]
glebokosc, tail_start = 0, None
for poz, rodzaj, _tag, _attrs, _koniec in sceny:
    glebokosc += 1 if rodzaj == "start" else -1
    if glebokosc == 0:
        tail_start = poz                                # początek domykającego </deck-stage>
        break
if tail_start is None:
    sys.exit(f"W szablonie {ORIG} nie znalazłem domknięcia </deck-stage>.")

# slajdy = sekcje <section data-label=…> wewnątrz deck-stage
dopasowania = [
    z for z in struktura.zdarzenia
    if z[1] == "start" and z[2] == "section" and "data-label" in z[3] and scena_start <= z[0] < tail_start
]
if not dopasowania:
    sys.exit(f"W szablonie {ORIG} nie ma żadnej sekcji <section data-label=…>.")
first_section = dopasowania[0][0]
head = html[:first_section]
tail = html[tail_start:]

# wytnij sekcje szablonu (tylko z zakresu body między head a tail)
spans = [(z[0], z[3]["data-label"]) for z in dopasowania]
spans.append((tail_start, None))
orig_sections = {}
for (start, label), (end, _) in zip(spans, spans[1:]):
    if label:
        orig_sections[label] = html[start:end].rstrip() + "\n"


def sekcja_szablonu(etykieta: str) -> str:
    """Sekcja wspólna z szablonu; etykieta mogła się w nim zmienić (np. „27 Thanks" → „30 Thanks")."""
    if etykieta in orig_sections:
        return orig_sections[etykieta]
    slowo = etykieta.split()[-1].lower()
    pasujace = [e for e in orig_sections if e.split()[-1].lower() == slowo]
    if len(pasujace) == 1:
        return orig_sections[pasujace[0]]
    dostepne = ", ".join(orig_sections) or "—"
    sys.exit(f'W szablonie {ORIG} nie ma sekcji „{etykieta}". Dostępne: {dostepne}')


def podmien(tekst: str, stare: str, nowe: str, co: str) -> str:
    """Podmiana idempotentna: szablonem bywa też deck już złożony, gdzie `nowe` po prostu już jest."""
    if stare in tekst:
        return tekst.replace(stare, nowe)
    if nowe in tekst:
        return tekst
    sys.exit(f"W szablonie nie ma ani starej, ani nowej wersji: {co}")


def wstaw_kroki(skrypt: str, tablica_js: str) -> str:
    """Podmienia `const GENERATION_STEPS = [...]` na tablicę z slides/inference_steps.js.

    Nawiasy liczymy ręcznie, bo tablica jest długa i ma w sobie zagnieżdżone `[…]`.
    """
    i = skrypt.find("const GENERATION_STEPS")
    if i < 0:
        sys.exit("Nie znalazłem GENERATION_STEPS w skrypcie decku (szablon bez animacji inferencji?).")
    j = skrypt.index("=", i) + 1
    while True:                                    # pomiń białe znaki i komentarze przed `[`
        while j < len(skrypt) and skrypt[j].isspace():
            j += 1
        if skrypt.startswith("//", j):
            koniec = skrypt.find("\n", j)
            if koniec < 0:
                sys.exit("GENERATION_STEPS w szablonie: komentarz bez końca linii (szablon zminifikowany?).")
            j = koniec + 1
        elif skrypt.startswith("/*", j):
            j = skrypt.index("*/", j) + 2
        else:
            break
    if not skrypt.startswith("[", j):
        sys.exit("GENERATION_STEPS w szablonie nie jest literałem tablicy — nie wiem, co podmienić.")
    glebokosc, k = 0, j
    while k < len(skrypt):
        if skrypt[k] == "[":
            glebokosc += 1
        elif skrypt[k] == "]":
            glebokosc -= 1
            if glebokosc == 0:
                break
        k += 1
    else:
        sys.exit("GENERATION_STEPS w szablonie: nie znalazłem końca tablicy.")
    k += 1
    if skrypt.startswith(";", k):
        k += 1
    return skrypt[:i] + "const GENERATION_STEPS = " + tablica_js + ";" + skrypt[k:]


# złóż slajdy
assert len(NOTES) == len(ORDER), f"notes {len(NOTES)} != slides {len(ORDER)}"
parts = []
for i, (kind, ref) in enumerate(ORDER, start=1):
    if kind == "orig":
        chunk = sekcja_szablonu(ref)
    else:
        chunk = (ROOT / "slides" / ref).read_text(encoding="utf-8")
    chunk = re.sub(r'(<div class="page-num"[^>]*>)[^<]*(</div>)', rf"\g<1>{i:02d}\g<2>", chunk)
    parts.append(chunk)

body = "\n\n".join(parts) + "\n\n"

# podmień speaker notes — szukamy strukturalnie, bo tag jest też wzmiankowany w komentarzach JS decku
notatki = [
    z for z in struktura.zdarzenia
    if z[1] == "start" and z[2] == "script" and z[3].get("id") == "speaker-notes" and z[0] < first_section
]
if not notatki:
    sys.exit("W szablonie nie ma <script id=\"speaker-notes\"> — nie wiem, gdzie wstawić notatki.")
notes_body_start = notatki[-1][4]
# koniec treści skryptu = domykający </script> wskazany przez parser (ignoruje „</script>" w łańcuchach JS)
domkniecia = [z[0] for z in struktura.zdarzenia if z[1] == "end" and z[2] == "script" and z[0] >= notes_body_start]
if not domkniecia:
    sys.exit("W szablonie <script id=\"speaker-notes\"> nie ma domknięcia </script>.")
notes_end = domkniecia[0]
head = (
    head[:notes_body_start]
    + "\n" + json.dumps(NOTES, ensure_ascii=False, indent=1) + "\n"
    + head[notes_end:]
)
# tytuł dokumentu i miniatura
head = re.sub(r"<title>.*?</title>", "<title>Następne słowo, następna choroba — AI w pediatrii</title>", head, flags=re.S)
head = podmien(head, ">Model Języka</text>", ">AI w pediatrii</text>", "napis na miniaturze")

# animacja inferencji: GENERATION_STEPS z slides/inference_steps.js + teksty bez miejscowości
steps_js = (ROOT / "slides" / "inference_steps.js").read_text(encoding="utf-8").strip()
tablica = re.search(r"^\[.*", steps_js, re.S | re.M)       # sama tablica, bez nagłówkowych komentarzy pliku
assert tablica, "slides/inference_steps.js: nie znalazłem tablicy kroków"
tail = wstaw_kroki(tail, tablica.group(0).rstrip().rstrip(";"))
tail = podmien(
    tail,
    "'Generowanie zakończone. Token STOP (!) sygnalizuje koniec nazwy. Wynik: „angioma cenine\"'",
    "'Generowanie zakończone. Token STOP (!) sygnalizuje koniec nazwy. Wynik: „pertus panalismus\"'",
    "tekst animacji: wynik po STOP",
)

# stopka slajdu końcowego
body = body.replace("Kamil Jędryczek · Model Języka", "Kamil Jędryczek · Sympozjum Nowoczesne Technologie w Pediatrii")
body = podmien(body, "Koniec wykładu", "Koniec prelekcji", "podpis slajdu końcowego")

OUT.write_text(head + body + tail, encoding="utf-8")
print(f"OK: {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(ORDER)} slajdów)")
