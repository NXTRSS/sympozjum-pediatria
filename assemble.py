#!/usr/bin/env python3
"""Składa prezentacja.html: infrastruktura + styl z ALK/Model_Jezyka.html,
slajdy = mix oryginalnych (mechanizm LM) i nowych (slides/*.html)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
ORIG = Path.home() / "Documents/ALK/prezentacje/Model_Jezyka.html"
OUT = ROOT / "prezentacja.html"

# Kolejność slajdów: ("orig", data-label z ALK) lub ("new", plik w slides/)
ORDER = [
    ("new", "n01_title.html"),
    ("new", "n02_alex.html"),
    ("new", "n03_jama.html"),
    ("new", "n04_skala.html"),
    ("orig", "04 Section LM"),
    ("orig", "05 LM Definition"),
    ("orig", "07 Data"),
    ("orig", "08 X and Y"),
    ("orig", "09 RNN Cell"),
    ("orig", "10 Inference Animation"),
    ("orig", "11 Sampling"),
    ("new", "n05_qr.html"),
    ("orig", "12 Section Scale"),
    ("orig", "13 Scale"),
    ("orig", "17 Base Model"),
    ("new", "n06_rozwiazanie.html"),
    ("new", "n07_whisper.html"),
    ("new", "n08_delphi_a.html"),
    ("new", "n09_delphi_b.html"),
    ("new", "n10_maidxo.html"),
    ("new", "n11_section_babynet.html"),
    ("new", "n12_hadlock.html"),
    ("new", "n13_babynet.html"),
    ("new", "n14_apel.html"),
    ("new", "n15_materialy.html"),
    ("orig", "27 Thanks"),
]

NOTES = [
    "Slajd tytułowy. Przedstawiam się: ML Engineer, nie lekarz. Obietnica: za 45 minut zobaczą Państwo od środka, jak działa AI — i dlaczego jest jednocześnie genialna i omylna. UWAGA: przed wejściem na scenę odpalić trening w Colabie!",
    "Zagadka, część 1: Alex, 3 lata bólu, 17 specjalistów. Mama wpisuje objawy i opis MRI do ChatGPT — pada 'tethered cord syndrome'. Neurochirurg potwierdza. Historia prawdziwa (Today.com 2023). Nie puenta, tylko połowa zagadki.",
    "Zagadka, część 2: JAMA Pediatrics 2024 — ten sam ChatGPT na 100 pediatrycznych case challenges: 83% błędów. Uczciwie: GPT-3.5. Pytanie do sali: jak obie historie mogą być prawdziwe? Odpowiedź znajdziemy metodą Państwa: diagnozą różnicową.",
    "Skala: to nie ciekawostka, to codzienność gabinetu. 25% Amerykanów (Gallup), 40 mln pytających ChatGPT o zdrowie, 79% rodziców. I meta-trick: mówię, że od 1. slajdu w tle trenuje się nasz model — wrócimy do niego.",
    "Przejście: żeby rozwiązać zagadkę, musimy zajrzeć pod maskę. Co to jest model języka?",
    "Definicja: rozkład prawdopodobieństwa nad sekwencją tokenów. Model pyta: co powinno być następne? Ta jedna idea to podstawa wszystkiego — od naszego mini-modelu po ChatGPT.",
    "Dane: nazwy + tokeny specjalne % (start) i ! (stop). W wersji na sympozjum trenujemy na łacińskich nazwach chorób — mechanizm identyczny jak dla miejscowości.",
    "X i Y: model uczy się przewidywać następny znak po każdym prefiksie. Jedno przejście przez sieć pokrywa całą nazwę.",
    "Komórka rekurencyjna: wchodzi aktualny znak + stan z poprzedniego kroku. Dlatego model 'pamięta' kontekst. Nie wchodzić głębiej w matematykę — sala nie musi.",
    "SERCE PRELEKCJI. Interaktywne demo krok po kroku: kontekst → rozkład prawdopodobieństwa → wybór tokenu → powtórz. Klikać powoli. Podkreślić: model NIGDY nie mówi 'nie wiem' — zawsze jest rozkład i zawsze coś wybierzemy. [TODO: podmienić na nazwę choroby wygenerowaną z notebooka]",
    "Greedy vs stochastic + temperatura. To wyjaśnia, czemu ChatGPT za każdym razem odpowiada inaczej — i czemu czasem 'odważniej' zmyśla.",
    "QR: cała sala generuje na telefonach (gradio.live), chętni dostają pełny kod (Colab). Dać sali 2-3 minuty zabawy, poprosić o odczytanie najlepszych nazw na głos.",
    "Przejście: od naszego modelu do prawdziwych LLM. Co się zmienia? Tylko skala.",
    "Ten sam cel — P(następny token | kontekst) — 10 rzędów wielkości różnicy. Nasz model: ~1M parametrów, nazwy chorób. GPT: biliony tokenów, cały internet.",
    "Base Model = symulator dokumentów internetowych. Nie 'wie', że jest asystentem — uzupełnia wzorce. Kluczowy pomost do halucynacji.",
    "ROZWIĄZANIE ZAGADKI: oba przypadki to ten sam mechanizm. Trafia po 17 lekarzach, bo w danych są miliony opisów i nie ma silosów specjalizacji. Myli się w 83%, bo generuje najbardziej prawdopodobne, nie prawdziwe. Model nie wie, kiedy nie wie.",
    "Whisper/Nabla (AP 2024): transkrypcja wizyt, 30 tys. klinicystów, halucynacje w ciszy — zmyślone leki, audio kasowane. Wniosek: czujność należy się też AI 'w tle' gabinetu, nie tylko chatbotom.",
    "Delphi-2M (Nature 2025): dosłownie nasz mechanizm, ale tokeny to kody ICD-10, pozycja to wiek, a token stop to... śmierć. GPT-2 o 2,2 mln parametrów — rząd wielkości naszego modelu z sali!",
    "Delphi-2M wyniki: 400k UK Biobank, walidacja 1,93M Duńczyków, >1000 chorób, AUC ~0,76. Uczciwie: rekrutacja 40-70 lat — brak dzieci w danych. Pediatria czeka na swój model.",
    "MAI-DxO (Microsoft 2025): panel wirtualnych lekarzy, 85,5% vs ~20% na 304 najtrudniejszych case NEJM. Zaznaczyć nierówność warunków (lekarze bez internetu i konsultacji). SLAJD DO CIĘCIA przy obsuwie.",
    "Zmiana tonu: sprawa osobista. Ile waży dziecko przed porodem? Tu mogę opowiedzieć historię z własnej rodziny — indukcja zalecona na podstawie szacunku masy.",
    "Hadlock: 4 ręczne pomiary, wzór z 1985, błąd ~7,5-10%, co trzecie oszacowanie poza ±10%, najgorzej przy makrosomii. A od tej liczby zależy indukcja i cesarka.",
    "BabyNet (Sano Kraków, MICCAI 2022; BabyNet++ 2023): model ogląda całe wideo USG z 3 płaszczyzn <24h przed porodem (wtedy znamy ground truth). MAPE 5,1% vs 6,3% ekspertów, zero zmienności międzyoperatorskiej. Open source, finansowanie FNP + UE. Wersja 2023: cała ciąża 16-38 tż.",
    "Apel: oś czasu komercjalizacji. My publikujemy i otwieramy kod (2022), Samsung kupuje Sonio (2024), GE wbudowuje SonoLyst, Chiny doganiają (2025). Puenta: nie czekajmy, aż nam sprzedadzą nasz własny pomysł. Konkret: pilotaże, Rejestr Lekarzy Innowatorów DIL.",
    "Materiały: jeden QR do repo — notebook, slajdy, źródła, zbiory danych. Zachęcić do puszczenia notebooka w domu.",
    "Dziękuję. Pytania.",
]

html = ORIG.read_text(encoding="utf-8")

first_section = html.index('<section data-label="01 ALK Title"')
head = html[:first_section]
tail_start = html.rindex("</deck-stage>")  # rindex: w komentarzach JS w <head> też występuje ten tag
tail = html[tail_start:]

# wytnij sekcje oryginału (tylko z zakresu body między head a tail)
spans = [
    (m.start(), m.group(1))
    for m in re.finditer(r'<section data-label="([^"]+)"', html)
    if first_section <= m.start() < tail_start
]
spans.append((tail_start, None))
orig_sections = {}
for (start, label), (end, _) in zip(spans, spans[1:]):
    if label:
        orig_sections[label] = html[start:end].rstrip() + "\n"

# złóż slajdy
assert len(NOTES) == len(ORDER), f"notes {len(NOTES)} != slides {len(ORDER)}"
parts = []
for i, (kind, ref) in enumerate(ORDER, start=1):
    if kind == "orig":
        chunk = orig_sections[ref]
    else:
        chunk = (ROOT / "slides" / ref).read_text(encoding="utf-8")
    chunk = re.sub(r'(<div class="page-num"[^>]*>)[^<]*(</div>)', rf"\g<1>{i:02d}\g<2>", chunk)
    parts.append(chunk)

body = "\n\n".join(parts) + "\n\n"

# podmień speaker notes — rindex, bo tag jest też wzmiankowany w komentarzu w deck-stage.js
notes_open = head.rindex('<script type="application/json" id="speaker-notes"')
notes_body_start = head.index(">", notes_open) + 1
notes_end = head.index("</script>", notes_body_start)
head = (
    head[:notes_body_start]
    + "\n" + json.dumps(NOTES, ensure_ascii=False, indent=1) + "\n"
    + head[notes_end:]
)
# tytuł dokumentu i miniatura
head = re.sub(r"<title>.*?</title>", "<title>Trafny po 17 lekarzach — AI w pediatrii</title>", head, flags=re.S)
head = head.replace(">Model Języka</text>", ">AI w pediatrii</text>")

# stopka slajdu końcowego
body = body.replace("Kamil Jędryczek · Model Języka", "Kamil Jędryczek · Sympozjum Nowoczesne Technologie w Pediatrii")

OUT.write_text(head + body + tail, encoding="utf-8")
print(f"OK: {OUT} ({OUT.stat().st_size/1e6:.1f} MB, {len(ORDER)} slajdów)")
