"""Słownik tokenów Delphi-2M (stałe wspólne dla wszystkich modułów).

Numeracja jak w data/ukb_simulated_data/labels.csv autorów: wiersz N pliku = token N modelu.
W plikach .bin tokeny są przesunięte o 1 (0 dokłada dopiero get_batch), patrz dane.py.
"""
from __future__ import annotations

from dataclasses import dataclass

PADDING = 0
BRAK_ZDARZENIA = 1          # „No event": wstawiany co ~5 lat, mówi modelowi, ile lat ma pacjent „teraz"
KOBIETA = 2
MEZCZYZNA = 3
STYL_ZYCIA = tuple(range(4, 13))   # BMI / palenie / alkohol × niski / średni / wysoki; w danych syntetycznych: 0 wpisów
PIERWSZA_CHOROBA = 13
SMIERC = 1269
LICZBA_TOKENOW = 1270
IGNOROWANE = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)   # = ignore_tokens w config/train_delphi_demo.py
DNI_W_ROKU = 365.25
CO_ILE_LAT_BRAK_ZDARZENIA = 5   # = no_event_token_rate w configu autorów


@dataclass(frozen=True)
class Zdarzenie:
    """Jeden wpis historii: token (płeć / choroba / zgon / „brak zdarzenia") i wiek w latach."""
    token: int
    wiek_lat: float
