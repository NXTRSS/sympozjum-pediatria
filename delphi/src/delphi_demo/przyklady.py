"""Gotowi pacjenci na pokaz (wymyśleni; żadnych danych prawdziwych osób)."""
from __future__ import annotations

PRZYKLADY = {
    "cukrzyca": dict(opis="Kobieta, 60 lat: cukrzyca typu 2 od 50. r.ż., nadciśnienie od 55. r.ż.",
                     plec="K", wiek_teraz=60, historia="E11 50, I10 55"),
    "bez_cukrzycy": dict(opis="Ta sama kobieta, ale BEZ cukrzycy (tylko nadciśnienie od 55. r.ż.)",
                         plec="K", wiek_teraz=60, historia="I10 55"),
    "astma_od_dziecka": dict(opis="Mężczyzna, 45 lat: astma od 6. r.ż., alergiczny nieżyt nosa od 8. r.ż., "
                                  "atopowe zapalenie skóry od 2. r.ż.",
                             plec="M", wiek_teraz=45, historia="L20 2, J45 6, J30 8"),
}
# Choroby, dla których rysujemy krzywe ryzyka (kody ICD-10; „Death" = zgon).
CELE = ["I21", "I50", "I63", "N18", "Death"]
CELE_ASTMA = ["J44", "J18", "F32", "Death"]
