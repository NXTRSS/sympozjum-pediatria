"""Pliki zrzutów dla decku zapasowego (assemble.py --zapas). Kamil dokłada je w slides/zapas/."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PLIKI_ZAPAS = [
    "slides/zapas/s14_lm_model_summary.png",
    "slides/zapas/s16_lm_krzywa_bledu.png",
    "slides/zapas/s17_lm_histogramy.png",
    "slides/Grdaio_app.png",
    "slides/Gradio_2.png",
    "slides/zapas/s30_delphi_pacjent.png",
    "slides/zapas/s31_delphi_trening.png",
    "slides/zapas/s32_delphi_app_prognoza.png",
    "slides/zapas/s33_delphi_krzywe.png",
    "slides/zapas/s34_delphi_generuj.png",
    "slides/zapas/s44_usg_nr1.png",
    "slides/zapas/s45_usg_nr3.png",
    "slides/zapas/s46_usg_fuvai_klatki.png",
]

BRAKUJACE = [p for p in PLIKI_ZAPAS if not (ROOT / p).is_file()]


@pytest.mark.skipif(
    bool(BRAKUJACE),
    reason="Brak plików zrzutów (dodaje Kamil): " + ", ".join(BRAKUJACE),
)
@pytest.mark.parametrize("sciezka", PLIKI_ZAPAS)
def test_zrzut_zapas_istnieje(sciezka: str):
    assert (ROOT / sciezka).is_file()


def test_lista_zrzutow_zapas_niepusta():
    assert len(PLIKI_ZAPAS) == 13