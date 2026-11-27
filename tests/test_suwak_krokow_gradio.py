"""Parametry suwaka kroków — maximum musi być > minimum (ograniczenie Gradio)."""


def _gora_suwaka(liczba_krokow: int) -> tuple[int, int]:
    """Ta sama logika co _suwak_krokow w uruchom_aplikacje (bez importu Gradio)."""
    if liczba_krokow <= 0:
        return 0, 1
    return 0, max(liczba_krokow - 1, 1)


def test_minimum_mniejsze_od_maximum():
    for n in (0, 1, 2, 5, 30):
        lo, hi = _gora_suwaka(n)
        assert lo < hi, (n, lo, hi)