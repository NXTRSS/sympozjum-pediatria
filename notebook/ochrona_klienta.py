"""Limit zapytań na klienta (okno 60 s)."""
from __future__ import annotations

from collections import defaultdict, deque


class LimitKlienta:
    def __init__(self, limit: int = 20, okno_s: float = 60.0):
        self.limit = limit
        self.okno_s = okno_s
        self.okna: dict[str, deque[float]] = defaultdict(deque)

    def wpusc(self, klucz: str, teraz: float) -> bool:
        kolejka = self.okna[klucz]
        while kolejka and teraz - kolejka[0] > self.okno_s:
            kolejka.popleft()
        if len(kolejka) >= self.limit:
            return False
        kolejka.append(teraz)
        return True