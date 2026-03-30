from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class VelocityTracker:
    """In-memory tracker for card transaction velocity and last-seen country."""

    window_seconds: int = 600  # 10 minutes
    _timestamps: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _last_country: dict[str, str] = field(default_factory=dict)

    def record(self, card_last_four: str, country: str) -> None:
        now = time.time()
        self._timestamps[card_last_four].append(now)
        self._last_country[card_last_four] = country

    def count_recent(self, card_last_four: str) -> int:
        now = time.time()
        cutoff = now - self.window_seconds
        timestamps = self._timestamps[card_last_four]
        # Prune old entries
        self._timestamps[card_last_four] = [t for t in timestamps if t > cutoff]
        return len(self._timestamps[card_last_four])

    def get_last_country(self, card_last_four: str) -> str | None:
        return self._last_country.get(card_last_four)
