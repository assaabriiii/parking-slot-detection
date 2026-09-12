"""Majority-vote temporal smoothing over a few frames per spot."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from parking_mvp.io_schema import OccupancyLabel, SpotStatus


class TemporalSmoother:
    def __init__(self, window: int = 5, min_votes: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.min_votes = int(min_votes)
        self._hist: dict[str, deque[OccupancyLabel]] = defaultdict(lambda: deque(maxlen=self.window))
        self._last: dict[str, OccupancyLabel] = {}

    def update(self, spots: list[SpotStatus]) -> list[SpotStatus]:
        smoothed: list[SpotStatus] = []
        for spot in spots:
            if spot.status != "unknown":
                self._hist[spot.spot_id].append(spot.status)
            label = self._vote(spot.spot_id, fallback=spot.status)
            self._last[spot.spot_id] = label
            smoothed.append(spot.model_copy(update={"status": label, "raw_status": spot.status}))
        return smoothed

    def _vote(self, spot_id: str, fallback: OccupancyLabel) -> OccupancyLabel:
        hist = list(self._hist[spot_id])
        if not hist:
            return fallback
        counts = Counter(hist)
        free_n = counts.get("free", 0)
        occ_n = counts.get("occupied", 0)
        if occ_n == free_n:
            return self._last.get(spot_id, fallback)
        winner: OccupancyLabel = "occupied" if occ_n > free_n else "free"
        if max(occ_n, free_n) < min(self.min_votes, len(hist)):
            return self._last.get(spot_id, winner)
        return winner
