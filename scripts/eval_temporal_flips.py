"""False-flip rate with vs without temporal majority vote.

Injects a one-frame occupancy flicker on an otherwise stable stall
(the brief's passing-car / shadow case).

  PYTHONPATH=src python scripts/eval_temporal_flips.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_mvp.io_schema import SpotStatus
from parking_mvp.temporal import TemporalSmoother


def _spot(status: str) -> SpotStatus:
    return SpotStatus(spot_id="s1", status=status, confidence=0.8, head="eval")


def flips(labels: list[str], window: int, min_votes: int) -> int:
    sm = TemporalSmoother(window=window, min_votes=min_votes)
    prev = None
    n = 0
    for label in labels:
        out = sm.update([_spot(label)])[0].status
        if prev is not None and out != prev:
            n += 1
        prev = out
    return n


def main() -> int:
    # 20 ticks occupied, a 1-tick free glitch (passing car), then occupied again.
    sequence = ["occupied"] * 8 + ["free"] + ["occupied"] * 8
    raw = flips(sequence, window=1, min_votes=1)
    smoothed = flips(sequence, window=5, min_votes=3)
    payload = {
        "scenario": "single-tick flicker on an occupied stall (passing car / shadow)",
        "n_ticks": len(sequence),
        "raw_flips_window1": raw,
        "smoothed_flips_window5_minvotes3": smoothed,
        "flip_reduction": raw - smoothed,
        "brief_note": "Phase 1 uses majority vote over a few frames to suppress momentary errors.",
    }
    dest = ROOT / "outputs" / "temporal_flips.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("wrote", dest, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
