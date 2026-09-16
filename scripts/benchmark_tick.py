"""CPU tick latency: read → preprocess → occupancy head → temporal vote.

  PYTHONPATH=src python scripts/benchmark_tick.py --config configs/default.yaml
  PYTHONPATH=src python scripts/benchmark_tick.py --config configs/default.yaml --roi-file configs/rois.example.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_mvp.io_util import list_frames, load_config
from parking_mvp.pipeline import OccupancyPipeline
from parking_mvp.preprocess import load_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure per-tick CPU latency.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--roi-file", default=None)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--out",
        default="outputs/benchmark_cpu.json",
        help="JSON path under the repo (or - for stdout only).",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    pipe = OccupancyPipeline(cfg, roi_path=args.roi_file or cfg["roi_file"])
    frames = list_frames(ROOT / cfg["source"] if not Path(cfg["source"]).is_absolute() else cfg["source"])
    images = [load_bgr(p) for p in frames]

    for _ in range(args.warmup):
        pipe.infer_bgr(images[0], source="warmup")

    times: list[float] = []
    last = None
    for i in range(args.repeats):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        last = pipe.infer_bgr(image, source=str(frames[i % len(frames)]), frame_index=i)
        times.append(time.perf_counter() - t0)

    times_sorted = sorted(times)
    p95_i = min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))
    payload = {
        "device": "cpu",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "occupancy_head_requested": cfg.get("occupancy_head"),
        "head_used": last.spots[0].head if last and last.spots else None,
        "notes": last.notes if last else None,
        "n_spots": len(pipe.rois.spots),
        "n_frames_source": len(frames),
        "repeats": args.repeats,
        "tick_seconds_config": pipe.tick_seconds,
        "target_latency_seconds": cfg.get("target_latency_seconds", 3),
        "mean_s": statistics.mean(times),
        "p95_s": times_sorted[p95_i],
        "max_s": max(times),
        "min_s": min(times),
        "under_3s": statistics.mean(times) < 3.0,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out != "-":
        dest = ROOT / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text + "\n", encoding="utf-8")
        print("wrote", dest, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
