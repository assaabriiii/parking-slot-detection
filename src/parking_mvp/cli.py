"""CLI: python -m parking_mvp --config configs/default.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

from parking_mvp.pipeline import OccupancyPipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parking occupancy MVP (files/video, no webcam by default).")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default=None, help="Override image, folder, or video path.")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    pipe = OccupancyPipeline.from_yaml(args.config)
    snap = pipe.run_source(args.source, write=not args.no_write)
    n_free = sum(1 for s in snap.spots if s.status == "free")
    n_occ = sum(1 for s in snap.spots if s.status == "occupied")
    print(f"{snap.camera_id} frame={snap.frame_index} free={n_free} occupied={n_occ} source={snap.source}")
    if not args.no_write:
        print(f"wrote {Path((pipe.cfg.get('output') or {}).get('last_status', 'outputs/last_status.json'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
