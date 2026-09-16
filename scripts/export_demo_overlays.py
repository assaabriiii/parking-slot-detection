"""Save overlay PNGs for every tick — demo evidence without Streamlit.

  PYTHONPATH=src python scripts/export_demo_overlays.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2

from parking_mvp.io_util import list_frames, load_config
from parking_mvp.pipeline import OccupancyPipeline
from parking_mvp.preprocess import load_bgr


def main() -> int:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    pipe = OccupancyPipeline(cfg)
    out = ROOT / "outputs" / "demo_overlays"
    out.mkdir(parents=True, exist_ok=True)
    frames = list_frames(ROOT / cfg["source"])
    for i, path in enumerate(frames):
        bgr = load_bgr(path)
        snap = pipe.infer_bgr(bgr, source=str(path), frame_index=i)
        vis = pipe.overlay_snapshot(bgr, snap)
        dest = out / f"tick_{i:03d}.png"
        cv2.imwrite(str(dest), vis)
        n_free = sum(1 for s in snap.spots if s.status == "free")
        print(f"{dest.name} free={n_free} occupied={len(snap.spots) - n_free}")
    pipe.write_status(snap)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
