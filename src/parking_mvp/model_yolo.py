"""YOLO occupancy stub.

Never instantiates Ultralytics with a Hub filename such as ``yolov8n.pt``.
That would trigger a weight download. Load only if a local file exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parking_mvp.io_schema import SpotStatus


class YOLODryRunError(RuntimeError):
    """Raised when YOLO is requested but no local weights were loaded."""


@dataclass
class YOLOOccupancyHead:
    weights: str | Path | None = None
    dry_run: bool = True
    model: Any = None

    def __post_init__(self) -> None:
        path = Path(self.weights) if self.weights else None
        if self.dry_run or path is None or not path.is_file():
            self.model = None
            self.dry_run = True
            return
        from ultralytics import YOLO  # imported only when a local file exists

        self.model = YOLO(str(path))
        self.dry_run = False

    @property
    def is_ready(self) -> bool:
        return self.model is not None and not self.dry_run

    def predict(self, spot_id: str, patch: np.ndarray) -> SpotStatus:
        if not self.is_ready:
            raise YOLODryRunError(
                "YOLO dry-run: no local weights. Place a .pt at the configured path "
                "and set yolo.dry_run: false. This stub never downloads yolov8n/cls."
            )
        result = self.model.predict(patch, verbose=False)[0]
        label = _label_from_yolo_result(result)
        conf = float(getattr(result, "probs", None).top1conf.item()) if getattr(result, "probs", None) is not None else 0.5
        return SpotStatus(spot_id=spot_id, status=label, confidence=conf, raw_status=label, head="yolo")


def _label_from_yolo_result(result: Any) -> str:
    names = result.names if hasattr(result, "names") else {}
    if getattr(result, "probs", None) is not None:
        top = int(result.probs.top1)
        name = str(names.get(top, top)).lower()
        if "free" in name or "empty" in name or name in {"0"}:
            return "free"
        return "occupied"
    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        return "occupied"
    return "free"
