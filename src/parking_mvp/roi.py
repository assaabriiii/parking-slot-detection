"""Load ROI polygons and crop patches from a frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Point = tuple[int, int]


@dataclass(frozen=True)
class SpotROI:
    spot_id: str
    polygon: tuple[Point, ...]

    def as_np(self) -> np.ndarray:
        return np.array(self.polygon, dtype=np.int32)

    def bounding_box(self, image_shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if image_shape is not None:
            h, w = int(image_shape[0]), int(image_shape[1])
            x0 = max(0, min(x0, w - 1))
            x1 = max(0, min(x1, w))
            y0 = max(0, min(y0, h - 1))
            y1 = max(0, min(y1, h))
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        return x0, y0, x1, y1


@dataclass(frozen=True)
class ROISet:
    camera_id: str
    image_width: int
    image_height: int
    spots: tuple[SpotROI, ...]
    notes: str | None = None

    def by_id(self) -> dict[str, SpotROI]:
        return {s.spot_id: s for s in self.spots}


def load_rois(path: str | Path) -> ROISet:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    spots = []
    for item in payload["spots"]:
        poly = tuple((int(x), int(y)) for x, y in item["polygon"])
        if len(poly) < 3:
            raise ValueError(f"spot {item.get('id')} needs at least 3 polygon vertices")
        spots.append(SpotROI(spot_id=str(item["id"]), polygon=poly))
    return ROISet(
        camera_id=str(payload.get("camera_id", "unknown")),
        image_width=int(payload.get("image_width", 0)),
        image_height=int(payload.get("image_height", 0)),
        spots=tuple(spots),
        notes=payload.get("notes"),
    )


def crop_spot(image: np.ndarray, roi: SpotROI) -> np.ndarray:
    x0, y0, x1, y1 = roi.bounding_box(image.shape)
    return image[y0:y1, x0:x1].copy()


def crop_all(image: np.ndarray, rois: ROISet) -> dict[str, np.ndarray]:
    return {spot.spot_id: crop_spot(image, spot) for spot in rois.spots}


def polygon_mask(image_shape: tuple[int, ...], roi: SpotROI) -> np.ndarray:
    """Binary mask for the polygon (requires OpenCV at call time)."""
    import cv2

    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi.as_np()], 255)
    return mask


def overlay_rois(
    image: np.ndarray,
    rois: ROISet,
    statuses: dict[str, str] | None = None,
) -> np.ndarray:
    import cv2

    canvas = image.copy()
    for spot in rois.spots:
        label = (statuses or {}).get(spot.spot_id, "")
        color = (40, 180, 40) if label == "free" else (40, 40, 200) if label == "occupied" else (200, 200, 40)
        cv2.polylines(canvas, [spot.as_np()], isClosed=True, color=color, thickness=2)
        x0, y0, _, _ = spot.bounding_box(image.shape)
        text = f"{spot.spot_id} {label}".strip()
        cv2.putText(canvas, text, (x0, max(12, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return canvas
