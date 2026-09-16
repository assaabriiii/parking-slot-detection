"""Load ROI polygons and crop patches from a frame."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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

    def scaled_to(self, width: int, height: int) -> ROISet:
        """Rescale polygons to a frame of a different size.

        Polygons are pixel coordinates on the frame they were drawn on, while
        preprocess caps resolution. A 1080p source is therefore downscaled and an
        unscaled ROI file would point at the wrong pixels.
        """
        if not self.image_width or not self.image_height:
            return self
        if (self.image_width, self.image_height) == (width, height):
            return self
        fx = width / self.image_width
        fy = height / self.image_height
        spots = tuple(
            replace(
                spot,
                polygon=tuple((int(round(x * fx)), int(round(y * fy))) for x, y in spot.polygon),
            )
            for spot in self.spots
        )
        return replace(self, image_width=width, image_height=height, spots=spots)


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


def order_quad(points: np.ndarray) -> np.ndarray:
    """Sort four corners into top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def warp_spot(image: np.ndarray, roi: SpotROI, out_size: int = 128) -> np.ndarray:
    """Perspective-warp a stall to a square patch.

    From an oblique camera a stall is a trapezoid, so its bounding box also holds
    pavement and parts of the neighbouring cars. Warping is what the patch
    classifier is trained on.
    """
    import cv2

    points = np.array(roi.polygon, dtype=np.float32)
    if len(points) != 4:
        patch = crop_spot(image, roi)
        return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    destination = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(order_quad(points), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def warp_all(image: np.ndarray, rois: ROISet, out_size: int = 128) -> dict[str, np.ndarray]:
    return {spot.spot_id: warp_spot(image, spot, out_size) for spot in rois.spots}


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
