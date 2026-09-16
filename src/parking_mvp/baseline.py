"""OpenCV occupancy heuristic on a spot crop (no learned weights)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parking_mvp.io_schema import OccupancyLabel, SpotStatus


@dataclass(frozen=True)
class BaselineConfig:
    # Painted stall markings inside the ROI put an empty stall near 100, so a
    # lower floor labels every stall occupied. See configs/default.yaml.
    laplacian_var_occupied_min: float = 300.0
    mean_occupied_max: float = 95.0


def patch_stats(patch: np.ndarray) -> tuple[float, float]:
    import cv2

    if patch.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    mean = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return mean, lap_var


def classify_patch(patch: np.ndarray, cfg: BaselineConfig | None = None) -> SpotStatus:
    cfg = cfg or BaselineConfig()
    mean, lap_var = patch_stats(patch)
    occupied = lap_var >= cfg.laplacian_var_occupied_min or mean <= cfg.mean_occupied_max
    status: OccupancyLabel = "occupied" if occupied else "free"
    # Confidence is a clipped mix of the two cues (lab heuristic, not calibrated).
    texture_score = min(1.0, lap_var / max(cfg.laplacian_var_occupied_min * 2.0, 1e-6))
    dark_score = min(1.0, max(0.0, (cfg.mean_occupied_max - mean) / max(cfg.mean_occupied_max, 1e-6)))
    if status == "occupied":
        confidence = max(0.55, 0.5 * texture_score + 0.5 * max(dark_score, 0.2))
    else:
        confidence = max(0.55, 1.0 - 0.5 * texture_score)
    return SpotStatus(
        spot_id="",
        status=status,
        confidence=float(min(1.0, confidence)),
        raw_status=status,
        head="opencv",
    )


class OpenCVOccupancyHead:
    def __init__(self, cfg: BaselineConfig | None = None) -> None:
        self.cfg = cfg or BaselineConfig()

    def predict(self, spot_id: str, patch: np.ndarray) -> SpotStatus:
        result = classify_patch(patch, self.cfg)
        return result.model_copy(update={"spot_id": spot_id})
