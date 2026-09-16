"""End-to-end occupancy pipeline: frame → ROI crops → head → temporal vote → JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from parking_mvp.baseline import BaselineConfig, OpenCVOccupancyHead
from parking_mvp.io_schema import OccupancySnapshot, SpotStatus
from parking_mvp.io_util import is_video, list_frames, load_config
from parking_mvp.model_patchcnn import PatchCNNDryRunError, PatchCNNOccupancyHead
from parking_mvp.model_yolo import YOLODryRunError, YOLOOccupancyHead
from parking_mvp.preprocess import load_bgr, preprocess
from parking_mvp.roi import ROISet, crop_all, load_rois, warp_all
from parking_mvp.temporal import TemporalSmoother

OccupancyHead = OpenCVOccupancyHead | PatchCNNOccupancyHead | YOLOOccupancyHead


def build_head(cfg: dict[str, Any]) -> OccupancyHead:
    kind = str(cfg.get("occupancy_head", "opencv")).lower()
    if kind == "patchcnn":
        patch_cfg = cfg.get("patchcnn") or {}
        head = PatchCNNOccupancyHead(
            weights=patch_cfg.get("weights"),
            dry_run=bool(patch_cfg.get("dry_run", True)),
            input_size=int(patch_cfg.get("input_size", 128)),
            device=str(patch_cfg.get("device", "cpu")),
        )
        if head.is_ready:
            return head
        if patch_cfg.get("fallback_to_baseline", True):
            return OpenCVOccupancyHead(_baseline_cfg(cfg))
        raise PatchCNNDryRunError(
            "patchcnn requested, dry-run, fallback_to_baseline is false"
        )
    if kind == "yolo":
        yolo_cfg = cfg.get("yolo") or {}
        head = YOLOOccupancyHead(
            weights=yolo_cfg.get("weights"),
            dry_run=bool(yolo_cfg.get("dry_run", True)),
        )
        if head.is_ready:
            return head
        if yolo_cfg.get("fallback_to_baseline", True):
            return OpenCVOccupancyHead(_baseline_cfg(cfg))
        raise YOLODryRunError("YOLO requested, dry-run, fallback_to_baseline is false")
    return OpenCVOccupancyHead(_baseline_cfg(cfg))


def _baseline_cfg(cfg: dict[str, Any]) -> BaselineConfig:
    raw = cfg.get("baseline") or {}
    defaults = BaselineConfig()
    return BaselineConfig(
        laplacian_var_occupied_min=float(
            raw.get("laplacian_var_occupied_min", defaults.laplacian_var_occupied_min)
        ),
        mean_occupied_max=float(raw.get("mean_occupied_max", defaults.mean_occupied_max)),
    )


class OccupancyPipeline:
    def __init__(self, cfg: dict[str, Any], roi_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.rois: ROISet = load_rois(roi_path or cfg["roi_file"])
        self.head = build_head(cfg)
        temporal = cfg.get("temporal") or {}
        self.smoother = TemporalSmoother(
            window=int(temporal.get("window", 5)),
            min_votes=int(temporal.get("min_votes", 3)),
        )
        self.camera_id = str(cfg.get("camera_id", self.rois.camera_id))
        self.tick_seconds = float(cfg.get("tick_seconds", 3.0))
        self._roi_cache: dict[tuple[int, int], ROISet] = {}

        requested = str(cfg.get("occupancy_head", "opencv")).lower()
        self.fallback_note: str | None = None
        if requested != "opencv" and isinstance(self.head, OpenCVOccupancyHead):
            self.fallback_note = (
                f"{requested} head had no local weights; used the OpenCV baseline"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> OccupancyPipeline:
        return cls(load_config(path))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        pp = self.cfg.get("preprocess") or {}
        return preprocess(
            image,
            max_width=int(pp.get("max_width", 1280)),
            max_height=int(pp.get("max_height", 720)),
            blur_ksize=int(pp.get("blur_ksize", 3)),
        )

    def rois_for(self, image: np.ndarray) -> ROISet:
        """ROIs rescaled to the frame actually being scored."""
        height, width = image.shape[:2]
        cached = self._roi_cache.get((width, height))
        if cached is None:
            cached = self.rois.scaled_to(width, height)
            self._roi_cache[(width, height)] = cached
        return cached

    def predict_frame(self, image: np.ndarray) -> list[SpotStatus]:
        """Score one preprocessed frame and advance the temporal vote."""
        rois = self.rois_for(image)
        if getattr(self.head, "crop_mode", "bbox") == "warp":
            crops = warp_all(image, rois, out_size=getattr(self.head, "input_size", 128))
        else:
            crops = crop_all(image, rois)

        spot_ids = [spot.spot_id for spot in rois.spots]
        predict_batch = getattr(self.head, "predict_batch", None)
        if predict_batch is not None:
            raw = predict_batch(spot_ids, [crops[spot_id] for spot_id in spot_ids])
        else:
            raw = [self.head.predict(spot_id, crops[spot_id]) for spot_id in spot_ids]
        return self.smoother.update(raw)

    def infer_image(self, image_path: str | Path, frame_index: int = 0) -> OccupancySnapshot:
        image = self._preprocess(load_bgr(image_path))
        return OccupancySnapshot.now(
            camera_id=self.camera_id,
            source=str(image_path),
            spots=self.predict_frame(image),
            frame_index=frame_index,
            tick_seconds=self.tick_seconds,
            notes=self.fallback_note,
        )

    def write_status(self, snapshot: OccupancySnapshot, out_path: str | Path | None = None) -> Path:
        dest = Path(out_path or (self.cfg.get("output") or {}).get("last_status", "outputs/last_status.json"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        return dest

    def run_source(self, source: str | Path | None = None, write: bool = True) -> OccupancySnapshot:
        src = source or self.cfg["source"]
        if is_video(src):
            return self._run_video(Path(src), write=write)
        frames = list_frames(src)
        snapshot = OccupancySnapshot.now(
            camera_id=self.camera_id,
            source=str(src),
            spots=[],
            tick_seconds=self.tick_seconds,
        )
        for i, frame in enumerate(frames):
            snapshot = self.infer_image(frame, frame_index=i)
        if write:
            self.write_status(snapshot)
        return snapshot

    def _run_video(self, path: Path, write: bool = True) -> OccupancySnapshot:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(fps * self.tick_seconds)))
        snapshot = OccupancySnapshot.now(
            camera_id=self.camera_id,
            source=str(path),
            spots=[],
            tick_seconds=self.tick_seconds,
        )
        idx = 0
        kept = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    idx += 1
                    continue
                image = self._preprocess(frame)
                snapshot = OccupancySnapshot.now(
                    camera_id=self.camera_id,
                    source=str(path),
                    spots=self.predict_frame(image),
                    frame_index=kept,
                    tick_seconds=self.tick_seconds,
                    notes=self.fallback_note,
                )
                kept += 1
                idx += 1
        finally:
            cap.release()
        if write:
            self.write_status(snapshot)
        return snapshot
