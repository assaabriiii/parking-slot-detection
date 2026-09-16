"""Patch-CNN occupancy head (approach A).

Mirrors the classifier trained in ``notebooks/parking_occupancy_mvp_colab.ipynb``.
Torch is imported only once a local weight file is found, so the default OpenCV
path still runs without it and nothing is ever fetched from a hub.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from parking_mvp.io_schema import OccupancyLabel, SpotStatus

# These must match the training transform or the weights read a different
# distribution than they were fitted on.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 128
CLASS_LABELS: tuple[OccupancyLabel, OccupancyLabel] = ("free", "occupied")


class PatchCNNDryRunError(RuntimeError):
    """Raised when the patch CNN is requested but no local weights were loaded."""


def preprocess_patch(patch: np.ndarray, input_size: int = INPUT_SIZE) -> np.ndarray:
    """BGR crop to normalised CHW float32, identical to the notebook transform."""
    import cv2

    if patch.size == 0:
        patch = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if patch.shape[:2] != (input_size, input_size):
        patch = cv2.resize(patch, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def build_patch_cnn() -> Any:
    """Architecture from the notebook. Layer names must stay aligned with the .pt."""
    import torch.nn as nn

    class PatchCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(192, 2))

        def forward(self, x: Any) -> Any:
            return self.head(self.features(x).flatten(1))

    return PatchCNN()


def _load_state_dict(path: Path) -> dict[str, Any]:
    import torch

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


@dataclass
class PatchCNNOccupancyHead:
    weights: str | Path | None = None
    dry_run: bool = True
    input_size: int = INPUT_SIZE
    device: str = "cpu"
    model: Any = None

    # The classifier is trained on perspective-warped stalls, not bounding boxes.
    crop_mode = "warp"

    def __post_init__(self) -> None:
        path = Path(self.weights) if self.weights else None
        if self.dry_run or path is None or not path.is_file():
            self.model = None
            self.dry_run = True
            return
        model = build_patch_cnn()
        model.load_state_dict(_load_state_dict(path))
        model.eval()
        self.model = model.to(self.device)
        self.dry_run = False

    @property
    def is_ready(self) -> bool:
        return self.model is not None and not self.dry_run

    def predict(self, spot_id: str, patch: np.ndarray) -> SpotStatus:
        return self.predict_batch([spot_id], [patch])[0]

    def predict_batch(self, spot_ids: list[str], patches: list[np.ndarray]) -> list[SpotStatus]:
        """One forward pass for the whole frame, which is what keeps a 10-30 spot
        camera inside the 3 s per-tick budget."""
        if not self.is_ready:
            raise PatchCNNDryRunError(
                "patchcnn dry-run: no local weights. Export approach_a_patchcnn.pt from "
                "the Colab notebook, place it at the configured path and set "
                "patchcnn.dry_run: false. This head never downloads weights."
            )
        if not spot_ids:
            return []

        import torch

        batch = np.stack([preprocess_patch(p, self.input_size) for p in patches])
        with torch.no_grad():
            logits = self.model(torch.from_numpy(batch).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        results: list[SpotStatus] = []
        for spot_id, row in zip(spot_ids, probs):
            index = int(np.argmax(row))
            label = CLASS_LABELS[index]
            results.append(
                SpotStatus(
                    spot_id=spot_id,
                    status=label,
                    confidence=float(row[index]),
                    raw_status=label,
                    head="patchcnn",
                )
            )
        return results
