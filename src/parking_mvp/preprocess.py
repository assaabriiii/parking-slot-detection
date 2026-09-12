"""Light frame preprocess (resize + optional blur)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_bgr(path: str | Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return image


def resize_to_max(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def maybe_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    import cv2

    if ksize is None or int(ksize) <= 1:
        return image
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)


def preprocess(image: np.ndarray, max_width: int = 1280, max_height: int = 720, blur_ksize: int = 3) -> np.ndarray:
    return maybe_blur(resize_to_max(image, max_width, max_height), blur_ksize)
