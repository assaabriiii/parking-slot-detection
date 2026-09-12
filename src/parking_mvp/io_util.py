"""Config helpers and frame-source listing (files only by default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def list_frames(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXT:
            return [path]
        return [path]
    if path.is_dir():
        files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXT]
        if not files:
            raise FileNotFoundError(f"no images in {path}")
        return files
    raise FileNotFoundError(f"source not found: {source}")


def is_video(path: str | Path) -> bool:
    return Path(path).is_file() and Path(path).suffix.lower() in VIDEO_EXT
