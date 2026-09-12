"""Pydantic schemas for occupancy outputs (last_status.json)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

OccupancyLabel = Literal["free", "occupied", "unknown"]


class SpotStatus(BaseModel):
    spot_id: str
    status: OccupancyLabel
    confidence: float = Field(ge=0.0, le=1.0)
    raw_status: OccupancyLabel | None = None
    head: str = "opencv"


class OccupancySnapshot(BaseModel):
    camera_id: str
    timestamp: datetime
    source: str
    frame_index: int = 0
    tick_seconds: float = 3.0
    spots: list[SpotStatus]
    notes: str | None = None

    @classmethod
    def now(
        cls,
        *,
        camera_id: str,
        source: str,
        spots: list[SpotStatus],
        frame_index: int = 0,
        tick_seconds: float = 3.0,
        notes: str | None = None,
    ) -> OccupancySnapshot:
        return cls(
            camera_id=camera_id,
            timestamp=datetime.now(timezone.utc),
            source=source,
            frame_index=frame_index,
            tick_seconds=tick_seconds,
            spots=spots,
            notes=notes,
        )
