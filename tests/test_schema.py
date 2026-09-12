from datetime import timezone

from parking_mvp.io_schema import OccupancySnapshot, SpotStatus


def test_snapshot_roundtrip():
    snap = OccupancySnapshot.now(
        camera_id="sample_cam",
        source="data/sample/frames/frame_000.png",
        spots=[
            SpotStatus(spot_id="spot_01", status="occupied", confidence=0.9, head="opencv"),
            SpotStatus(spot_id="spot_02", status="free", confidence=0.7, head="opencv"),
        ],
        frame_index=0,
        tick_seconds=3.0,
    )
    payload = snap.model_dump(mode="json")
    again = OccupancySnapshot.model_validate(payload)
    assert again.camera_id == "sample_cam"
    assert again.timestamp.tzinfo is not None
    assert again.timestamp.tzinfo.utcoffset(again.timestamp) == timezone.utc.utcoffset(again.timestamp)
    assert [s.status for s in again.spots] == ["occupied", "free"]
    text = snap.model_dump_json()
    assert "spot_01" in text
