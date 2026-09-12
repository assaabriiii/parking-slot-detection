from parking_mvp.model_yolo import YOLODryRunError, YOLOOccupancyHead
import numpy as np
import pytest


def test_yolo_dry_run_does_not_load_missing_weights():
    head = YOLOOccupancyHead(weights="weights/occupancy.pt", dry_run=True)
    assert head.model is None
    assert head.is_ready is False
    with pytest.raises(YOLODryRunError):
        head.predict("spot_01", np.zeros((16, 16, 3), dtype=np.uint8))


def test_missing_file_stays_dry_run_even_if_flag_false():
    head = YOLOOccupancyHead(weights="weights/does_not_exist.pt", dry_run=False)
    assert head.dry_run is True
    assert head.model is None
