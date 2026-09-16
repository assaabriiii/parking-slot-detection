import numpy as np
import pytest

from parking_mvp.baseline import OpenCVOccupancyHead
from parking_mvp.model_patchcnn import PatchCNNDryRunError
from parking_mvp.pipeline import OccupancyPipeline, build_head


def base_cfg(**overrides):
    cfg = {
        "camera_id": "test_cam",
        "roi_file": "data/sample/spots.json",
        "temporal": {"window": 3, "min_votes": 2},
    }
    cfg.update(overrides)
    return cfg


def test_patchcnn_without_weights_falls_back_to_baseline():
    head = build_head(base_cfg(
        occupancy_head="patchcnn",
        patchcnn={"weights": "weights/missing.pt", "dry_run": True},
    ))
    assert isinstance(head, OpenCVOccupancyHead)


def test_patchcnn_without_fallback_raises():
    with pytest.raises(PatchCNNDryRunError):
        build_head(base_cfg(
            occupancy_head="patchcnn",
            patchcnn={"weights": "weights/missing.pt", "dry_run": True,
                      "fallback_to_baseline": False},
        ))


def test_pipeline_records_the_fallback_in_notes():
    pipe = OccupancyPipeline(base_cfg(
        occupancy_head="patchcnn",
        patchcnn={"weights": "weights/missing.pt", "dry_run": True},
    ))
    assert pipe.fallback_note is not None
    assert "patchcnn" in pipe.fallback_note


def test_opencv_head_leaves_notes_empty():
    pipe = OccupancyPipeline(base_cfg(occupancy_head="opencv"))
    assert pipe.fallback_note is None


def test_predict_frame_scores_every_spot_and_rescales_rois():
    pipe = OccupancyPipeline(base_cfg())
    # spots.json declares 640x360; feed half that and the ROIs must follow.
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    statuses = pipe.predict_frame(image)
    assert [s.spot_id for s in statuses] == [s.spot_id for s in pipe.rois.spots]
    assert pipe.rois_for(image).spots[0].polygon[0] == (10, 90)


def test_roi_rescaling_is_cached_per_frame_size():
    pipe = OccupancyPipeline(base_cfg())
    small = np.zeros((180, 320, 3), dtype=np.uint8)
    assert pipe.rois_for(small) is pipe.rois_for(small)


def test_iter_source_scores_every_sample_frame():
    pipe = OccupancyPipeline(base_cfg(source="data/sample/frames"))
    snaps = list(pipe.iter_source("data/sample/frames"))
    assert len(snaps) == 3
    assert [s.frame_index for s in snaps] == [0, 1, 2]
    assert all(len(s.spots) == 5 for s in snaps)
