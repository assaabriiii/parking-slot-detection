"""Locks the occupancy pattern that README and docs/acceptance_criteria.md promise.

The stub previously read every stall as occupied: the painted stall marking sits
inside each ROI, so an empty stall still scored ~100 Laplacian variance against a
threshold of 80.
"""

import yaml

from parking_mvp.baseline import BaselineConfig, patch_stats
from parking_mvp.pipeline import OccupancyPipeline
from parking_mvp.preprocess import load_bgr, preprocess
from parking_mvp.roi import crop_all, load_rois

DOCUMENTED = {
    "spot_01": "occupied",
    "spot_02": "free",
    "spot_03": "occupied",
    "spot_04": "free",
    "spot_05": "occupied",
}


def test_default_config_reproduces_the_documented_pattern():
    pipe = OccupancyPipeline(yaml.safe_load(open("configs/default.yaml", encoding="utf-8")))
    snapshot = pipe.run_source(write=False)
    assert {s.spot_id: s.status for s in snapshot.spots} == DOCUMENTED


def test_code_defaults_match_the_shipped_config():
    # A caller constructing BaselineConfig() directly must not get the old
    # threshold that mislabelled every empty stall.
    cfg = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))["baseline"]
    defaults = BaselineConfig()
    assert defaults.laplacian_var_occupied_min == cfg["laplacian_var_occupied_min"]
    assert defaults.mean_occupied_max == cfg["mean_occupied_max"]


def test_threshold_sits_clear_of_both_classes():
    image = preprocess(load_bgr("data/sample/frames/frame_002.png"))
    rois = load_rois("data/sample/spots.json")
    crops = crop_all(image, rois)
    threshold = BaselineConfig().laplacian_var_occupied_min

    variances = {
        spot.spot_id: patch_stats(crops[spot.spot_id])[1] for spot in rois.spots
    }
    free = [v for spot_id, v in variances.items() if DOCUMENTED[spot_id] == "free"]
    occupied = [v for spot_id, v in variances.items() if DOCUMENTED[spot_id] == "occupied"]

    assert max(free) < threshold < min(occupied)
    # Keep a real margin on both sides rather than a value that barely separates.
    assert max(free) * 2 < threshold
    assert min(occupied) > threshold * 2
