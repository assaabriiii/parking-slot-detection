from pathlib import Path

from parking_mvp.roi import SpotROI, crop_spot, load_rois
import numpy as np


def test_load_sample_rois():
    rois = load_rois(Path("data/sample/spots.json"))
    assert rois.camera_id == "sample_cam"
    assert len(rois.spots) == 5
    assert rois.spots[0].spot_id == "spot_01"
    assert len(rois.spots[0].polygon) == 4


def test_example_has_twelve_spots():
    rois = load_rois(Path("configs/rois.example.json"))
    assert len(rois.spots) == 12


def test_bounding_box_and_crop():
    roi = SpotROI(spot_id="a", polygon=((10, 20), (40, 20), (40, 50), (10, 50)))
    assert roi.bounding_box() == (10, 20, 40, 50)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:50, 10:40] = 7
    crop = crop_spot(image, roi)
    assert crop.shape == (30, 30, 3)
    assert int(crop.mean()) == 7


def test_bbox_clamped_to_image():
    roi = SpotROI(spot_id="b", polygon=((-5, -5), (200, -5), (200, 200), (-5, 200)))
    assert roi.bounding_box((50, 60)) == (0, 0, 60, 50)
