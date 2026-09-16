from pathlib import Path

from parking_mvp.roi import ROISet, SpotROI, crop_spot, load_rois, order_quad, warp_spot
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


def test_order_quad_sorts_corners_clockwise():
    scrambled = np.array([[40, 50], [10, 20], [40, 20], [10, 50]], dtype=np.float32)
    assert order_quad(scrambled).tolist() == [[10, 20], [40, 20], [40, 50], [10, 50]]


def test_warp_spot_returns_square_patch_of_the_stall():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:50, 10:40] = 9
    roi = SpotROI(spot_id="a", polygon=((10, 20), (40, 20), (40, 50), (10, 50)))
    patch = warp_spot(image, roi, out_size=32)
    assert patch.shape == (32, 32, 3)
    # Corners map onto the polygon vertices, so the outer edge samples the
    # boundary and bilinear interpolation bleeds one pixel in. The interior
    # must be the stall.
    assert np.all(patch[2:-2, 2:-2] == 9)


def test_warp_spot_ignores_vertex_order():
    image = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    corners = ((5, 5), (45, 8), (48, 40), (8, 44))
    straight = warp_spot(image, SpotROI(spot_id="a", polygon=corners), out_size=24)
    rotated = warp_spot(image, SpotROI(spot_id="a", polygon=corners[2:] + corners[:2]), out_size=24)
    assert np.array_equal(straight, rotated)


def test_warp_spot_falls_back_for_non_quadrilaterals():
    image = np.full((40, 40, 3), 5, dtype=np.uint8)
    roi = SpotROI(spot_id="tri", polygon=((5, 5), (30, 8), (18, 30)))
    assert warp_spot(image, roi, out_size=16).shape == (16, 16, 3)


def test_scaled_to_moves_polygons_with_the_frame():
    # A 1080p frame is downscaled by preprocess, so ROI pixels must follow it.
    rois = ROISet(
        camera_id="cam",
        image_width=1920,
        image_height=1080,
        spots=(SpotROI(spot_id="a", polygon=((192, 108), (384, 108), (384, 216), (192, 216))),),
    )
    scaled = rois.scaled_to(1280, 720)
    assert scaled.image_width == 1280
    assert scaled.spots[0].polygon == ((128, 72), (256, 72), (256, 144), (128, 144))


def test_scaled_to_is_a_noop_for_matching_or_unknown_sizes():
    rois = load_rois(Path("data/sample/spots.json"))
    assert rois.scaled_to(rois.image_width, rois.image_height) is rois
    unsized = ROISet(camera_id="c", image_width=0, image_height=0, spots=rois.spots)
    assert unsized.scaled_to(640, 480) is unsized
