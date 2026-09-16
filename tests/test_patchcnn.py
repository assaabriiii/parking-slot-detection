import numpy as np
import pytest

from parking_mvp.model_patchcnn import (
    CLASS_LABELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PatchCNNDryRunError,
    PatchCNNOccupancyHead,
    preprocess_patch,
)


def test_dry_run_does_not_import_torch_or_load_weights():
    head = PatchCNNOccupancyHead(weights="weights/approach_a_patchcnn.pt", dry_run=True)
    assert head.model is None
    assert head.is_ready is False
    with pytest.raises(PatchCNNDryRunError):
        head.predict("spot_01", np.zeros((16, 16, 3), dtype=np.uint8))


def test_missing_file_stays_dry_run_even_if_flag_false():
    head = PatchCNNOccupancyHead(weights="weights/does_not_exist.pt", dry_run=False)
    assert head.dry_run is True
    assert head.model is None


def test_head_requests_warped_crops():
    # The classifier is trained on perspective-warped stalls, so the pipeline must
    # not hand it bounding boxes.
    assert PatchCNNOccupancyHead.crop_mode == "warp"


def test_free_is_class_zero():
    assert CLASS_LABELS == ("free", "occupied")


def test_preprocess_patch_matches_training_transform():
    patch = np.full((40, 90, 3), 255, dtype=np.uint8)
    tensor = preprocess_patch(patch, input_size=128)
    assert tensor.shape == (3, 128, 128)
    assert tensor.dtype == np.float32
    # White patch: every channel sits at (1 - mean) / std, in RGB order.
    expected = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    for channel in range(3):
        assert tensor[channel].mean() == pytest.approx(expected[channel], abs=1e-3)


def test_preprocess_patch_converts_bgr_to_rgb():
    patch = np.zeros((20, 20, 3), dtype=np.uint8)
    patch[:, :, 0] = 255  # blue in BGR
    tensor = preprocess_patch(patch)
    assert tensor[2].mean() > tensor[0].mean()


def test_preprocess_patch_survives_empty_crop():
    tensor = preprocess_patch(np.zeros((0, 0, 3), dtype=np.uint8))
    assert tensor.shape == (3, 128, 128)
