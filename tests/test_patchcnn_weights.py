"""Weight-compatibility tests: the .pt exported by the Colab notebook must load here.

Skipped when torch is absent, which is the default install.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from parking_mvp.model_patchcnn import (  # noqa: E402
    PatchCNNOccupancyHead,
    build_patch_cnn,
)

NOTEBOOK = Path("notebooks/parking_occupancy_mvp_colab.ipynb")


def notebook_patch_cnn():
    """Instantiate the architecture straight out of the notebook cell."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = next(
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code" and "class PatchCNN(nn.Module)" in "".join(c["source"])
    )
    start = source.index("class PatchCNN(nn.Module)")
    end = source.index("def patch_to_tensor")
    namespace = {"nn": torch.nn, "torch": torch}
    exec(source[start:end], namespace)
    return namespace["PatchCNN"]()


def test_architecture_matches_the_notebook():
    ours = build_patch_cnn().state_dict()
    theirs = notebook_patch_cnn().state_dict()
    assert list(ours) == list(theirs), "layer names drifted; exported .pt would not load"
    assert {k: tuple(v.shape) for k, v in ours.items()} == {
        k: tuple(v.shape) for k, v in theirs.items()
    }


def test_head_loads_a_notebook_style_checkpoint(tmp_path):
    # The notebook saves a bare state_dict via torch.save(net.state_dict(), ...).
    weights = tmp_path / "approach_a_patchcnn.pt"
    torch.save(notebook_patch_cnn().state_dict(), weights)

    head = PatchCNNOccupancyHead(weights=weights, dry_run=False)
    assert head.is_ready is True

    patch = np.full((60, 30, 3), 120, dtype=np.uint8)
    status = head.predict("spot_01", patch)
    assert status.spot_id == "spot_01"
    assert status.status in ("free", "occupied")
    assert 0.0 <= status.confidence <= 1.0
    assert status.head == "patchcnn"


def test_batch_and_single_predictions_agree(tmp_path):
    weights = tmp_path / "w.pt"
    torch.save(notebook_patch_cnn().state_dict(), weights)
    head = PatchCNNOccupancyHead(weights=weights, dry_run=False)

    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 255, (40, 40, 3), dtype=np.uint8) for _ in range(5)]
    ids = [f"spot_{i:02d}" for i in range(5)]

    batched = head.predict_batch(ids, patches)
    assert [s.spot_id for s in batched] == ids
    for spot_id, patch, expected in zip(ids, patches, batched):
        single = head.predict(spot_id, patch)
        assert single.status == expected.status
        assert single.confidence == pytest.approx(expected.confidence, abs=1e-5)


def test_empty_batch_is_allowed(tmp_path):
    weights = tmp_path / "w.pt"
    torch.save(notebook_patch_cnn().state_dict(), weights)
    head = PatchCNNOccupancyHead(weights=weights, dry_run=False)
    assert head.predict_batch([], []) == []


def test_pipeline_runs_end_to_end_with_real_weights(tmp_path):
    from parking_mvp.pipeline import OccupancyPipeline

    weights = tmp_path / "w.pt"
    torch.save(notebook_patch_cnn().state_dict(), weights)
    pipe = OccupancyPipeline({
        "camera_id": "cam",
        "roi_file": "data/sample/spots.json",
        "occupancy_head": "patchcnn",
        "patchcnn": {"weights": str(weights), "dry_run": False},
    })
    assert pipe.fallback_note is None

    image = np.zeros((360, 640, 3), dtype=np.uint8)
    statuses = pipe.predict_frame(image)
    assert len(statuses) == 5
    assert all(s.head == "patchcnn" for s in statuses)
