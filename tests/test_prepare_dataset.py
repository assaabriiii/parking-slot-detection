import json
from pathlib import Path

from scripts.prepare_dataset import from_cnrpark_patches, from_metapklot_coco, from_pklot_xml


def test_cnrpark_folder_labels(tmp_path: Path):
    busy = tmp_path / "A" / "busy"
    free = tmp_path / "B" / "free"
    busy.mkdir(parents=True)
    free.mkdir(parents=True)
    (busy / "20150703_1425_32.jpg").write_bytes(b"x")
    (free / "20150703_0910_01.jpg").write_bytes(b"x")
    rows = from_cnrpark_patches(tmp_path, limit=None)
    by_status = {r["status"] for r in rows}
    assert by_status == {"occupied", "free"}


def test_pklot_xml(tmp_path: Path):
    xml = tmp_path / "lot.xml"
    xml.write_text(
        """<?xml version="1.0"?><parking>
        <space id="1" occupied="1"><contour>
          <point x="1" y="2"/><point x="3" y="2"/><point x="3" y="4"/><point x="1" y="4"/>
        </contour></space>
        <space id="2" occupied="0"><contour>
          <point x="5" y="5"/><point x="6" y="5"/><point x="6" y="6"/><point x="5" y="6"/>
        </contour></space>
        </parking>
        """,
        encoding="utf-8",
    )
    rows = from_pklot_xml(tmp_path, limit=None)
    assert rows[0]["status"] == "occupied"
    assert rows[1]["status"] == "free"
    assert rows[0]["polygon"][0] == [1, 2]


def test_metapklot_coco(tmp_path: Path):
    payload = {
        "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "occupied"}],
        "images": [{"id": 1, "file_name": "UFPR04/Sunny/a.jpg"}],
        "annotations": [
            {
                "id": 9,
                "image_id": 1,
                "category_id": 0,
                "bbox": [0, 0, 10, 10],
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                "car_id": -1,
            }
        ],
    }
    path = tmp_path / "spots.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = from_metapklot_coco(path, None)
    assert rows[0]["status"] == "free"
    assert len(rows[0]["polygon"]) == 4
