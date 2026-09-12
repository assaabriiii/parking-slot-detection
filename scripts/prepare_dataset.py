"""Convert public parking datasets into a JSONL manifest. Path wiring only.

Examples (run later, after you download):

  python scripts/prepare_dataset.py cnrpark-patches --root data/cnrpark --out data/cnrpark/manifest.jsonl
  python scripts/prepare_dataset.py pklot-xml --root data/pklot/PKLot --out data/pklot/manifest.jsonl --limit 50
  python scripts/prepare_dataset.py metapklot-coco --json spots.json --out data/metapklot/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def from_cnrpark_patches(root: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for jpg in sorted(root.rglob("*.jpg")):
        parts = [p.lower() for p in jpg.parts]
        if "busy" in parts:
            label = "occupied"
        elif "free" in parts:
            label = "free"
        else:
            continue
        rows.append(
            {
                "dataset": "cnrpark",
                "path": str(jpg),
                "status": label,
                "kind": "patch",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def from_pklot_xml(root: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    xml_files = sorted(root.rglob("*.xml"))
    for xml_path in xml_files:
        image_path = xml_path.with_suffix(".jpg")
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        for space in tree.getroot().findall("space"):
            occupied = space.attrib.get("occupied", "0") == "1"
            contour = []
            node = space.find("contour")
            if node is not None:
                for pt in node.findall("point"):
                    contour.append([int(float(pt.attrib["x"])), int(float(pt.attrib["y"]))])
            rows.append(
                {
                    "dataset": "pklot",
                    "path": str(image_path),
                    "xml": str(xml_path),
                    "spot_id": space.attrib.get("id"),
                    "status": "occupied" if occupied else "free",
                    "polygon": contour,
                    "kind": "full_frame_spot",
                }
            )
            if limit and len(rows) >= limit:
                return rows
    return rows


def from_metapklot_coco(json_path: Path, limit: int | None) -> list[dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in payload.get("images", [])}
    cat = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
    rows: list[dict] = []
    for ann in payload.get("annotations", []):
        image = images.get(ann.get("image_id"), {})
        cat_id = int(ann.get("category_id", 0))
        name = str(cat.get(cat_id, cat_id)).lower()
        if name in {"empty", "free"} or cat_id == 0:
            status = "free"
        else:
            status = "occupied"
        seg = ann.get("segmentation") or []
        polygon = []
        if seg and isinstance(seg[0], list) and seg[0]:
            flat = seg[0]
            if len(flat) >= 6 and not isinstance(flat[0], list):
                polygon = [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]
        rows.append(
            {
                "dataset": "metapklot",
                "path": image.get("file_name"),
                "image_id": ann.get("image_id"),
                "spot_id": ann.get("id"),
                "status": status,
                "bbox": ann.get("bbox"),
                "polygon": polygon,
                "car_id": ann.get("car_id"),
                "kind": "coco_spot",
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build occupancy manifests from public datasets.")
    parser.add_argument("command", choices=["cnrpark-patches", "pklot-xml", "metapklot-coco"])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="MetaPKLot COCO JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.command == "cnrpark-patches":
        if args.root is None:
            parser.error("--root required")
        rows = from_cnrpark_patches(args.root, args.limit)
    elif args.command == "pklot-xml":
        if args.root is None:
            parser.error("--root required")
        rows = from_pklot_xml(args.root, args.limit)
    else:
        if args.json is None:
            parser.error("--json required")
        rows = from_metapklot_coco(args.json, args.limit)

    _write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
