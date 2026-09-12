# Label / ROI guide (lab, public images only)

Do **not** record street cameras for this phase. Draw polygons on CNRPark-EXT full frames, PKLot stills, or the synthetic stub in `data/sample/`.

## File format

Same schema as `data/sample/spots.json` and `configs/rois.example.json`:

```json
{
  "camera_id": "cam_01",
  "image_width": 1280,
  "image_height": 720,
  "spots": [
    {
      "id": "spot_01",
      "polygon": [[x, y], [x, y], [x, y], [x, y]]
    }
  ]
}
```

- Coordinates are **pixel xy** on the frame you will feed the pipeline (after any resize you keep consistent).
- Use **4+ vertices** per stall; order should be clockwise or counter-clockwise, not self-intersecting.
- Scope: **10–30 spots**, one **fixed** camera.
- Label is **not** stored in the ROI file. Occupancy is predicted (free vs occupied). PKLot XML / MetaPKLot COCO already store occupied flags for training later (out of scope here).

## How to draw

1. Open one representative **daylight** frame (night is out of scope).
2. Trace the stall as it appears in that camera (foreshortened trapezoids are expected).
3. Keep IDs stable (`spot_01` …). Temporal smoothing keys on `spot_id`.
4. Copy `configs/rois.example.json` (12-spot grid) and replace polygons.
5. Point `roi_file` in `configs/default.yaml` at your JSON.

Optional: overlay with a short snippet after install:

```python
from pathlib import Path
from parking_mvp.preprocess import load_bgr
from parking_mvp.roi import load_rois, overlay_rois
import cv2
img = load_bgr("data/sample/frames/frame_000.png")
rois = load_rois("data/sample/spots.json")
cv2.imwrite("outputs/roi_preview.png", overlay_rois(img, rois))
```

## PKLot XML (reference)

Each image has a sibling `.xml` with `<space occupied="0|1">` and `<contour><point x y>`. `scripts/prepare_dataset.py pklot-xml` maps those polygons into JSONL. You can promote one XML into `spots.json` for a single camera/day.

## MetaPKLot COCO

`category_id` 0 = empty, 1 = occupied. Use `segmentation` polygons as ROIs. Prefer one camera subset so IDs stay in the 10–30 range.

## What not to label

- Drive lanes, sidewalks, or “maybe a car” regions.
- Spots that are not visible in this camera.
- Night / IR frames for this MVP.
