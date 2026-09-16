# Architecture — parking occupancy lab MVP (phase 1)

## Purpose

Classify each marked stall in a **single fixed camera** as **free** or **occupied**. Input is an image folder or video from **public datasets** (or the synthetic stub). Output is `outputs/last_status.json` plus an optional Streamlit view.

Night / IR performance is **out of scope**.

## Data flow

```text
frame (file/video)
    → preprocess (resize, light blur)
    → fixed ROI polygons
    → crop per spot
    → occupancy head
         ├─ OpenCV heuristic (default)
         └─ YOLO stub (local .pt only; never auto-download)
    → temporal majority vote (few frames)
    → {spot_id, free|occupied, timestamp}
    → last_status.json / Streamlit
```

Update cadence when a human later runs video: **2–5 s** (`tick_seconds: 3`). Intended response **< 3 s** per tick; this repo does not benchmark runtime.

## Modules (`src/parking_mvp`)

| Module | Role |
| --- | --- |
| `preprocess` | Load BGR, cap resolution |
| `roi` | JSON polygons, bbox crop, perspective warp, rescale, overlay |
| `baseline` | Laplacian variance + mean intensity |
| `model_patchcnn` | Approach-A patch classifier; imports torch only if a **local** `.pt` exists |
| `model_yolo` | Load Ultralytics only if a **local** weight file exists |
| `temporal` | Per-spot deque, majority, tie → previous label |
| `pipeline` | Glue; writes JSON; no webcam index in default config |
| `io_schema` | Pydantic snapshot |

ROI polygons are pixel coordinates on the frame they were drawn on. `preprocess`
caps resolution, so `ROISet.scaled_to` rescales them to whatever frame is actually
scored — without it a 1080p source silently shifts every ROI.

## Occupancy heads

**OpenCV (default).** Empty pavement is brighter and lower-texture than a vehicle. Thresholds live in `configs/default.yaml`. This is a lab baseline, not a production classifier.

**Patch CNN (approach A).** The classifier trained in `notebooks/parking_occupancy_mvp_colab.ipynb`. Each stall polygon is perspective-warped to a 128×128 square and classified free/occupied, with class 0 = free. One batched forward pass covers the whole frame, which is what keeps a 10–30 spot camera inside the 3 s tick budget. Torch is imported only when a local `.pt` is present, so the default install stays numpy + OpenCV.

Export `approach_a_patchcnn.pt` from the notebook, place it at `patchcnn.weights`, `pip install -r requirements-patchcnn.txt`, and set `patchcnn.dry_run: false`. `input_size` must match the training crop size or the weights see the wrong scale.

**YOLO (optional).** Classification or detection on the crop. Instantiating `YOLO("yolov8n.pt")` would fetch weights; the stub **refuses** that.

Any learned head that has no local weights **falls back** to OpenCV (`fallback_to_baseline`) and the fallback is recorded in the snapshot `notes`, so a run can never silently look like it used a model it did not load.

## Datasets (public only)

- **CNRPark-EXT** — patch occupancy (primary). ODbL v1.0. http://cnrpark.it/
- **PKLot** — full frames + XML polygons + occupied. CC BY 4.0. HF mirror `teenygrad/pklot`
- **MetaPKLot** — COCO cars/spots over PKLot + CNRPark. Mixed licenses.

See `scripts/download_data.md`. Do not capture real streets.

## UI

- `app/streamlit_app.py` — lab demo (preferred).
- `app/fastapi_app.py` — `GET /status` reads the JSON file.

## Non-goals (phase 1)

No live street recording, no training loop, no multi-camera fusion, no payment/ANPR, no night model, no phase 2/3 productization.
