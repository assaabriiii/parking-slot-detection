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
| `roi` | JSON polygons, bbox crop, overlay |
| `baseline` | Laplacian variance + mean intensity |
| `model_yolo` | Load Ultralytics only if a **local** weight file exists |
| `temporal` | Per-spot deque, majority, tie → previous label |
| `pipeline` | Glue; writes JSON; no webcam index in default config |
| `io_schema` | Pydantic snapshot |

## Occupancy heads

**OpenCV (default).** Empty pavement is brighter and lower-texture than a vehicle. Thresholds live in `configs/default.yaml`. This is a lab baseline, not a production classifier.

**YOLO (optional).** Classification or detection on the crop. Instantiating `YOLO("yolov8n.pt")` would fetch weights; the stub **refuses** that. If `occupancy_head: yolo` and weights are missing, the pipeline **falls back** to OpenCV (`yolo.fallback_to_baseline`).

Training is a non-goal. If you later fine-tune `yolov8n-cls` on CNRPark patches, put the `.pt` under `weights/` and set `dry_run: false`.

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
