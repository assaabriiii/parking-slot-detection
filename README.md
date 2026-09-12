# Parking occupancy lab MVP (phase 1)

Fixed-camera stall occupancy (**free** / **occupied**) from **public datasets** only. No self-captured street video. This repo is a scaffold: code, configs, stub data, and download docs. **Night scenes are out of scope.**

Intended loop when you run it later: update every **2–5 s**, respond in **< 3 s** (documented target, not benchmarked here).

## What you get

- OpenCV occupancy heuristic on ROI crops
- YOLO **stub** that never auto-downloads weights
- Temporal majority vote
- Streamlit lab UI + optional FastAPI `/status`
- Tiny synthetic `data/sample/` (5 spots, 3 frames)
- Wiring for **CNRPark-EXT**, **PKLot**, **MetaPKLot**

Do not train models in this phase.

## Google Colab (later)

```bash
# from the repo root after upload / clone
python -m pip install -r requirements.txt
python scripts/make_sample_data.py
PYTHONPATH=src python -m parking_mvp --config configs/default.yaml
# inspect outputs/last_status.json
```

Streamlit in Colab (optional):

```bash
python -m pip install streamlit
# then use a Colab streamlit tunnel of your choice, or run locally:
streamlit run app/streamlit_app.py
```

Public datasets (optional, large). Read `scripts/download_data.md` first. Commands in `scripts/download_data.sh` stay **commented** so they are not run by accident.

```bash
python scripts/prepare_dataset.py cnrpark-patches --root data/cnrpark --out data/cnrpark/manifest.jsonl --limit 100
```

Offline tests (optional):

```bash
PYTHONPATH=src:tests:. pytest
```

(`PYTHONPATH` must include the repo root so `scripts.prepare_dataset` imports in tests.)

## Local layout

```text
configs/default.yaml          # source, ROIs, head, temporal window
configs/rois.example.json     # 12-spot example
data/sample/spots.json        # 5 synthetic stalls
data/sample/frames/           # generated PNGs
src/parking_mvp/              # pipeline
app/streamlit_app.py
app/fastapi_app.py            # optional
scripts/download_data.md
tests/
docs/architecture.md
docs/acceptance_criteria.md
```

Default occupancy on the stub (dark / striped rectangles = occupied):

`spot_01` occupied, `spot_02` free, `spot_03` occupied, `spot_04` free, `spot_05` occupied.

## Config knobs

- `occupancy_head`: `opencv` (default) or `yolo`
- `yolo.weights`: local file only; `dry_run: true` skips Ultralytics load
- `temporal.window` / `min_votes`
- `tick_seconds`: `3` (use as frame stride on video: `fps * tick_seconds`)

Copy `configs/rois.example.json` and follow `scripts/label_guide.md` to mark 10–30 stalls on a **daylight** CNRPark or PKLot frame.

## Licenses (data)

| Dataset | License | Role |
| --- | --- | --- |
| CNRPark-EXT | ODbL v1.0 | Primary patch occupancy |
| PKLot | CC BY 4.0 | Polygons + occupied + weather |
| MetaPKLot | mixed (PKLot CC BY, CNRPark ODbL) | COCO cars/spots |

## Non-goals

No street recording, no training, no production multi-camera system, no night model.
