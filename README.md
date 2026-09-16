# Parking occupancy lab MVP (phase 1)

Fixed-camera stall occupancy (**free** / **occupied**) from **public datasets** only. No self-captured street video. **Night scenes are out of scope.**

This repo is the lab sample for the phase-1 feasibility brief (10–30 spots, one fixed camera, ≥90% in suitable daylight, update every 2–5 s, response &lt; 3 s). Code and evaluation protocols are in place; the numbers and demo still have to be **run and recorded** — see [How to cover the brief](#how-to-cover-the-brief).

## What you get

- OpenCV occupancy heuristic on ROI crops (default, no weights)
- **Patch CNN** head (approach A) — loads a **local** `.pt` from the Colab notebook; never downloads
- YOLO **stub** that never auto-downloads weights
- Temporal majority vote
- Streamlit lab UI + optional FastAPI `/status`
- Tiny synthetic `data/sample/` (5 spots, 3 frames)
- Colab notebook that trains/compares A vs YOLO-overlap vs YOLO-seg on PKLot, with **lot/date splits** (not a leaky random-frame split)

Default occupancy on the stub (dark / striped rectangles = occupied):

`spot_01` occupied, `spot_02` free, `spot_03` occupied, `spot_04` free, `spot_05` occupied.

## Quick start (no model)

```bash
python -m pip install -r requirements.txt
python scripts/make_sample_data.py
PYTHONPATH=src python -m parking_mvp --config configs/default.yaml
# inspect outputs/last_status.json
```

Optional UI:

```bash
python -m pip install streamlit
streamlit run app/streamlit_app.py
```

Offline tests:

```bash
PYTHONPATH=src:tests:. pytest
```

(`PYTHONPATH` must include the repo root so `scripts.prepare_dataset` imports in tests.)

## How to cover the brief

Do these **in order**. The accuracy claim depends on the Colab run; everything after it is wasted if that split is still leaky or the `.pt` is missing.

### 1. Honest accuracy (≥90% target) — Colab

Upload `notebooks/parking_occupancy_mvp_colab.ipynb` **and** `requirements.txt`. Runtime → GPU (T4). Run all.

The notebook reports **three protocols**. Use them as follows:

| Protocol | What it is | Use in the grant report |
| --- | --- | --- |
| `P1_same_lot_unseen_days` | same camera, **unseen dates** | **This is the ≥90% claim** |
| `P2_unseen_lot` | a lot never seen in training | domain-transfer / risk #2, not the headline |
| `P3_leaky_random_frames` | old random-frame split | only to show why 100% was invalid |

Also copy:

- `reports/feasibility_report.md`
- `outputs/approach_a_patchcnn.pt`
- the protocol table and weather-stress table (cloudy / rainy)

If the log says only **one lot** was found, raise `MAX_HUB_SAMPLES` and re-run. Do not quote P3 as the result.

### 2. Wire the winner into the software

```bash
mkdir -p weights
# copy the Colab file to:
#   weights/approach_a_patchcnn.pt
python -m pip install -r requirements-patchcnn.txt
```

In `configs/default.yaml`:

```yaml
occupancy_head: patchcnn
patchcnn:
  weights: weights/approach_a_patchcnn.pt
  dry_run: false
  device: cpu
```

Then:

```bash
PYTHONPATH=src python -m parking_mvp --config configs/default.yaml
```

If `outputs/last_status.json` has `notes` about falling back to OpenCV, the `.pt` did not load. Do **not** commit `*.pt` (gitignored). Keep a copy outside git or on a drive.

### 3. Latency &lt; 3 s (CPU, not Colab GPU)

Time **one full tick** on a laptop CPU with ~24 spots: read → preprocess → warp ROIs → batched Patch CNN → temporal vote. Colab T4 times do not satisfy the brief’s “lightweight hardware” line.

Record mean and p95. Target: &lt; 3 s per tick. Config already uses `tick_seconds: 3`.

### 4. Update cadence 2–5 s (demo)

Streamlit today scores **one still**. To cover the demo requirement:

- Point `source` at a frame folder or video
- Advance every `tick_seconds` (3)
- Overlay ROIs green/red and show the status table + timestamp

Until that loop exists, a video run via `python -m parking_mvp --source path/to/clip.mp4` plus saved overlays in `outputs/` is the fallback evidence.

Temporal vote (`temporal.window: 5`) should be measured as **false flip rate with vs without** smoothing (passing cars, shadows).

### 5. Street / oblique gap (PKLot is a parking lot)

The brief is **curbside, oblique, 10–50 m**. PKLot is overhead lots. Cover the gap without recording streets:

1. One public oblique set (e.g. ACPDS) or a licensed public traffic still
2. Draw **10–30** polygons → `rois.json` (not the 12-spot dummy grid)
3. Run the PKLot Patch CNN **zero-shot** on that scene
4. Put the accuracy drop in the report as a known limitation / Project 2 input

Follow `scripts/label_guide.md`. Night / rain are out of scope; report them as limits.

### 6. Paper pack the activity table asks for

| Brief deliverable | How to cover it |
| --- | --- |
| Feasibility report | Colab `feasibility_report.md` after the protocol run → copy into `docs/` |
| Requirements | Short checklist: 10–30 spots, 1080p, 2–5 s, ≥90% on P1, night out |
| Camera geometry | One page: 10–50 m, oblique, roughly how many pixels a car is at 1080p |
| Labeling guide | `scripts/label_guide.md` |
| Test report | P1/P2/P3 + CPU latency + weather stress + YOLO occupied-recall failure |
| Project 2 plan | From P2 + oblique drop: per-camera calib, street data, night later |

Do **not** claim night accuracy, city-wide deployment, or street accuracy without the oblique number.

## Config knobs

- `occupancy_head`: `opencv` (default), `patchcnn`, or `yolo`
- `patchcnn.weights`: local `.pt` from the notebook; `input_size` must match training (128)
- `yolo.weights`: local file only; `dry_run: true` skips Ultralytics
- Missing learned weights → fallback to `opencv`, recorded in snapshot `notes`
- `temporal.window` / `min_votes`
- `tick_seconds`: `3` (video stride = `fps * tick_seconds`)

Copy `configs/rois.example.json` and follow `scripts/label_guide.md` for 10–30 stalls on a **daylight** public frame.

## Public datasets (optional, large)

Read `scripts/download_data.md` first. Commands in `scripts/download_data.sh` stay **commented**.

```bash
python scripts/prepare_dataset.py cnrpark-patches --root data/cnrpark --out data/cnrpark/manifest.jsonl --limit 100
```

| Dataset | License | Role |
| --- | --- | --- |
| CNRPark-EXT | ODbL v1.0 | Primary patch occupancy |
| PKLot | CC BY 4.0 | Polygons + occupied + weather (Colab) |
| MetaPKLot | mixed (PKLot CC BY, CNRPark ODbL) | COCO cars/spots |

## Local layout

```text
configs/default.yaml          # source, ROIs, head, temporal window
configs/rois.example.json     # 12-spot example
data/sample/spots.json        # 5 synthetic stalls
notebooks/parking_occupancy_mvp_colab.ipynb
src/parking_mvp/              # pipeline + patchcnn + yolo stub
app/streamlit_app.py
docs/architecture.md
docs/acceptance_criteria.md
```

## Non-goals (this phase)

No self-captured street recording in git, no production multi-camera system, no night model, no claiming 100% from a same-day random split.
