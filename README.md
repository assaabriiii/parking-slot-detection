# Parking occupancy lab MVP (phase 1)

Fixed-camera stall occupancy (**free** / **occupied**) from **public datasets** only. No self-captured street video. **Night scenes are out of scope.**

This repo is the lab sample for the phase-1 feasibility brief (10–30 spots, one fixed camera, ≥90% in suitable daylight, update every 2–5 s, response &lt; 3 s).

**Done in-repo:** pipeline, Patch CNN head, leak-free Colab protocols, CPU latency, temporal flicker test, sequence demo, paper pack under `docs/`.

**Colab P1 (2026-09-16):** Patch CNN **99.7%** on lot `pucpr`, unseen days, sunny, 24 spots. Weights: `weights/approach_a_patchcnn.pt` (gitignored). Default config uses `occupancy_head: patchcnn` and falls back to OpenCV if that file is missing.

**Still open:** unseen-lot (P2) transfer; public oblique / curb-side number.

## What you get

- OpenCV occupancy heuristic on ROI crops (fallback if no `.pt`)
- **Patch CNN** head (approach A) — default when `weights/approach_a_patchcnn.pt` is present
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

Status: **local lab pack executed** (latency, temporal, overlays, docs). **Accuracy ≥90% is not closed** until Colab P1 is re-run.

### 0. Re-run the local pack

```bash
python -m pip install -r requirements.txt pytest
PYTHONPATH=src:tests:. pytest
PYTHONPATH=src python scripts/benchmark_tick.py
PYTHONPATH=src python scripts/eval_temporal_flips.py
PYTHONPATH=src python scripts/export_demo_overlays.py
streamlit run app/streamlit_app.py   # Sequence mode; Play advances every tick_seconds
```

Paper pack (already written):

- `docs/requirements.md`
- `docs/camera_geometry.md`
- `docs/feasibility_report.md`
- `docs/test_report.md`
- `docs/project2_plan.md`
- `docs/lab_results/` — JSON + overlay PNGs from this machine

### 1. Honest accuracy (≥90% target) — Colab (open)

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

### 3. Latency &lt; 3 s — **done on this lab CPU**

JSON in `docs/lab_results/`. OpenCV ~0.5 ms (12 spots); Patch CNN architecture ~26 ms (12 spots, random weights). Re-run:

```bash
PYTHONPATH=src python scripts/benchmark_tick.py --roi-file configs/rois.example.json
```

Colab T4 times still do not replace this.

### 4. Update cadence 2–5 s — **lab demo done**

`streamlit run app/streamlit_app.py` → **Sequence (2–5 s tick)** → Play.

Fallback: `PYTHONPATH=src python scripts/export_demo_overlays.py`

Temporal flicker: `PYTHONPATH=src python scripts/eval_temporal_flips.py` (2 raw flips → 0 smoothed).

### 5. Street / oblique gap — **procedure only**

The brief is **curbside, oblique, 10–50 m**. PKLot is overhead lots. Cover the gap without recording streets:

1. One public oblique set (e.g. ACPDS) or a licensed public traffic still
2. Draw **10–30** polygons → `rois.json` (not the 12-spot dummy grid)
3. Run the PKLot Patch CNN **zero-shot** on that scene
4. Put the accuracy drop in the report as a known limitation / Project 2 input

Follow `scripts/label_guide.md`. Night / rain are out of scope; report them as limits.

### 6. Paper pack — **written**

| Brief deliverable | File |
| --- | --- |
| Feasibility report | `docs/feasibility_report.md` |
| Requirements | `docs/requirements.md` |
| Camera geometry | `docs/camera_geometry.md` |
| Labeling guide | `scripts/label_guide.md` |
| Test report | `docs/test_report.md` |
| Project 2 plan | `docs/project2_plan.md` |

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
