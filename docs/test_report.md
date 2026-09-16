# Test report — phase 1 lab

Machine: macOS arm64 (Apple Silicon), CPU. Date: 2026-09-16.

## Unit / offline

```text
PYTHONPATH=src:tests:. pytest
```

42 tests (ROI warp/scale, PatchCNN dry-run + weight-layout vs notebook, temporal vote, sample occupied/free pattern, dataset parsers, YOLO stub).

## Stub occupancy pattern (OpenCV)

Documented: spot_01/03/05 occupied, 02/04 free. Reproduced by `tests/test_sample_pattern.py` and overlay export (`free=2 occupied=3` on all three stub frames).

## Latency vs R8 (&lt; 3 s)

See `docs/lab_results/benchmark_cpu*.json`.

- OpenCV, 5 and 12 ROIs: **&lt; 1 ms** mean.
- Patch CNN, same architecture as Colab, random init, 5 and 12 ROIs: **12–26 ms** mean.

Pass for compute. **Fail as a substitute for a trained-model field test** — weights were untrained.

## Temporal vs R10

`docs/lab_results/temporal_flips.json`: one-tick flicker produces 2 raw flips, **0** after majority vote.

## Demo vs R7 / R11

Sequence mode implemented in `app/streamlit_app.py`. Overlay ticks exported without a browser:

- `outputs/demo_overlays/tick_000.png` … `tick_002.png`

## Accuracy vs R9 (≥90%)

Colab 2026-09-16, lot=`pucpr`, 24 spots, sunny, date-split P1:

| Check | Result |
| --- | --- |
| P1 same-lot unseen days | **99.7% acc / 99.7% balanced / 99.5% occupied recall** (n=576) |
| P2 unseen lot | **Not run** (Hub sample only had `pucpr`) |
| P3 leaky random frames | 97.0% (13 dates leak; not the claim) |
| YOLO overlap occupied recall | 9.8% on P1 test — not a usable occupancy head on PKLot |

**R9 is met under P1 conditions** (fixed PKLot camera, unseen days, daylight). Street / new-camera accuracy remains open.

Trained weights live at `weights/approach_a_patchcnn.pt` (gitignored). Default config: `occupancy_head: patchcnn`, `dry_run: false`, fallback to OpenCV if the file is absent.

## Lighting / weather

Notebook can score cloudy / rainy as stress sets. **Not executed** in this lab session. Rain remains out of scope for the target, in scope as a documented limit.

## Street / oblique

Not executed. Procedure: public ACPDS (or similar) stills, 10–30 polygons, zero-shot Patch CNN, report drop vs P1.

## Defects found and fixed in this phase

1. ROI pixels not following 1080p→720p preprocess (`ROISet.scaled_to`).
2. Patch CNN missing from `src/` (notebook-only).
3. `pydantic` missing from `requirements.txt`.
4. OpenCV threshold 80 labelled every stub stall occupied (paint lines); now 300.
5. Random-frame split inflated accuracy to 100%.
