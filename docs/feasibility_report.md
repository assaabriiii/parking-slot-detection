# Feasibility report — phase 1 (project form 26 Mordad)

Team: فناوران شرق. Scope: one fixed camera, 10–30 stalls, binary occupancy, laboratory MVP. Night is out of scope.

## Question

Can a simple fixed camera plus a light vision pipeline report stall **free / occupied** well enough in daylight to justify Project 2 (engineering sample + field pilot)?

## Approach chosen

1. **OpenCV heuristic** on each ROI (texture + darkness) — always-on baseline, no weights.
2. **Patch CNN (approach A)** — perspective-warp each polygon to 128×128, classify free/occupied. Matches the original spec’s per-ROI classifier. Trained in `notebooks/parking_occupancy_mvp_colab.ipynb` on public **PKLot** (CC BY 4.0).
3. **YOLOv8n boxes / masks + ROI overlap** — zero-shot COCO, no parking labels. Kept as a comparison; occupied recall on overhead lots was poor in earlier Colab runs (~4–10%).

Default software path is **patchcnn** when `weights/approach_a_patchcnn.pt` exists (`dry_run: false`). If the file is missing, the pipeline falls back to OpenCV and records that in snapshot `notes`. Nothing auto-downloads.

## Data

| Source | License | Use |
| --- | --- | --- |
| Synthetic `data/sample/` | in-repo | wiring, demo, CPU timing |
| PKLot via Voxel51 Hub | CC BY 4.0 | labelled polygons + occupancy |
| CNRPark-EXT / MetaPKLot | ODbL / mixed | optional; see `scripts/download_data.md` |

**Domain gap:** PKLot is elevated parking lots, not oblique curb at 10–50 m. That gap is Risk #2 in the form. Geometry of a 1080p / 60° camera is in `docs/camera_geometry.md`. An oblique public set (ACPDS or similar) is still required before claiming street accuracy.

No self-captured street video is in git.

## Evaluation protocols (accuracy)

An earlier Colab run split frames of **one lot on one day** at random and reported **100%** Patch-CNN accuracy. That split leaks the same parked cars into train and test. It is **not** a phase-1 result.

The notebook now reports:

| Protocol | Meaning | Role |
| --- | --- | --- |
| P1 same lot, unseen dates | deployment on a fixed camera | **carries the ≥90% target** |
| P2 unseen lot | new camera | transfer / Project 2 calibration |
| P3 leaky random frames | old protocol | quantify inflation only |

**Status (Colab 2026-09-16, GPU T4, 2000 Hub samples):** only lot `pucpr` (sunny), 120 frames / 24 spots / 15 dates, occupied rate 47%. P2 skipped (no second lot in the pull).

| Protocol | n (stalls) | Accuracy | Balanced acc | Occupied recall | Notes |
| --- | --- | --- | --- | --- | --- |
| **P1 same lot, unseen days** | 576 | **99.7%** | **99.7%** | **99.5%** | 9 train / 3 val / 3 test dates; **this is the ≥90% claim** |
| P2 unseen lot | — | — | — | — | Hub sample had only `pucpr` |
| P3 leaky random frames | 576 | 97.0% | 97.1% | 96.9% | 13 dates on both sides; not the headline |

Patch-CNN mean latency on Colab GPU: **22 ms**/frame (24 spots). YOLO-overlap test: 38.9% acc, occupied recall 9.8%. YOLO-seg: 34.5% acc, occupied recall 3.1%. Winner: **A_PatchCNN**.

R9 is **closed on PKLot / one fixed lot / unseen days / sunny**. It is **not** a street-curb or unseen-camera result.

## Latency (this machine, closed)

Target: &lt; 3 s per tick on limited hardware. Measured with `scripts/benchmark_tick.py` on macOS arm CPU. Full JSON: `docs/lab_results/`.

| Head | Spots | Mean tick | p95 | &lt; 3 s |
| --- | --- | --- | --- | --- |
| OpenCV | 5 | 0.44 ms | 0.60 ms | yes |
| OpenCV | 12 | 0.48 ms | 0.72 ms | yes |
| Patch CNN (random weights, architecture only) | 5 | 12 ms | 14 ms | yes |
| Patch CNN (random weights) | 12 | 26 ms | 29 ms | yes |

Random weights measure **compute**, not accuracy. A trained `.pt` from Colab has the same architecture, so tick time stays on the same order (tens of ms), well under 3 s. Config `tick_seconds: 3` is the **update cadence**, not the compute time.

## Temporal vote

Scenario: occupied stall, one-tick free glitch (passing car / shadow), occupied again.

| | Flips |
| --- | --- |
| Raw (window=1) | 2 |
| Smoothed (window=5, min_votes=3) | 0 |

Source: `docs/lab_results/temporal_flips.json`.

## Demo

- Still + **sequence** mode: `streamlit run app/streamlit_app.py` (advances every `tick_seconds`).
- Static evidence: `scripts/export_demo_overlays.py` → `outputs/demo_overlays/tick_*.png` (pattern free=2 occupied=3 on the stub).
- FastAPI `GET /status` reads `outputs/last_status.json`.

## Limitations

- P1 99.7% is same-lot, unseen **days**, PKLot overhead, sunny only — not a new camera and not curb-side street.
- P2 (unseen lot) was not run; FiftyOne pull contained only `pucpr`.
- No oblique street zero-shot number.
- YOLO occupied recall on overhead cars is a known miss (~3–10%).
- Night / heavy rain not evaluated (out of scope).
- Streamlit sequence on the **3-frame stub** is a lab loop, not a 10–30 stall street mount.

## Go / no-go for Project 2

**Conditional go:** the pipeline, ROI tooling, light CNN, CPU budget, and temporal vote are in place. **Do not** fund a city pilot until P1 ≥90% is logged on Colab and an oblique scene is scored. Project 2 should assume **per-camera ROI + fine-tune**, not a universal street model.
