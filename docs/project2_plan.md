# Project 2 plan (from phase-1 risks)

Phase 1 is feasibility + lab MVP. Project 2 in the form is: engineering sample, higher robustness, **field pilot** (~11 months in the original table).

## What phase 1 already hands over

- Fixed-camera ROI JSON schema and warp/rescale.
- OpenCV baseline + Patch CNN head in `src/parking_mvp`.
- Colab comparison A/B/C with **date/lot** protocols.
- CPU tick budget: tens of milliseconds for 12 stalls (well under 3 s).
- Temporal majority vote with a measured flicker test.
- Explicit domain-gap write-up (lot vs curb).

## What Project 2 must fix

1. **Close P1.** If Colab P1 is ≥90% on PKLot unseen days, keep Patch CNN as the default head. If not, widen lots/dates or replace the tiny CNN with the mAlexNet / MobileNet line from the literature (still per-ROI).
2. **Per-camera calibration.** P2 (unseen lot) is expected to drop. Field install = new `rois.json` + optional fine-tune on 1–2 days of that camera, not a global model.
3. **Oblique curb data.** Label 50–100 frames of a public oblique set or the pilot street (10–30 stalls). Measure zero-shot then fine-tuned accuracy. This is the form’s Risk #2.
4. **YOLO path.** Keep only if occupied recall on **small/distant** cars is raised (higher-res detector, tiling, or masks). Do not ship overlap@0.4 with ~8% occupied recall.
5. **Live 2–5 s** on a 1080p stream (file or RTSP), with overlay + CSV log, on a small CPU/NUC — not Colab GPU.
6. **Lighting.** Daytime shade and overcast in the pilot. Night stays a later increment unless the sponsor changes the form.
7. **ROI drift.** Mechanical mount + occasional ROI redraw; optional homography if the camera is bumped.
8. **Ops.** Health check, disk log rotation, no multi-site fusion yet (that is Project 3).

## Suggested 90-day slice after P1 numbers exist

| Week | Output |
| --- | --- |
| 1–2 | Colab P1/P2 table in `docs/test_report.md`; freeze `.pt` |
| 3–4 | Oblique public stills + 10–30 ROI; zero-shot score |
| 5–8 | Fine-tune on that camera; error gallery (shade, occlusion) |
| 9–12 | NUC/laptop demo 3 s tick on a recorded 1080p clip; field plan |

## Out of Project 2

City-wide dashboard, payment, ANPR, multi-camera fusion, commercial packaging (Project 3 in the form).
