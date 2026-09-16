# Camera geometry (phase 1)

The brief assumes one **simple fixed camera**, **≥ 1080p**, **10–30 stalls**, **oblique** view, **10–50 m** from the stalls. This note checks whether that geometry is enough for a per-stall crop, not a full photogrammetric survey.

## Setup

- Sensor: 1920×1080 (the brief’s minimum).
- Typical stall: ~5.0 m long × ~2.3 m wide.
- Typical passenger car: ~4.5 m long × ~1.8 m wide.
- Horizontal field of view: **~60°** (a common cheap-camera / webcam equivalent).

Pinhole width of an object:

`pixels ≈ (object_width / (2 * distance * tan(FOV/2))) * image_width`

## Pixels on a car (width 1.8 m)

| Distance | Car width in pixels (approx.) | Notes |
| --- | --- | --- |
| 10 m | ~320 | Near row; easy for a 128×128 patch |
| 25 m | ~130 | Mid range |
| 50 m | ~65 | Far row; still larger than the 128 px warp **after** crop of the stall, but the stall itself is small on the frame |

A **2.3 m stall width** at 50 m is ~83 px. After a perspective warp to 128×128 the Patch CNN still gets a filled square, but **far stalls have fewer source pixels**, so interpolation blurs texture. This is a real limit at the 50 m end of the brief and belongs in the risk register (angle / distance), not as a night problem.

## 10–30 stalls in one frame

At 25–40 m, a 60° FOV covers roughly **25–45 m** of curb. That is enough for **10–30** parallel stalls if they sit in 1–2 rows in the lower half of the frame (the usual oblique parking-lot / curb composition). A bird’s-eye lot with 100 stalls (PKLot) is **out of scope** for a single 10–30 ROI set; we subsample to ≤24 spots per camera in the notebook.

## Installation notes for a later field camera

1. Mount so stalls are **trapezoids**, not a vertical wall of bumpers (too little pavement) and not nadir (domain of PKLot).
2. Draw ROI polygons on the **same resolution** the pipeline will see, or set `image_width` / `image_height` in the ROI JSON so `ROISet.scaled_to` can follow preprocess (1080p → 1280×720 cap).
3. Avoid pointing at headlights / IR at night; night is out of scope.
4. Keep the camera **rigid**. The MVP has no tracking; ROI drift is a Project 2 item.

## Implication for this lab

PKLot frames are elevated parking lots, not 10–50 m curb. Geometry here **justifies 1080p** for the intended street mount. It does **not** substitute for an oblique street evaluation (see `docs/test_report.md`).
