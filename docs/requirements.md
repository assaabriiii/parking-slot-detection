# Phase-1 requirements (from the 26 Mordad project form)

This is the requirements sheet for **project 1 of 3**: feasibility, conceptual design, and a laboratory MVP. It is not a production product spec.

## In scope

| ID | Requirement | Target |
| --- | --- | --- |
| R1 | Single **fixed** camera (no PTZ, no multi-camera fusion) | one street section |
| R2 | Image / video input, minimum **1080p** | HD or above |
| R3 | Coverage | **10–30** stalls, oblique view, camera-to-stalls **10–50 m** |
| R4 | Output per stall | `free` or `occupied` + `spot_id` + timestamp |
| R5 | ROI | polygon or rectangle, manual or semi-automatic (not a stall detector) |
| R6 | Pipeline | frame → preprocess → ROI → occupancy decision → log |
| R7 | Update cadence | **2–5 s** (`tick_seconds: 3`) |
| R8 | Response time | **< 3 s** per tick on limited hardware |
| R9 | Accuracy | **≥ 90%** in suitable **daylight** and viewing angle |
| R10 | Temporal stability | majority vote over a few frames |
| R11 | Lab UI | Streamlit (preferred) or FastAPI showing colour-coded ROIs |
| R12 | Architecture | extensible later to several cameras (not implemented now) |

## Operating conditions (this phase)

In scope: daytime lighting, some shade, some lighting change, partial occlusion.

Out of scope (document as limits, do not tune): heavy rain, deep night / IR, dense occlusion / snow, city-wide deployment, payment / ANPR.

## Acceptance mapping

| ID | How we check it |
| --- | --- |
| R1–R6 | `python -m parking_mvp` writes `outputs/last_status.json` |
| R3 | `configs/rois.example.json` has 12 spots; a real scene must use 10–30 polygons |
| R7 | Streamlit **Sequence** mode and video stride `fps * tick_seconds` |
| R8 | `scripts/benchmark_tick.py` on CPU → `outputs/benchmark_cpu.json` |
| R9 | Colab notebook protocol **P1** (same lot, unseen dates). Not P3. |
| R10 | `scripts/eval_temporal_flips.py` |
| R11 | `streamlit run app/streamlit_app.py` |
| Street vs lot gap | zero-shot on an oblique public scene (ACPDS or similar); not claimed as R9 |

## Explicit non-requirements (phase 1)

Training in the default README path is optional (Colab). Webcam index `0` is not the default source. Night accuracy must not be claimed.
