# Acceptance criteria (phase 1 lab MVP)

Use this list when a human runs the repo (e.g. Google Colab). CI runs pytest only.

## Must pass

- [x] `data/sample/` contains stub frames and `spots.json` with **3–5** fake stalls labeled in the README (occupied/free pattern).
- [x] `configs/rois.example.json` defines **12** spots.
- [x] `python -m parking_mvp --config configs/default.yaml` on the stub writes `outputs/last_status.json` with `spot_id`, `status` (`free` \| `occupied`), `confidence`, UTC `timestamp`.
- [x] Default source is a **folder of images**, not a webcam index.
- [x] OpenCV head runs with **no** `.pt` file and reproduces the README pattern
      (`spot_02` / `spot_04` free, the rest occupied).
- [x] YOLO path with missing weights does **not** download `yolov8n*.pt` (dry-run / fallback).
- [x] `occupancy_head: patchcnn` with missing weights imports **no** torch, falls back
      to OpenCV, and records the fallback in the snapshot `notes`.
- [x] `model_patchcnn` layer names match the notebook, so the exported
      `approach_a_patchcnn.pt` loads without edits.
- [x] ROI polygons are rescaled when preprocess resizes the frame (1080p source).
- [x] Temporal window is configurable (default 5 frames); flicker test in `scripts/eval_temporal_flips.py`.
- [x] `pytest` (offline) covers ROI load/crop, temporal vote, schema, YOLO dry-run, dataset parsers on tiny temp files.
- [x] Streamlit still + sequence mode; overlay export via `scripts/export_demo_overlays.py`.
- [x] CPU tick latency logged in `docs/lab_results/` (under 3 s on this lab machine).
- [x] `scripts/download_data.md` lists CNRPark-EXT, PKLot (HF), MetaPKLot with **ODbL** / **CC BY** notes and expected folders.
- [x] README states night is out of scope; tick **2–5 s**; latency measured in `docs/lab_results/`.
- [x] No self-captured street footage in the repo.
- [x] Paper pack: `docs/requirements.md`, `docs/camera_geometry.md`, `docs/feasibility_report.md`, `docs/test_report.md`, `docs/project2_plan.md`.

## Still human / GPU

- [x] Colab notebook **P1** table (≥90% claim) after leak-free split (99.7% on `pucpr` unseen days).
- [x] Local `weights/approach_a_patchcnn.pt` from that run (`dry_run: false` in default.yaml; file gitignored).
- [ ] Oblique public scene, 10–30 ROIs, zero-shot score.

## Should document

- [x] How to point `roi_file` at PKLot XML-derived polygons or MetaPKLot COCO (`scripts/label_guide.md`).
- [x] How to skip 4.6 GB PKLot until needed.

## Explicit fails

- Training commands (`yolo train`, custom loops) in the default README path.
- Default config using `source: 0`.
- Shipping CNRPark/PKLot binaries in git.
- Claiming night accuracy.
- Claiming 100% from a same-day random-frame split.
