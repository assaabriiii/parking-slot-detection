# Acceptance criteria (phase 1 lab MVP)

Use this list when a human runs the repo (e.g. Google Colab). The scaffold itself does not execute inference, servers, or dataset downloads.

## Must pass

- [ ] `data/sample/` contains stub frames and `spots.json` with **3–5** fake stalls labeled in the README (occupied/free pattern).
- [ ] `configs/rois.example.json` defines **12** spots.
- [ ] `python -m parking_mvp --config configs/default.yaml` on the stub writes `outputs/last_status.json` with `spot_id`, `status` (`free` \| `occupied`), `confidence`, UTC `timestamp`.
- [ ] Default source is a **folder of images**, not a webcam index.
- [ ] OpenCV head runs with **no** `.pt` file and reproduces the README pattern
      (`spot_02` / `spot_04` free, the rest occupied).
- [ ] YOLO path with missing weights does **not** download `yolov8n*.pt` (dry-run / fallback).
- [ ] `occupancy_head: patchcnn` with missing weights imports **no** torch, falls back
      to OpenCV, and records the fallback in the snapshot `notes`.
- [ ] `model_patchcnn` layer names match the notebook, so the exported
      `approach_a_patchcnn.pt` loads without edits.
- [ ] ROI polygons are rescaled when preprocess resizes the frame (1080p source).
- [ ] Temporal window is configurable (default 5 frames).
- [ ] `pytest` (offline) covers ROI load/crop, temporal vote, schema, YOLO dry-run, dataset parsers on tiny temp files.
- [ ] Streamlit file exists; starting it is optional and **not** required for this checklist in CI.
- [ ] `scripts/download_data.md` lists CNRPark-EXT, PKLot (HF), MetaPKLot with **ODbL** / **CC BY** notes and expected folders.
- [ ] README states night is out of scope; tick **2–5 s**; target latency **< 3 s** (not measured here).
- [ ] No self-captured street footage in the repo.

## Should document

- [ ] How to point `roi_file` at PKLot XML-derived polygons or MetaPKLot COCO (`scripts/label_guide.md`).
- [ ] How to skip 4.6 GB PKLot until needed.

## Explicit fails

- Training commands (`yolo train`, custom loops) in the default README path.
- Default config using `source: 0`.
- Shipping CNRPark/PKLot binaries in git.
- Claiming night accuracy.
