# Agent Prompt — Street Parking Occupancy Detection MVP (Project 1)

Paste everything below into your coding agent (Claude Code, Cursor agent, ChatGPT with code execution, etc.) running against a Google Colab notebook.

---

## ROLE

You are a computer-vision engineer building a **feasibility-stage MVP**, not a production system. Optimize for "does this concept work at all, with what accuracy" — not for robustness, scale, or edge cases outside what's specified below.

## PROJECT CONTEXT

Team "فناوران شرق" is running a grant-funded feasibility study (Project 1 of 3) to test whether a **single simple fixed camera** + lightweight computer vision can detect free/occupied street parking spots accurately enough to justify further investment. This is Project 1 only: conceptual design + MVP. Do not build anything from Project 2/3 (multi-camera fusion, city-wide deployment, hardened edge devices, commercial packaging).

## SCOPE & HARD CONSTRAINTS (from the approved project spec — do not exceed or under-deliver)

- **Input:** a single static/fixed camera's video or image stream. Minimum quality 1080p. No PTZ, no multi-camera fusion.
- **Coverage:** 10 to 30 parking spots, all visible from one fixed, oblique viewing angle, at an assumed camera-to-spots distance of 10–50 m. One street or one section of one street — do NOT try to generalize to arbitrary streets or arbitrary weather/lighting.
- **Output per spot:** binary state — "free" or "occupied" — tagged with the spot's ID/ROI and a timestamp.
- **Update cadence target:** re-evaluate each spot's status every 2–5 seconds; per-frame processing latency target: under 3 seconds.
- **Accuracy target:** ≥90% under reasonable daytime lighting and a suitable viewing angle. Explicitly OUT of scope: heavy rain, deep night with poor lighting, dense occlusion/snow cover — do not tune for these, just note them as known limitations in your final report.
- **ROI definition:** each parking spot's region of interest must be definable as a polygon or rectangle, manually or semi-automatically — this does NOT need to be a trained detector for "where are the parking spots," just a config step.
- **Demo requirement:** a lightweight visual demo (Streamlit or FastAPI + simple HTML) showing the frame with each ROI color-coded free/occupied, updating live, so a non-technical reviewer can see the concept work.
- **Explicitly not required:** commercial-grade UI, user accounts, multi-camera support, edge-device optimization, mobile app, database persistence beyond a simple log/CSV or in-memory state.

## DATA SOURCING — DO NOT ASK ME TO RECORD VIDEO

I will not be capturing street footage myself. You must find and use existing, freely available public data. Concretely:

1. **Search for and shortlist public datasets/footage** suitable for a *fixed-camera, oblique-angle, on-street or parking-lot* occupancy task. Prioritize, in this order:
   - **CNRPark + CNRPark-EXT** — labeled parking-spot patches (occupied/free) from fixed overhead-ish cameras across weather/lighting conditions — good for training/validating the per-spot classifier itself.
   - **PKLot** — large labeled parking-spot patch dataset (sunny/cloudy/rainy subsets) — same purpose as above, more data.
   - **ACPDS (Action Camera Parking Dataset)** or similar oblique/street-level parking datasets if you find them — closer to true street-parking camera geometry than PKLot/CNRPark.
   - Any **public YouTube live-cam or dashcam/street-cam footage** of on-street parking (traffic cams, city open-data camera feeds, "parking lot timelapse" clips) that you can legally download (yt-dlp or direct download) to use as the *demo input stream* — this doesn't need labels, since it's just for showing the live ROI overlay, not for training.
   - A general vehicle-detection dataset (COCO, or a pretrained COCO model) if you go the detection-based route (see Approach B below) — no street-parking-specific labels needed for this route at all.
2. State clearly, in your final report, exactly which dataset(s) you used, their licenses, and any constraints on redistribution.
3. If you cannot find a truly street-angle labeled dataset, say so explicitly rather than silently substituting a bird's-eye parking-lot dataset without flagging the domain-gap risk (this matches Risk #2 in the project's own risk register: reduced accuracy from unfamiliar viewing angle / occlusion / lighting).

## TWO ACCEPTABLE TECHNICAL APPROACHES — PICK ONE, JUSTIFY YOUR CHOICE

**Approach A — Per-spot patch classifier (matches the original spec most literally):**
- For each ROI, crop the patch each frame.
- Train/fine-tune a small classifier (MobileNetV2 or a small CNN) on CNRPark-EXT/PKLot patches to output occupied/free.
- Pro: directly matches the project doc's described pipeline. Con: trained on overhead parking-lot images, so on a real oblique street-camera frame accuracy may be lower — call this out.

**Approach B — Detection + ROI-overlap (more robust with zero street-specific training data):**
- Run a pretrained lightweight detector (YOLOv8n or similar, pretrained on COCO, class = "car"/"truck"/"bus") on each frame.
- For each ROI polygon, compute IoU (or overlap ratio) between the ROI and any detected vehicle box/mask. Threshold to decide occupied vs. free.
- Pro: no training data problem at all, generalizes to whatever demo footage you find, matches the "امکان‌سنجی" (feasibility, not final product) framing of this phase well. Con: sensitive to occlusion and to the IoU/overlap threshold — tune this on a small validation set of frames you manually label from your demo footage.

**Default recommendation: build Approach B first** (fastest path to a working demo with no street-specific labeled data), and treat Approach A as a stretch goal / comparison if time allows, since the project brief explicitly frames Project 1 as testing basic feasibility, not delivering a final model.

## STEP-BY-STEP TASKS

1. **Environment setup** (Colab): install `ultralytics`, `opencv-python`, `numpy`, `streamlit` (or `fastapi`+`uvicorn`), `yt-dlp` (if pulling demo footage from a URL), and any dataset-download helper you need.
2. **Data acquisition**: download/search the datasets above; download one clip or image sequence to serve as the live demo input. Save everything under a clear `data/` folder structure.
3. **ROI definition tool**: write a small script/notebook cell that lets me click-and-define polygons on a sample frame (`cv2` mouse callback or a simple `matplotlib`-based clicker), and saves them to `rois.json` as `{id, polygon_points}`. Pre-fill it with a reasonable default set of 10–30 ROIs on the actual demo frame you chose, so it runs end-to-end without me needing to click anything, but leave the tool in place so I can redefine ROIs for a different frame.
4. **Preprocessing pipeline**: frame read → resize/normalize → (Approach A: per-ROI crop; Approach B: run detector on full frame) → decision logic → per-spot state + timestamp.
5. **Model / detector**: implement your chosen approach; if Approach A, include the training script and report train/val accuracy on CNRPark-EXT/PKLot; if Approach B, include the IoU-threshold tuning step and show a few tuning trials.
6. **Evaluation**: manually label ground truth for ~50–100 frames from your demo footage (free/occupied per spot) and report accuracy, precision/recall per class, per-frame processing latency, and whether the ≥90%/<3s targets are met. Be honest if they aren't — this is the actual point of a feasibility study.
7. **Live demo**: a Streamlit (preferred, simplest in Colab via `pyngrok`/`localtunnel`) app that plays through the demo footage (or loops it) and overlays each ROI in green (free) / red (occupied) with a timestamp and a simple status table.
8. **Final report** (markdown, in the notebook or a separate file): dataset(s) + licenses used, approach chosen and why, architecture diagram in words, evaluation results against the targets above, and a short "limitations & risks" section covering: lighting/weather sensitivity, occlusion, domain gap between training data and the real street angle, and what Project 2 would need to fix.

## DELIVERABLES CHECKLIST

- [ ] `rois.json` with the defined parking-spot polygons
- [ ] Working detection/classification pipeline (script or notebook cells)
- [ ] Evaluation results (accuracy, precision/recall, latency) vs. targets
- [ ] Streamlit (or FastAPI+simple HTML) live demo runnable in Colab
- [ ] Short markdown report as described in step 8
- [ ] Explicit list of datasets/sources used, with links and licenses

## TONE / WORKING STYLE

Move fast, favor the simplest thing that could plausibly hit the accuracy target, and clearly flag every assumption or shortcut you take (e.g., "used dataset X because Y wasn't available," "threshold chosen empirically, not exhaustively tuned"). This is a 5-month feasibility project condensed into a Colab notebook — the standard is "convincing proof of concept," not production code.
