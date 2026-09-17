#!/usr/bin/env python3
"""Download a few PKLot PUCPR (same camera as Colab P1) frames and overlay Patch CNN."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_mvp.io_schema import OccupancySnapshot
from parking_mvp.model_patchcnn import PatchCNNOccupancyHead
from parking_mvp.preprocess import preprocess
from parking_mvp.roi import ROISet, SpotROI, overlay_rois, warp_all

HUB = "Voxel51/PKLot"
SAMPLES_URL = f"https://huggingface.co/datasets/{HUB}/resolve/main/samples.json"
OUT = ROOT / "docs" / "lab_results" / "pucpr_overlays"
WEIGHTS = ROOT / "weights" / "approach_a_patchcnn.pt"
MAX_SPOTS = 24
N_FRAMES = 4


def occ_label(raw: str) -> str:
    s = str(raw or "").lower().replace(" ", "_")
    if s in ("not_occupied", "not-occupied", "vacant", "free"):
        return "free"
    if s in ("occupied", "busy"):
        return "occupied"
    return "unknown"


def iter_samples(url: str):
    """Yield objects from the top-level 'samples' array without loading the whole file."""
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        buf = ""
        started = False
        depth = 0
        in_str = False
        esc = False
        for chunk in resp.iter_content(chunk_size=1 << 16):
            text = chunk.decode("utf-8", errors="ignore")
            i = 0
            if not started:
                buf += text
                key = '"samples"'
                k = buf.find(key)
                if k < 0:
                    buf = buf[-32:]
                    continue
                bracket = buf.find("[", k)
                if bracket < 0:
                    continue
                started = True
                text = buf[bracket + 1 :]
                buf = ""
            while i < len(text):
                ch = text[i]
                if depth == 0:
                    if ch == "{":
                        depth = 1
                        buf = "{"
                    elif ch == "]":
                        return
                    i += 1
                    continue
                buf += ch
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            yield json.loads(buf)
                            buf = ""
                i += 1


def sample_spaces(sample: dict, width: int, height: int, selected_ids: list[str] | None):
    spaces = []
    polylines = ((sample.get("parking_spaces") or {}).get("polylines")) or []
    for pl in polylines:
        status = occ_label(pl.get("occupancy_status"))
        if status == "unknown":
            continue
        pts_norm = pl["points"][0]
        poly = [[int(round(x * width)), int(round(y * height))] for x, y in pts_norm]
        sid = int(pl.get("index") or pl.get("space_id") or (len(spaces) + 1))
        spaces.append({"id": f"S{sid:03d}", "polygon": poly, "gt": status})
    spaces.sort(key=lambda s: float(np.mean([p[1] for p in s["polygon"]])))
    if selected_ids is None:
        if len(spaces) > MAX_SPOTS:
            idx = np.linspace(0, len(spaces) - 1, MAX_SPOTS).round().astype(int)
            selected_ids = [spaces[j]["id"] for j in sorted(set(idx.tolist()))]
        else:
            selected_ids = [s["id"] for s in spaces]
    by_id = {s["id"]: s for s in spaces}
    return [by_id[i] for i in selected_ids if i in by_id], selected_ids


def sample_occupied_rate(sample: dict) -> float:
    polylines = ((sample.get("parking_spaces") or {}).get("polylines")) or []
    labels = [occ_label(pl.get("occupancy_status")) for pl in polylines]
    known = [x for x in labels if x != "unknown"]
    if not known:
        return 0.0
    return sum(x == "occupied" for x in known) / len(known)


def pick_samples() -> list[dict]:
    picked: list[dict] = []
    seen_dates: set[str] = set()
    for sample in iter_samples(SAMPLES_URL):
        if sample.get("source") != "pucpr":
            continue
        weather = ((sample.get("weather") or {}).get("label") or "").lower()
        if weather != "sunny":
            continue
        rate = sample_occupied_rate(sample)
        # Skip empty dawn lots so the report shows the same occupied/mixed scene as P1.
        if rate < 0.20 or rate > 0.90:
            continue
        date = str((sample.get("date") or {}).get("$date") or "")[:10]
        if date in seen_dates:
            continue
        filepath = sample["filepath"]
        picked.append(sample)
        seen_dates.add(date)
        print("picked", filepath, "date", date, f"occ_rate={rate:.2f}")
        if len(picked) >= N_FRAMES:
            break
    if len(picked) < N_FRAMES:
        raise RuntimeError(f"only found {len(picked)} pucpr sunny samples")
    return picked


def banner(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    h = 22 * len(lines) + 12
    cv2.rectangle(out, (0, 0), (out.shape[1], h), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)
    OUT.mkdir(parents=True, exist_ok=True)
    samples = pick_samples()
    head = PatchCNNOccupancyHead(weights=WEIGHTS, dry_run=False, input_size=128, device="cpu")
    if not head.is_ready:
        raise RuntimeError("Patch CNN weights did not load")

    selected_ids = None
    summary = []
    for i, sample in enumerate(samples, start=1):
        rel = sample["filepath"]
        local = Path(
            hf_hub_download(repo_id=HUB, repo_type="dataset", filename=rel)
        )
        bgr = cv2.imread(str(local))
        if bgr is None:
            raise FileNotFoundError(local)
        h, w = bgr.shape[:2]
        spaces, selected_ids = sample_spaces(sample, w, h, selected_ids)
        rois = ROISet(
            camera_id="pklot_pucpr",
            image_width=w,
            image_height=h,
            spots=tuple(SpotROI(s["id"], tuple((x, y) for x, y in s["polygon"])) for s in spaces),
        )
        image = preprocess(bgr, max_width=1280, max_height=720)
        rois_s = rois.scaled_to(image.shape[1], image.shape[0])
        crops = warp_all(image, rois_s, out_size=128)
        ids = [s.spot_id for s in rois_s.spots]
        preds = head.predict_batch(ids, [crops[sid] for sid in ids])
        pred_map = {p.spot_id: p.status for p in preds}
        gt_map = {s["id"]: s["gt"] for s in spaces}
        n = len(ids)
        n_ok = sum(pred_map[sid] == gt_map[sid] for sid in ids)
        n_occ_p = sum(v == "occupied" for v in pred_map.values())
        n_occ_g = sum(v == "occupied" for v in gt_map.values())
        ts = sample.get("parking_timestamp", {}).get("$date", "")
        date = str((sample.get("date") or {}).get("$date") or "")[:10]
        stem = Path(rel).stem

        vis_p = overlay_rois(image, rois_s, pred_map)
        vis_p = banner(
            vis_p,
            [
                f"PKLot PUCPR  camera=pucpr  weather=sunny  {date}",
                f"PatchCNN prediction  occupied={n_occ_p}/{n}  match_GT={n_ok}/{n}",
            ],
        )
        vis_g = overlay_rois(image, rois_s, gt_map)
        vis_g = banner(
            vis_g,
            [
                f"PKLot PUCPR  camera=pucpr  weather=sunny  {date}",
                f"Ground truth  occupied={n_occ_g}/{n}",
            ],
        )
        pred_path = OUT / f"pucpr_{i:02d}_pred_{stem}.png"
        gt_path = OUT / f"pucpr_{i:02d}_gt_{stem}.png"
        cv2.imwrite(str(pred_path), vis_p)
        cv2.imwrite(str(gt_path), vis_g)
        row = {
            "index": i,
            "file": rel,
            "date": date,
            "timestamp": ts,
            "n_spots": n,
            "gt_occupied": n_occ_g,
            "pred_occupied": n_occ_p,
            "correct": n_ok,
            "pred_png": pred_path.name,
            "gt_png": gt_path.name,
        }
        summary.append(row)
        print(row)

    (OUT / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
