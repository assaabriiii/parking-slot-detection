#!/usr/bin/env python3
"""Generate notebooks/parking_occupancy_mvp_colab.ipynb — Voxel51/PKLot + 3 approaches."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "parking_occupancy_mvp_colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(source: str) -> list[str]:
    text = source.strip("\n") + "\n"
    return text.splitlines(keepends=True) or ["\n"]


CELLS = [
    md(
        r"""
# Parking Occupancy MVP — Voxel51/PKLot + 3 Approaches (Colab)

**Dataset:** [`Voxel51/PKLot`](https://huggingface.co/datasets/Voxel51/PKLot) (CC BY 4.0, Almeida et al. 2015)

### How to run
1. Upload this notebook **and** `requirements.txt` to [Google Colab](https://colab.research.google.com/)
2. **Runtime → GPU (T4)**
3. Run the **install cell once** — it will **auto-restart** the runtime after fixing Pillow (expected).
4. After reconnect, **Run all** again (install cell will no-op).

| # | Approach | Idea | Needs parking training? |
|---|----------|------|-------------------------|
| **A** | Patch CNN classifier | Crop each ROI → free/occupied small CNN | Yes (on PKLot patches) |
| **B** | YOLOv8n boxes + ROI overlap | Detect cars, score ROI coverage | No (COCO pretrained) |
| **C** | YOLOv8n-seg masks + ROI IoU | Segment cars, IoU with ROI mask | No (COCO pretrained) |

### Why Approach A is the strongest candidate
- [martin-marek/parking-space-occupancy](https://github.com/martin-marek/parking-space-occupancy) uses perspective ROI pooling + a CNN and reports **98%** on unseen parking lots.
- [fabiocarrara/deep-parking](https://github.com/fabiocarrara/deep-parking) uses a lightweight per-slot CNN on PKLot/CNRPark.
- [Eighonet/parking-research](https://github.com/Eighonet/parking-research) compares patch classifiers (ResNet, MobileNet, CarNet, mAlexNet, ViTs) with intersection detectors.
- [visualbuffer/parkingslot](https://github.com/visualbuffer/parkingslot) notes Mask R-CNN sees small/distant objects better than YOLO, while a classifier still determines occupancy.

This revision therefore improves A with perspective-warped ROIs, balanced sampling,
stratified frame splits, and balanced-accuracy model selection. B/C remain useful zero-shot baselines.
"""
    ),
    md("## 0 · Install (Pillow fixed last — Colab Python 3.13)"),
    code(
        r"""
# Upload requirements.txt to /content/ (Files sidebar) before running.
# This cell installs deps, repairs Pillow, then HARD-RESTARTS the runtime once.
# That restart is required on Colab Python 3.13 — otherwise torchvision/fiftyone keep a broken PIL in memory.
import os
import subprocess
import sys
from pathlib import Path

FLAG = Path("/content/.parking_mvp_deps_ok")
candidates = [
    Path("requirements.txt"),
    Path("/content/requirements.txt"),
    Path("/content/parking-slot-detection/requirements.txt"),
]
req = next((p for p in candidates if p.is_file()), None)

if FLAG.exists():
    print("Deps already installed (flag:", FLAG, ")")
    print("Continue with the next cells.")
else:
    if req is None:
        raise FileNotFoundError(
            "requirements.txt not found. Upload it to /content/ then re-run this cell."
        )
    print("Installing from", req.resolve())
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "-r", str(req)])
    # fiftyone/ultralytics often leave a mixed Pillow install — wipe and reinstall LAST
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "pillow", "PIL"], stderr=subprocess.DEVNULL)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall", "pillow==11.2.1"]
    )
    FLAG.write_text("ok\n")
    print("Pillow repaired. Restarting runtime now (normal)...")
    print("After reconnect: Runtime → Run all")
    os.kill(os.getpid(), 9)
"""
    ),
    code(
        r"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.display import display, Markdown
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

# No torchvision / no PIL imports here — they trigger Colab's broken _Ink error.

ROOT = Path("/content/parking_mvp") if Path("/content").exists() else Path.cwd() / "parking_mvp_colab_run"
DATA, FRAMES, OUT, REPORTS = ROOT / "data", ROOT / "data" / "frames", ROOT / "outputs", ROOT / "reports"
for p in (DATA, FRAMES, OUT, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("ROOT =", ROOT, "| device =", DEVICE)
"""
    ),
    md(
        r"""
## 1 · Load Voxel51/PKLot

**Important:** `datasets.load_dataset("Voxel51/PKLot")` exposes **images only** (no spot polygons).
The same Hub repo stores polygons + occupancy via **FiftyOne**. We:

1. Call `load_dataset` as requested (and keep a small image subset)
2. Load a capped labeled view with FiftyOne for ROIs + GT
"""
    ),
    code(
        r'''
from datasets import load_dataset
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

MAX_FRAMES = 80          # keep Colab disk/time reasonable (full set is 12k / ~4GB)
MAX_SPOTS_PER_FRAME = 24 # project scope: 10–30 spots
WEATHER = "sunny"        # daytime feasibility target

# --- 1a. User-requested API (images) ---
# Full parquet is ~4GB. Use streaming for a quick preview; labeled work uses FiftyOne below.
print("Loading via datasets.load_dataset('Voxel51/PKLot', streaming=True) ...")
ds = load_dataset("Voxel51/PKLot", split="train", streaming=True)
print(ds)
preview_rows = list(ds.take(3))
fig, axes = plt.subplots(1, len(preview_rows), figsize=(12, 4))
if len(preview_rows) == 1:
    axes = [axes]
for i, (ax, row) in enumerate(zip(axes, preview_rows)):
    ax.imshow(row["image"])
    ax.set_title(f"datasets stream row {i}")
    ax.axis("off")
plt.suptitle("load_dataset('Voxel51/PKLot') — image column (streaming preview)")
plt.tight_layout()
plt.show()

# Non-streaming handle (same API the Hub card shows). Comment in if you have disk for ~4GB:
# ds_full = load_dataset("Voxel51/PKLot", split="train")

# --- 1b. Labeled subset (polygons + occupancy) via FiftyOne on the SAME repo ---
print("\nLoading labeled subset via FiftyOne load_from_hub (annotations)...")
fo_ds = load_from_hub("Voxel51/PKLot", max_samples=400, overwrite=True)
view = fo_ds.match(fo.ViewField("weather.label") == WEATHER)
if len(view) < 20:
    print("Few sunny samples; using all weather.")
    view = fo_ds.view()

# Stay on ONE lot so space_ids are comparable across frames
sources = view.distinct("source")
best_source = max(sources, key=lambda s: len(view.match(fo.ViewField("source") == s)))
view = view.match(fo.ViewField("source") == best_source).sort_by("parking_timestamp").limit(MAX_FRAMES)
print(f"Using lot={best_source!r}, weather filter={WEATHER!r}, frames={len(view)}")


def _pl_attr(pl, name, default=None):
    if hasattr(pl, name) and getattr(pl, name) is not None:
        return getattr(pl, name)
    attrs = getattr(pl, "attributes", None) or {}
    if name in attrs:
        val = attrs[name]
        return getattr(val, "value", val)
    try:
        return pl[name]
    except Exception:
        return default


def occ_label(raw) -> str:
    s = str(raw or "unknown").lower().replace(" ", "_")
    if s in ("not_occupied", "not-occupied", "vacant", "free"):
        return "free"
    if s in ("occupied", "busy"):
        return "occupied"
    return "unknown"


if FRAMES.exists():
    shutil.rmtree(FRAMES)
FRAMES.mkdir(parents=True, exist_ok=True)

manifest = []
selected_space_ids = None
for i, sample in enumerate(tqdm(view, desc="Export labeled frames")):
    img = cv2.imread(sample.filepath)
    if img is None:
        continue
    h, w = img.shape[:2]
    dst = FRAMES / f"frame_{i:04d}.jpg"
    cv2.imwrite(str(dst), img)

    spaces = []
    if sample.parking_spaces is not None:
        for pl in sample.parking_spaces.polylines:
            status = occ_label(_pl_attr(pl, "occupancy_status"))
            if status == "unknown":
                continue
            pts = [[float(x), float(y)] for x, y in pl.points[0]]
            poly = [[int(round(x * w)), int(round(y * h))] for x, y in pts]
            sid = _pl_attr(pl, "space_id", len(spaces) + 1)
            spaces.append({
                "id": f"S{int(sid):03d}",
                "polygon": poly,
                "polygon_norm": pts,
                "gt": status,
            })

    # PKLot has ~100 spots/frame. Keep ≤ MAX_SPOTS_PER_FRAME but SAMPLE ACROSS DEPTH
    # (do NOT take only the largest polygons — those are always the near/bottom row).
    def _yc(s):
        pts = np.array(s["polygon"], np.float32)
        return float(pts[:, 1].mean())

    spaces = sorted(spaces, key=_yc)  # top of image → bottom
    if selected_space_ids is None and len(spaces) > MAX_SPOTS_PER_FRAME:
        # even stride across near+far rows
        idx = np.linspace(0, len(spaces) - 1, MAX_SPOTS_PER_FRAME).round().astype(int)
        selected_space_ids = [spaces[j]["id"] for j in sorted(set(idx.tolist()))]
    elif selected_space_ids is None:
        selected_space_ids = [s["id"] for s in spaces]

    # Keep the same physical spaces on every frame from this fixed camera.
    by_id = {s["id"]: s for s in spaces}
    spaces = [by_id[sid] for sid in selected_space_ids if sid in by_id]

    manifest.append({
        "frame": dst.name,
        "source": sample.source,
        "weather": sample.weather.label if sample.weather else None,
        "width": w,
        "height": h,
        "spaces": spaces,
    })

(DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"Exported {len(manifest)} frames → {FRAMES}")
print("Example spots:", len(manifest[0]["spaces"]), "GT free/occ:",
      sum(s["gt"]=="free" for s in manifest[0]["spaces"]),
      sum(s["gt"]=="occupied" for s in manifest[0]["spaces"]))
'''
    ),
    md("## 2 · `rois.json` (reference frame)"),
    code(
        r"""
def load_bgr(path: Path) -> np.ndarray:
    im = cv2.imread(str(path))
    if im is None:
        raise FileNotFoundError(path)
    return im


def overlay_rois(image, spots, statuses=None):
    vis = image.copy()
    for spot in spots:
        pts = np.array(spot["polygon"], np.int32)
        st = (statuses or {}).get(spot["id"], spot.get("gt", "unknown"))
        color = (0, 200, 0) if st == "free" else (0, 0, 220) if st == "occupied" else (160, 160, 160)
        cv2.polylines(vis, [pts], True, color, 2)
        c = pts.mean(0).astype(int)
        cv2.putText(vis, spot["id"], tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return vis


ref = manifest[0]
ref_img = load_bgr(FRAMES / ref["frame"])
rois = {
    "camera_id": f"pklot_{ref['source']}",
    "image_size": {"width": ref["width"], "height": ref["height"]},
    "source_frame": ref["frame"],
    "spots": [{"id": s["id"], "polygon": s["polygon"], "polygon_norm": s["polygon_norm"]} for s in ref["spaces"]],
    "notes": "From Voxel51/PKLot polygons on reference frame",
}
(DATA / "rois.json").write_text(json.dumps(rois, indent=2))
print("Wrote", DATA / "rois.json", "n=", len(rois["spots"]))

# Show GT ROIs on several frames so you're not stuck looking at one empty lot photo
preview_ids = list(range(min(6, len(manifest))))
# Prefer frames that actually have occupied spots
preview_ids = sorted(
    range(len(manifest)),
    key=lambda i: sum(s["gt"] == "occupied" for s in manifest[i]["spaces"]),
    reverse=True,
)[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, i in zip(axes.ravel(), preview_ids):
    entry = manifest[i]
    img = load_bgr(FRAMES / entry["frame"])
    gt = {s["id"]: s["gt"] for s in entry["spaces"]}
    vis = overlay_rois(img, entry["spaces"], gt)
    n_occ = sum(v == "occupied" for v in gt.values())
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{entry['frame']} | occupied={n_occ}/{len(gt)}")
    ax.axis("off")
plt.suptitle("Reference ROIs across multiple frames (green=free, red=occupied)")
plt.tight_layout()
plt.show()
"""
    ),
    md("## 3 · Shared helpers (metrics, crops, masks)"),
    code(
        r"""
def polygon_mask(hw, polygon):
    m = np.zeros(hw, np.uint8)
    cv2.fillPoly(m, [np.array(polygon, np.int32)], 1)
    return m


def crop_roi(image_bgr, polygon, out_size=128):
    # Perspective-warp a four-point parking polygon to a square ROI.
    pts = np.array(polygon, np.float32)
    if len(pts) != 4:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        patch = image_bgr[max(0, y):y + h, max(0, x):x + w]
        return cv2.resize(patch, (out_size, out_size))

    ordered = np.zeros((4, 2), np.float32)
    sums, diffs = pts.sum(1), np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(
        image_bgr, matrix, (out_size, out_size), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def prf(y_true, y_pred, cls):
    tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
    fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
    fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec


def summarize(y_true, y_pred, latencies, name):
    if not y_true:
        return {"approach": name, "n": 0}
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    po, ro = prf(y_true, y_pred, "occupied")
    pf, rf = prf(y_true, y_pred, "free")
    f1o = 2 * po * ro / (po + ro) if po + ro else 0.0
    return {
        "approach": name,
        "n": len(y_true),
        "accuracy": acc,
        "precision_occupied": po,
        "recall_occupied": ro,
        "precision_free": pf,
        "recall_free": rf,
        "f1_occupied": f1o,
        "balanced_accuracy": (ro + rf) / 2,
        "n_free": sum(y == "free" for y in y_true),
        "n_occupied": sum(y == "occupied" for y in y_true),
        "latency_mean_s": float(np.mean(latencies)) if latencies else None,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else None,
    }


def occupancy_rate(entry):
    labels = [s["gt"] for s in entry["spaces"] if s["gt"] in ("free", "occupied")]
    return sum(y == "occupied" for y in labels) / max(len(labels), 1)


def stratified_frame_split(entries, seed=42):
    # 60/20/20 split balanced by frame-level occupied ratio.
    buckets = {0: [], 1: [], 2: [], 3: []}
    for i, entry in enumerate(entries):
        rate = occupancy_rate(entry)
        bucket = 0 if rate == 0 else 1 if rate <= 0.25 else 2 if rate <= 0.5 else 3
        buckets[bucket].append(i)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for indices in buckets.values():
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            destination = position % 5
            (val if destination == 3 else test if destination == 4 else train).append(index)
    return sorted(train), sorted(val), sorted(test)


train_idx, val_idx, test_idx = stratified_frame_split(manifest)


def split_summary(name, indices):
    labels = [s["gt"] for i in indices for s in manifest[i]["spaces"]]
    occupied = sum(y == "occupied" for y in labels)
    print(
        f"{name}: frames={len(indices)} spots={len(labels)} "
        f"free={len(labels)-occupied} occupied={occupied} "
        f"occupied_rate={occupied/max(len(labels),1):.1%}"
    )


split_summary("train", train_idx)
split_summary("validation", val_idx)
split_summary("test", test_idx)
"""
    ),
    md(
        r"""
## 4 · Approach A — Patch CNN classifier (no torchvision)

Same idea as mAlexNet / MobileNet papers: crop each ROI → binary free/occupied.
Uses a **small pure-PyTorch CNN** (no `torchvision` / Pillow) so Colab 3.13 does not hit `_Ink`.
"""
    ),
    code(
        r"""
class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(192, 2))

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


def patch_to_tensor(bgr: np.ndarray, train: bool = False) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if train and random.random() < 0.5:
        rgb = rgb[:, ::-1, :].copy()
    if train:
        gain = random.uniform(0.85, 1.15)
        bias = random.uniform(-0.06, 0.06)
        rgb = np.clip(rgb * gain + bias, 0.0, 1.0)
    # ImageNet-ish normalize
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class PatchDS(Dataset):
    def __init__(self, items, train=False):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        arr, y = self.items[i]
        return patch_to_tensor(arr, train=self.train), y


def collect_patches(indices, max_per_frame=24):
    items = []
    for i in indices:
        entry = manifest[i]
        img = load_bgr(FRAMES / entry["frame"])
        for s in entry["spaces"][:max_per_frame]:
            if s["gt"] not in ("free", "occupied"):
                continue
            patch = crop_roi(img, s["polygon"])
            y = 0 if s["gt"] == "free" else 1
            items.append((patch, y))
    return items


train_items = collect_patches(train_idx)
val_items = collect_patches(val_idx)
print(f"Approach A patches: train={len(train_items)} val={len(val_items)}")

class_counts = np.bincount([y for _, y in train_items], minlength=2)
sample_weights = [1.0 / max(class_counts[y], 1) for _, y in train_items]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_items), replacement=True)
print(f"Train classes: free={class_counts[0]} occupied={class_counts[1]} (balanced sampler enabled)")

train_loader = DataLoader(
    PatchDS(train_items, train=True), batch_size=64, sampler=sampler, num_workers=0
)
val_loader = DataLoader(PatchDS(val_items, train=False), batch_size=64, shuffle=False, num_workers=0)

net = PatchCNN().to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

EPOCHS_A = 10
best_balanced_acc = -1.0
best_state = None
for epoch in range(EPOCHS_A):
    net.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    net.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = net(x).argmax(1)
            val_true.extend(y.cpu().tolist())
            val_pred.extend(pred.cpu().tolist())
    recalls = []
    for cls in (0, 1):
        denom = sum(y == cls for y in val_true)
        recalls.append(sum(t == cls and p == cls for t, p in zip(val_true, val_pred)) / max(denom, 1))
    balanced_acc = sum(recalls) / 2
    accuracy = sum(t == p for t, p in zip(val_true, val_pred)) / max(len(val_true), 1)
    if balanced_acc > best_balanced_acc:
        best_balanced_acc = balanced_acc
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    print(
        f"A epoch {epoch+1}/{EPOCHS_A} val_acc={accuracy:.3f} "
        f"val_bal_acc={balanced_acc:.3f} recalls(free/occ)={recalls}"
    )

net.load_state_dict(best_state)
torch.save(net.state_dict(), OUT / "approach_a_patchcnn.pt")


@torch.no_grad()
def predict_a_frame(image_bgr, spaces):
    net.eval()
    if not spaces:
        return {}
    batch = torch.stack([
        patch_to_tensor(crop_roi(image_bgr, s["polygon"])) for s in spaces
    ]).to(DEVICE)
    preds = net(batch).argmax(1).cpu().tolist()
    return {
        s["id"]: ("free" if pred == 0 else "occupied")
        for s, pred in zip(spaces, preds)
    }


def eval_approach_a(indices):
    y_true, y_pred, lats = [], [], []
    for i in indices:
        entry = manifest[i]
        img = load_bgr(FRAMES / entry["frame"])
        t0 = time.perf_counter()
        st = predict_a_frame(img, entry["spaces"])
        lats.append(time.perf_counter() - t0)
        for s in entry["spaces"]:
            y_true.append(s["gt"])
            y_pred.append(st[s["id"]])
    return summarize(y_true, y_pred, lats, "A_PatchCNN")


metrics_a = eval_approach_a(test_idx)
print(json.dumps(metrics_a, indent=2))
"""
    ),
    md(
        r"""
## 5 · Approach B — YOLOv8n detection + ROI overlap

Feasibility default: no parking-specific training. Occupied if any vehicle box covers ≥ threshold of the ROI.
"""
    ),
    code(
        r"""
from ultralytics import YOLO

VEHICLE_IDS = {2, 5, 7}  # COCO car, bus, truck
yolo_det = YOLO("yolov8n.pt")


@dataclass
class Det:
    xyxy: np.ndarray
    conf: float


def detect_boxes(image_bgr, conf=0.25):
    res = yolo_det.predict(image_bgr, conf=conf, verbose=False)[0]
    out = []
    if res.boxes is None:
        return out
    for b in res.boxes:
        if int(b.cls.item()) in VEHICLE_IDS:
            out.append(Det(b.xyxy.cpu().numpy().reshape(4), float(b.conf.item())))
    return out


def overlap_ratio(roi_m, box_m):
    a = float(roi_m.sum())
    return float((roi_m & box_m).sum()) / a if a else 0.0


def decide_b(image_bgr, spaces, thr=0.25):
    h, w = image_bgr.shape[:2]
    dets = detect_boxes(image_bgr)
    box_masks = []
    for d in dets:
        m = np.zeros((h, w), np.uint8)
        x1, y1, x2, y2 = map(int, d.xyxy)
        m[max(0, y1):min(h, y2 + 1), max(0, x1):min(w, x2 + 1)] = 1
        box_masks.append(m)
    statuses, scores = {}, {}
    for s in spaces:
        rm = polygon_mask((h, w), s["polygon"])
        best = max((overlap_ratio(rm, bm) for bm in box_masks), default=0.0)
        scores[s["id"]] = best
        statuses[s["id"]] = "occupied" if best >= thr else "free"
    return statuses, scores, dets


def eval_b_thr(thr, indices):
    y_true, y_pred, lats = [], [], []
    for i in indices:
        entry = manifest[i]
        img = load_bgr(FRAMES / entry["frame"])
        t0 = time.perf_counter()
        st, _, _ = decide_b(img, entry["spaces"], thr=thr)
        lats.append(time.perf_counter() - t0)
        for s in entry["spaces"]:
            y_true.append(s["gt"])
            y_pred.append(st[s["id"]])
    return summarize(y_true, y_pred, lats, f"B_YOLOv8n_overlap@{thr}")


b_trials = []
for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    m = eval_b_thr(thr, val_idx)
    b_trials.append(m)
    print(
        f"B thr={thr:.2f} acc={m.get('accuracy', 0):.3f} "
        f"bal_acc={m.get('balanced_accuracy', 0):.3f} "
        f"occ_recall={m.get('recall_occupied', 0):.3f}"
    )

b_df = pd.DataFrame(b_trials).sort_values(
    ["balanced_accuracy", "f1_occupied"], ascending=False
)
display(b_df)
BEST_B_THR = float(str(b_df.iloc[0]["approach"]).split("@")[-1])
metrics_b = eval_b_thr(BEST_B_THR, test_idx)
metrics_b["approach"] = f"B_YOLOv8n_overlap@{BEST_B_THR}"
metrics_b["best_thr"] = BEST_B_THR
print("TEST", json.dumps(metrics_b, indent=2))
"""
    ),
    md(
        r"""
## 6 · Approach C — YOLOv8n-seg masks + ROI IoU

Same zero-shot idea as B, but uses **instance masks** so partial overlaps / odd angles are scored more precisely than axis-aligned boxes.
"""
    ),
    code(
        r"""
yolo_seg = YOLO("yolov8n-seg.pt")


def detect_masks(image_bgr, conf=0.25):
    res = yolo_seg.predict(image_bgr, conf=conf, verbose=False)[0]
    h, w = image_bgr.shape[:2]
    masks = []
    if res.boxes is None or res.masks is None:
        return masks
    for cls_t, mdata in zip(res.boxes.cls, res.masks.data):
        if int(cls_t.item()) not in VEHICLE_IDS:
            continue
        m = mdata.cpu().numpy()
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((m > 0.5).astype(np.uint8))
    return masks


def iou_mask(a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union else 0.0


def decide_c(image_bgr, spaces, thr=0.15):
    h, w = image_bgr.shape[:2]
    masks = detect_masks(image_bgr)
    statuses, scores = {}, {}
    for s in spaces:
        rm = polygon_mask((h, w), s["polygon"])
        # use max IoU OR coverage of ROI by any mask (coverage often better for occupancy)
        best_iou = max((iou_mask(rm, vm) for vm in masks), default=0.0)
        best_cov = max((overlap_ratio(rm, vm) for vm in masks), default=0.0)
        best = max(best_iou, best_cov)
        scores[s["id"]] = best
        statuses[s["id"]] = "occupied" if best >= thr else "free"
    return statuses, scores, masks


def eval_c_thr(thr, indices):
    y_true, y_pred, lats = [], [], []
    for i in indices:
        entry = manifest[i]
        img = load_bgr(FRAMES / entry["frame"])
        t0 = time.perf_counter()
        st, _, _ = decide_c(img, entry["spaces"], thr=thr)
        lats.append(time.perf_counter() - t0)
        for s in entry["spaces"]:
            y_true.append(s["gt"])
            y_pred.append(st[s["id"]])
    return summarize(y_true, y_pred, lats, f"C_YOLOv8nseg_iou@{thr}")


c_trials = []
for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    m = eval_c_thr(thr, val_idx)
    c_trials.append(m)
    print(
        f"C thr={thr:.2f} acc={m.get('accuracy', 0):.3f} "
        f"bal_acc={m.get('balanced_accuracy', 0):.3f} "
        f"occ_recall={m.get('recall_occupied', 0):.3f}"
    )

c_df = pd.DataFrame(c_trials).sort_values(
    ["balanced_accuracy", "f1_occupied"], ascending=False
)
display(c_df)
BEST_C_THR = float(str(c_df.iloc[0]["approach"]).split("@")[-1])
metrics_c = eval_c_thr(BEST_C_THR, test_idx)
metrics_c["approach"] = f"C_YOLOv8nseg_iou@{BEST_C_THR}"
metrics_c["best_thr"] = BEST_C_THR
print("TEST", json.dumps(metrics_c, indent=2))
"""
    ),
    md("## 7 · Comparison table + pick winner"),
    code(
        r'''
results = [metrics_a, metrics_b, metrics_c]
# Accuracy alone rewarded the broken "always free" models in the old run.
# Rank by balanced accuracy, then occupied F1.
cmp = pd.DataFrame(results).sort_values(
    ["balanced_accuracy", "f1_occupied"], ascending=False
)
display(cmp)

winner = cmp.iloc[0].to_dict()
print(
    "WINNER:", winner["approach"],
    f"balanced_acc={winner.get('balanced_accuracy', 0):.1%}",
    f"acc={winner.get('accuracy', 0):.1%}",
    f"occupied_recall={winner.get('recall_occupied', 0):.1%}",
)

(OUT / "comparison.json").write_text(json.dumps({
    "dataset": "Voxel51/PKLot",
    "lot": manifest[0]["source"],
    "n_frames_total": len(manifest),
    "n_test_frames": len(test_idx),
    "approaches": results,
    "winner": winner,
    "note": "datasets.load_dataset provides images; FiftyOne load_from_hub on the same repo provides polygons+occupancy.",
}, indent=2))

# Qualitative: several DIFFERENT test frames (not just one mostly-empty image)
def frame_occ_count(i):
    return sum(s["gt"] == "occupied" for s in manifest[i]["spaces"])

# Prefer occupied-heavy frames, then fill with remaining test frames
ranked = sorted(test_idx, key=frame_occ_count, reverse=True)
qual_idx = []
for i in ranked:
    if i not in qual_idx:
        qual_idx.append(i)
    if len(qual_idx) >= 6:
        break
print("Qualitative frames:", [(manifest[i]["frame"], frame_occ_count(i)) for i in qual_idx])

# Grid: rows = frames, cols = GT | A | B | C
n_show = len(qual_idx)
fig, axes = plt.subplots(n_show, 4, figsize=(18, 3.2 * n_show))
if n_show == 1:
    axes = np.array([axes])
col_titles = ["GT", "A PatchCNN", f"B YOLO @{BEST_B_THR}", f"C YOLO-seg @{BEST_C_THR}"]

for row, fi in enumerate(qual_idx):
    entry = manifest[fi]
    img = load_bgr(FRAMES / entry["frame"])
    gt = {s["id"]: s["gt"] for s in entry["spaces"]}
    st_a = predict_a_frame(img, entry["spaces"])
    st_b, _, _ = decide_b(img, entry["spaces"], thr=BEST_B_THR)
    st_c, _, _ = decide_c(img, entry["spaces"], thr=BEST_C_THR)
    for col, st in enumerate([gt, st_a, st_b, st_c]):
        ax = axes[row, col]
        vis = overlay_rois(img, entry["spaces"], st)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        title = col_titles[col] if row == 0 else ""
        n_occ = sum(v == "occupied" for v in st.values())
        ax.set_title(f"{title} | {entry['frame']} occ={n_occ}" if title else f"{entry['frame']} occ={n_occ}", fontsize=9)
        ax.axis("off")

plt.suptitle("Multi-frame qualitative comparison (green=free, red=occupied)", y=1.01)
plt.tight_layout()
plt.show()

# Also save each qualitative frame overlay for the winner
winner_name = str(winner["approach"])
for fi in qual_idx:
    entry = manifest[fi]
    img = load_bgr(FRAMES / entry["frame"])
    if winner_name.startswith("A_"):
        st = predict_a_frame(img, entry["spaces"])
    elif winner_name.startswith("B_"):
        st, _, _ = decide_b(img, entry["spaces"], thr=BEST_B_THR)
    else:
        st, _, _ = decide_c(img, entry["spaces"], thr=BEST_C_THR)
    vis = overlay_rois(img, entry["spaces"], st)
    out_path = OUT / f"qual_{entry['frame']}"
    cv2.imwrite(str(out_path), vis)
print("Saved winner overlays to", OUT)
'''
    ),
    md("## 8 · Feasibility report"),
    code(
        r'''
report = f"""# Feasibility Report — PKLot 3-Approach Comparison

## Dataset
- **Hub:** [Voxel51/PKLot](https://huggingface.co/datasets/Voxel51/PKLot)
- **Load API:** `datasets.load_dataset("Voxel51/PKLot")` (images)
- **Labels:** FiftyOne `load_from_hub("Voxel51/PKLot")` (polygons + occupancy) — required because the datasets parquet export is **image-only**
- **License:** CC BY 4.0 (cite Almeida et al., ESWA 2015)
- **Subset:** lot=`{manifest[0]["source"]}`, weather≈{WEATHER}, frames={len(manifest)}, spots/frame≤{MAX_SPOTS_PER_FRAME}

## Approaches
1. **A — Balanced perspective Patch CNN:** warp each polygon to a square, balance both classes, then classify free/occupied.
2. **B — YOLOv8n + ROI overlap:** zero-shot COCO detector (feasibility default).
3. **C — YOLOv8n-seg + ROI IoU/coverage:** zero-shot masks for tighter geometry.

## GitHub evidence
- `martin-marek/parking-space-occupancy`: R-CNN-style perspective ROI pooling + CNN; reports 98% on unseen lots.
- `fabiocarrara/deep-parking`: lightweight per-slot CNN validated on PKLot/CNRPark.
- `Eighonet/parking-research`: broad patch and intersection-based model comparison.
- `visualbuffer/parkingslot`: Mask R-CNN/YOLO for localization, ResNet/VGG for occupancy; reports YOLO misses some small corner objects.

## Results (held-out frames)
{cmp.to_markdown(index=False)}

**Winner (ranked by balanced accuracy):** `{winner["approach"]}` · balanced accuracy={winner.get("balanced_accuracy", float("nan")):.1%} · raw accuracy={winner.get("accuracy", float("nan")):.1%} · occupied recall={winner.get("recall_occupied", float("nan")):.1%} · mean latency={winner.get("latency_mean_s", float("nan")):.3f}s

Targets: ≥90% accuracy, <3s / frame.

## Limitations
- PKLot is parking-lot elevated cams, **not** curb-side street parking (domain gap).
- Full Hub set is ~12k images / ~4GB — this notebook uses a capped subset for Colab.
- Approach A can overfit the lot it was trained on; retrain for a new camera.
- Approaches B/C need ROI polygons defined once per fixed camera.

## Project 2
- Collect target-street labeled frames; re-tune thresholds / fine-tune patch CNN.
- Add temporal smoothing; consider homography for oblique street views.
"""
(REPORTS / "feasibility_report.md").write_text(report)
display(Markdown(report))
print("Saved", REPORTS / "feasibility_report.md")
'''
    ),
]


def main() -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": CELLS,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"Wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
