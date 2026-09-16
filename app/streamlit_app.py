"""Streamlit lab demo.

  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from parking_mvp.io_util import list_frames, load_config
from parking_mvp.pipeline import OccupancyPipeline
from parking_mvp.preprocess import load_bgr


def _pipeline(config_path: str) -> OccupancyPipeline:
    cfg = load_config(config_path)
    key = f"pipe:{config_path}:{cfg.get('occupancy_head')}:{cfg.get('patchcnn')}"
    if key not in st.session_state:
        st.session_state[key] = OccupancyPipeline.from_yaml(config_path)
    return st.session_state[key]


def _save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    tmp = ROOT / "outputs" / f"_upload{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded.getbuffer())
    return tmp


def _roi_mismatch_warning(pipe: OccupancyPipeline, bgr) -> None:
    rois = pipe.rois
    h, w = bgr.shape[:2]
    if not rois.image_width or not rois.image_height:
        return
    if (w, h) == (rois.image_width, rois.image_height):
        return
    st.warning(
        f"ROI file is drawn on {rois.image_width}×{rois.image_height} "
        f"({rois.camera_id}), but this frame is {w}×{h}. Polygons are scaled, "
        f"which is only valid if this is the same camera. For a new photo, "
        f"draw new polygons (see scripts/label_guide.md) and set roi_file."
    )


def _render(pipe: OccupancyPipeline, snap, vis) -> None:
    pipe.write_status(snap)
    col_img, col_tbl = st.columns([2, 1])
    with col_img:
        st.image(vis[:, :, ::-1], caption=snap.source, use_container_width=True)
    with col_tbl:
        st.subheader("Spots")
        st.table(
            [
                {
                    "spot": s.spot_id,
                    "status": s.status,
                    "confidence": round(s.confidence, 2),
                    "raw": s.raw_status,
                }
                for s in snap.spots
            ]
        )
        n_free = sum(1 for s in snap.spots if s.status == "free")
        n_occ = sum(1 for s in snap.spots if s.status == "occupied")
        st.metric("Free", n_free)
        st.metric("Occupied", n_occ)
        st.caption(f"tick={snap.tick_seconds}s · frame={snap.frame_index} · {snap.timestamp.isoformat()}")
        if snap.notes:
            st.info(snap.notes)
        st.json(snap.model_dump(mode="json"))


def main() -> None:
    st.set_page_config(page_title="Parking occupancy MVP", layout="wide")
    st.title("Street parking occupancy (lab MVP)")
    st.caption(
        "Fixed camera, ROI polygons, binary free/occupied. Public datasets only. "
        "Night scenes are out of scope. Sequence mode advances every tick_seconds (default 3)."
    )

    default_cfg = str(ROOT / "configs" / "default.yaml")
    config_path = st.sidebar.text_input("Config YAML", default_cfg)
    cfg = load_config(config_path)
    pipe = _pipeline(config_path)
    tick = float(cfg.get("tick_seconds", 3.0))

    mode = st.sidebar.radio("Mode", ["Still frame", "Sequence (2–5 s tick)"], index=0)
    uploaded = st.sidebar.file_uploader("Frame (optional)", type=["jpg", "jpeg", "png"])
    sample_dir = ROOT / cfg["source"]
    try:
        sample_files = list_frames(sample_dir)
    except FileNotFoundError:
        sample_files = []

    upload_path = _save_upload(uploaded) if uploaded is not None else None

    # An uploaded still always wins: Sequence otherwise silently scores the stub.
    if upload_path is not None:
        if mode.startswith("Sequence"):
            st.info(
                "An uploaded frame is selected, so Sequence is paused. "
                "Clear the upload to play the config source folder."
            )
        bgr = load_bgr(upload_path)
        _roi_mismatch_warning(pipe, bgr)
        snap = pipe.infer_bgr(bgr, source=str(upload_path) + f" ({uploaded.name})")
        _render(pipe, snap, pipe.overlay_snapshot(bgr, snap))
        return

    if mode == "Still frame":
        choice = st.sidebar.selectbox(
            "Sample frame",
            options=[str(p.relative_to(ROOT)) for p in sample_files] or ["(none)"],
        )
        frame_path = ROOT / choice if sample_files else None
        if frame_path is None or not frame_path.is_file():
            st.warning("No frame selected. Generate the stub with `python scripts/make_sample_data.py`.")
            return
        bgr = load_bgr(frame_path)
        _roi_mismatch_warning(pipe, bgr)
        snap = pipe.infer_bgr(bgr, source=str(frame_path))
        _render(pipe, snap, pipe.overlay_snapshot(bgr, snap))
        return

    if not sample_files:
        st.warning("No frames in the config source. Generate the stub with `python scripts/make_sample_data.py`.")
        return

    if "seq_i" not in st.session_state:
        st.session_state.seq_i = 0
    if "seq_playing" not in st.session_state:
        st.session_state.seq_playing = False

    cols = st.sidebar.columns(3)
    if cols[0].button("Prev"):
        st.session_state.seq_i = (st.session_state.seq_i - 1) % len(sample_files)
        st.session_state.seq_playing = False
    if cols[1].button("Play"):
        st.session_state.seq_playing = True
    if cols[2].button("Pause"):
        st.session_state.seq_playing = False

    st.sidebar.write(f"Tick {tick:.1f}s · {len(sample_files)} frames · index {st.session_state.seq_i}")

    frame_path = sample_files[st.session_state.seq_i]
    bgr = load_bgr(frame_path)
    _roi_mismatch_warning(pipe, bgr)
    snap = pipe.infer_bgr(bgr, source=str(frame_path), frame_index=st.session_state.seq_i)
    _render(pipe, snap, pipe.overlay_snapshot(bgr, snap))

    if st.session_state.seq_playing:
        time.sleep(max(0.1, tick))
        st.session_state.seq_i = (st.session_state.seq_i + 1) % len(sample_files)
        st.rerun()


if __name__ == "__main__":
    main()
