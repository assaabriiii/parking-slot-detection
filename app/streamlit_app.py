"""Streamlit lab demo. Do not launch in CI/agent sessions; a human runs this later.

  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from parking_mvp.io_util import load_config
from parking_mvp.pipeline import OccupancyPipeline
from parking_mvp.preprocess import load_bgr
from parking_mvp.roi import overlay_rois


def _pipeline(config_path: str) -> OccupancyPipeline:
    return OccupancyPipeline.from_yaml(config_path)


def main() -> None:
    st.set_page_config(page_title="Parking occupancy MVP", layout="wide")
    st.title("Street parking occupancy (lab MVP)")
    st.caption(
        "Fixed camera, ROI polygons, binary free/occupied. Public datasets only. "
        "Night scenes are out of scope. Target tick 2–5 s, response < 3 s (not measured here)."
    )

    default_cfg = str(ROOT / "configs" / "default.yaml")
    config_path = st.sidebar.text_input("Config YAML", default_cfg)
    cfg = load_config(config_path)
    pipe = _pipeline(config_path)

    uploaded = st.sidebar.file_uploader("Frame (optional)", type=["jpg", "jpeg", "png"])
    sample_dir = ROOT / cfg["source"]
    sample_files = sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg"))
    choice = st.sidebar.selectbox(
        "Sample frame",
        options=[str(p.relative_to(ROOT)) for p in sample_files] or ["(none)"],
    )

    if uploaded is not None:
        tmp = ROOT / "outputs" / "_upload.png"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(uploaded.getbuffer())
        frame_path = tmp
    else:
        frame_path = ROOT / choice if sample_files else None

    if frame_path is None or not frame_path.is_file():
        st.warning("No frame selected. Generate the stub with `python scripts/make_sample_data.py`.")
        return

    snap = pipe.infer_image(frame_path)
    pipe.write_status(snap)

    statuses = {s.spot_id: s.status for s in snap.spots}
    vis = overlay_rois(load_bgr(frame_path), pipe.rois, statuses)

    col_img, col_tbl = st.columns([2, 1])
    with col_img:
        st.image(vis[:, :, ::-1], caption=str(frame_path), use_container_width=True)
    with col_tbl:
        st.subheader("Spots")
        st.table(
            [
                {"spot": s.spot_id, "status": s.status, "confidence": round(s.confidence, 2), "raw": s.raw_status}
                for s in snap.spots
            ]
        )
        n_free = sum(1 for s in snap.spots if s.status == "free")
        n_occ = sum(1 for s in snap.spots if s.status == "occupied")
        st.metric("Free", n_free)
        st.metric("Occupied", n_occ)
        st.json(snap.model_dump(mode="json"))


if __name__ == "__main__":
    main()
