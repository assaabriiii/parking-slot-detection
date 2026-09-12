"""Optional FastAPI stub. Do not start uvicorn in agent sessions.

  uvicorn app.fastapi_app:app --reload
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_mvp.io_util import load_config

app = FastAPI(title="Parking occupancy MVP", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def last_status() -> JSONResponse:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    path = ROOT / (cfg.get("output") or {}).get("last_status", "outputs/last_status.json")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"run the pipeline first; missing {path}")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
