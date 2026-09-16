#!/usr/bin/env bash
# Local lab pack: tests, CPU latency, temporal flips, overlay ticks.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:tests:.
python -m pytest -q
python scripts/benchmark_tick.py --config configs/default.yaml
python scripts/benchmark_tick.py --config configs/default.yaml --roi-file configs/rois.example.json --out outputs/benchmark_cpu_12spots.json
python scripts/eval_temporal_flips.py
python scripts/export_demo_overlays.py
echo "lab pack ok"
