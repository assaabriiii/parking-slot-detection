#!/usr/bin/env bash
# Download public parking datasets. Do NOT run this from the agent session.
# Review sizes first (PKLot ~4.6 GB compressed; CNR-EXT full frames ~1.1 GB).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${ROOT}/data"
mkdir -p "${DATA}/cnrpark" "${DATA}/pklot" "${DATA}/metapklot"

echo "CNRPark-EXT (ODbL v1.0) — http://cnrpark.it/"
echo "  Patches 150x150 are enough for occupancy-head wiring (~0.5 GB combined)."
echo "  Skip CNR-EXT_FULL_IMAGE unless you need full frames + camera CSVs."
# curl -L -o "${DATA}/cnrpark/CNRPark-Patches-150x150.zip" "http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip"
# curl -L -o "${DATA}/cnrpark/CNR-EXT-Patches-150x150.zip" "http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip"
# curl -L -o "${DATA}/cnrpark/CNRPark+EXT.csv" "http://cnrpark.it/dataset/CNRPark+EXT.csv"
# curl -L -o "${DATA}/cnrpark/splits.zip" "http://cnrpark.it/dataset/splits.zip"
# curl -L -o "${DATA}/cnrpark/CNR-EXT_FULL_IMAGE_1000x750.tar" "http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar"

echo "PKLot (CC BY 4.0) — Hugging Face mirror of UFPR archive"
echo "  ~4.6 GB compressed. Prefer a single camera/day subset after extract."
# curl -L -o "${DATA}/pklot/PKLot.tar.gz" "https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz"
# Original host (often unreachable): http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz

echo "MetaPKLot — COCO-style cars/spots over PKLot + CNRPark-EXT"
echo "  Shallow clone; annotation tarballs are under annotations/"
# git clone --depth 1 https://github.com/DSBD-Research/MetaPKLot-Dataset.git "${DATA}/metapklot/MetaPKLot-Dataset"
# huggingface-cli download DSBD-Research/MetaPKLot-Dataset --repo-type dataset --local-dir "${DATA}/metapklot/hf"

echo "Commands are commented on purpose. Uncomment locally / in Colab after checking disk."
exit 0
