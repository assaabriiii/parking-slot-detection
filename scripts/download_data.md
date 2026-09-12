# Public dataset download (human / Colab later)

This MVP must **not** use self-captured street video. Use only the sources below.
Do **not** download multi-GB archives unless you need them for path wiring.

Suggested layout after extract (gitignored except `data/sample/`):

```text
data/
  sample/                 # tiny synthetic stub (in git)
  cnrpark/
  pklot/
  metapklot/
```

`scripts/download_data.sh` lists the same URLs with commands commented out.

---

## 1. CNRPark-EXT — occupancy patches (primary)

- Site: http://cnrpark.it/
- License: **Open Data Commons Open Database License (ODbL) v1.0**
- Paper: Amato et al., “Deep learning for decentralized parking lot occupancy detection”, ESWA 2017

Exact files:

| File | URL | Size (approx) | Role |
| --- | --- | --- | --- |
| CNRPark patches | http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip | 36.6 MB | 150×150 free/busy crops, cameras A/B |
| CNR-EXT patches | http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip | 449.5 MB | weather splits, 9 cameras |
| Metadata CSV | http://cnrpark.it/dataset/CNRPark+EXT.csv | 18.1 MB | one row per patch |
| Experiment splits | http://cnrpark.it/dataset/splits.zip | 27.2 MB | list files (`path label`) |
| Full frames | http://cnrpark.it/dataset/CNR-EXT_FULL_IMAGE_1000x750.tar | 1.1 GB | 1000×750 frames + per-camera bbox CSVs |

Patch layout (CNRPark):

```text
<camera>/<free|busy>/YYYYMMDD_HHMM_<SLOT_ID>.jpg
```

CNR-EXT patches:

```text
PATCHES/<SUNNY|OVERCAST|RAINY>/<YYYY-MM-DD>/camera<ID>/<W>_<date>_<time>_C0<cam>_<slot>.jpg
```

Full-frame bbox CSVs use coordinates on the **2592×1944** original; rescale by `1000/2592` and `750/1944` to match the released 1000×750 JPEGs.

Colab (patches only, skip full frames unless needed):

```bash
mkdir -p data/cnrpark && cd data/cnrpark
curl -L -O http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip
curl -L -O http://cnrpark.it/dataset/CNR-EXT-Patches-150x150.zip
curl -L -O http://cnrpark.it/dataset/CNRPark+EXT.csv
unzip -q CNRPark-Patches-150x150.zip
unzip -q CNR-EXT-Patches-150x150.zip
```

Then:

```bash
python scripts/prepare_dataset.py cnrpark-patches \
  --root data/cnrpark \
  --out data/cnrpark/manifest.jsonl
```

---

## 2. PKLot — UFPR (Hugging Face mirror)

- Mirror: https://huggingface.co/datasets/teenygrad/pklot
- Original (often down): http://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
- License: **CC BY 4.0**
- Paper: Almeida et al., “PKLot — A robust dataset for parking lot classification”, ESWA 2015

Archive (byte-identical, ~4.6 GB compressed):

```text
https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz
```

Extracted layout:

```text
PKLot/PKLot/{PUCPR,UFPR04,UFPR05}/{Cloudy,Rainy,Sunny}/{YYYY-MM-DD}/*.jpg
PKLot/PKLot/{...}/*.xml   # sibling XML: parking-space polygons + occupied flag
```

Colab (only if you have disk + time):

```bash
mkdir -p data/pklot
curl -L -o data/pklot/PKLot.tar.gz \
  https://huggingface.co/datasets/teenygrad/pklot/resolve/main/PKLot.tar.gz
tar -xzf data/pklot/PKLot.tar.gz -C data/pklot
python scripts/prepare_dataset.py pklot-xml \
  --root data/pklot/PKLot \
  --out data/pklot/manifest.jsonl \
  --limit 50
```

---

## 3. MetaPKLot — COCO-style cars + spots

- GitHub: https://github.com/DSBD-Research/MetaPKLot-Dataset
- Hugging Face: https://huggingface.co/datasets/DSBD-Research/MetaPKLot-Dataset
- Licenses: **PKLot images CC BY 4.0**; **CNRPark-EXT images ODbL v1.0**; PLds images are **not** redistributed
- Paper: de Almeida et al., MetaPKLot, Neural Computing and Applications, 2026, https://doi.org/10.1007/s00521-026-12398-0

```bash
git clone --depth 1 https://github.com/DSBD-Research/MetaPKLot-Dataset.git data/metapklot/MetaPKLot-Dataset
# example annotation extract
tar -xf data/metapklot/MetaPKLot-Dataset/annotations/original/spots/PKLot/ufpr04_spots.tar.xz \
  -C data/metapklot
python scripts/prepare_dataset.py metapklot-coco \
  --json path/to/extracted_spots.json \
  --out data/metapklot/spots_manifest.jsonl
```

Spot `category_id`: `0` empty / free, `1` occupied. `car_id = -1` means no vehicle.

---

## License reminder

Share-alike (ODbL) applies to CNRPark-EXT derived databases. Attribute PKLot (CC BY). Do not mix in private street captures for this phase.
