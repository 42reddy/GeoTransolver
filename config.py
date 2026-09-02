"""
Single place to point every script at data -- edit this file (not each
script's params block) when moving between local and Kaggle.

Local: defaults below match this repo's local data/ layout, so every
script's path params resolve to a real path and each file runs as-is via
an IDE "Run" button (no CLI args, nothing else to set).

Kaggle (ShapeNet car, prebuilt cache): upload the `data/cache/` folder you
built locally as a Kaggle dataset, attach it to the notebook, then change
CACHE_DIR below to Path("/kaggle/input/<your-dataset-slug>/cache") --
train.py and visualize_sample.py both import CACHE_DIR from here, so that
one edit repoints both.

Kaggle (BlendedNet, download + build in-notebook): BLENDEDNET_RAW_DIR and
BLENDEDNET_CACHE_DIR below already point under /kaggle/working/ (the only
writable disk in a Kaggle notebook -- /kaggle/input is read-only), so
download_blendednet.py and blendednet_dataset.py run as-is: run
download_blendednet.py first to fetch + extract into BLENDEDNET_RAW_DIR,
then blendednet_dataset.py to build BLENDEDNET_CACHE_DIR from it. Note
/kaggle/working has a ~20GB output quota -- the raw BlendedNet archive
(25-60GB) may not fit alongside its extracted copy and the cache; delete
the downloaded archive/parts after extraction if you hit the quota, or
set SKIP_EXTRACT/point OUT_DIR at Kaggle's larger temp scratch space if
available.
"""

from pathlib import Path

RAW_DIR = Path("data/raw/extracted/training_data")           # shapenet_car_dataset.py raw meshes
CACHE_DIR = Path("/kaggle/input/datasets/reddy42/shape-net/cache")                                # cached .npz + manifest.json
CKPT_DIR = Path("checkpoints")                                # training checkpoints

BLENDEDNET_RAW_DIR = Path("/kaggle/working/data/raw/extracted")   # download_blendednet.py output / blendednet_dataset.py raw meshes
BLENDEDNET_CACHE_DIR = Path("/kaggle/working/data/cache_blendednet")  # separate from CACHE_DIR -- different schema
