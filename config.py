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
download_blendednet.py first to fetch the archive into BLENDEDNET_RAW_DIR,
then blendednet_dataset.py to stream cases out of it (one at a time,
deleting each case's raw files as it's cached, then deleting the archive
once done) into BLENDEDNET_CACHE_DIR. Note /kaggle/working has a ~20GB
output quota -- the raw BlendedNet archive alone (25-60GB) may still not
fit; point OUT_DIR at Kaggle's larger temp scratch space if available.
"""

from pathlib import Path

RAW_DIR = Path("data/raw/extracted/training_data")           # shapenet_car_dataset.py raw meshes
CACHE_DIR = Path("/kaggle/input/datasets/reddy42/shape-net/cache")                                # cached .npz + manifest.json
CKPT_DIR = Path("checkpoints")                                # training checkpoints

# Holds the downloaded archive(s) only -- download_blendednet.py no longer
# extracts to disk; blendednet_dataset.py streams cases straight out of the
# zip/tar.gz one at a time so the full unpacked dataset never has to fit
# alongside the archive and the cache.
BLENDEDNET_RAW_DIR = Path("/data/raw")
BLENDEDNET_CACHE_DIR = Path("/data/cache_blendednet")