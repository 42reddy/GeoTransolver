"""
Single place to point every script at data -- edit this file (not each
script's params block) when moving between local and Kaggle.

Local: defaults below match this repo's local data/ layout, so every
script's path params resolve to a real path and each file runs as-is via
an IDE "Run" button (no CLI args, nothing else to set).

Kaggle: upload the `data/cache/` folder you built locally as a Kaggle
dataset, attach it to the notebook, then change CACHE_DIR below to
Path("/kaggle/input/<your-dataset-slug>/cache") -- train.py and
visualize_sample.py both import CACHE_DIR from here, so that one edit
repoints both.
"""

from pathlib import Path

RAW_DIR = Path("data/raw/extracted/training_data")           # shapenet_car_dataset.py raw meshes
CACHE_DIR = Path("/kaggle/input/datasets/reddy42/shape-net")                                # cached .npz + manifest.json
CKPT_DIR = Path("checkpoints")                                # training checkpoints

BLENDEDNET_RAW_DIR = Path("data/raw/extracted")               # blendednet_dataset.py raw meshes
BLENDEDNET_CACHE_DIR = Path("data/cache_blendednet")           # separate from CACHE_DIR -- different schema
