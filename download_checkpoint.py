"""
Pull a committed Kaggle notebook's output (the checkpoint zip) down to this
machine via kagglehub. Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(already present if `kaggle` CLI / kagglehub has been used before on this
machine).

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No CLI flags -- edit the PARAMS block below (NOTEBOOK slug + the output
filename Kaggle listed, usually visible in the notebook's Output tab after
a committed "Save & Run All"), then run this file.
"""

import zipfile
from pathlib import Path

import kagglehub

# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
NOTEBOOK = "reddy42/notebookf54c15e6dc"   # <kaggle-username>/<notebook-slug>
OUTPUT_FILE = "model.zip"                 # filename as listed in the notebook's Output tab
DOWNLOAD_DIR = Path("./geotransolver")    # kagglehub's own cache/download dir
EXTRACT_DIR = Path("checkpoints")         # where to unzip the checkpoint locally


def main():
    zip_path = kagglehub.notebook_output_download(NOTEBOOK, path=OUTPUT_FILE, output_dir=str(DOWNLOAD_DIR))
    print(f"downloaded -> {zip_path}")

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(EXTRACT_DIR)
    print(f"extracted {len(names)} file(s) -> {EXTRACT_DIR}/")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
