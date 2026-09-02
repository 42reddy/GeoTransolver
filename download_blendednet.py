"""
Download the BlendedNet / BlendedNet++ blended-wing-body (BWB) aircraft
aerodynamics datasets from Harvard Dataverse.

Usage:
    python download_blendednet.py --dataset blendednet   --out_dir data/raw
    python download_blendednet.py --dataset blendednet++ --out_dir data/raw
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import requests

DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

DATASETS = {
    "blendednet": {
        "doi": "10.7910/DVN/VJT9EP",
        "files": [(11905766, "BlendedNet_Dataset_Released.zip")],
    },
    "blendednet++": {
        "doi": "10.7910/DVN/ICIDK4",
        "files": [
            (13355836, "blendednet++_dataset.tar.gz.part-00"),
            (13355834, "blendednet++_dataset.tar.gz.part-01"),
            (13355835, "blendednet++_dataset.tar.gz.part-02"),
            (13355839, "blendednet++_dataset.tar.gz.part-03"),
            (13355828, "blendednet++_dataset.tar.gz.part-04"),
            (13355829, "blendednet++_dataset.tar.gz.part-05"),
            (13355837, "blendednet++_dataset.tar.gz.part-06"),
            (13355832, "blendednet++_dataset.tar.gz.part-07"),
            (13355841, "blendednet++_dataset.tar.gz.part-08"),
            (13355830, "blendednet++_dataset.tar.gz.part-09"),
            (13355840, "blendednet++_dataset.tar.gz.part-10"),
            (13355838, "blendednet++_dataset.tar.gz.part-11"),
            (13355831, "blendednet++_dataset.tar.gz.part-12"),
            (13355833, "blendednet++_dataset.tar.gz.part-13"),
        ],
    },
}


def download_file(file_id: int, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = DATAVERSE_FILE_URL.format(file_id=file_id)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [{dest.name}] {done / total:.1%}", end="", flush=True)
    print()
    tmp.rename(dest)


def extract(raw_dir: Path, dataset: str) -> Path:
    out_dir = raw_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    if dataset == "blendednet":
        zip_path = raw_dir / "BlendedNet_Dataset_Released.zip"
        shutil.unpack_archive(str(zip_path), str(out_dir))
    else:
        parts = sorted(raw_dir.glob("blendednet++_dataset.tar.gz.part-*"))
        combined = raw_dir / "blendednet++_dataset.tar.gz"
        if not combined.exists():
            with open(combined, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out)
        subprocess.run(["tar", "-xzf", str(combined), "-C", str(out_dir)], check=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS.keys(), default="blendednet")
    ap.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip_extract", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[args.dataset]
    print(f"Downloading {args.dataset} (doi:{spec['doi']}) -> {args.out_dir}")
    for file_id, name in spec["files"]:
        download_file(file_id, args.out_dir / name)

    if not args.skip_extract:
        print("Extracting...")
        out_dir = extract(args.out_dir, args.dataset)
        print(f"Extracted to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
