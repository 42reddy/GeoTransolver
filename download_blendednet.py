"""
Download the BlendedNet / BlendedNet++ blended-wing-body (BWB) aircraft
aerodynamics datasets from Harvard Dataverse.

Usage:
No CLI flags -- edit the PARAMS block below main() directly, then run this
file (IDE run button or `python download_blendednet.py`). Set DATASET to
"blendednet" or "blendednet++".
"""

import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests

from config import BLENDEDNET_RAW_DIR

DATAVERSE_FILE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"

# Dataverse's WAF 403s the default "python-requests/x.y" User-Agent; a
# browser-like one is enough to get through to the S3 redirect.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

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


def download_file(
    file_id: int, dest: Path, chunk_size: int = 1 << 20, max_retries: int = 8
) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = DATAVERSE_FILE_URL.format(file_id=file_id)

    # Large Dataverse/S3 files routinely trip a read timeout mid-stream even
    # on a healthy connection. Retry the request, resuming via Range from
    # however much of tmp is already on disk, instead of failing outright.
    for attempt in range(1, max_retries + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        headers = dict(REQUEST_HEADERS)
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                if resume_from and r.status_code == 200:
                    # server ignored Range and is sending the whole file again
                    resume_from = 0
                r.raise_for_status()
                content_length = int(r.headers.get("content-length", 0))
                total = (
                    content_length + resume_from
                    if r.status_code == 206
                    else content_length
                )
                done = resume_from
                mode = "ab" if r.status_code == 206 else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r  [{dest.name}] {done / total:.1%}",
                                end="",
                                flush=True,
                            )
            print()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise
            wait = min(2**attempt, 60)
            print(
                f"\n  [{dest.name}] attempt {attempt}/{max_retries} failed "
                f"({e!r}); retrying in {wait}s, resuming from "
                f"{tmp.stat().st_size if tmp.exists() else 0} bytes..."
            )
            time.sleep(wait)
            continue

        final_size = tmp.stat().st_size
        if total and final_size != total:
            raise IOError(
                f"{dest.name}: incomplete download ({final_size} of {total} bytes). "
                f"This is usually the disk filling up mid-write -- check `df -h` on the "
                f"target volume, free up space, then rerun (it will resume from {final_size} bytes)."
            )
        tmp.rename(dest)
        return

    raise IOError(f"{dest.name}: exhausted {max_retries} retries")


def extract(raw_dir: Path, dataset: str) -> Path:
    out_dir = raw_dir / "extracted"
    out_dir.mkdir(exist_ok=True)
    if dataset == "blendednet":
        zip_path = raw_dir / "BlendedNet_Dataset_Released.zip"
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                zip_path.unlink()
                raise IOError(
                    f"{zip_path.name} is corrupt (bad CRC on {bad!r}) and has been "
                    f"deleted -- this means the download completed but the bytes on "
                    f"disk don't match (usually the disk filled up mid-write). Check "
                    f"`df -h` on the target volume, free up space, then rerun the "
                    f"download from scratch."
                )
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


# ==========================================================================
# Params -- edit these directly, no CLI flags. OUT_DIR is derived from
# BLENDEDNET_RAW_DIR in config.py (extract() writes to OUT_DIR/"extracted",
# which is where BLENDEDNET_RAW_DIR points) -- edit config.py to switch
# between local and Kaggle, not this file.
# ==========================================================================
DATASET = "blendednet"    # "blendednet" or "blendednet++"
OUT_DIR = BLENDEDNET_RAW_DIR.parent
SKIP_EXTRACT = False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spec = DATASETS[DATASET]
    print(f"Downloading {DATASET} (doi:{spec['doi']}) -> {OUT_DIR}")
    for file_id, name in spec["files"]:
        download_file(file_id, OUT_DIR / name)

    if not SKIP_EXTRACT:
        print("Extracting...")
        out_dir = extract(OUT_DIR, DATASET)
        print(f"Extracted to {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
