"""
Download the ShapeNet-Car aerodynamics dataset (Umetani & Bickel, SIGGRAPH
2018, "Learning Three-dimensional Flow for Interactive Aerodynamic Design")
-- the same 3D benchmark used in the Transolver paper itself
(github.com/thuml/Transolver/tree/main/Car-Design-ShapeNetCar).

889 car shapes, each with a downsampled surface mesh (~3,682 points) storing
the converged RANS surface pressure field. Raw archive is ~1.9 GB, well
under the 10 GB you asked for.

--------------------------------------------------------------------------
Verified archive layout (inspected via HTTP Range requests, not guessed)
--------------------------------------------------------------------------
mlcfd_data.zip
  mlcfd_data/training_data/param0.tar.gz  (100 cases -- the held-out test fold)
  mlcfd_data/training_data/param1.tar.gz  (each of param1..param8 ~= training)
  ...
  mlcfd_data/training_data/param8.tar.gz

Each param{i}.tar.gz contains, per case:
  param{i}/<case_hash>/quadpress_smpl.vtk   surface mesh + pressure  <- we keep this
  param{i}/<case_hash>/hexvelo_smpl.vtk     volume mesh + velocity  <- skipped (not needed for a surface-field task, and ~13x bigger)
  param{i}/<case_hash>/press.npy, velo.npy, param1.txt, param2.txt, cd.txt  <- skipped, redundant with the vtk / not needed

We only extract quadpress_smpl.vtk from each tar.gz, which shrinks the
on-disk footprint after extraction to ~190 MB total (vs. ~1.9 GB if you
extracted everything).

Usage:
    python download_shapenet_car.py --out_dir data/raw
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests

URL = "http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip"

# Some hosts reject the default python-requests User-Agent; harmless to send
# a browser-like one regardless of whether this particular host needs it.
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

PARAM_TARS = [f"mlcfd_data/training_data/param{i}.tar.gz" for i in range(9)]


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = dict(REQUEST_HEADERS)
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if resume_from and r.status_code == 200:
            resume_from = 0  # server ignored Range, sending from scratch
        r.raise_for_status()
        content_length = int(r.headers.get("content-length", 0))
        total = content_length + resume_from if r.status_code == 206 else content_length
        done = resume_from
        mode = "ab" if r.status_code == 206 else "wb"
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [{dest.name}] {done / total:.1%}", end="", flush=True)
    print()

    final_size = tmp.stat().st_size
    if total and final_size != total:
        raise IOError(
            f"{dest.name}: incomplete download ({final_size} of {total} bytes). "
            f"Rerun the same command to resume from {final_size} bytes."
        )
    tmp.rename(dest)


def extract(raw_dir: Path, out_dir: Path, keep_archives: bool) -> Path:
    zip_path = raw_dir / "mlcfd_data.zip"
    training_dir = out_dir / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)

    print("Reading outer zip's central directory...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in PARAM_TARS:
            i = member.split("param")[-1].split(".")[0]
            tar_path = raw_dir / f"param{i}.tar.gz"
            if not tar_path.exists():
                print(f"  extracting {member} -> {tar_path}")
                with zf.open(member) as src, open(tar_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            print(f"  unpacking quadpress_smpl.vtk from param{i}.tar.gz")
            with tarfile.open(tar_path, mode="r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith("quadpress_smpl.vtk")]
                tf.extractall(training_dir, members=members)

            if not keep_archives:
                tar_path.unlink()

    if not keep_archives:
        zip_path.unlink()

    return training_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--keep_archives", action="store_true",
                     help="keep the downloaded .zip/.tar.gz after extraction (uses more disk)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShapeNet-Car -> {args.out_dir}")
    download_file(URL, args.out_dir / "mlcfd_data.zip")

    if not args.skip_extract:
        print("Extracting (surface meshes only)...")
        extracted = extract(args.out_dir, args.out_dir / "extracted", args.keep_archives)
        print(f"Extracted to {extracted}")
    print("Done.")


if __name__ == "__main__":
    main()
