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


def extract(raw_dir: Path, out_dir: Path, keep_archives: bool, surface_only: bool) -> Path:
    zip_path = raw_dir / "mlcfd_data.zip"
    training_dir = out_dir / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    wanted_suffixes = ("quadpress_smpl.vtk",) if surface_only else ("quadpress_smpl.vtk", "hexvelo_smpl.vtk")

    print("Reading outer zip's central directory...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in PARAM_TARS:
            i = member.split("param")[-1].split(".")[0]
            tar_path = raw_dir / f"param{i}.tar.gz"
            if not tar_path.exists():
                print(f"  extracting {member} -> {tar_path}")
                with zf.open(member) as src, open(tar_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            print(f"  unpacking {', '.join(wanted_suffixes)} from param{i}.tar.gz")
            with tarfile.open(tar_path, mode="r:gz") as tf:
                members = [m for m in tf.getmembers() if m.name.endswith(wanted_suffixes)]
                tf.extractall(training_dir, members=members)

            if not keep_archives:
                tar_path.unlink()

    if not keep_archives:
        zip_path.unlink()

    return training_dir


# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
OUT_DIR = Path("data/raw")
SKIP_EXTRACT = False
KEEP_ARCHIVES = False    # True -> keep the downloaded .zip/.tar.gz after extraction (uses more disk)
SURFACE_ONLY = False     # True -> extract only quadpress_smpl.vtk (surface+pressure); skips
                          # hexvelo_smpl.vtk (volume+velocity) -- no real SDF, pressure-only targets


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ShapeNet-Car -> {OUT_DIR}")
    download_file(URL, OUT_DIR / "mlcfd_data.zip")

    if not SKIP_EXTRACT:
        print("Extracting...")
        extracted = extract(OUT_DIR, OUT_DIR / "extracted", KEEP_ARCHIVES, SURFACE_ONLY)
        print(f"Extracted to {extracted}")
    print("Done.")


if __name__ == "__main__":
    main()
