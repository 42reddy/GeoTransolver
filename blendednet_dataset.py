"""
Turn extracted BlendedNet / BlendedNet++ CFD cases into fixed-size point
clouds and expose them as a torch Dataset whose tensors plug straight into
Transolver.forward(pos, features) from transolver.py.

--------------------------------------------------------------------------
What a "case" is
--------------------------------------------------------------------------
Each case is one BWB geometry solved at one flight condition, run to
steady-state RANS convergence (density/turbulence residuals below the
dataset's threshold). The "converged field" we predict is the point-wise
surface state *at that converged solution*:

    Cp                pressure coefficient
    Cfx, Cfy, Cfz      skin-friction coefficient components

stored on the surface mesh as VTK point-data arrays -- this is different
from an intermediate/unconverged iterate, which the dataset does not
release.

--------------------------------------------------------------------------
What we feed the model
--------------------------------------------------------------------------
    pos        (N, 3)  surface point coordinates x, y, z
    geometry   (N, 3)  per-point surface normal (nx, ny, nz) -- GeoTransolver's
                        geometry stream (context tokens + local ball-query
                        features + direct per-point input)
    condition  (C,)    the case's geometry parameters and flight conditions,
                        one vector per case (not broadcast to points --
                        GeoTransolver reads it as a single cross-attention
                        context token, see geotransolver/geotransolver.py)
    target     (N, F)  the converged Cp / Cfx / Cfy / Cfz fields, whichever
                        point-data arrays the VTK file actually contains
                        (auto-detected, see discover_fields())
    constants  (K,)    per-case aerodynamic coefficients (C1, C3, ... --
                        integrated quantities like CL/CD/CM), a regression
                        target predicted from the pooled point features,
                        not an input (see split_metadata())

GeoTransolver consumes a fixed point count per batch item ((B, N, ...)
tensors), so every case is randomly subsampled (or resampled with
replacement, if a case has fewer points than --num_points) to the same N.

--------------------------------------------------------------------------
Why discovery instead of hardcoded paths
--------------------------------------------------------------------------
The Dataverse archives are 25-60 GB; this script was written against the
dataset's *documented* structure (papers arXiv:2509.07209, arXiv:2512.03280)
without unpacking the full archive to verify exact file/folder names. So
instead of hardcoding paths like "case_0001/surface.vtk", find_cases() below
recursively globs for surface-mesh files and pairs each with a metadata
file in the same directory. Run this script once with `--inspect` on your
actual extracted data first -- it prints what got matched and what field
names/metadata keys were found, so you can fix the matching heuristics
below if your archive's layout differs.

Requires: pyvista (`pip install pyvista`) for VTK I/O.

"""

import json
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import BLENDEDNET_CACHE_DIR, BLENDEDNET_RAW_DIR

SURFACE_EXTS = (".vtk", ".vtu", ".vtp")
META_EXTS = (".json", ".dat", ".csv", ".yaml", ".yml")

PRESSURE_TOKENS = ("cp", "pressure")
FRICTION_TOKENS = ("cf", "friction", "shear")

# Metadata keys matching this pattern (C1, C2, C3, ...) are per-case
# aerodynamic coefficients -- regression targets, not input conditioning.
CONST_KEY_PATTERN = re.compile(r"^c\d+$", re.IGNORECASE)


def split_metadata(meta: dict, const_keys: list[str] | None = None) -> tuple[dict, dict]:
    """Split a case's metadata into (condition, constants): constants are
    the keys matching CONST_KEY_PATTERN (or the explicit `const_keys`),
    everything else is flight/geometry conditioning."""
    if const_keys is None:
        const_keys = [k for k in meta if CONST_KEY_PATTERN.match(k)]
    condition = {k: v for k, v in meta.items() if k not in const_keys}
    constants = {k: meta[k] for k in const_keys if k in meta}
    return condition, constants


# --------------------------------------------------------------------------
# Archive discovery: the raw dataset is never fully extracted to disk --
# find_archive() locates the downloaded zip/tar.gz, and build_cache() below
# streams it one case at a time (extract just that case's files -> process
# into the .npz cache -> delete the raw files) so disk only ever holds the
# archive + one case's raw mesh + the cache being built, instead of the
# archive + a full extracted copy + the cache all at once.
# --------------------------------------------------------------------------
def find_archive(raw_dir: Path, dataset: str = "blendednet") -> Path:
    if dataset == "blendednet":
        path = raw_dir / "BlendedNet_Dataset_Released.zip"
    else:
        path = raw_dir / "blendednet++_dataset.tar.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run download_blendednet.py first "
            f"(DATASET={dataset!r})."
        )
    return path


def _case_dir(member_name: str) -> str:
    return str(Path(member_name).parent)


def _is_mesh(name: str) -> bool:
    return Path(name).suffix.lower() in SURFACE_EXTS


def _is_meta(name: str) -> bool:
    return Path(name).suffix.lower() in META_EXTS


def _group_cases(names):
    """Pair each mesh file with a metadata file sharing its directory."""
    by_dir = {}
    for n in names:
        if n.endswith("/"):
            continue
        by_dir.setdefault(_case_dir(n), []).append(n)
    cases = []
    for d, members in by_dir.items():
        mesh = next((m for m in members if _is_mesh(m)), None)
        if mesh is None:
            continue
        meta = next((m for m in members if _is_meta(m)), None)
        cases.append({"case_id": Path(mesh).stem, "mesh": mesh, "meta": meta})
    return cases


def find_cases(raw_dir: Path, dataset: str = "blendednet"):
    """List candidate cases without extracting anything (used by --inspect)."""
    archive = find_archive(raw_dir, dataset)
    if dataset == "blendednet":
        with zipfile.ZipFile(archive) as zf:
            return _group_cases(zf.namelist())
    with tarfile.open(archive, mode="r|gz") as tf:
        return _group_cases(m.name for m in tf if m.isfile())


def discover_fields(mesh, tokens, requested: list[str] | None = None):
    names = list(mesh.point_data.keys())
    if requested:
        missing = [r for r in requested if r not in names]
        if missing:
            raise ValueError(f"Requested fields {missing} not in {names}")
        return requested
    return [n for n in names if any(t in n.lower() for t in tokens)]


def load_metadata(meta_path: Path | None):
    """Best-effort parse of a case's geometry/flight-condition file.

    Supports JSON directly. For .dat/.csv, falls back to a generic
    "one key-value pair per line" parse (whitespace or comma separated).
    Adjust this if your metadata files use a different format --
    run with --inspect to see the raw values this pulls out.
    """
    if meta_path is None:
        return {}
    if meta_path.suffix.lower() == ".json":
        return json.loads(meta_path.read_text())
    values = {}
    for line in meta_path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) == 2:
            key, val = parts
            try:
                values[key] = float(val)
            except ValueError:
                continue
    return values


def load_case(case: dict, global_feat_keys: list[str] | None, const_keys: list[str] | None,
              fields: list[str] | None):
    import pyvista as pv

    mesh = pv.read(str(case["mesh_path"]))
    pos = np.asarray(mesh.points, dtype=np.float32)

    normal_key = next((k for k in mesh.point_data.keys() if "normal" in k.lower()), None)
    if normal_key is not None:
        normals = np.asarray(mesh.point_data[normal_key], dtype=np.float32)
    else:
        mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        normals = np.asarray(mesh.point_data["Normals"], dtype=np.float32)

    field_names = discover_fields(mesh, PRESSURE_TOKENS + FRICTION_TOKENS, fields)
    if not field_names:
        raise ValueError(
            f"No Cp/Cf-like fields found in {case['mesh_path']}; "
            f"available point_data: {list(mesh.point_data.keys())}. Pass --fields explicitly."
        )
    target = np.stack([np.asarray(mesh.point_data[f], dtype=np.float32).reshape(len(pos), -1)[:, 0]
                        for f in field_names], axis=-1)

    meta = load_metadata(case["meta_path"])
    condition_meta, const_meta = split_metadata(meta, const_keys)
    if global_feat_keys is None:
        global_feat_keys = sorted(condition_meta.keys())
    if const_keys is None:
        const_keys = sorted(const_meta.keys())
    condition = np.array([condition_meta.get(k, 0.0) for k in global_feat_keys], dtype=np.float32)
    constants = np.array([const_meta.get(k, 0.0) for k ind const_keys], dtype=np.float32)

    return pos, normals, condition, target, constants, field_names, global_feat_keys, const_keys


def subsample(pos, geometry, target, num_points: int, rng: np.random.Generator):
    n = len(pos)
    replace = n < num_points
    idx = rng.choice(n, size=num_points, replace=replace)
    return pos[idx], geometry[idx], target[idx]


# --------------------------------------------------------------------------
# Cache builder -- streams cases straight out of the archive. Each case is
# extracted to a scratch dir, turned into a cached .npz, then the scratch
# dir is deleted before the next case is touched, so raw mesh data for at
# most one (zip) or a couple (tar.gz, see below) cases sits on disk at once.
#
# A corrupt individual case (e.g. a bad zip CRC on one member -- this
# archive has a handful) is logged and skipped, not fatal. Consecutive
# failures are tracked, though: if MAX_CONSECUTIVE_FAILURES cases in a row
# all fail, that's not "a few corrupt files" anymore, it's more likely a
# bug in the extraction/parsing code (or a systemically broken download),
# so we stop and raise instead of silently skipping the entire dataset.
# --------------------------------------------------------------------------
MAX_CONSECUTIVE_FAILURES = 20


class TooManyConsecutiveFailures(RuntimeError):
    pass


def _process_one_case(case_id, mesh_path, meta_path, global_feat_keys, const_keys, fields,
                       num_points, rng, cache_dir):
    fs_case = {"case_id": case_id, "mesh_path": mesh_path, "meta_path": meta_path}
    pos, normals, condition, target, constants, field_names, gkeys, ckeys = load_case(
        fs_case, global_feat_keys, const_keys, fields)
    pos, normals, target = subsample(pos, normals, target, num_points, rng)
    np.savez_compressed(cache_dir / f"{case_id}.npz", pos=pos, geometry=normals,
                         condition=condition, target=target, constants=constants)
    return field_names, gkeys, ckeys


class _ConsecutiveFailureTracker:
    """Counts consecutive per-case failures across whichever build_fn is
    running and raises once too many happen in a row; a success resets it."""

    def __init__(self):
        self.consecutive = 0
        self.total_failed = 0

    def ok(self):
        self.consecutive = 0

    def failed(self, case_id: str, error: Exception):
        self.consecutive += 1
        self.total_failed += 1
        print(f"  [skip] {case_id}: {error}")
        if self.consecutive >= MAX_CONSECUTIVE_FAILURES:
            raise TooManyConsecutiveFailures(
                f"{self.consecutive} cases in a row failed to process (most recently "
                f"{case_id!r}: {error}). That's more than a couple of corrupt files -- "
                f"stopping so this can be looked at, rather than silently skipping the "
                f"rest of the dataset. Check the error above; if it really is just "
                f"scattered corrupt cases, raise MAX_CONSECUTIVE_FAILURES and rerun "
                f"(already-cached cases are left in place)."
            )


def _build_cache_zip(archive: Path, cache_dir: Path, num_points, seed,
                      global_feat_keys, const_keys, fields):
    rng = np.random.default_rng(seed)
    manifest = []
    tracker = _ConsecutiveFailureTracker()
    with zipfile.ZipFile(archive) as zf:
        cases = _group_cases(zf.namelist())
        print(f"Found {len(cases)} candidate cases in {archive.name}")
        for i, case in enumerate(cases):
            with tempfile.TemporaryDirectory(prefix="blendednet_case_") as tmp:
                tmp = Path(tmp)
                mesh_path = tmp / Path(case["mesh"]).name
                meta_path = None
                try:
                    with zf.open(case["mesh"]) as src, open(mesh_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if case["meta"]:
                        meta_path = tmp / Path(case["meta"]).name
                        with zf.open(case["meta"]) as src, open(meta_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    field_names, gkeys, ckeys = _process_one_case(
                        case["case_id"], mesh_path, meta_path, global_feat_keys, const_keys,
                        fields, num_points, rng, cache_dir)
                except Exception as e:
                    tracker.failed(case["case_id"], e)
                    continue
            # tmp dir (raw mesh/meta for this case) is deleted on context exit here
            tracker.ok()
            manifest.append(case["case_id"])
            global_feat_keys, const_keys = gkeys, ckeys
            if i % 50 == 0:
                print(f"  [{i}/{len(cases)}] {case['case_id']}: fields={field_names} "
                      f"condition={gkeys} constants={ckeys}")
    print(f"{tracker.total_failed} case(s) skipped as corrupt/unparseable")
    return manifest, global_feat_keys, const_keys


def _build_cache_targz(archive: Path, cache_dir: Path, num_points, seed,
                        global_feat_keys, const_keys, fields):
    """Single sequential pass over the tar.gz (gzip streams can't be seeked
    to an arbitrary member cheaply). Cases are assumed to be stored as
    contiguous per-case directories, so a case is finalized -- processed and
    its raw files deleted -- as soon as the stream moves to the next
    directory."""
    rng = np.random.default_rng(seed)
    manifest = []
    tracker = _ConsecutiveFailureTracker()
    tmp_root = Path(tempfile.mkdtemp(prefix="blendednet_targz_"))
    pending = {}
    i = 0

    def flush(case_dir):
        nonlocal global_feat_keys, const_keys, i
        info = pending.pop(case_dir)
        raw_dir = tmp_root / case_dir
        if info["mesh"] is not None:
            case_id = info["mesh"].stem
            try:
                field_names, gkeys, ckeys = _process_one_case(
                    case_id, info["mesh"], info["meta"], global_feat_keys, const_keys,
                    fields, num_points, rng, cache_dir)
                manifest.append(case_id)
                global_feat_keys, const_keys = gkeys, ckeys
                tracker.ok()
                if i % 50 == 0:
                    print(f"  [{i}] {case_id}: fields={field_names} "
                          f"condition={gkeys} constants={ckeys}")
                i += 1
            except Exception as e:
                tracker.failed(case_id, e)
        shutil.rmtree(raw_dir, ignore_errors=True)

    try:
        with tarfile.open(archive, mode="r|gz") as tf:
            last_dir = None
            for member in tf:
                if not member.isfile():
                    continue
                d = _case_dir(member.name)
                if last_dir is not None and d != last_dir and last_dir in pending:
                    flush(last_dir)
                if _is_mesh(member.name) or _is_meta(member.name):
                    out_path = tmp_root / member.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    slot = "mesh" if _is_mesh(member.name) else "meta"
                    pending.setdefault(d, {"mesh": None, "meta": None})[slot] = out_path
                last_dir = d
            if last_dir in pending:
                flush(last_dir)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"{tracker.total_failed} case(s) skipped as corrupt/unparseable")
    return manifest, global_feat_keys, const_keys


def build_cache(raw_dir: Path, cache_dir: Path, num_points: int, seed: int,
                 global_feat_keys: list[str] | None, const_keys: list[str] | None,
                 fields: list[str] | None, dataset: str = "blendednet",
                 delete_archive_when_done: bool = True):
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = find_archive(raw_dir, dataset)
    build_fn = _build_cache_zip if dataset == "blendednet" else _build_cache_targz
    manifest, global_feat_keys, const_keys = build_fn(
        archive, cache_dir, num_points, seed, global_feat_keys, const_keys, fields)

    (cache_dir / "manifest.json").write_text(json.dumps({
        "case_ids": manifest,
        "num_points": num_points,
        "global_feat_keys": global_feat_keys,
        "const_keys": const_keys,
    }, indent=2))
    print(f"Cached {len(manifest)} cases -> {cache_dir}")

    if delete_archive_when_done:
        archive.unlink()
        print(f"Deleted raw archive {archive} (all cases cached)")


def inspect(raw_dir: Path, dataset: str = "blendednet", n: int = 3):
    cases = find_cases(raw_dir, dataset)
    archive = find_archive(raw_dir, dataset)
    print(f"Found {len(cases)} mesh files in {archive.name}")
    import pyvista as pv
    opener = zipfile.ZipFile(archive) if dataset == "blendednet" else tarfile.open(archive, "r:gz")
    with opener as ar, tempfile.TemporaryDirectory(prefix="blendednet_inspect_") as tmp:
        tmp = Path(tmp)
        for case in cases[:n]:
            mesh_name = Path(case["mesh"]).name
            mesh_path = tmp / mesh_name
            if dataset == "blendednet":
                with ar.open(case["mesh"]) as src, open(mesh_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                with ar.extractfile(case["mesh"]) as src, open(mesh_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            meta = {}
            if case["meta"]:
                meta_path = tmp / Path(case["meta"]).name
                if dataset == "blendednet":
                    with ar.open(case["meta"]) as src, open(meta_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    with ar.extractfile(case["meta"]) as src, open(meta_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                meta = load_metadata(meta_path)
            mesh = pv.read(str(mesh_path))
            print(f"\ncase_id={case['case_id']}")
            print(f"  mesh:   {case['mesh']}  ({mesh.n_points} points)")
            print(f"  fields: {list(mesh.point_data.keys())}")
            print(f"  meta:   {case['meta']} -> {meta}")
            mesh_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class BlendedNetDataset(Dataset):
    """Loads cached .npz point clouds, normalizes with stats computed once
    over `case_ids` (pass the training split's ids when constructing the
    training set; reuse the same returned `stats` for val/test)."""

    def __init__(self, cache_dir: str | Path, case_ids: list[str] | None = None,
                 stats: dict | None = None):
        self.cache_dir = Path(cache_dir)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.case_ids = case_ids or manifest["case_ids"]
        self.stats = stats or self._compute_stats()

    def _compute_stats(self):
        conditions, targets, constants = [], [], []
        for cid in self.case_ids:
            d = np.load(self.cache_dir / f"{cid}.npz")
            conditions.append(d["condition"])
            targets.append(d["target"])
            constants.append(d["constants"])
        conditions = np.stack(conditions, axis=0)
        targets = np.concatenate(targets, axis=0)
        constants = np.stack(constants, axis=0)
        return {
            "cond_mean": conditions.mean(0), "cond_std": conditions.std(0) + 1e-8,
            "target_mean": targets.mean(0), "target_std": targets.std(0) + 1e-8,
            "const_mean": constants.mean(0), "const_std": constants.std(0) + 1e-8,
        }

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        d = np.load(self.cache_dir / f"{self.case_ids[idx]}.npz")
        pos = d["pos"].astype(np.float32)
        pos = pos - pos.mean(0, keepdims=True)  # center per-case, shape only
        scale = np.linalg.norm(pos, axis=-1).max() + 1e-8
        pos = pos / scale

        geometry = d["geometry"].astype(np.float32)  # unit-norm surface normals, left raw
        condition = (d["condition"] - self.stats["cond_mean"]) / self.stats["cond_std"]
        target = (d["target"] - self.stats["target_mean"]) / self.stats["target_std"]
        constants = (d["constants"] - self.stats["const_mean"]) / self.stats["const_std"]
        return (
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(geometry),
            torch.from_numpy(condition.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(constants.astype(np.float32)),
        )


# ==========================================================================
# Params -- edit these directly, no CLI flags. raw_dir/cache_dir come from
# config.py (edit that file to switch between local and Kaggle).
# ==========================================================================
DATASET = "blendednet"     # "blendednet" or "blendednet++" -- must match download_blendednet.py
NUM_POINTS = 4096
SEED = 0
FIELDS = None              # explicit VTK point_data array names to use as targets, or None
GLOBAL_FEAT_KEYS = None    # explicit metadata keys to use as the condition vector, or None
CONST_KEYS = None          # explicit metadata keys to use as constants targets, or None (auto: C1, C2, ...)
INSPECT = False            # True -> sanity-check discovery heuristics, no processing
DEMO = False                # True -> skip building the cache, one forward pass through GeoTransolver instead
DELETE_ARCHIVE_WHEN_DONE = True  # True -> delete the raw archive once every case is cached


def main():
    if INSPECT:
        inspect(BLENDEDNET_RAW_DIR, DATASET)
        return

    if DEMO:
        from geotransolver import GeoTransolver
        ds = BlendedNetDataset(BLENDEDNET_CACHE_DIR)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        pos, geometry, condition, target, constants = next(iter(loader))
        model = GeoTransolver(
            space_dim=pos.shape[-1], geom_dim=geometry.shape[-1], cond_dim=condition.shape[-1],
            out_channels=target.shape[-1], num_constants=constants.shape[-1],
        )
        field_pred, const_pred = model(pos, geometry, condition)
        loss = (torch.nn.functional.mse_loss(field_pred, target)
                + torch.nn.functional.mse_loss(const_pred, constants))
        print(f"pos {pos.shape}, geometry {geometry.shape}, condition {condition.shape}, "
              f"target {target.shape}, constants {constants.shape}")
        print(f"field_pred {field_pred.shape}, const_pred {const_pred.shape}, mse loss {loss.item():.4f}")
        return

    build_cache(BLENDEDNET_RAW_DIR, BLENDEDNET_CACHE_DIR, NUM_POINTS, SEED,
                GLOBAL_FEAT_KEYS, CONST_KEYS, FIELDS, DATASET, DELETE_ARCHIVE_WHEN_DONE)


if __name__ == "__main__":
    main()
