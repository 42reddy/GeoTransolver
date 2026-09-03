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
# Discovery: find (surface_mesh, metadata) pairs under raw_dir
# --------------------------------------------------------------------------
def find_cases(raw_dir: Path):
    cases = []
    for mesh_path in sorted(raw_dir.rglob("*")):
        if mesh_path.suffix.lower() not in SURFACE_EXTS:
            continue
        meta_path = None
        for sibling in mesh_path.parent.iterdir():
            if sibling.suffix.lower() in META_EXTS:
                meta_path = sibling
                break
        cases.append(
            {
                "case_id": mesh_path.stem,
                "mesh_path": mesh_path,
                "meta_path": meta_path,
            }
        )
    return cases


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
    constants = np.array([const_meta.get(k, 0.0) for k in const_keys], dtype=np.float32)

    return pos, normals, condition, target, constants, field_names, global_feat_keys, const_keys


def subsample(pos, geometry, target, num_points: int, rng: np.random.Generator):
    n = len(pos)
    replace = n < num_points
    idx = rng.choice(n, size=num_points, replace=replace)
    return pos[idx], geometry[idx], target[idx]


# --------------------------------------------------------------------------
# Cache builder
# --------------------------------------------------------------------------
def build_cache(raw_dir: Path, cache_dir: Path, num_points: int, seed: int,
                 global_feat_keys: list[str] | None, const_keys: list[str] | None,
                 fields: list[str] | None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} candidate cases under {raw_dir}")

    manifest = []
    for i, case in enumerate(cases):
        try:
            pos, normals, condition, target, constants, field_names, gkeys, ckeys = load_case(
                case, global_feat_keys, const_keys, fields)
        except Exception as e:
            print(f"  [skip] {case['case_id']}: {e}")
            continue
        pos, normals, target = subsample(pos, normals, target, num_points, rng)
        out_path = cache_dir / f"{case['case_id']}.npz"
        np.savez_compressed(out_path, pos=pos, geometry=normals, condition=condition,
                             target=target, constants=constants)
        manifest.append(case["case_id"])
        global_feat_keys, const_keys = gkeys, ckeys  # lock in on first successful case
        if i % 50 == 0:
            print(f"  [{i}/{len(cases)}] {case['case_id']}: fields={field_names} "
                  f"condition={gkeys} constants={ckeys}")

    (cache_dir / "manifest.json").write_text(json.dumps({
        "case_ids": manifest,
        "num_points": num_points,
        "global_feat_keys": global_feat_keys,
        "const_keys": const_keys,
    }, indent=2))
    print(f"Cached {len(manifest)} cases -> {cache_dir}")


def inspect(raw_dir: Path, n: int = 3):
    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} mesh files under {raw_dir}")
    import pyvista as pv
    for case in cases[:n]:
        mesh = pv.read(str(case["mesh_path"]))
        meta = load_metadata(case["meta_path"])
        print(f"\ncase_id={case['case_id']}")
        print(f"  mesh:   {case['mesh_path']}  ({mesh.n_points} points)")
        print(f"  fields: {list(mesh.point_data.keys())}")
        print(f"  meta:   {case['meta_path']} -> {meta}")


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
NUM_POINTS = 4096
SEED = 0
FIELDS = None              # explicit VTK point_data array names to use as targets, or None
GLOBAL_FEAT_KEYS = None    # explicit metadata keys to use as the condition vector, or None
CONST_KEYS = None          # explicit metadata keys to use as constants targets, or None (auto: C1, C2, ...)
INSPECT = False            # True -> sanity-check discovery heuristics, no processing
DEMO = False                # True -> skip building the cache, one forward pass through GeoTransolver instead


def main():
    if INSPECT:
        inspect(BLENDEDNET_RAW_DIR)
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
                GLOBAL_FEAT_KEYS, CONST_KEYS, FIELDS)


if __name__ == "__main__":
    main()
