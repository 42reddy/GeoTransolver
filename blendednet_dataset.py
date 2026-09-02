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
    pos      (N, 3)  surface point coordinates x, y, z
    features (N, C)  per-point surface normal (nx, ny, nz), concatenated
                      with the case's geometry parameters and flight
                      conditions broadcast to every point (so the model
                      knows *which* geometry and *which* flow regime it is
                      looking at -- the coordinates alone don't encode that)
    target   (N, F)  the converged Cp / Cfx / Cfy / Cfz fields, whichever
                      point-data arrays the VTK file actually contains
                      (auto-detected, see discover_fields())

Transolver consumes a fixed point count per batch item ((B, N, ...)
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

Usage:
    # 1. sanity-check what the discovery heuristics find, no processing
    python blendednet_dataset.py --raw_dir data/raw/extracted --inspect

    # 2. build the cached, fixed-size point-cloud .npz files
    python blendednet_dataset.py --raw_dir data/raw/extracted \\
        --cache_dir data/cache --num_points 4096

    # 3. quick end-to-end wiring check against Transolver
    python blendednet_dataset.py --cache_dir data/cache --demo
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SURFACE_EXTS = (".vtk", ".vtu", ".vtp")
META_EXTS = (".json", ".dat", ".csv", ".yaml", ".yml")

PRESSURE_TOKENS = ("cp", "pressure")
FRICTION_TOKENS = ("cf", "friction", "shear")


# --------------------------------------------------------------------------
# Discovery: find (surface_mesh, metadata) pairs under raw_dir
# --------------------------------------------------------------------------
def find_cases(raw_dir: Path) -> list[dict]:
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


def discover_fields(mesh, tokens, requested: list[str] | None = None) -> list[str]:
    names = list(mesh.point_data.keys())
    if requested:
        missing = [r for r in requested if r not in names]
        if missing:
            raise ValueError(f"Requested fields {missing} not in {names}")
        return requested
    return [n for n in names if any(t in n.lower() for t in tokens)]


def load_metadata(meta_path: Path | None) -> dict:
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


def load_case(case: dict, global_feat_keys: list[str] | None, fields: list[str] | None):
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
    if global_feat_keys is None:
        global_feat_keys = sorted(meta.keys())
    global_feats = np.array([meta.get(k, 0.0) for k in global_feat_keys], dtype=np.float32)
    global_feats = np.broadcast_to(global_feats, (len(pos), len(global_feat_keys)))

    features = np.concatenate([normals, global_feats], axis=-1)
    return pos, features, target, field_names, global_feat_keys


def subsample(pos, features, target, num_points: int, rng: np.random.Generator):
    n = len(pos)
    replace = n < num_points
    idx = rng.choice(n, size=num_points, replace=replace)
    return pos[idx], features[idx], target[idx]


# --------------------------------------------------------------------------
# Cache builder
# --------------------------------------------------------------------------
def build_cache(raw_dir: Path, cache_dir: Path, num_points: int, seed: int,
                 global_feat_keys: list[str] | None, fields: list[str] | None):
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} candidate cases under {raw_dir}")

    manifest = []
    for i, case in enumerate(cases):
        try:
            pos, features, target, field_names, gkeys = load_case(case, global_feat_keys, fields)
        except Exception as e:
            print(f"  [skip] {case['case_id']}: {e}")
            continue
        pos, features, target = subsample(pos, features, target, num_points, rng)
        out_path = cache_dir / f"{case['case_id']}.npz"
        np.savez_compressed(out_path, pos=pos, features=features, target=target)
        manifest.append(case["case_id"])
        global_feat_keys = gkeys  # lock in on first successful case
        if i % 50 == 0:
            print(f"  [{i}/{len(cases)}] {case['case_id']}: fields={field_names} global={gkeys}")

    (cache_dir / "manifest.json").write_text(json.dumps({
        "case_ids": manifest,
        "num_points": num_points,
        "global_feat_keys": global_feat_keys,
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
        feats, targets = [], []
        for cid in self.case_ids:
            d = np.load(self.cache_dir / f"{cid}.npz")
            feats.append(d["features"])
            targets.append(d["target"])
        feats = np.concatenate(feats, axis=0)
        targets = np.concatenate(targets, axis=0)
        return {
            "feat_mean": feats.mean(0), "feat_std": feats.std(0) + 1e-8,
            "target_mean": targets.mean(0), "target_std": targets.std(0) + 1e-8,
        }

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        d = np.load(self.cache_dir / f"{self.case_ids[idx]}.npz")
        pos = d["pos"].astype(np.float32)
        pos = pos - pos.mean(0, keepdims=True)  # center per-case, shape only
        scale = np.linalg.norm(pos, axis=-1).max() + 1e-8
        pos = pos / scale

        features = (d["features"] - self.stats["feat_mean"]) / self.stats["feat_std"]
        target = (d["target"] - self.stats["target_mean"]) / self.stats["target_std"]
        return (
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=Path, help="extracted BlendedNet root")
    ap.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--num_points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fields", nargs="*", default=None,
                     help="explicit VTK point_data array names to use as targets")
    ap.add_argument("--global_feat_keys", nargs="*", default=None,
                     help="explicit metadata keys to broadcast as conditioning features")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--demo", action="store_true", help="one forward pass through Transolver")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.raw_dir)
        return

    if args.demo:
        from transolver import Transolver
        ds = BlendedNetDataset(args.cache_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        pos, features, target = next(iter(loader))
        model = Transolver(space_dim=3, in_channels=features.shape[-1], out_channels=target.shape[-1])
        pred = model(pos, features)
        loss = torch.nn.functional.mse_loss(pred, target)
        print(f"pos {pos.shape}, features {features.shape}, target {target.shape}")
        print(f"pred {pred.shape}, mse loss {loss.item():.4f}")
        return

    build_cache(args.raw_dir, args.cache_dir, args.num_points, args.seed,
                args.global_feat_keys, args.fields)


if __name__ == "__main__":
    main()
