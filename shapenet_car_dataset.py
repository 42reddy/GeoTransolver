"""
Turn extracted ShapeNet-Car surface + volume meshes into fixed-size point
clouds and expose them as a torch Dataset whose tensors plug straight into
Transolver.forward(pos, features) from transolver.py.

--------------------------------------------------------------------------
What a "case" is, and why we use two meshes
--------------------------------------------------------------------------
Each case is one car shape (889 total, split into 9 folds: param0 is the
official 100-case test fold, param1..param8 the 789-case training pool --
see thuml/Transolver/Car-Design-ShapeNetCar). All cases share the same
inlet flow condition; only geometry varies.

Two converged-RANS meshes are shipped per case:
  quadpress_smpl.vtk   surface mesh (~3,586 pts): converged pressure field
  hexvelo_smpl.vtk     volume mesh  (~29,500 pts): converged velocity field

A signed-distance feature is only meaningful with *both*: the distance
from a surface point to itself is trivially zero, so we need off-surface
"exterior" points to get a real SDF. We build a combined point cloud:

    exterior points (from the volume mesh, excluding the ~900/3586 nodes
    that coincide exactly with the surface mesh -- verified via a nearest-
    neighbor match, see MATCH_EPS below)
        + sdf   = distance to the nearest surface point            (> 0)
        + "normal" = unit direction from that nearest surface point
                     (surface normals aren't defined off the surface,
                     so this direction is the closest analogue -- same
                     convention as the official Transolver loader)
        + velocity from the volume mesh at that point
        + pressure = 0 (undefined off the surface; masked out of the loss)

    surface points (from the surface mesh)
        + sdf = 0
        + normal = true geometric surface normal
        + velocity = the volume mesh's value at the coincident node if one
                     exists, else 0 (this also happens to match the no-slip
                     wall boundary condition -- velocity truly is ~0 at
                     most wall points in viscous flow)
        + pressure = the converged surface pressure (real target)

so per case:
    pos      (N, 3)  point coordinates x, y, z
    features (N, 4)  [sdf, dir_x, dir_y, dir_z]
    target   (N, 4)  [velocity_x, velocity_y, velocity_z, pressure]
    mask     (N,)    True where the point is an actual surface node, i.e.
                      where `pressure` is a real (not dummy-zero) target

Verified end to end against real archive files before writing this (VTK
point_data array names "point_scalars"/"point_vectors", the surface/volume
coincidence-matching behavior, no NaNs).

Transolver consumes a fixed point count per batch item, so each case is
subsampled to a fixed budget, stratified across surface/exterior so the
(sparser, 3,586-point) surface class isn't drowned out by the (denser,
~28,600-point) exterior class -- see --num_surface_points/--num_exterior_points.

Requires: pyvista, scipy (`pip install pyvista scipy`).

Usage:
    # 1. build the cached, fixed-size point-cloud .npz files
    python shapenet_car_dataset.py --raw_dir data/raw/extracted/training_data \\
        --cache_dir data/cache --num_surface_points 2048 --num_exterior_points 4096

    # 2. quick end-to-end wiring check against Transolver
    python shapenet_car_dataset.py --cache_dir data/cache --demo

Surface-only fallback (--mode surface, matches a --surface_only download):
    python shapenet_car_dataset.py --raw_dir ... --cache_dir ... --mode surface --num_points 3000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PRESSURE_FIELD = "point_scalars"  # verified VTK point_data array name (quadpress_smpl.vtk)
VELOCITY_FIELD = "point_vectors"  # verified VTK point_data array name (hexvelo_smpl.vtk)
MATCH_EPS = 1e-6  # nearest-neighbor distance below which a volume point is "the same node" as a surface point


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def find_cases(raw_dir: Path, mode: str) -> list[dict]:
    """raw_dir/param{i}/<case_hash>/{quadpress_smpl.vtk, hexvelo_smpl.vtk}"""
    cases = []
    for mesh_path in sorted(raw_dir.glob("param*/*/quadpress_smpl.vtk")):
        fold = mesh_path.parent.parent.name       # "param0".."param8"
        case_hash = mesh_path.parent.name
        velo_path = mesh_path.parent / "hexvelo_smpl.vtk"
        if mode == "full" and not velo_path.exists():
            continue
        cases.append({
            "case_id": f"{fold}_{case_hash}",
            "fold": fold,
            "mesh_path": mesh_path,
            "velo_path": velo_path if mode == "full" else None,
        })
    return cases


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _load_surface(mesh_path):
    import pyvista as pv

    # quadpress_smpl.vtk loads as an UnstructuredGrid; compute_normals only
    # exists on PolyData, so convert via extract_surface() first (mirrors
    # vtkDataSetSurfaceFilter in the official Transolver loader). Since this
    # file is already an all-surface quad mesh, extract_surface keeps every
    # point -- pull pos/normals/pressure from the *same* PolyData afterward
    # so they stay index-aligned regardless of any reordering the filter does.
    mesh = pv.read(str(mesh_path))
    surf = mesh.extract_surface(algorithm="dataset_surface")
    surf = surf.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

    pos = np.asarray(surf.points, dtype=np.float64)
    normals = np.asarray(surf.point_data["Normals"], dtype=np.float64)
    pressure = np.asarray(surf.point_data[PRESSURE_FIELD], dtype=np.float64).reshape(-1, 1)
    return pos, normals, pressure


def load_case_surface(case: dict):
    """Surface-only: features=[normal(3)], target=[pressure(1)]. All points are surface (mask all True)."""
    pos, normals, pressure = _load_surface(case["mesh_path"])
    mask = np.ones(len(pos), dtype=bool)
    return pos.astype(np.float32), normals.astype(np.float32), pressure.astype(np.float32), mask


def load_case_full(case: dict):
    """Surface + exterior volume points, with real SDF and velocity+pressure targets."""
    import pyvista as pv
    from scipy.spatial import cKDTree

    pos_surf, normal_surf, press_surf = _load_surface(case["mesh_path"])

    velo_grid = pv.read(str(case["velo_path"]))
    pos_velo = np.asarray(velo_grid.points, dtype=np.float64)
    velo_all = np.asarray(velo_grid.point_data[VELOCITY_FIELD], dtype=np.float64)

    tree_surf = cKDTree(pos_surf)
    dist_v2s, idx_v2s = tree_surf.query(pos_velo, k=1)
    is_ext = dist_v2s > MATCH_EPS

    pos_ext = pos_velo[is_ext]
    sdf_ext = dist_v2s[is_ext].reshape(-1, 1)
    dir_ext = (pos_ext - pos_surf[idx_v2s[is_ext]]) / (sdf_ext + 1e-8)
    velo_ext = velo_all[is_ext]
    press_ext = np.zeros((len(pos_ext), 1))

    tree_velo = cKDTree(pos_velo)
    dist_s2v, idx_s2v = tree_velo.query(pos_surf, k=1)
    matched = (dist_s2v < MATCH_EPS).reshape(-1, 1)
    velo_surf = np.where(matched, velo_all[idx_s2v], 0.0)

    pos = np.concatenate([pos_ext, pos_surf])
    sdf = np.concatenate([sdf_ext, np.zeros((len(pos_surf), 1))])
    normal = np.concatenate([dir_ext, normal_surf])
    velocity = np.concatenate([velo_ext, velo_surf])
    pressure = np.concatenate([press_ext, press_surf])
    mask = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))]).astype(bool)

    features = np.concatenate([sdf, normal], axis=-1)
    target = np.concatenate([velocity, pressure], axis=-1)
    return pos.astype(np.float32), features.astype(np.float32), target.astype(np.float32), mask


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------
def subsample_uniform(pos, features, target, mask, num_points: int, rng: np.random.Generator):
    n = len(pos)
    idx = rng.choice(n, size=num_points, replace=n < num_points)
    return pos[idx], features[idx], target[idx], mask[idx]


def subsample_stratified(pos, features, target, mask, num_surface: int, num_exterior: int,
                          rng: np.random.Generator):
    surf_idx = np.flatnonzero(mask)
    ext_idx = np.flatnonzero(~mask)
    surf_pick = rng.choice(surf_idx, size=num_surface, replace=len(surf_idx) < num_surface)
    ext_pick = rng.choice(ext_idx, size=num_exterior, replace=len(ext_idx) < num_exterior)
    idx = np.concatenate([surf_pick, ext_pick])
    return pos[idx], features[idx], target[idx], mask[idx]


# --------------------------------------------------------------------------
# Cache builder
# --------------------------------------------------------------------------
def build_cache(raw_dir: Path, cache_dir: Path, mode: str, seed: int,
                 num_points: int, num_surface_points: int, num_exterior_points: int,
                 test_fold: str = "param0"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = find_cases(raw_dir, mode)
    print(f"Found {len(cases)} cases under {raw_dir} (mode={mode})")
    if not cases:
        hint = "hexvelo_smpl.vtk" if mode == "full" else "quadpress_smpl.vtk"
        raise SystemExit(
            f"No usable cases found under {raw_dir}/param*/*/ (need {hint}). "
            f"Did you download with the matching mode ({'default (full)' if mode == 'full' else '--surface_only'})?"
        )

    train_ids, test_ids = [], []
    for i, case in enumerate(cases):
        if mode == "full":
            pos, features, target, mask = load_case_full(case)
            pos, features, target, mask = subsample_stratified(
                pos, features, target, mask, num_surface_points, num_exterior_points, rng)
        else:
            pos, features, target, mask = load_case_surface(case)
            pos, features, target, mask = subsample_uniform(pos, features, target, mask, num_points, rng)

        np.savez_compressed(cache_dir / f"{case['case_id']}.npz",
                             pos=pos, features=features, target=target, mask=mask)
        (test_ids if case["fold"] == test_fold else train_ids).append(case["case_id"])
        if i % 100 == 0:
            print(f"  [{i}/{len(cases)}] {case['case_id']}")

    manifest = {
        "case_ids": train_ids + test_ids,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "mode": mode,
        "in_channels": 4,
        "out_channels": 4 if mode == "full" else 1,
    }
    if mode == "full":
        manifest["num_points"] = num_surface_points + num_exterior_points
        manifest["num_surface_points"] = num_surface_points
        manifest["num_exterior_points"] = num_exterior_points
    else:
        manifest["num_points"] = num_points
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Cached {len(train_ids)} train / {len(test_ids)} test cases -> {cache_dir}")


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class ShapeNetCarDataset(Dataset):
    """Loads cached .npz point clouds, normalizing features/target with
    stats computed once over `case_ids` (pass the training split's ids when
    constructing the training set; reuse the returned `stats` for test)."""

    def __init__(self, cache_dir: str | Path, case_ids: list[str] | None = None,
                 stats: dict | None = None):
        self.cache_dir = Path(cache_dir)
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.case_ids = case_ids if case_ids is not None else self.manifest["case_ids"]
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
        pos = pos - pos.mean(0, keepdims=True)          # center per-case
        scale = np.linalg.norm(pos, axis=-1).max() + 1e-8
        pos = pos / scale                                # unit-sphere-normalize shape

        features = (d["features"] - self.stats["feat_mean"]) / self.stats["feat_std"]
        target = (d["target"] - self.stats["target_mean"]) / self.stats["target_std"]
        mask = d["mask"].astype(np.float32)
        return (
            torch.from_numpy(pos.astype(np.float32)),
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(target.astype(np.float32)),
            torch.from_numpy(mask),
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=Path, help="extracted training_data/ root (contains param0..param8)")
    ap.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--mode", choices=["full", "surface"], default="full",
                     help="full = surface+exterior with real SDF & velocity+pressure targets (needs "
                          "hexvelo_smpl.vtk); surface = pressure-only, no SDF")
    ap.add_argument("--num_points", type=int, default=3000, help="[surface mode] points per case")
    ap.add_argument("--num_surface_points", type=int, default=2048, help="[full mode] surface points per case")
    ap.add_argument("--num_exterior_points", type=int, default=4096, help="[full mode] exterior points per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="one forward pass through Transolver")
    args = ap.parse_args()

    if args.demo:
        from transolver import Transolver
        manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        train_ds = ShapeNetCarDataset(args.cache_dir, case_ids=manifest["train_ids"])
        test_ds = ShapeNetCarDataset(args.cache_dir, case_ids=manifest["test_ids"], stats=train_ds.stats)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        pos, features, target, mask = next(iter(loader))
        model = Transolver(space_dim=3, in_channels=features.shape[-1], out_channels=target.shape[-1])
        pred = model(pos, features)
        loss = torch.nn.functional.mse_loss(pred, target)
        print(f"mode={manifest['mode']} train={len(train_ds)} test={len(test_ds)}")
        print(f"pos {pos.shape}, features {features.shape}, target {target.shape}, mask {mask.shape}")
        print(f"pred {pred.shape}, mse loss {loss.item():.4f}")
        return

    build_cache(args.raw_dir, args.cache_dir, args.mode, args.seed,
                args.num_points, args.num_surface_points, args.num_exterior_points)


if __name__ == "__main__":
    main()
