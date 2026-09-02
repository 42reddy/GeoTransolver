"""
Turn extracted ShapeNet-Car surface meshes into fixed-size point clouds and
expose them as a torch Dataset whose tensors plug straight into
Transolver.forward(pos, features) from transolver.py.

--------------------------------------------------------------------------
What a "case" is
--------------------------------------------------------------------------
Each case is one car shape (889 total, split into 9 folds: param0 is the
official 100-case test fold, param1..param8 the 789-case training pool --
see thuml/Transolver/Car-Design-ShapeNetCar). All cases share the same
inlet flow condition; only geometry varies. Each case's surface was solved
to a converged RANS solution and downsampled to ~3,682 surface points
storing the converged pressure field -- that's what we predict.

    pos      (N, 3)  surface point coordinates x, y, z
    features (N, 3)  per-point surface normal (nx, ny, nz)
    target   (N, 1)  converged pressure coefficient (VTK point_data
                      array "point_scalars", verified by inspection)

Transolver consumes a fixed point count per batch item, so every case is
randomly subsampled (or resampled with replacement, if fewer points than
--num_points) to the same N.

Requires: pyvista (`pip install pyvista`).

Usage:
    # 1. build the cached, fixed-size point-cloud .npz files
    python shapenet_car_dataset.py --raw_dir data/raw/extracted/training_data \\
        --cache_dir data/cache --num_points 3000

    # 2. quick end-to-end wiring check against Transolver
    python shapenet_car_dataset.py --cache_dir data/cache --demo
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PRESSURE_FIELD = "point_scalars"  # verified VTK point_data array name


# --------------------------------------------------------------------------
# Discovery + loading
# --------------------------------------------------------------------------
def find_cases(raw_dir: Path) -> list[dict]:
    """raw_dir/param{i}/<case_hash>/quadpress_smpl.vtk"""
    cases = []
    for mesh_path in sorted(raw_dir.glob("param*/*/quadpress_smpl.vtk")):
        fold = mesh_path.parent.parent.name       # "param0".."param8"
        case_hash = mesh_path.parent.name
        cases.append({"case_id": f"{fold}_{case_hash}", "fold": fold, "mesh_path": mesh_path})
    return cases


def load_case(case: dict):
    import pyvista as pv

    mesh = pv.read(str(case["mesh_path"]))
    pos = np.asarray(mesh.points, dtype=np.float32)
    target = np.asarray(mesh.point_data[PRESSURE_FIELD], dtype=np.float32).reshape(-1, 1)

    mesh = mesh.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    normals = np.asarray(mesh.point_data["Normals"], dtype=np.float32)
    return pos, normals, target


def subsample(pos, features, target, num_points: int, rng: np.random.Generator):
    n = len(pos)
    idx = rng.choice(n, size=num_points, replace=n < num_points)
    return pos[idx], features[idx], target[idx]


# --------------------------------------------------------------------------
# Cache builder
# --------------------------------------------------------------------------
def build_cache(raw_dir: Path, cache_dir: Path, num_points: int, seed: int,
                 test_fold: str = "param0"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = find_cases(raw_dir)
    print(f"Found {len(cases)} cases under {raw_dir}")
    if not cases:
        raise SystemExit(
            f"No quadpress_smpl.vtk files found under {raw_dir}/param*/*/. "
            f"Did you run download_shapenet_car.py first?"
        )

    train_ids, test_ids = [], []
    for i, case in enumerate(cases):
        pos, normals, target = load_case(case)
        pos, normals, target = subsample(pos, normals, target, num_points, rng)
        np.savez_compressed(cache_dir / f"{case['case_id']}.npz", pos=pos, features=normals, target=target)
        (test_ids if case["fold"] == test_fold else train_ids).append(case["case_id"])
        if i % 100 == 0:
            print(f"  [{i}/{len(cases)}] {case['case_id']}")

    (cache_dir / "manifest.json").write_text(json.dumps({
        "case_ids": train_ids + test_ids,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "num_points": num_points,
    }, indent=2))
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
        manifest = json.loads((self.cache_dir / "manifest.json").read_text())
        self.case_ids = case_ids if case_ids is not None else manifest["case_ids"]
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
    ap.add_argument("--raw_dir", type=Path, help="extracted training_data/ root (contains param0..param8)")
    ap.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--num_points", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="one forward pass through Transolver")
    args = ap.parse_args()

    if args.demo:
        from transolver import Transolver
        manifest = json.loads((args.cache_dir / "manifest.json").read_text())
        train_ds = ShapeNetCarDataset(args.cache_dir, case_ids=manifest["train_ids"])
        test_ds = ShapeNetCarDataset(args.cache_dir, case_ids=manifest["test_ids"], stats=train_ds.stats)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        pos, features, target = next(iter(loader))
        model = Transolver(space_dim=3, in_channels=features.shape[-1], out_channels=target.shape[-1])
        pred = model(pos, features)
        loss = torch.nn.functional.mse_loss(pred, target)
        print(f"train={len(train_ds)} test={len(test_ds)}")
        print(f"pos {pos.shape}, features {features.shape}, target {target.shape}")
        print(f"pred {pred.shape}, mse loss {loss.item():.4f}")
        return

    build_cache(args.raw_dir, args.cache_dir, args.num_points, args.seed)


if __name__ == "__main__":
    main()
