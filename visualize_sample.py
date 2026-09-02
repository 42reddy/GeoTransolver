"""
Plot one cached ShapeNet-Car sample in 3D so you can eyeball the point cloud,
input fields (SDF, surface normal), and output fields (velocity, pressure)
before spending time training on them.

Reads straight from the cached .npz (raw, unnormalized units) written by
shapenet_car_dataset.py -- not through ShapeNetCarDataset, so what you see
here is the physical geometry/fields, not the per-case-centered/normalized
tensors the model actually trains on.

Usage:
    python visualize_sample.py --cache_dir data/cache --case_id param1_abcd1234
    python visualize_sample.py --cache_dir data/cache            # random case
    python visualize_sample.py --cache_dir data/cache --save out.png
"""

import argparse
import json
from pathlib import Path

import numpy as np


def plot_sample(pos, features, target, mask, title=""):
    import matplotlib.pyplot as plt

    n_feat = features.shape[-1]
    n_tgt = target.shape[-1]
    sdf = features[:, 0] if n_feat == 4 else None
    normal = features[:, -3:]
    velocity = target[:, :3] if n_tgt == 4 else None
    pressure = target[:, -1] if n_tgt in (1, 4) else None
    surf = mask.astype(bool)

    panels = [("geometry (surface / exterior)", "mask")]
    if sdf is not None:
        panels.append(("input: sdf", "sdf"))
    panels.append(("input: surface normal", "normal"))
    if velocity is not None:
        panels.append(("output: |velocity|", "velocity"))
    panels.append(("output: pressure (surface only)", "pressure"))

    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    fig.suptitle(title)

    def scatter(ax, c, cmap="viridis", sub=None, **kw):
        p = pos if sub is None else pos[sub]
        cc = c if sub is None else c[sub]
        s = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=cc, cmap=cmap, s=3, **kw)
        ax.set_box_aspect(np.ptp(pos, axis=0))
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        return s

    for i, (label, kind) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.set_title(label)

        if kind == "mask":
            s = scatter(ax, surf.astype(int), cmap="coolwarm", vmin=0, vmax=1)
            fig.colorbar(s, ax=ax, shrink=0.6, ticks=[0, 1], label="0=exterior 1=surface")

        elif kind == "sdf":
            s = scatter(ax, sdf, cmap="magma")
            fig.colorbar(s, ax=ax, shrink=0.6, label="distance to surface")

        elif kind == "normal":
            # quiver on a subsample of surface points only -- dense arrows
            # over thousands of points is unreadable.
            surf_idx = np.flatnonzero(surf)
            n_show = min(400, len(surf_idx))
            pick = np.random.default_rng(0).choice(surf_idx, size=n_show, replace=False)
            p, d = pos[pick], normal[pick]
            ax.quiver(p[:, 0], p[:, 1], p[:, 2], d[:, 0], d[:, 1], d[:, 2],
                      length=0.05, normalize=True, color="tab:blue", linewidth=0.7)
            scatter(ax, surf.astype(int), cmap="Greys", vmin=-1, vmax=1, alpha=0.15)

        elif kind == "velocity":
            speed = np.linalg.norm(velocity, axis=-1)
            s = scatter(ax, speed, cmap="plasma")
            fig.colorbar(s, ax=ax, shrink=0.6, label="|velocity|")

        elif kind == "pressure":
            s = scatter(ax, pressure, cmap="coolwarm", sub=surf)
            fig.colorbar(s, ax=ax, shrink=0.6, label="pressure")

    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--case_id", type=str, default=None, help="defaults to a random case from the manifest")
    ap.add_argument("--seed", type=int, default=0, help="used to pick the random case_id when none is given")
    ap.add_argument("--save", type=Path, default=None, help="save the figure here instead of showing it")
    args = ap.parse_args()

    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    case_id = args.case_id or np.random.default_rng(args.seed).choice(manifest["case_ids"])

    d = np.load(args.cache_dir / f"{case_id}.npz")
    pos, features, target, mask = d["pos"], d["features"], d["target"], d["mask"]
    print(f"case {case_id}: {len(pos)} points, features {features.shape}, target {target.shape}, "
          f"{mask.sum()} surface / {(~mask.astype(bool)).sum()} exterior")

    fig = plot_sample(pos, features, target, mask, title=f"{case_id}  (mode={manifest['mode']})")

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
