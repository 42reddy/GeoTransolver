"""
Evaluate a trained Transolver checkpoint on the held-out test set using the
same metric the Transolver paper reports for this exact benchmark (relative
L2 error per sample, averaged over the test set) -- so the number printed
here is directly comparable to published results. Also renders one random
test car as a simple, reliable static 3D plot (continuous shaded surface,
not point scatter) for a qualitative sanity check.

--------------------------------------------------------------------------
Metric: relative L2 error
--------------------------------------------------------------------------
For each test case, flatten prediction/target into one vector and compute
    ||pred - target||_2 / ||target||_2
then average that ratio over all test cases. This is the standard metric
in the neural-PDE-solver literature (FNO, Transolver, etc.) and is what
thuml/Transolver reports for this Car-Design-ShapeNetCar task: ~0.075
relative L2 on surface pressure. Pressure is only a real target at surface
points (mask==1), so it's computed there only; velocity is a real target
everywhere, so it's computed over all points.

--------------------------------------------------------------------------
Visualization
--------------------------------------------------------------------------
The cache is a point cloud with no mesh connectivity, so there's nothing
to shade continuously out of the box. We rebuild a triangulated car-body
surface once with PyVista's `reconstruct_surface()` (a pure CPU geometry
filter -- fits an implicit function to the surface points and triangulates
its zero level set) and `interpolate()` the scalar fields onto it. That
part never touches a renderer, so it can't crash from a missing display.

The actual drawing is done with plain matplotlib (`Poly3DCollection`),
not VTK/pyvista's own plotting -- matplotlib's Agg backend works
headlessly everywhere (Kaggle, Colab, CI) with no virtual framebuffer or
GPU context required, which is what makes this "simple and reliable"
compared to a full VTK render pipeline.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No CLI flags -- edit the PARAMS block below, then run this file (IDE run
button or `python eval_visualize.py`). Paths come from config.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from config import CACHE_DIR, CKPT_DIR
from shapenet_car_dataset import ShapeNetCarDataset
from transolver import Transolver
from train import DIM, DEPTH, HEADS, DIM_HEAD, NUM_SLICES, MLP_RATIO, DROPOUT

# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
CASE_ID = None              # None -> pick a random test case for the plot (seeded by SEED)
SEED = 0
CKPT_NAME = "best.pt"       # falls back to "last.pt" if best.pt isn't there
OUT_PNG = Path("eval_sample.png")

TRANSOLVER_PAPER_PRESSURE_REL_L2 = 0.075  # thuml/Transolver's reported number on this benchmark


# --------------------------------------------------------------------------
# Metric
# --------------------------------------------------------------------------
def relative_l2(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """||pred - target||_2 / ||target||_2, flattened over all points/channels given."""
    pred, target = pred.reshape(-1), target.reshape(-1)
    return float(np.linalg.norm(pred - target) / (np.linalg.norm(target) + eps))


def split_fields(target: np.ndarray, pred: np.ndarray, out_channels: int) -> dict:
    if out_channels == 4:
        vel_gt, press_gt = target[:, :3], target[:, 3]
        vel_pred, press_pred = pred[:, :3], pred[:, 3]
        return {
            "press_gt": press_gt, "press_pred": press_pred,
            "vel_gt": vel_gt, "vel_pred": vel_pred,
            "speed_gt": np.linalg.norm(vel_gt, axis=-1), "speed_pred": np.linalg.norm(vel_pred, axis=-1),
        }
    return {"press_gt": target[:, 0], "press_pred": pred[:, 0]}


@torch.no_grad()
def predict_case(model, ds: ShapeNetCarDataset, idx: int, device):
    pos, features, target, mask = ds[idx]
    pred_norm = model(pos.unsqueeze(0).to(device), features.unsqueeze(0).to(device))[0].cpu().numpy()
    pred_phys = pred_norm * ds.stats["target_std"] + ds.stats["target_mean"]
    return pred_phys


@torch.no_grad()
def evaluate_test_set(model, ds: ShapeNetCarDataset, device, out_channels: int):
    model.eval()
    press_rel, vel_rel, speed_rel = [], [], []
    for i, case_id in enumerate(tqdm(ds.case_ids, desc="eval test set")):
        pred_phys = predict_case(model, ds, i, device)
        raw = np.load(ds.cache_dir / f"{case_id}.npz")
        surf = raw["mask"].astype(bool) if out_channels == 4 else np.ones(len(raw["target"]), dtype=bool)
        fields = split_fields(raw["target"], pred_phys, out_channels)

        press_rel.append(relative_l2(fields["press_pred"][surf], fields["press_gt"][surf]))
        if out_channels == 4:
            vel_rel.append(relative_l2(fields["vel_pred"], fields["vel_gt"]))
            speed_rel.append(relative_l2(fields["speed_pred"], fields["speed_gt"]))
    return press_rel, vel_rel, speed_rel


# --------------------------------------------------------------------------
# Continuous surface reconstruction (pure CPU geometry, no renderer)
# --------------------------------------------------------------------------
def build_colored_surface(pos_surf: np.ndarray, scalars: dict[str, np.ndarray]):
    import pyvista as pv

    cloud = pv.PolyData(pos_surf)
    for name, vals in scalars.items():
        cloud[name] = vals

    mesh = cloud.reconstruct_surface(nbr_sz=30, sample_spacing=None)
    mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

    diag = np.linalg.norm(pos_surf.max(0) - pos_surf.min(0))
    mesh = mesh.interpolate(cloud, radius=0.05 * diag, sharpness=5, strategy="closest_point")
    return mesh


def plot_continuous(pos_surf, gt: dict, pred: dict, out_png: Path, title: str, show_velocity: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    press_gt, press_pred = gt["pressure"], pred["pressure"]
    press_err = np.abs(press_pred - press_gt)
    scalars = {"press_gt": press_gt, "press_pred": press_pred, "press_err": press_err}

    panels = [
        ("press_gt", "pressure -- ground truth", "coolwarm", None, False),
        ("press_pred", "pressure -- prediction", "coolwarm", None, True),
        ("press_err", "pressure -- |error|", "inferno", (0.0, float(press_err.max()) + 1e-8), True),
    ]
    p_clim = (float(min(press_gt.min(), press_pred.min())), float(max(press_gt.max(), press_pred.max())))
    panels[0] = (panels[0][0], panels[0][1], panels[0][2], p_clim, panels[0][4])
    panels[1] = (panels[1][0], panels[1][1], panels[1][2], p_clim, panels[1][4])

    if show_velocity:
        speed_gt, speed_pred = gt["speed"], pred["speed"]
        speed_err = np.abs(speed_pred - speed_gt)
        scalars.update({"speed_gt": speed_gt, "speed_pred": speed_pred, "speed_err": speed_err})
        s_clim = (float(min(speed_gt.min(), speed_pred.min())), float(max(speed_gt.max(), speed_pred.max())))
        panels += [
            ("speed_gt", "|velocity| -- ground truth", "viridis", s_clim, False),
            ("speed_pred", "|velocity| -- prediction", "viridis", s_clim, True),
            ("speed_err", "|velocity| -- |error|", "inferno", (0.0, float(speed_err.max()) + 1e-8), True),
        ]

    mesh = build_colored_surface(pos_surf, scalars)
    verts = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    tri_verts = verts[faces]
    lo, hi = verts.min(0), verts.max(0)

    ncols = 3
    nrows = 2 if show_velocity else 1
    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
    fig.suptitle(title, fontsize=11)

    for i, (scalar_name, panel_title, cmap, clim, show_bar) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        face_vals = mesh.point_data[scalar_name][faces].mean(axis=1)

        poly = Poly3DCollection(tri_verts, linewidths=0)
        poly.set_array(face_vals)
        poly.set_cmap(cmap)
        poly.set_clim(*clim)
        ax.add_collection3d(poly)

        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=15, azim=-60)
        ax.set_axis_off()
        ax.set_title(panel_title, fontsize=10)
        if show_bar:
            fig.colorbar(poly, ax=ax, shrink=0.6, pad=0.02)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text())
    out_channels = manifest["out_channels"]

    ckpt_path = CKPT_DIR / CKPT_NAME
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "last.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found in {CKPT_DIR} (looked for {CKPT_NAME} and last.pt). "
                          f"Train a model first (train.py).")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    stats_path = CKPT_DIR / "norm_stats.npz"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} not found -- it's written by train.py alongside the checkpoint "
                          f"and is needed to denormalize predictions consistently with training.")
    stats = dict(np.load(stats_path))

    model = Transolver(
        space_dim=3,
        in_channels=manifest["in_channels"],
        out_channels=out_channels,
        dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
        num_slices=NUM_SLICES, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ds = ShapeNetCarDataset(CACHE_DIR, case_ids=manifest["test_ids"], stats=stats)

    # ---------------- full test-set metric (comparable to the paper) ----------------
    press_rel, vel_rel, speed_rel = evaluate_test_set(model, ds, device, out_channels)
    press_rel = np.array(press_rel)
    print(f"\ncheckpoint epoch {ckpt.get('epoch', '?')} | {len(ds)} test cases")
    print(f"  pressure relative L2:  mean={press_rel.mean():.4f}  std={press_rel.std():.4f}  "
          f"(thuml/Transolver reports ~{TRANSOLVER_PAPER_PRESSURE_REL_L2:.3f} on this benchmark)")
    if out_channels == 4:
        vel_rel, speed_rel = np.array(vel_rel), np.array(speed_rel)
        print(f"  velocity relative L2:   mean={vel_rel.mean():.4f}  std={vel_rel.std():.4f}")
        print(f"  |velocity| relative L2: mean={speed_rel.mean():.4f}  std={speed_rel.std():.4f}")

    # ---------------- qualitative plot on one case ----------------
    case_id = CASE_ID or str(np.random.default_rng(SEED).choice(manifest["test_ids"]))
    idx = ds.case_ids.index(case_id)
    pred_phys = predict_case(model, ds, idx, device)

    raw = np.load(CACHE_DIR / f"{case_id}.npz")
    pos_raw = raw["pos"]
    surf = raw["mask"].astype(bool) if out_channels == 4 else np.ones(len(pos_raw), dtype=bool)
    fields = split_fields(raw["target"], pred_phys, out_channels)
    case_press_rel = relative_l2(fields["press_pred"][surf], fields["press_gt"][surf])

    gt = {"pressure": fields["press_gt"][surf]}
    pred = {"pressure": fields["press_pred"][surf]}
    if out_channels == 4:
        gt["speed"], pred["speed"] = fields["speed_gt"][surf], fields["speed_pred"][surf]

    title = f"{case_id}  (epoch {ckpt.get('epoch', '?')}, pressure rel. L2={case_press_rel:.4f})"
    plot_continuous(pos_raw[surf], gt, pred, OUT_PNG, title, show_velocity=out_channels == 4)
    print(f"\nsaved qualitative plot for case {case_id} -> {OUT_PNG}")


if __name__ == "__main__":
    main()
