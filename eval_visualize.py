"""
Run a trained Transolver checkpoint on one random test-set car, then render
ground truth / prediction / error as smooth continuous 3D surfaces (not
point scatter) -- a triangulated reconstruction of the car body, painted
with per-vertex color via PyVista/VTK.

--------------------------------------------------------------------------
How the continuous surface is built
--------------------------------------------------------------------------
The cache only stores a subsampled point cloud, not mesh connectivity, so
there's no ready-made triangulation to color. We rebuild one:
  1. `reconstruct_surface()` fits an implicit function to the *surface*
     points (mask==1) and triangulates its zero level set -> a smooth,
     closed, continuous car-body mesh (this is what makes the plot look
     like a shaded surface instead of a point cloud).
  2. Each scalar field (pressure, |velocity|, and their errors) is known
     only at the original point-cloud locations, so it's transferred onto
     the new mesh vertices with `interpolate()` (inverse-distance weighted
     over nearby points) -- this is what makes the coloring continuous
     across the surface rather than isolated dots.

Pressure is a surface-only quantity in this dataset already (mask==1), so
it reconstructs directly. Velocity is defined everywhere (surface +
exterior) but we render its magnitude on the same car-body surface (i.e.
near-wall velocity) so both fields share one mesh and one figure.

Output: an interactive HTML (rotate/zoom in a browser, no server needed)
via VTK.js, plus a static PNG for a quick look.

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

from config import CACHE_DIR, CKPT_DIR
from shapenet_car_dataset import ShapeNetCarDataset
from transolver import Transolver
from train import DIM, DEPTH, HEADS, DIM_HEAD, NUM_SLICES, MLP_RATIO, DROPOUT

# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
CASE_ID = None              # None -> pick a random test case (seeded by SEED)
SEED = 0
CKPT_NAME = "best.pt"       # falls back to "last.pt" if best.pt isn't there
OUT_HTML = Path("eval_sample.html")
OUT_PNG = Path("eval_sample.png")


# --------------------------------------------------------------------------
# Eval: one forward pass, denormalized back to physical units
# --------------------------------------------------------------------------
@torch.no_grad()
def eval_sample(model, ds, device):
    pos, features, target, mask = ds[0]
    pos_b = pos.unsqueeze(0).to(device)
    features_b = features.unsqueeze(0).to(device)

    model.eval()
    pred_norm = model(pos_b, features_b)[0].cpu().numpy()

    target_mean, target_std = ds.stats["target_mean"], ds.stats["target_std"]
    pred_phys = pred_norm * target_std + target_mean
    return pred_phys, mask.numpy().astype(bool)


# --------------------------------------------------------------------------
# Continuous surface reconstruction + scalar interpolation
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


def add_panel(plotter, mesh, scalar_name, title, cmap, clim, row, col, bar_title=None):
    plotter.subplot(row, col)
    plotter.add_text(title, font_size=10)
    plotter.add_mesh(mesh, scalars=scalar_name, cmap=cmap, clim=clim,
                      smooth_shading=True, show_scalar_bar=bar_title is not None,
                      scalar_bar_args={"title": bar_title or "", "n_labels": 3,
                                        "height": 0.45, "width": 0.09,
                                        "position_x": 0.88, "position_y": 0.05,
                                        "vertical": True, "fmt": "%.1f"})
    plotter.set_background("white")


def plot_continuous(pos_surf, gt, pred, out_html: Path, out_png: Path, title: str):
    import pyvista as pv

    press_gt, press_pred = gt["pressure"], pred["pressure"]
    press_err = np.abs(press_pred - press_gt)
    speed_gt, speed_pred = gt["speed"], pred["speed"]
    speed_err = np.abs(speed_pred - speed_gt)

    mesh = build_colored_surface(pos_surf, {
        "press_gt": press_gt, "press_pred": press_pred, "press_err": press_err,
        "speed_gt": speed_gt, "speed_pred": speed_pred, "speed_err": speed_err,
    })

    p_clim = (float(min(press_gt.min(), press_pred.min())), float(max(press_gt.max(), press_pred.max())))
    s_clim = (float(min(speed_gt.min(), speed_pred.min())), float(max(speed_gt.max(), speed_pred.max())))
    pe_clim = (0.0, float(press_err.max()) + 1e-8)
    se_clim = (0.0, float(speed_err.max()) + 1e-8)

    plotter = pv.Plotter(shape=(2, 3), window_size=(1700, 1000), off_screen=True)
    plotter.link_views()

    add_panel(plotter, mesh, "press_gt", "pressure -- ground truth", "coolwarm", p_clim, 0, 0)
    add_panel(plotter, mesh, "press_pred", "pressure -- prediction", "coolwarm", p_clim, 0, 1, bar_title="pressure")
    add_panel(plotter, mesh, "press_err", "pressure -- |error|", "inferno", pe_clim, 0, 2, bar_title="|error|")

    add_panel(plotter, mesh, "speed_gt", "|velocity| -- ground truth", "viridis", s_clim, 1, 0)
    add_panel(plotter, mesh, "speed_pred", "|velocity| -- prediction", "viridis", s_clim, 1, 1, bar_title="|velocity|")
    add_panel(plotter, mesh, "speed_err", "|velocity| -- |error|", "inferno", se_clim, 1, 2, bar_title="|error|")

    plotter.subplot(0, 0)
    plotter.add_text(title, position="lower_edge", font_size=9)

    plotter.view_isometric()
    plotter.camera.azimuth += 25
    plotter.camera.elevation += 10
    plotter.camera.zoom(1.3)

    plotter.show(screenshot=str(out_png), auto_close=False)
    plotter.export_html(str(out_html))
    plotter.close()


# --------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text())

    ckpt_path = CKPT_DIR / CKPT_NAME
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "last.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint found in {CKPT_DIR} (looked for {CKPT_NAME} and last.pt). "
                          f"Train a model first (train.py).")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    stats_path = CKPT_DIR / "norm_stats.npz"
    stats = dict(np.load(stats_path)) if stats_path.exists() else None

    case_id = CASE_ID or str(np.random.default_rng(SEED).choice(manifest["test_ids"]))
    ds = ShapeNetCarDataset(CACHE_DIR, case_ids=[case_id], stats=stats)
    if stats is None:
        print("warning: no norm_stats.npz found, recomputed stats from this single case "
              "(will not match training normalization exactly)")

    model = Transolver(
        space_dim=3,
        in_channels=manifest["in_channels"],
        out_channels=manifest["out_channels"],
        dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
        num_slices=NUM_SLICES, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    pred_phys, mask = eval_sample(model, ds, device)

    raw = np.load(CACHE_DIR / f"{case_id}.npz")
    pos_raw, target_raw = raw["pos"], raw["target"]

    out_channels = manifest["out_channels"]
    if out_channels == 4:
        vel_gt, press_gt = target_raw[:, :3], target_raw[:, 3]
        vel_pred, press_pred = pred_phys[:, :3], pred_phys[:, 3]
        speed_gt, speed_pred = np.linalg.norm(vel_gt, axis=-1), np.linalg.norm(vel_pred, axis=-1)
    else:
        press_gt, press_pred = target_raw[:, 0], pred_phys[:, 0]
        speed_gt = speed_pred = np.zeros_like(press_gt)  # surface-only mode has no velocity target

    surf = mask if out_channels == 4 else np.ones(len(pos_raw), dtype=bool)
    pos_surf = pos_raw[surf]
    gt = {"pressure": press_gt[surf], "speed": speed_gt[surf]}
    pred = {"pressure": press_pred[surf], "speed": speed_pred[surf]}

    press_mse = float(np.mean((press_pred[surf] - press_gt[surf]) ** 2))
    print(f"case {case_id}: {surf.sum()} surface points | pressure MSE (physical units) = {press_mse:.6f}")

    title = f"{case_id}  (epoch {ckpt.get('epoch', '?')}, pressure MSE={press_mse:.4f})"
    plot_continuous(pos_surf, gt, pred, OUT_HTML, OUT_PNG, title)
    print(f"saved interactive plot -> {OUT_HTML}")
    print(f"saved screenshot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
