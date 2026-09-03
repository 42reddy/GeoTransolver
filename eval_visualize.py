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
Metric: drag / lift coefficients (Cd, Cl)
--------------------------------------------------------------------------
thuml/Transolver's own evaluation protocol for this task also reports
relative error and Spearman rank correlation of the drag coefficient
(a car-design use case cares about *ranking* shapes by drag, not just
per-point field error), so we compute the same thing here from first
principles, using only what's actually in the cache -- no external
constants:

  freestream velocity (direction + speed): estimated per-case as the mean
  velocity at the exterior points farthest from the surface (highest sdf)
  -- i.e. read directly off the ground-truth flow field instead of
  assumed. Sanity check against the dataset's known 72 km/h (~20 m/s)
  driving condition (Umetani & Bickel 2018) confirms this: ~20.7-20.8 m/s.

  vertical axis: found from the two directions orthogonal to the flow
  (PCA of surface points projected onto that plane) by taking the one
  with *smaller* extent -- cars are always shorter than they are wide.

  reference area: the frontal silhouette area, i.e. the convex-hull area
  of the surface points projected onto the plane perpendicular to flow.

  force: pressure-only surface integral, F = -sum(p * n_outward * area)
  over a triangulated reconstruction of the surface. This is *form drag
  only* -- the cache has no wall-shear-stress field, so viscous friction
  drag isn't included. That's a real limitation, not a rounding error;
  Cd/Cl here are a proxy, not the literal total force.

  units: the cached "pressure" is kinematic pressure p/rho (m^2/s^2),
  not Pa -- confirmed empirically, its max (~207) sits right at the
  theoretical stagnation value 0.5*v_inf^2 (~216) for v_inf~20.8 m/s.
  Because of that, rho cancels analytically in Cd = F/(0.5*rho*v_inf^2*A):
  no density value needs to be assumed at all.

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
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from config import CACHE_DIR, CKPT_DIR
from shapenet_car_dataset import ShapeNetCarDataset
from transolver import Transolver
from train import DIM, DEPTH, HEADS, DIM_HEAD, NUM_SLICES, MLP_RATIO, DROPOUT

# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
VIS_CASE_IDS = None         # None -> pick NUM_VIS_SAMPLES random test cases (seeded by SEED)
NUM_VIS_SAMPLES = 10
SEED = 0
CKPT_NAME = "best.pt"       # falls back to "last.pt" if best.pt isn't there
OUT_DIR = Path("eval_samples")

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


def timed_predict_case(model, ds: ShapeNetCarDataset, idx: int, device):
    """Same as predict_case, but returns (pred_phys, inference_seconds) for
    just the forward pass -- torch.cuda.synchronize() brackets it so
    timing isn't just measuring how fast the GPU queues the kernel."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    pred_phys = predict_case(model, ds, idx, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred_phys, time.perf_counter() - t0


@torch.no_grad()
def evaluate_test_set(model, ds: ShapeNetCarDataset, device, out_channels: int):
    model.eval()
    metrics = {"press_rel": [], "vel_rel": [], "speed_rel": [], "Cd_gt": [], "Cd_pred": [], "Cl_gt": [], "Cl_pred": []}
    for i, case_id in enumerate(tqdm(ds.case_ids, desc="eval test set")):
        pred_phys = predict_case(model, ds, i, device)
        raw = np.load(ds.cache_dir / f"{case_id}.npz")
        mask = raw["mask"].astype(bool)
        surf = mask if out_channels == 4 else np.ones(len(raw["target"]), dtype=bool)
        fields = split_fields(raw["target"], pred_phys, out_channels)

        metrics["press_rel"].append(relative_l2(fields["press_pred"][surf], fields["press_gt"][surf]))
        if out_channels == 4:
            metrics["vel_rel"].append(relative_l2(fields["vel_pred"], fields["vel_gt"]))
            metrics["speed_rel"].append(relative_l2(fields["speed_pred"], fields["speed_gt"]))

            aero = compute_case_aero(raw["pos"], raw["features"], raw["target"], pred_phys, mask)
            for k in ("Cd_gt", "Cd_pred", "Cl_gt", "Cl_pred"):
                metrics[k].append(aero[k])
    return metrics


# --------------------------------------------------------------------------
# Continuous surface reconstruction (pure CPU geometry, no renderer)
# --------------------------------------------------------------------------
def build_colored_surface(pos_surf: np.ndarray, scalars: dict[str, np.ndarray], smooth: bool = True):
    import pyvista as pv

    cloud = pv.PolyData(pos_surf)
    for name, vals in scalars.items():
        cloud[name] = vals

    mesh = cloud.reconstruct_surface(nbr_sz=30, sample_spacing=None)
    if smooth:
        # cosmetic only (for the plot) -- skipped when this mesh feeds a
        # force integral, since it displaces vertices and would bias area
        # and normal estimates.
        mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1)

    diag = np.linalg.norm(pos_surf.max(0) - pos_surf.min(0))
    mesh = mesh.interpolate(cloud, radius=0.05 * diag, sharpness=5, strategy="closest_point")
    return mesh


# --------------------------------------------------------------------------
# Drag / lift coefficients from surface pressure (see module docstring)
# --------------------------------------------------------------------------
def compute_case_aero(pos: np.ndarray, features: np.ndarray, target: np.ndarray,
                       pred: np.ndarray, mask: np.ndarray):
    """Returns a dict of freestream info + Cd/Cl for both gt and predicted
    pressure, or None if this case has no velocity field to estimate a
    freestream direction/speed from (surface-only mode)."""
    ext = ~mask
    if not ext.any():
        return None

    sdf = features[ext, 0]
    far = np.argsort(-sdf)[:min(200, ext.sum())]
    v_inf_vec = target[ext][far, :3].mean(0)
    v_inf = float(np.linalg.norm(v_inf_vec))
    drag_dir = v_inf_vec / (v_inf + 1e-8)

    pos_surf = pos[mask]
    centered = pos_surf - pos_surf.mean(0)
    proj = centered - np.outer(centered @ drag_dir, drag_dir)
    eigvals, eigvecs = np.linalg.eigh(proj.T @ proj)
    order = np.argsort(eigvals)          # ascending: [~0 (along flow), smaller extent, larger extent]
    vertical_dir = eigvecs[:, order[1]]  # smaller-extent orthogonal axis -> "up" (car height < width)
    axis = np.argmax(np.abs(vertical_dir))
    if vertical_dir[axis] < 0:           # pin sign so it's reproducible case-to-case
        vertical_dir = -vertical_dir
    lateral_dir = np.cross(drag_dir, vertical_dir)
    lateral_dir /= np.linalg.norm(lateral_dir) + 1e-8
    vertical_dir = np.cross(lateral_dir, drag_dir)  # re-orthogonalize

    coords_2d = np.stack([pos_surf @ lateral_dir, pos_surf @ vertical_dir], axis=-1)
    A_ref = float(ConvexHull(coords_2d).volume)  # 2D hull "volume" == area

    mesh = build_colored_surface(pos_surf, {"press_gt": target[mask, 3], "press_pred": pred[mask, 3]},
                                  smooth=False)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
    mesh = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    faces = mesh.faces.reshape(-1, 4)[:, 1:4]
    normals, areas = mesh.cell_data["Normals"], mesh.cell_data["Area"]
    q = 0.5 * v_inf ** 2

    out = {
        "v_inf": v_inf, "A_ref": A_ref,
        "drag_dir": drag_dir, "vertical_dir": vertical_dir, "lateral_dir": lateral_dir,
        "center": pos_surf.mean(0),
    }
    for name in ("press_gt", "press_pred"):
        face_press = mesh.point_data[name][faces].mean(axis=1)
        force = -(face_press[:, None] * areas[:, None] * normals).sum(axis=0)
        tag = "gt" if name == "press_gt" else "pred"
        out[f"Cd_{tag}"] = float(force @ drag_dir / (q * A_ref))
        out[f"Cl_{tag}"] = float(force @ vertical_dir / (q * A_ref))
    return out


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

    # ---------------- full test-set metrics (comparable to the paper) ----------------
    metrics = evaluate_test_set(model, ds, device, out_channels)
    press_rel = np.array(metrics["press_rel"])
    print(f"\ncheckpoint epoch {ckpt.get('epoch', '?')} | {len(ds)} test cases")
    print(f"  pressure relative L2:  mean={press_rel.mean():.4f}  std={press_rel.std():.4f}  "
          f"(thuml/Transolver reports ~{TRANSOLVER_PAPER_PRESSURE_REL_L2:.3f} on this benchmark)")
    if out_channels == 4:
        vel_rel, speed_rel = np.array(metrics["vel_rel"]), np.array(metrics["speed_rel"])
        print(f"  velocity relative L2:   mean={vel_rel.mean():.4f}  std={vel_rel.std():.4f}")
        print(f"  |velocity| relative L2: mean={speed_rel.mean():.4f}  std={speed_rel.std():.4f}")

        Cd_gt, Cd_pred = np.array(metrics["Cd_gt"]), np.array(metrics["Cd_pred"])
        Cl_gt, Cl_pred = np.array(metrics["Cl_gt"]), np.array(metrics["Cl_pred"])
        Cd_rel_err = np.abs(Cd_pred - Cd_gt) / (np.abs(Cd_gt) + 1e-8)
        Cl_rel_err = np.abs(Cl_pred - Cl_gt) / (np.abs(Cl_gt) + 1e-8)
        Cd_corr = spearmanr(Cd_gt, Cd_pred).correlation
        print(f"  Cd (drag, pressure-only form drag -- no wall-shear data):")
        print(f"    gt   mean={Cd_gt.mean():.4f}  std={Cd_gt.std():.4f}")
        print(f"    pred mean={Cd_pred.mean():.4f}  std={Cd_pred.std():.4f}")
        print(f"    relative error: mean={Cd_rel_err.mean():.4f}  median={np.median(Cd_rel_err):.4f}")
        print(f"    Spearman rank correlation (gt vs pred, across cars): {Cd_corr:.4f}")
        print(f"  Cl (lift): gt mean={Cl_gt.mean():.4f}  pred mean={Cl_pred.mean():.4f}  "
              f"relative error: mean={Cl_rel_err.mean():.4f}  median={np.median(Cl_rel_err):.4f}")

    # ---------------- qualitative plots + per-sample inference time ----------------
    vis_case_ids = VIS_CASE_IDS or list(
        np.random.default_rng(SEED).choice(manifest["test_ids"], size=NUM_VIS_SAMPLES, replace=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nrendering {len(vis_case_ids)} qualitative samples -> {OUT_DIR}/")
    infer_times = []
    for i, case_id in enumerate(vis_case_ids):
        idx = ds.case_ids.index(case_id)
        pred_phys, infer_s = timed_predict_case(model, ds, idx, device)
        infer_times.append(infer_s)

        raw = np.load(CACHE_DIR / f"{case_id}.npz")
        pos_raw, mask_raw = raw["pos"], raw["mask"].astype(bool)
        surf = mask_raw if out_channels == 4 else np.ones(len(pos_raw), dtype=bool)
        fields = split_fields(raw["target"], pred_phys, out_channels)
        case_press_rel = relative_l2(fields["press_pred"][surf], fields["press_gt"][surf])

        gt = {"pressure": fields["press_gt"][surf]}
        pred = {"pressure": fields["press_pred"][surf]}
        title = (f"{case_id}  (epoch {ckpt.get('epoch', '?')}, pressure rel. L2={case_press_rel:.4f}, "
                  f"inference={infer_s * 1000:.1f} ms)")
        if out_channels == 4:
            gt["speed"], pred["speed"] = fields["speed_gt"][surf], fields["speed_pred"][surf]
            aero = compute_case_aero(pos_raw, raw["features"], raw["target"], pred_phys, mask_raw)
            title += f"\nCd gt={aero['Cd_gt']:.3f} pred={aero['Cd_pred']:.3f} | Cl gt={aero['Cl_gt']:.3f} pred={aero['Cl_pred']:.3f}"
        else:
            aero = None

        out_png = OUT_DIR / f"{i:02d}_{case_id}.png"
        plot_continuous(pos_raw[surf], gt, pred, out_png, title, show_velocity=out_channels == 4)

        line = f"  [{i}] {case_id}: inference={infer_s * 1000:.1f} ms  pressure rel. L2={case_press_rel:.4f}"
        if aero is not None:
            line += f"  Cd gt/pred={aero['Cd_gt']:.3f}/{aero['Cd_pred']:.3f}"
        print(line)

    infer_times = np.array(infer_times)
    print(f"\ninference time over {len(infer_times)} samples: "
          f"mean={infer_times.mean() * 1000:.1f} ms  std={infer_times.std() * 1000:.1f} ms  "
          f"min={infer_times.min() * 1000:.1f} ms  max={infer_times.max() * 1000:.1f} ms")


if __name__ == "__main__":
    main()
