"""
Product app: pick one of a fixed set of ShapeNet-Car geometries and see the
GeoTransolver-inferred CFD flow field, alongside aerodynamic metrics.

Inference is lazy: the sidebar gallery is populated from thumbnails built
offline (data_prep/build_geometry.py, no model involved), and the trained
checkpoint only runs the moment a user clicks a car -- so the first click on
each car is a real, visible measurement of inference speed, not a replay of
something precomputed. Results are cached per case after that first click so
switching fields on an already-viewed car is instant.

The 3D viewer is a custom three.js scene (embedded via gr.HTML's `head` /
`js_on_load`), not Gradio's built-in gr.Model3D, mainly to get full control
over rendering thin, per-point-colored streamlines on top of the car body.

Run: `python webapp/app.py` from anywhere (paths are resolved from this
file's location via settings.py).
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch

import gradio as gr

WEBAPP_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBAPP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WEBAPP_DIR))

from settings import (  # noqa: E402
    ASSETS_DIR, MANIFEST_PATH, LOCAL_CACHE_DIR, LOCAL_CKPT_DIR,
    HF_CKPT_REPO_ID, HF_CKPT_REPO_TYPE, HF_DATA_REPO_ID, HF_DATA_REPO_TYPE, HF_TOKEN,
)
from shapenet_car_dataset import ShapeNetCarDataset  # noqa: E402
from geotransolver import GeoTransolver  # noqa: E402
from eval_visualize import (  # noqa: E402
    build_colored_surface, compute_case_aero, predict_case, relative_l2, timed_predict_case,
)
# Model hyperparameters live in train.py (single source of truth) so the
# architecture built here always matches whatever checkpoint train.py last
# produced -- a hardcoded second copy here is exactly how DIM/DEPTH/etc.
# would silently drift from the weights and break state_dict loading.
from train import (  # noqa: E402
    DIM, DEPTH, HEADS, DIM_HEAD, NUM_SLICES, MLP_RATIO, DROPOUT,
    LOCAL_RADII, LOCAL_NEIGHBORS, LOCAL_HIDDEN,
)

# ==========================================================================
# Params
# ==========================================================================
CKPT_NAME = "GeoTransolver_Shapenetcar.pt"  # falls back to "last.pt" if best.pt isn't there

PRESSURE_CMAP, VELOCITY_CMAP, ERROR_CMAP = "coolwarm", "viridis", "inferno"
BODY_COLOR_U8 = np.array([120, 120, 130, 255], dtype=np.uint8)

# Velocity is only meaningful in the exterior volume, not on the surface --
# the no-slip boundary condition pins surface velocity to ~0 everywhere. The
# exterior points have no mesh connectivity and are sparse/scattered, so
# they're alpha-shape Delaunay-tetrahedralized into a real 3D field first
# (alpha discards oversized "bridging" tetrahedra that would otherwise
# connect points on opposite sides of the car body, cutting straight through
# it), then seeded with a grid of streamlines just upstream of the car and
# traced through that field with VTK's adaptive RK45 integrator (accurate
# and, unlike a hand-rolled fixed-step Euler integrator, requires no manual
# dense-grid resampling step). Point data (speed) is interpolated along each
# line automatically, so color varies along its length exactly as it would
# in a real CFD visualization -- accelerating over the hood, slowing in the
# wake.
FLOW_VOLUME_ALPHA = 0.35        # ~2.7x the median exterior-point spacing on this dataset (empirically tuned)
STREAMLINE_SEED_GRID = (20, 14)  # (lateral, vertical) particle seeds upstream of the car
STREAMLINE_DOMAIN_CROSSINGS = 2.5  # integrate for this many domain-lengths of (curving) travel

FIELD_CHOICES = [
    "Predicted pressure", "Ground-truth pressure", "Pressure error",
    "Predicted velocity", "Ground-truth velocity", "Velocity error",
]
DEFAULT_FIELD = "Predicted pressure"
VIEWER_HEIGHT = 520

FIELD_MAP = {
    "Predicted pressure": {"type": "mesh", "color": "pred_pressure", "cmap": PRESSURE_CMAP, "unit": "Pressure (m²/s²)"},
    "Ground-truth pressure": {"type": "mesh", "color": "gt_pressure", "cmap": PRESSURE_CMAP, "unit": "Pressure (m²/s²)"},
    "Pressure error": {"type": "mesh", "color": "err_pressure", "cmap": ERROR_CMAP, "unit": "Pressure error (m²/s²)"},
    "Predicted velocity": {"type": "stream", "which": "pred", "color": "pred_velocity", "cmap": VELOCITY_CMAP, "unit": "Speed (m/s)"},
    "Ground-truth velocity": {"type": "stream", "which": "gt", "color": "gt_velocity", "cmap": VELOCITY_CMAP, "unit": "Speed (m/s)"},
    "Velocity error": {"type": "stream", "which": "gt", "color": "err_velocity", "cmap": ERROR_CMAP, "unit": "Speed error (m/s)"},
}

# 12-stop hex gradient per colormap, embedded once and reused by every case's
# colorbar payload -- built here from the same matplotlib colormap objects
# used to color the geometry, so the legend always matches exactly.
CMAP_STOPS = {
    name: [mcolors.to_hex(matplotlib.colormaps[name](t)) for t in np.linspace(0, 1, 12)]
    for name in (PRESSURE_CMAP, VELOCITY_CMAP, ERROR_CMAP)
}

CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 0.5rem; }
#header p { color: var(--body-text-color-subdued); margin-top: 0; font-size: 1.05rem; }
#gallery .grid-wrap { max-height: 520px; overflow-y: auto; }
#metrics { padding: 0.75rem 1rem; border-radius: 12px; background: var(--block-background-fill);
           border: 1px solid var(--block-border-color); }
#flow-viewer { border-radius: 12px; overflow: hidden; border: 1px solid var(--block-border-color); }
"""


# ==========================================================================
# Model + dataset -- loaded once at startup, never re-loaded per request.
# Loading weights is fast; it's the forward pass that's deferred to the
# user's first click on each car (see compute_case, below).
# ==========================================================================
def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(device: torch.device, manifest: dict) -> GeoTransolver:
    ckpt_path = LOCAL_CKPT_DIR / CKPT_NAME
    if not ckpt_path.exists():
        ckpt_path = LOCAL_CKPT_DIR / "last.pt"
    if not ckpt_path.exists():
        if not HF_CKPT_REPO_ID:
            raise SystemExit(
                f"No checkpoint found in {LOCAL_CKPT_DIR} (looked for {CKPT_NAME} and last.pt), and "
                f"HF_CKPT_REPO_ID isn't set to download one. Either drop a checkpoint in {LOCAL_CKPT_DIR}, "
                f"or set HF_CKPT_REPO_ID (Space Settings -> Variables and secrets) to the HF model repo "
                f"holding {CKPT_NAME}."
            )
        from huggingface_hub import hf_hub_download
        print(f"no local checkpoint -- downloading hf://{HF_CKPT_REPO_ID}/{CKPT_NAME} ...")
        ckpt_path = Path(hf_hub_download(
            repo_id=HF_CKPT_REPO_ID, repo_type=HF_CKPT_REPO_TYPE, filename=CKPT_NAME,
            local_dir=LOCAL_CKPT_DIR, token=HF_TOKEN,
        ))
    model = GeoTransolver(
        space_dim=3, geom_dim=manifest["in_channels"], cond_dim=None,
        out_channels=manifest["out_channels"], num_constants=None,
        dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
        num_slices=NUM_SLICES, mlp_ratio=MLP_RATIO, dropout=DROPOUT,
        local_radii=LOCAL_RADII, local_neighbors=LOCAL_NEIGHBORS, local_hidden=LOCAL_HIDDEN,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"{MANIFEST_PATH} not found. Run `python webapp/data_prep/build_geometry.py` first "
            f"to generate the sidebar thumbnails."
        )
    return json.loads(MANIFEST_PATH.read_text())["cases"]


def ensure_case_cache(case_ids: list[str]):
    """Download any of the gallery's per-case .npz files (ground-truth pos/
    features/target/mask) not already sitting in LOCAL_CACHE_DIR -- these
    aren't committed to the repo (see settings.py), so on a fresh Space
    checkout every one of them is missing on first launch."""
    missing = [cid for cid in case_ids if not (LOCAL_CACHE_DIR / f"{cid}.npz").exists()]
    if not missing:
        return
    if not HF_DATA_REPO_ID:
        raise SystemExit(
            f"{len(missing)} case .npz file(s) missing from {LOCAL_CACHE_DIR}, and HF_DATA_REPO_ID isn't "
            f"set to download them. Either populate {LOCAL_CACHE_DIR} directly, or set HF_DATA_REPO_ID "
            f"(Space Settings -> Variables and secrets) to the HF repo holding them."
        )
    from huggingface_hub import hf_hub_download
    print(f"downloading {len(missing)} case file(s) from hf://{HF_DATA_REPO_ID} ...")
    for cid in missing:
        hf_hub_download(repo_id=HF_DATA_REPO_ID, repo_type=HF_DATA_REPO_TYPE, filename=f"{cid}.npz",
                         local_dir=LOCAL_CACHE_DIR, token=HF_TOKEN)


CASES = load_manifest()
GALLERY_ITEMS = [(str(ASSETS_DIR / c["thumbnail"]), c["display_name"]) for c in CASES]

DEVICE = pick_device()
ensure_case_cache([c["case_id"] for c in CASES])
DATA_MANIFEST = json.loads((LOCAL_CACHE_DIR / "manifest.json").read_text())
NORM_STATS = dict(np.load(LOCAL_CKPT_DIR / "norm_stats.npz"))
MODEL = load_model(DEVICE, DATA_MANIFEST)
DS = ShapeNetCarDataset(LOCAL_CACHE_DIR, case_ids=[c["case_id"] for c in CASES], stats=NORM_STATS)

# A device's first forward pass pays a one-time cost (MPS compiles its Metal
# kernels on first use -- measured ~1.8s here vs. ~0.4s steady-state on the
# same input size). Paying that during startup, not on the user's first
# click, is what actually fixes "inference feels slow" -- the model itself
# isn't slower than before, the first click was just unluckily the one
# absorbing a one-time cost that used to be hidden inside a longer offline
# build script.
_t0 = time.perf_counter()
predict_case(MODEL, DS, 0, DEVICE)
print(f"model warm-up: {time.perf_counter() - _t0:.2f}s (one-time, not shown to users)")

LIVE_CACHE: dict[int, dict] = {}


# ==========================================================================
# Live per-case compute: real model forward pass + mesh/slice construction,
# triggered once (and cached) the first time a case is selected.
# ==========================================================================
def to_rgba_u8(values: np.ndarray, cmap_name: str, clim: tuple[float, float]) -> np.ndarray:
    norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
    rgba = matplotlib.colormaps[cmap_name](norm(values))
    return (rgba * 255).astype(np.uint8)


def write_bin(array: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(str(out_path))


def asset_url(abs_path: Path) -> str:
    return f"/gradio_api/file={abs_path}"


def export_mesh(mesh_pv, case_dir: Path, prefix: str) -> tuple[str, str, int, int]:
    pos = np.ascontiguousarray(mesh_pv.points, dtype=np.float32)
    faces = np.ascontiguousarray(mesh_pv.faces.reshape(-1, 4)[:, 1:4], dtype=np.uint32)
    pos_path, faces_path = case_dir / f"{prefix}_pos.bin", case_dir / f"{prefix}_faces.bin"
    write_bin(pos, pos_path)
    write_bin(faces, faces_path)
    return asset_url(pos_path), asset_url(faces_path), len(pos), len(faces)


def build_flow_volume(pos_ext: np.ndarray, vel_pred: np.ndarray, vel_gt: np.ndarray):
    speed_pred, speed_gt = np.linalg.norm(vel_pred, axis=-1), np.linalg.norm(vel_gt, axis=-1)
    cloud = pv.PolyData(pos_ext)
    cloud["velocity_pred"], cloud["velocity_gt"] = vel_pred, vel_gt
    cloud["speed_pred"], cloud["speed_gt"] = speed_pred, speed_gt
    cloud["speed_err"] = np.abs(speed_pred - speed_gt)
    return cloud.delaunay_3d(alpha=FLOW_VOLUME_ALPHA)


def streamline_seed_grid(pos_ext: np.ndarray, center: np.ndarray, drag_dir: np.ndarray,
                          lateral_dir: np.ndarray, vertical_dir: np.ndarray) -> np.ndarray:
    """A grid of seed points in a plane just upstream of the car.

    Sized to a fraction of the exterior point cloud's own lateral/vertical
    extent and placed at a fraction of its upstream longitudinal extent --
    *not* close to the full extent on either axis, even though that would
    still nominally sit inside the overall bounding box: the alpha-shape
    filtering in build_flow_volume trims tetrahedra more aggressively where
    the scattered points thin out near the domain's outer boundary, so a
    seed grid sized to the full extent lands mostly outside the *actually
    interpolatable* region -- empirically, ~65% of seeds were dead on
    arrival before this was tuned down. (along=0.8, span=0.55) keeps
    ~96-100% of seeds valid at t=0 across the cases checked."""
    rel = pos_ext - center
    along_min = float((rel @ drag_dir).min())
    lat_span = float(np.abs(rel @ lateral_dir).max()) * 0.55
    vert_rel = rel @ vertical_dir
    vert_lo, vert_hi = float(vert_rel.min()) * 0.275, float(vert_rel.max()) * 0.55

    n_lat, n_vert = STREAMLINE_SEED_GRID
    lat_grid, vert_grid = np.meshgrid(np.linspace(-lat_span, lat_span, n_lat), np.linspace(vert_lo, vert_hi, n_vert))
    return (center + (along_min * 0.8) * drag_dir
            + lat_grid.reshape(-1, 1) * lateral_dir + vert_grid.reshape(-1, 1) * vertical_dir)


def polylines_to_segments(poly) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """VTK packs however many polylines a PolyData holds into one flat array
    (count, id0, id1, ..., count, id0, id1, ...); unpack it into parallel
    start/end point-index arrays, one pair per line *segment* -- what
    THREE.LineSegments expects -- so consecutive streamlines never get
    joined by a stray connecting line the way a single continuous THREE.Line
    would draw them."""
    lines = np.asarray(poly.lines)
    starts, ends = [], []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1: i + 1 + n]
        starts.append(ids[:-1])
        ends.append(ids[1:])
        i += 1 + n
    if not starts:
        return None, None
    return np.concatenate(starts), np.concatenate(ends)


def export_streamlines(poly, case_dir: Path, prefix: str, color_fields: dict) -> dict | None:
    """color_fields: {field_key: (point_data_name, cmap_name, clim)}. Every
    entry shares the same line geometry (just recolored), so positions are
    written once and reused across all of them -- same pattern as the
    pressure mesh's shared positions + per-field color arrays."""
    starts, ends = polylines_to_segments(poly)
    if starts is None or len(starts) == 0:
        return None

    pts = np.asarray(poly.points, dtype=np.float32)
    pos = np.empty((len(starts) * 2, 3), dtype=np.float32)
    pos[0::2], pos[1::2] = pts[starts], pts[ends]
    pos_path = case_dir / f"{prefix}_pos.bin"
    write_bin(pos, pos_path)

    colors = {}
    for field_key, (value_name, cmap_name, clim) in color_fields.items():
        vals = np.asarray(poly.point_data[value_name], dtype=np.float32)
        seg_vals = np.empty(len(starts) * 2, dtype=np.float32)
        seg_vals[0::2], seg_vals[1::2] = vals[starts], vals[ends]
        out_path = case_dir / f"{prefix}_col_{field_key}.bin"
        write_bin(to_rgba_u8(seg_vals, cmap_name, clim), out_path)
        colors[field_key] = asset_url(out_path)

    return {"pos": asset_url(pos_path), "nSegments": len(starts), "colors": colors}


def compute_case(idx: int) -> dict:
    if idx in LIVE_CACHE:
        return LIVE_CACHE[idx]

    case = CASES[idx]
    case_id = case["case_id"]
    case_dir = ASSETS_DIR / case_id

    pred_phys, infer_s = timed_predict_case(MODEL, DS, idx, DEVICE)
    raw = np.load(LOCAL_CACHE_DIR / f"{case_id}.npz")
    pos, features, target, mask = raw["pos"], raw["features"], raw["target"], raw["mask"].astype(bool)
    pos_surf, pos_ext = pos[mask], pos[~mask]

    aero = compute_case_aero(pos, features, target, pred_phys, mask)

    # -------- pressure surface mesh --------
    press_gt, press_pred = target[mask, 3], pred_phys[mask, 3]
    press_err = np.abs(press_pred - press_gt)
    mesh = build_colored_surface(pos_surf, {"press_gt": press_gt, "press_pred": press_pred, "press_err": press_err})
    mesh_pos_url, mesh_faces_url, n_verts, n_faces = export_mesh(mesh, case_dir, "mesh")

    p_clim = (float(min(press_gt.min(), press_pred.min())), float(max(press_gt.max(), press_pred.max())))
    perr_clim = (0.0, float(press_err.max()) + 1e-8)
    mesh_color_fields = {
        "pred_pressure": ("press_pred", PRESSURE_CMAP, p_clim),
        "gt_pressure": ("press_gt", PRESSURE_CMAP, p_clim),
        "err_pressure": ("press_err", ERROR_CMAP, perr_clim),
    }
    mesh_colors = {}
    for field_key, (scalar_name, cmap_name, clim) in mesh_color_fields.items():
        out_path = case_dir / f"mesh_color_{field_key}.bin"
        write_bin(to_rgba_u8(mesh.point_data[scalar_name], cmap_name, clim), out_path)
        mesh_colors[field_key] = asset_url(out_path)
    body_path = case_dir / "mesh_color_body.bin"
    write_bin(np.tile(BODY_COLOR_U8, (n_verts, 1)), body_path)
    mesh_colors["body"] = asset_url(body_path)

    # -------- velocity: thin streamlines through the exterior flow field --------
    vel_pred_vec, vel_gt_vec = pred_phys[~mask, :3], target[~mask, :3]
    speed_pred, speed_gt = np.linalg.norm(vel_pred_vec, axis=-1), np.linalg.norm(vel_gt_vec, axis=-1)
    speed_err = np.abs(speed_pred - speed_gt)
    v_clim = (float(min(speed_gt.min(), speed_pred.min())), float(max(speed_gt.max(), speed_pred.max())))
    verr_clim = (0.0, float(speed_err.max()) + 1e-8)

    flow_volume = build_flow_volume(pos_ext, vel_pred_vec, vel_gt_vec)
    diag = float(np.linalg.norm(pos.max(0) - pos.min(0)))
    seeds = pv.PolyData(streamline_seed_grid(pos_ext, aero["center"], aero["drag_dir"],
                                              aero["lateral_dir"], aero["vertical_dir"]))
    max_length = STREAMLINE_DOMAIN_CROSSINGS * diag
    stream_kwargs = dict(integration_direction="forward", max_length=max_length, terminal_speed=0.05)

    sl_pred = flow_volume.streamlines_from_source(seeds, vectors="velocity_pred", **stream_kwargs)
    sl_gt = flow_volume.streamlines_from_source(seeds, vectors="velocity_gt", **stream_kwargs)

    streamlines = {
        "pred": export_streamlines(sl_pred, case_dir, "stream_pred",
                                    {"pred_velocity": ("speed_pred", VELOCITY_CMAP, v_clim)}),
        "gt": export_streamlines(sl_gt, case_dir, "stream_gt", {
            "gt_velocity": ("speed_gt", VELOCITY_CMAP, v_clim),
            "err_velocity": ("speed_err", ERROR_CMAP, verr_clim),
        }),
    }

    press_rel_l2 = relative_l2(press_pred, press_gt)
    vel_rel_l2 = relative_l2(pred_phys[:, :3], target[:, :3])

    result = {
        "display_name": case["display_name"],
        "mesh": {"pos": mesh_pos_url, "faces": mesh_faces_url, "nVerts": n_verts, "nFaces": n_faces, "colors": mesh_colors},
        "streamlines": streamlines,
        "clims": {
            "pred_pressure": p_clim, "gt_pressure": p_clim, "err_pressure": perr_clim,
            "pred_velocity": v_clim, "gt_velocity": v_clim, "err_velocity": verr_clim,
        },
        "Cd_gt": aero["Cd_gt"], "Cd_pred": aero["Cd_pred"],
        "v_inf": aero["v_inf"],
        "press_rel_l2": press_rel_l2, "vel_rel_l2": vel_rel_l2,
        "inference_ms": infer_s * 1000,
    }
    LIVE_CACHE[idx] = result
    return result


def metrics_markdown(case: dict) -> str:
    return f"""
### {case['display_name']}

| metric | predicted | ground truth |
|---|---|---|
| Drag coefficient (Cd) | {case['Cd_pred']:.3f} | {case['Cd_gt']:.3f} |

**Pressure error (rel. L2):** {case['press_rel_l2']:.2%}
**Velocity error (rel. L2):** {case['vel_rel_l2']:.2%}
**Inference time:** {case['inference_ms']:.1f} ms
**Freestream speed:** {case['v_inf']:.1f} m/s
"""


PLACEHOLDER_METRICS = "### Select a car\nClick a thumbnail on the left to run inference on it."


def nice_ticks(lo: float, hi: float, target_n: int = 5) -> list[float]:
    """Round, evenly-spaced tick values covering [lo, hi] -- e.g. 0/5/10/15/20
    rather than the raw range's own 0/5.98/11.95/17.93/23.9 -- the classic
    'nice numbers' axis-ticking trick: pick whichever of {1, 2, 2.5, 5, 10}
    x 10^k lands closest to the ideal step size for the requested tick
    count, then step through the range at that size."""
    if hi <= lo:
        return [round(lo, 2)]
    raw_step = (hi - lo) / (target_n - 1)
    mag = 10 ** np.floor(np.log10(raw_step))
    nice_norm = min([1, 2, 2.5, 5, 10], key=lambda c: abs(c - raw_step / mag))
    step = nice_norm * mag
    start = np.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-6:
        if v >= lo - step * 1e-6:
            ticks.append(v)
        v += step
    if step >= 1:
        return [round(t) for t in ticks]
    decimals = max(0, -int(np.floor(np.log10(step))))
    return [round(t, decimals) for t in ticks]


def colorbar_payload(case: dict, field_label: str) -> dict:
    meta = FIELD_MAP[field_label]
    lo, hi = case["clims"][meta["color"]]
    return {"label": meta["unit"], "ticks": nice_ticks(lo, hi), "lo": lo, "hi": hi, "stops": CMAP_STOPS[meta["cmap"]]}


def viewer_payload(idx: int | None, field_label: str) -> str:
    if idx is None:
        return json.dumps({})
    case = compute_case(idx)
    return json.dumps({
        "field": field_label,
        "mesh": case["mesh"],
        "streamlines": case["streamlines"],
        "colorbar": colorbar_payload(case, field_label),
    })


def on_select_case(evt: gr.SelectData, field_label: str):
    idx = evt.index
    case = compute_case(idx)
    return idx, metrics_markdown(case), viewer_payload(idx, field_label)


def on_field_change(selected_index: int | None, field_label: str):
    if selected_index is None:
        return PLACEHOLDER_METRICS, json.dumps({})
    return metrics_markdown(compute_case(selected_index)), viewer_payload(selected_index, field_label)


# ==========================================================================
# Custom three.js viewer: the car surface (colored by pressure, or a neutral
# gray body while a velocity field is showing) plus, for velocity fields,
# thin streamlines whose color varies along their own length (speed/error,
# interpolated per-point by VTK during tracing) -- rendered as THREE.LineSegments
# so each streamline is visually independent, plus an HTML/CSS colorbar
# (not a 3D object) showing what each color means for the current field.
# ==========================================================================
THREE_JS_VERSION = "0.128.0"
HTML_HEAD = f"""
<script src="https://cdn.jsdelivr.net/npm/three@{THREE_JS_VERSION}/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@{THREE_JS_VERSION}/examples/js/controls/OrbitControls.js"></script>
"""

JS_ON_LOAD = f"""
const FIELD_MAP = {json.dumps(FIELD_MAP)};
const VIEWER_HEIGHT = {VIEWER_HEIGHT};

function whenThreeReady(cb) {{
  if (typeof THREE !== 'undefined' && THREE.OrbitControls) {{ cb(); }}
  else {{ setTimeout(() => whenThreeReady(cb), 50); }}
}}

whenThreeReady(() => {{
  const container = element.querySelector('#flow-viewer');
  const width = container.clientWidth || 800;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  const camera = new THREE.PerspectiveCamera(35, width / VIEWER_HEIGHT, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, VIEWER_HEIGHT);
  container.appendChild(renderer.domElement);

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(-1, 1.6, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4);
  fill.position.set(1, 0.5, -1.5);
  scene.add(fill);

  const fetchCache = {{}};
  async function fetchBin(url) {{
    if (!(url in fetchCache)) {{
      fetchCache[url] = fetch(url).then(r => r.arrayBuffer());
    }}
    return await fetchCache[url];
  }}

  function disposeObject(obj) {{
    if (!obj) return;
    scene.remove(obj);
    obj.geometry.dispose();
    obj.material.dispose();
  }}

  function fitCameraTo(object) {{
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const dir = new THREE.Vector3(0.7, 0.35, -0.9).normalize();
    camera.position.copy(center).addScaledVector(dir, maxDim * 1.7);
    camera.near = maxDim * 0.01;
    camera.far = maxDim * 30;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
  }}

  const colorbarEl = element.querySelector('#colorbar');
  const COLORBAR_WIDTH = 280;
  function updateColorbar(info) {{
    if (!info) {{ colorbarEl.style.display = 'none'; return; }}
    colorbarEl.style.display = 'block';
    const stops = info.stops.map((c, i) => `${{c}} ${{(i / (info.stops.length - 1) * 100).toFixed(1)}}%`).join(', ');
    const span = info.hi - info.lo;
    const marks = info.ticks.map(v => {{
      const pct = span > 0 ? ((v - info.lo) / span) * 100 : 0;
      const tx = pct < 5 ? '0%' : pct > 95 ? '-100%' : '-50%';
      return `<div style="position:absolute;left:${{pct}}%;top:0;width:1px;height:4px;background:#999;"></div>
              <span style="position:absolute;left:${{pct}}%;top:6px;transform:translateX(${{tx}});color:#111;">${{v}}</span>`;
    }}).join('');
    colorbarEl.innerHTML = `
      <div style="margin-bottom:4px;color:#111;">${{info.label}}</div>
      <div style="width:${{COLORBAR_WIDTH}}px;height:10px;border-radius:4px;background:linear-gradient(to right, ${{stops}});"></div>
      <div style="position:relative;width:${{COLORBAR_WIDTH}}px;height:16px;margin-top:2px;font-size:11px;">${{marks}}</div>`;
  }}

  // Two payloads can land close together (e.g. a car click still mid-flight
  // when a field click's response arrives) and their `show()` calls'
  // fetches can then resolve out of order. A monotonic generation token,
  // checked immediately before every buffer mutation (not just before
  // deciding whether to keep going -- by the time an awaited fetch
  // resolves, the mutation it's about to perform is the thing that needs
  // gating, not just the control flow after it), guarantees the *last*
  // requested (car, field) always wins the screen even if it settles first.
  let requestGen = 0;

  const viewer = {{
    scene, camera, renderer, controls,
    meshObj: null, meshKey: null,
    streamObj: null, streamKey: null,

    async loadMesh(m, gen) {{
      if (this.meshKey === m.pos) return;
      const [posBuf, faceBuf] = await Promise.all([fetchBin(m.pos), fetchBin(m.faces)]);
      if (gen !== requestGen) return;
      this.meshKey = m.pos;
      disposeObject(this.meshObj);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(posBuf), 3));
      geo.setIndex(new THREE.BufferAttribute(new Uint32Array(faceBuf), 1));
      geo.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(m.nVerts * 4), 4, true));
      geo.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({{ vertexColors: true, side: THREE.DoubleSide, roughness: 0.85, metalness: 0.0 }});
      this.meshObj = new THREE.Mesh(geo, mat);
      scene.add(this.meshObj);
      fitCameraTo(this.meshObj);
    }},

    async setMeshColor(colorUrl, gen) {{
      const buf = await fetchBin(colorUrl);
      if (gen !== requestGen || !this.meshObj) return;
      this.meshObj.geometry.attributes.color.array.set(new Uint8Array(buf));
      this.meshObj.geometry.attributes.color.needsUpdate = true;
    }},

    hideStream() {{
      if (this.streamObj) this.streamObj.visible = false;
    }},

    async loadStream(s, gen) {{
      if (this.streamKey === s.pos) {{ this.streamObj.visible = true; return; }}
      const posBuf = await fetchBin(s.pos);
      if (gen !== requestGen) return;
      this.streamKey = s.pos;
      disposeObject(this.streamObj);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(posBuf), 3));
      geo.setAttribute('color', new THREE.BufferAttribute(new Uint8Array(s.nSegments * 2 * 4), 4, true));
      const mat = new THREE.LineBasicMaterial({{ vertexColors: true, linewidth: 1 }});
      this.streamObj = new THREE.LineSegments(geo, mat);
      scene.add(this.streamObj);
    }},

    async setStreamColor(colorUrl, gen) {{
      const buf = await fetchBin(colorUrl);
      if (gen !== requestGen || !this.streamObj) return;
      this.streamObj.geometry.attributes.color.array.set(new Uint8Array(buf));
      this.streamObj.geometry.attributes.color.needsUpdate = true;
    }},

    async show(data) {{
      if (!data.mesh) return;
      const gen = ++requestGen;
      const f = FIELD_MAP[data.field];
      updateColorbar(data.colorbar);
      await this.loadMesh(data.mesh, gen);
      if (gen !== requestGen) return;
      if (f.type === 'mesh') {{
        this.hideStream();
        await this.setMeshColor(data.mesh.colors[f.color], gen);
      }} else {{
        await this.setMeshColor(data.mesh.colors.body, gen);
        if (gen !== requestGen) return;
        const s = data.streamlines[f.which];
        if (s) {{
          await this.loadStream(s, gen);
          if (gen !== requestGen) return;
          await this.setStreamColor(s.colors[f.color], gen);
        }} else {{
          this.hideStream();
        }}
      }}
    }},
  }};
  window.flowViewer = viewer;

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }}
  animate();

  window.addEventListener('resize', () => {{
    const w = container.clientWidth || width;
    camera.aspect = w / VIEWER_HEIGHT;
    camera.updateProjectionMatrix();
    renderer.setSize(w, VIEWER_HEIGHT);
  }});

  // `value` on this component is a JSON payload Python writes to on every
  // car/field selection -- gr.State can't be read from js= event callbacks
  // (confirmed empirically: it always arrives as null), so this component's
  // own value is the sync channel instead. `html_template` (in app.py)
  // doesn't reference ${{value}}, so these updates never touch the
  // DOM/canvas -- only `watch` observes them.
  watch('value', () => {{
    try {{
      viewer.show(JSON.parse(props.value));
    }} catch (e) {{ console.error('viewer sync payload parse failed', e); }}
  }});
}});
"""

with gr.Blocks(title="") as demo:
    selected_index = gr.State(None)

    with gr.Column(elem_id="header"):
        gr.Markdown("Interactive CFD flow-field prediction on car geometries")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            metrics = gr.Markdown(PLACEHOLDER_METRICS, elem_id="metrics")
            gallery = gr.Gallery(GALLERY_ITEMS, label="Cars", elem_id="gallery", columns=2,
                                  height=520, object_fit="cover", allow_preview=False)
        with gr.Column(scale=3):
            field = gr.Radio(FIELD_CHOICES, value=DEFAULT_FIELD, label="Flow field")
            viewer_sync = gr.HTML(
                value=json.dumps({}),
                html_template=(
                    f'<div id="flow-viewer" style="position:relative;width:100%;height:{VIEWER_HEIGHT}px;">'
                    f'<div id="colorbar" style="display:none;position:absolute;left:14px;bottom:14px;'
                    f'padding:8px 10px;background:rgba(255,255,255,0.88);border-radius:8px;'
                    f'font:12px/1.3 -apple-system,sans-serif;color:#111;box-shadow:0 1px 4px rgba(0,0,0,0.15);"></div>'
                    f'</div>'
                ),
                head=HTML_HEAD, js_on_load=JS_ON_LOAD,
            )

    # Both handlers read/write `selected_index` and write the viewer payload;
    # sharing one concurrency slot forces them to run strictly in submission
    # order instead of racing (e.g. a car click still mid-inference losing
    # to a fast follow-up field click that reads the not-yet-updated
    # `selected_index` state and overwrites the screen with the wrong car).
    gallery.select(on_select_case, inputs=[field], outputs=[selected_index, metrics, viewer_sync],
                    concurrency_id="viewer_update", concurrency_limit=1)
    field.change(on_field_change, inputs=[selected_index, field], outputs=[metrics, viewer_sync],
                  concurrency_id="viewer_update", concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=CUSTOM_CSS,
                allowed_paths=[str(ASSETS_DIR)])
