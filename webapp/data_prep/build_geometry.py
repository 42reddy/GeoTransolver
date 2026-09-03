"""
Offline precompute for the product app's sidebar: for each car picked by
select_cases.py, reconstructs a continuous surface mesh from the raw point
cloud and renders a plain thumbnail PNG. That's all -- no model, no
checkpoint, no inference. app.py runs the trained model live, the moment a
user actually clicks a car, so the sidebar can populate instantly and the
first click genuinely demonstrates inference speed rather than replaying a
precomputed result.

No CLI flags -- edit the PARAMS block below, then run this file (IDE run
button or `python webapp/data_prep/build_geometry.py` from the repo root).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

WEBAPP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = WEBAPP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WEBAPP_DIR))

from settings import ASSETS_DIR, LOCAL_CACHE_DIR  # noqa: E402
from data_prep.select_cases import select_case_ids  # noqa: E402
from eval_visualize import build_colored_surface  # noqa: E402

BODY_COLOR = (0.55, 0.55, 0.6)


def render_thumbnail(mesh_pv, out_path: Path):
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:4]
    verts = mesh_pv.points

    fig = plt.figure(figsize=(3, 2.2))
    ax = fig.add_subplot(projection="3d")
    poly = Poly3DCollection(verts[faces], linewidths=0, facecolor=BODY_COLOR)
    ax.add_collection3d(poly)
    lo, hi = verts.min(0), verts.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=15, azim=-60)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def main():
    case_ids = select_case_ids()
    print(f"selected {len(case_ids)} cases: {case_ids}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, case_id in enumerate(case_ids):
        raw = np.load(LOCAL_CACHE_DIR / f"{case_id}.npz")
        pos_surf = raw["pos"][raw["mask"].astype(bool)]
        mesh = build_colored_surface(pos_surf, {})

        case_dir = ASSETS_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        thumb_path = case_dir / "thumbnail.png"
        render_thumbnail(mesh, thumb_path)

        entries.append({
            "case_id": case_id,
            "display_name": f"Car {idx + 1:02d}",
            "thumbnail": str(thumb_path.relative_to(ASSETS_DIR)),
        })
        print(f"  [{idx + 1}/{len(case_ids)}] {case_id}")

    (ASSETS_DIR / "manifest.json").write_text(json.dumps({"cases": entries}, indent=2))
    print(f"\nwrote {len(entries)} cases -> {ASSETS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
