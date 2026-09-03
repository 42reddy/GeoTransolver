"""
Picks a fixed, deterministic set of car geometries from the ShapeNet-Car
test fold (param0) to ship in the product's sidebar. Test-fold cars have
ground truth in the cache, so the app can show real accuracy metrics (not
just predictions) for every car it displays.

No CLI flags -- edit NUM_CASES/SEED below and rerun build_assets.py.
"""

import json

import numpy as np

from settings import LOCAL_CACHE_DIR

NUM_CASES = 12
SEED = 0


def select_case_ids() -> list[str]:
    manifest = json.loads((LOCAL_CACHE_DIR / "manifest.json").read_text())
    rng = np.random.default_rng(SEED)
    return list(rng.choice(manifest["test_ids"], size=NUM_CASES, replace=False))
