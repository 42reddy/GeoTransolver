"""
Train GeoTransolver (geotransolver/geotransolver.py) to predict the
converged surface CFD field on BlendedNet, using the cache built by
blendednet_dataset.py (packed as blendednet_cache.tar.gz).

--------------------------------------------------------------------------
Task recap
--------------------------------------------------------------------------
Each BlendedNet case is one BWB geometry at one flight condition. The
cache stores a fixed-N surface point cloud per case:

    pos        (N, 3)  surface xyz, recentered and scaled in the Dataset
    geometry   (N, 3)  surface normals (GeoTransolver geometry stream)
    condition  (C,)    per-case geometry/flight metadata (may be empty)
    target     (N, F)  converged surface fields (Cp, Cfx, Cfy, Cfz)
    constants  (K,)    per-case aero coefficients C1/C3/... (may be empty)

Unlike ShapeNet-Car, every cached point is a real surface sample, so the
field loss is unmasked MSE. If the cache actually has a condition vector
and/or constant targets, they are wired into GeoTransolver (global
context token + constants head) and included in the loss. This packed
cache currently has C=0 and K=0 (no metadata keys were stored), so the
model is built with cond_dim=None and num_constants=None -- geometry
cross-attention only, field head only -- until you rebuild the cache with
those keys populated.

The packed archive extracts to cache_blendednet/ (manifest.json + one
.npz per case). First run unpacks BLENDEDNET_CACHE_ARCHIVE into
BLENDEDNET_CACHE_DIR's parent if the extracted manifest is missing.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No CLI flags -- edit the PARAMS block below directly, then run this file
(IDE run button or `python train_blendednet.py`). Data/checkpoint paths
come from config.py.
"""

import json
import math
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from blendednet_dataset import BlendedNetDataset
from config import BLENDEDNET_CACHE_ARCHIVE, BLENDEDNET_CACHE_DIR, BLENDEDNET_CKPT_DIR
from geotransolver import GeoTransolver


# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
RESUME = None          # Path to a checkpoint .pt to resume from, or None

# optimization
EPOCHS = 25
BATCH_SIZE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
VAL_FRACTION = 0.1     # carved out of the non-test pool
TEST_FRACTION = 0.1    # held-out from all case_ids (manifest has no official split)
NUM_WORKERS = 2
AMP = True
SEED = 0
CONST_LOSS_WEIGHT = 1.0  # only used when the cache has constant targets

# model
DIM = 512
DEPTH = 12
HEADS = 8
DIM_HEAD = 64
NUM_SLICES = 96
MLP_RATIO = 4.0
DROPOUT = 0.05
LOCAL_RADII = (0.05, 0.25)
LOCAL_NEIGHBORS = (8, 32)
LOCAL_HIDDEN = 32


# --------------------------------------------------------------------------
# Cache: unpack blendednet_cache.tar.gz once if the extracted dir is missing
# --------------------------------------------------------------------------
def ensure_cache(cache_dir: Path, archive: Path):
    if (cache_dir / "manifest.json").exists():
        return cache_dir
    if not archive.exists():
        raise FileNotFoundError(
            f"BlendedNet cache not found at {cache_dir} and archive "
            f"{archive} is missing. Run blendednet_dataset.py first, or "
            f"place blendednet_cache.tar.gz at that archive path."
        )
    dest = cache_dir.parent
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} -> {dest} (one-time unpack of the packed cache)")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    if not (cache_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Extracted {archive} into {dest}, but {cache_dir / 'manifest.json'} "
            f"is still missing -- check the archive's top-level folder name."
        )
    return cache_dir


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def compute_loss(field_pred, field_target, const_pred, const_target, const_weight: float):
    """Unmasked field MSE, plus constants MSE when the constants head is on."""
    field_loss = F.mse_loss(field_pred, field_target)
    breakdown = {"field_mse": field_loss.item()}
    if const_pred is None:
        return field_loss, breakdown
    const_loss = F.mse_loss(const_pred, const_target)
    breakdown["const_mse"] = const_loss.item()
    return field_loss + const_weight * const_loss, breakdown


# --------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay, stepped per optimizer step
# --------------------------------------------------------------------------
def make_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


# --------------------------------------------------------------------------
# Train / eval loops
# --------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None, amp=False, grad_clip=1.0,
              pbar=None, phase="", const_weight=1.0):
    """pbar, if given, is a single tqdm bar (shared across train+val for the
    epoch) that gets ticked one step per batch.

    Autocast uses bf16 (not fp16): bf16 has the same exponent range as
    fp32, so it can't overflow to inf the way fp16 can with a model this
    size, which removes the need for a GradScaler."""
    train = optimizer is not None
    model.train(mode=train)
    total_loss, total_items = 0.0, 0
    breakdown_sum = {}

    for pos, geometry, condition, target, constants in loader:
        pos, geometry, condition, target, constants = (
            t.to(device, non_blocking=True)
            for t in (pos, geometry, condition, target, constants)
        )
        cond_arg = condition if condition.shape[-1] > 0 else None

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                field_pred, const_pred = model(pos, geometry, cond_arg)
                loss, breakdown = compute_loss(
                    field_pred, target, const_pred, constants, const_weight
                )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = pos.shape[0]
        total_loss += loss.item() * bs
        total_items += bs
        for k, v in breakdown.items():
            breakdown_sum[k] = breakdown_sum.get(k, 0.0) + v * bs
        if pbar is not None:
            pbar.set_postfix_str(f"{phase} loss={loss.item():.4f}", refresh=False)
            pbar.update(1)

    avg = total_loss / total_items
    avg_breakdown = {k: v / total_items for k, v in breakdown_sum.items()}
    return avg, avg_breakdown


def _nonzero_dim(size: int) -> int | None:
    return size if size > 0 else None


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BLENDEDNET_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cache_dir = ensure_cache(BLENDEDNET_CACHE_DIR, BLENDEDNET_CACHE_ARCHIVE)

    # ---------------- data ----------------
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    rng = np.random.default_rng(SEED)
    all_ids = list(manifest["case_ids"])
    rng.shuffle(all_ids)
    n_test = max(1, int(len(all_ids) * TEST_FRACTION))
    n_val = max(1, int((len(all_ids) - n_test) * VAL_FRACTION))
    test_ids = all_ids[:n_test]
    val_ids = all_ids[n_test:n_test + n_val]
    fit_ids = all_ids[n_test + n_val:]

    fit_ds = BlendedNetDataset(cache_dir, case_ids=fit_ids)
    val_ds = BlendedNetDataset(cache_dir, case_ids=val_ids, stats=fit_ds.stats)
    test_ds = BlendedNetDataset(cache_dir, case_ids=test_ids, stats=fit_ds.stats)
    np.savez(BLENDEDNET_CKPT_DIR / "norm_stats.npz", **fit_ds.stats)
    print(f"data: fit={len(fit_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"num_points={manifest.get('num_points')} "
          f"condition_keys={manifest.get('global_feat_keys')} "
          f"const_keys={manifest.get('const_keys')}")

    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    sample_pos, sample_geom, sample_cond, sample_tgt, sample_const = fit_ds[0]
    cond_dim = _nonzero_dim(sample_cond.shape[-1])
    num_constants = _nonzero_dim(sample_const.shape[-1])
    geom_dim = sample_geom.shape[-1]
    out_channels = sample_tgt.shape[-1]

    # ---------------- model ----------------
    model = GeoTransolver(
        space_dim=sample_pos.shape[-1],
        geom_dim=geom_dim,
        cond_dim=cond_dim,
        out_channels=out_channels,
        num_constants=num_constants,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        num_slices=NUM_SLICES,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        local_radii=LOCAL_RADII,
        local_neighbors=LOCAL_NEIGHBORS,
        local_hidden=LOCAL_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: dim={DIM} depth={DEPTH} heads={HEADS} "
          f"dim_head={DIM_HEAD} num_slices={NUM_SLICES} "
          f"geom_dim={geom_dim} cond_dim={cond_dim} out_channels={out_channels} "
          f"num_constants={num_constants} -> {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * max(1, len(fit_loader))
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warmup_steps))
    amp_enabled = AMP and device.type == "cuda"

    start_epoch = 0
    best_val = float("inf")
    if RESUME is not None:
        ckpt = torch.load(RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        tqdm.write(f"resumed from {RESUME} at epoch {start_epoch}, best_val={best_val:.4f}")

    # ---------------- train ----------------
    bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
    for epoch in range(start_epoch, EPOCHS):
        n_steps = len(fit_loader) + len(val_loader)
        with tqdm(total=n_steps, desc=f"epoch {epoch}", mininterval=1.0, leave=True, bar_format=bar_fmt) as pbar:
            train_loss, train_bd = run_epoch(
                model, fit_loader, device, optimizer=optimizer, amp=amp_enabled,
                grad_clip=GRAD_CLIP, pbar=pbar, phase="train", const_weight=CONST_LOSS_WEIGHT,
            )
            for _ in range(len(fit_loader)):
                scheduler.step()
            val_loss, val_bd = run_epoch(
                model, val_loader, device, amp=amp_enabled, pbar=pbar, phase="val",
                const_weight=CONST_LOSS_WEIGHT,
            )

            is_best = val_loss < best_val
            best_val = min(best_val, val_loss)
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(lr=f"{lr_now:.1e}", train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              best="*" if is_best else "")
        if is_best:
            tqdm.write(f"epoch {epoch}: new best val {val_loss:.4f} "
                       f"(train {train_loss:.4f} {train_bd}, val {val_bd})")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "manifest": manifest,
            "cond_dim": cond_dim,
            "num_constants": num_constants,
            "out_channels": out_channels,
            "geom_dim": geom_dim,
        }
        torch.save(ckpt, BLENDEDNET_CKPT_DIR / "last.pt")
        if is_best:
            torch.save(ckpt, BLENDEDNET_CKPT_DIR / "best.pt")

    # ---------------- final held-out test evaluation ----------------
    best = torch.load(BLENDEDNET_CKPT_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    with tqdm(total=len(test_loader), desc="test", mininterval=1.0, leave=True, bar_format=bar_fmt) as pbar:
        test_loss, test_bd = run_epoch(
            model, test_loader, device, amp=amp_enabled, pbar=pbar, phase="test",
            const_weight=CONST_LOSS_WEIGHT,
        )
    tqdm.write(f"final test (best checkpoint, epoch {best['epoch']}): loss {test_loss:.4f} {test_bd}")


if __name__ == "__main__":
    main()
