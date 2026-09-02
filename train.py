"""
Train Transolver (transolver.py) to predict the converged CFD field on the
ShapeNet-Car dataset, using the cache built by shapenet_car_dataset.py.

--------------------------------------------------------------------------
Task recap
--------------------------------------------------------------------------
--mode full (default, from shapenet_car_dataset.py --mode full):
    in_channels=4  [sdf, dir_x, dir_y, dir_z]
    out_channels=4 [velocity_x, velocity_y, velocity_z, pressure]
    pressure is only a real target at surface points (mask==1); it's
    dummy-zeroed at exterior points, so the loss masks it out there.
    Velocity is valid everywhere (real CFD value off-surface, ~0 at the
    no-slip wall on-surface) so it's never masked.

--mode surface (from shapenet_car_dataset.py --mode surface):
    in_channels=3 [normal_x, normal_y, normal_z]   (built by the surface-
    only fallback; note the "4" in-channel default below only applies to
    full mode -- the manifest tells us the actual dims either way)
    out_channels=1 [pressure], mask is all-True (plain MSE).

Model size is read from CLI flags with "sufficiently large" defaults
(dim=512, depth=12, heads=8, dim_head=64, num_slices=64) -- about 45-50M
parameters, printed at startup so you can judge it against your GPU.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
No CLI flags -- edit the PARAMS block below directly, then run this file
(IDE run button or `python train.py`). Data/checkpoint paths come from
config.py (edit that file to switch between local and Kaggle).
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CACHE_DIR, CKPT_DIR
from shapenet_car_dataset import ShapeNetCarDataset
from transolver import Transolver


# ==========================================================================
# Params -- edit these directly, no CLI flags
# ==========================================================================
RESUME = None          # Path to a checkpoint .pt to resume from, or None

# optimization
EPOCHS = 200
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_FRAC = 0.05
GRAD_CLIP = 1.0
VAL_FRACTION = 0.1     # carved out of the train split, not test
NUM_WORKERS = 2
AMP = True
SEED = 0

# model size ("sufficiently large": ~45-50M params at these defaults)
DIM = 512
DEPTH = 12
HEADS = 8
DIM_HEAD = 64
NUM_SLICES = 64
MLP_RATIO = 4.0
DROPOUT = 0.05


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, out_channels: int):
    """Masked MSE. In full mode (out_channels==4) velocity (channels 0:3) is
    always valid; pressure (channel 3) is only valid where mask==1. In
    surface mode (out_channels==1) mask is all-ones, so this reduces to a
    plain MSE on pressure."""
    m = mask.unsqueeze(-1)  # (B, N, 1)
    denom = m.sum().clamp(min=1.0)

    if out_channels == 4:
        velo_loss = F.mse_loss(pred[..., :3], target[..., :3])
        press_loss = ((pred[..., 3:4] - target[..., 3:4]) ** 2 * m).sum() / denom
        loss = velo_loss + press_loss
        return loss, {"velo_mse": velo_loss.item(), "press_mse": press_loss.item()}

    press_loss = ((pred - target) ** 2 * m).sum() / denom
    return press_loss, {"press_mse": press_loss.item()}


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
def run_epoch(model, loader, out_channels, device, optimizer=None, scaler=None, grad_clip=1.0,
              pbar=None, phase=""):
    """pbar, if given, is a single tqdm bar (shared across train+val for the
    epoch) that gets ticked one step per batch -- keeps the whole epoch on
    one progress line instead of spawning a bar per phase."""
    train = optimizer is not None
    model.train(mode=train)
    total_loss, total_items = 0.0, 0
    breakdown_sum = {}

    for pos, features, target, mask in loader:
        pos, features, target, mask = (t.to(device, non_blocking=True) for t in (pos, features, target, mask))

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                pred = model(pos, features)
                loss, breakdown = compute_loss(pred, target, mask, out_channels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
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


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text())
    rng = np.random.default_rng(SEED)
    train_ids = list(manifest["train_ids"])
    rng.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * VAL_FRACTION))
    val_ids, fit_ids = train_ids[:n_val], train_ids[n_val:]
    test_ids = manifest["test_ids"]

    fit_ds = ShapeNetCarDataset(CACHE_DIR, case_ids=fit_ids)
    val_ds = ShapeNetCarDataset(CACHE_DIR, case_ids=val_ids, stats=fit_ds.stats)
    test_ds = ShapeNetCarDataset(CACHE_DIR, case_ids=test_ids, stats=fit_ds.stats)
    np.savez(CKPT_DIR / "norm_stats.npz", **fit_ds.stats)
    print(f"data: fit={len(fit_ds)} val={len(val_ds)} test={len(test_ds)} mode={manifest['mode']}")

    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ---------------- model ----------------
    out_channels = manifest["out_channels"]
    model = Transolver(
        space_dim=3,
        in_channels=manifest["in_channels"],
        out_channels=out_channels,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        num_slices=NUM_SLICES,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: dim={DIM} depth={DEPTH} heads={HEADS} "
          f"dim_head={DIM_HEAD} num_slices={NUM_SLICES} -> {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * max(1, len(fit_loader))
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")
    if RESUME is not None:
        # weights_only=True (the torch>=2.6 default) can't unpickle the plain
        # dicts/lists we store (manifest, params); this is our own just-
        # written checkpoint, not an untrusted download, so full unpickling
        # is safe.
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
            train_loss, train_bd = run_epoch(model, fit_loader, out_channels, device,
                                              optimizer=optimizer, scaler=scaler, grad_clip=GRAD_CLIP,
                                              pbar=pbar, phase="train")
            for _ in range(len(fit_loader)):
                scheduler.step()
            val_loss, val_bd = run_epoch(model, val_loader, out_channels, device, pbar=pbar, phase="val")

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
        }
        torch.save(ckpt, CKPT_DIR / "last.pt")
        if is_best:
            torch.save(ckpt, CKPT_DIR / "best.pt")

    # ---------------- final held-out test evaluation ----------------
    best = torch.load(CKPT_DIR / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    with tqdm(total=len(test_loader), desc="test", mininterval=1.0, leave=True, bar_format=bar_fmt) as pbar:
        test_loss, test_bd = run_epoch(model, test_loader, out_channels, device, pbar=pbar, phase="test")
    tqdm.write(f"final test (best checkpoint, epoch {best['epoch']}): loss {test_loss:.4f} {test_bd}")


if __name__ == "__main__":
    main()
