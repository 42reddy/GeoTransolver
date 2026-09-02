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
    python train.py --cache_dir data/cache --ckpt_dir checkpoints \\
        --epochs 200 --batch_size 4 --lr 1e-3
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shapenet_car_dataset import ShapeNetCarDataset
from transolver import Transolver


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
def run_epoch(model, loader, out_channels, device, optimizer=None, scaler=None, grad_clip=1.0):
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

    avg = total_loss / total_items
    avg_breakdown = {k: v / total_items for k, v in breakdown_sum.items()}
    return avg, avg_breakdown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache_dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint .pt to resume from")

    # optimization
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_fraction", type=float, default=0.1, help="carved out of the train split, not test")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=0)

    # model size ("sufficiently large": ~45-50M params at these defaults)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim_head", type=int, default=64)
    ap.add_argument("--num_slices", type=int, default=64)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.05)

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    rng = np.random.default_rng(args.seed)
    train_ids = list(manifest["train_ids"])
    rng.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * args.val_fraction))
    val_ids, fit_ids = train_ids[:n_val], train_ids[n_val:]
    test_ids = manifest["test_ids"]

    fit_ds = ShapeNetCarDataset(args.cache_dir, case_ids=fit_ids)
    val_ds = ShapeNetCarDataset(args.cache_dir, case_ids=val_ids, stats=fit_ds.stats)
    test_ds = ShapeNetCarDataset(args.cache_dir, case_ids=test_ids, stats=fit_ds.stats)
    np.savez(args.ckpt_dir / "norm_stats.npz", **fit_ds.stats)
    print(f"data: fit={len(fit_ds)} val={len(val_ds)} test={len(test_ds)} mode={manifest['mode']}")

    fit_loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---------------- model ----------------
    out_channels = manifest["out_channels"]
    model = Transolver(
        space_dim=3,
        in_channels=manifest["in_channels"],
        out_channels=out_channels,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        num_slices=args.num_slices,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: dim={args.dim} depth={args.depth} heads={args.heads} "
          f"dim_head={args.dim_head} num_slices={args.num_slices} -> {n_params / 1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(fit_loader))
    warmup_steps = int(total_steps * args.warmup_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch = 0
    best_val = float("inf")
    if args.resume is not None:
        # weights_only=True (the torch>=2.6 default) can't unpickle the plain
        # dicts/lists we store (manifest, args); this is our own just-written
        # checkpoint, not an untrusted download, so full unpickling is safe.
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"resumed from {args.resume} at epoch {start_epoch}, best_val={best_val:.4f}")

    # ---------------- train ----------------
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_bd = run_epoch(model, fit_loader, out_channels, device,
                                          optimizer=optimizer, scaler=scaler, grad_clip=args.grad_clip)
        for _ in range(len(fit_loader)):
            scheduler.step()
        val_loss, val_bd = run_epoch(model, val_loader, out_channels, device)
        dt = time.time() - t0

        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:4d}/{args.epochs} | lr {lr_now:.2e} | "
              f"train {train_loss:.4f} {train_bd} | val {val_loss:.4f} {val_bd} | "
              f"{dt:.1f}s{'  * best' if is_best else ''}")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "manifest": manifest,
        }
        torch.save(ckpt, args.ckpt_dir / "last.pt")
        if is_best:
            torch.save(ckpt, args.ckpt_dir / "best.pt")

    # ---------------- final held-out test evaluation ----------------
    best = torch.load(args.ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_loss, test_bd = run_epoch(model, test_loader, out_channels, device)
    print(f"final test (best checkpoint, epoch {best['epoch']}): loss {test_loss:.4f} {test_bd}")


if __name__ == "__main__":
    main()
