"""Radius-restricted k-nearest-neighbor search, chunked over queries to
bound peak memory to O(chunk * N) instead of O(N^2). Pure PyTorch, no
compiled/CUDA extension.
"""

import torch


def ball_query(pos: torch.Tensor, radius: float, k: int, chunk_size: int = 2048) -> torch.Tensor:
    """
    pos: (B, N, 3) query and key positions (same point set).
    returns idx: (B, N, k) long, indices of the k nearest points within
    `radius`; slots with fewer than k valid neighbors are padded with the
    query point's own index (always a valid neighbor, distance 0).
    """
    B, N, _ = pos.shape
    idx = torch.empty(B, N, k, dtype=torch.long, device=pos.device)
    with torch.autocast(pos.device.type, enabled=False):
        pos = pos.float()
        radius_sq = radius ** 2
        pos_sq = (pos ** 2).sum(dim=-1)  # (B, N)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = pos[:, start:end]
            chunk_sq = pos_sq[:, start:end]
            
            # dist_sq = x^2 + y^2 - 2xy
            dist_sq = chunk_sq.unsqueeze(2) + pos_sq.unsqueeze(1) - 2 * torch.bmm(chunk, pos.transpose(1, 2))
            
            dist_sq = dist_sq.masked_fill(dist_sq > radius_sq, float("inf"))
            knn_dist_sq, knn_idx = torch.topk(dist_sq, k, dim=-1, largest=False)
            self_idx = torch.arange(start, end, device=pos.device).view(1, -1, 1).expand(B, -1, k)
            idx[:, start:end] = torch.where(torch.isinf(knn_dist_sq), self_idx, knn_idx)
    return idx


def gather_neighbors(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """feat: (B, N, C), idx: (B, N, K) -> (B, N, K, C)"""
    B, N, K = idx.shape
    C = feat.shape[-1]
    idx_flat = idx.reshape(B, N * K, 1).expand(-1, -1, C)
    gathered = torch.gather(feat, 1, idx_flat)
    return gathered.reshape(B, N, K, C)
