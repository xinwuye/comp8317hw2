import torch

from .hilbert import encode as hilbert_encode_
from .z_order import xyz2key as z_order_encode_


def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks / (values.numel() - 1)


def _estimate_point_curvature(
    coord: torch.Tensor, batch: torch.Tensor, knn_k: int, eps: float
) -> torch.Tensor:
    import pointops

    point_curvature = torch.zeros(coord.shape[0], device=coord.device, dtype=torch.float32)
    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    counts = torch.bincount(batch, minlength=batch_size)
    start = 0
    for count in counts.tolist():
        end = start + count
        if count < 3:
            start = end
            continue
        sample_coord = coord[start:end].contiguous()
        nsample = min(knn_k + 1, count)
        if nsample < 3:
            start = end
            continue
        sample_offset = torch.tensor([count], device=coord.device, dtype=torch.int32)
        idx, _ = pointops.knn_query(nsample, sample_coord, sample_offset)
        local_points = sample_coord[idx.long()]
        centered = local_points - local_points.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / max(nsample - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov)
        point_curvature[start:end] = eigvals[:, 0].clamp_min(0) / eigvals.sum(
            dim=1
        ).clamp_min(eps)
        start = end
    return point_curvature


def _z_order_encode(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    return z_order_encode_(x, y, z, b=None, depth=depth)


def _hilbert_encode(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


@torch.inference_mode()
def gahs_encode(
    grid_coord: torch.Tensor,
    coord: torch.Tensor,
    batch: torch.Tensor = None,
    depth: int = 16,
    order: str = "gahs",
    serialization_cfg=None,
):
    assert order in {"gahs", "gahs-trans"}
    assert coord is not None, "GAHS requires point coordinates for curvature estimation."

    cfg = {} if serialization_cfg is None else dict(serialization_cfg)
    knn_k = int(cfg.get("knn_k", 16))
    density_weight = float(cfg.get("density_weight", 0.5))
    curvature_weight = float(cfg.get("curvature_weight", 0.5))
    hilbert_quantile = float(cfg.get("hilbert_quantile", 0.7))
    fine_bits_cap = int(cfg.get("fine_bits_cap", 6))
    eps = float(cfg.get("eps", 1e-6))

    if batch is None:
        batch = torch.zeros(grid_coord.shape[0], device=grid_coord.device, dtype=torch.long)
    else:
        batch = batch.long()

    if order == "gahs-trans":
        grid_coord = grid_coord[:, [1, 0, 2]]
        coord = coord[:, [1, 0, 2]]

    grid_coord = grid_coord.int().contiguous()
    coord = coord.float().contiguous()

    if grid_coord.shape[0] == 0:
        return torch.zeros(0, device=grid_coord.device, dtype=torch.long)

    fine_bits = min(fine_bits_cap, max(1, depth - 1))
    coarse_bits = depth - fine_bits

    coarse_coord = grid_coord >> fine_bits
    fine_coord = grid_coord & ((1 << fine_bits) - 1)

    if coarse_bits > 0:
        global_code = _z_order_encode(coarse_coord, depth=coarse_bits)
    else:
        global_code = torch.zeros(grid_coord.shape[0], device=grid_coord.device, dtype=torch.long)

    block_key = global_code | (batch << (coarse_bits * 3))
    _, block_inverse, block_counts = torch.unique(
        block_key, sorted=True, return_inverse=True, return_counts=True
    )

    point_curvature = _estimate_point_curvature(coord, batch, knn_k=knn_k, eps=eps)
    block_curvature_sum = torch.zeros(
        block_counts.shape[0], device=coord.device, dtype=torch.float32
    )
    block_curvature_sum.index_add_(0, block_inverse, point_curvature)
    block_curvature = block_curvature_sum / block_counts.to(torch.float32).clamp_min(1)

    point_order = torch.argsort(block_inverse)
    idx_ptr = torch.cat(
        [block_counts.new_zeros(1), torch.cumsum(block_counts, dim=0)], dim=0
    )
    head_indices = point_order[idx_ptr[:-1]]
    block_batch = batch[head_indices]
    block_density = block_counts.to(torch.float32)

    density_rank = torch.zeros_like(block_density)
    curvature_rank = torch.zeros_like(block_curvature)
    block_use_hilbert = torch.zeros(
        block_counts.shape[0], device=grid_coord.device, dtype=torch.bool
    )

    for sample_id in torch.unique(block_batch, sorted=True):
        sample_mask = block_batch == sample_id
        density_rank[sample_mask] = _rank_normalize(block_density[sample_mask])
        curvature_rank[sample_mask] = _rank_normalize(block_curvature[sample_mask])
        block_score = (
            density_weight * density_rank[sample_mask]
            + curvature_weight * curvature_rank[sample_mask]
        )
        threshold = torch.quantile(block_score, hilbert_quantile)
        block_use_hilbert[sample_mask] = block_score >= threshold

    local_code_z = _z_order_encode(fine_coord, depth=fine_bits)
    local_code_h = _hilbert_encode(fine_coord, depth=fine_bits)
    local_code = torch.where(block_use_hilbert[block_inverse], local_code_h, local_code_z)

    return (global_code << (fine_bits * 3)) | local_code
