import torch


def differentiable_cylinder(cylinder_params, patch_size=96, sharpness=200) :
    half_patch = patch_size / 2
    B = cylinder_params.shape[0]
    device = cylinder_params.device

    # Unpack and decode parameters
    radius = cylinder_params[:, 0] * 1.5 + 5.5           # (B,)
    length = cylinder_params[:, 1] * 10.0 + 35.0         # (B,)
    direction = cylinder_params[:, 2:5]                  # (B, 3)
    centroid = cylinder_params[:, 5:] * half_patch       # (B, 3)

    # Normalize direction
    direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)  # (B, 3)

    # Create voxel grid (D, H, W, 3)
    coords = torch.linspace(0, patch_size - 1, patch_size, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')  # (D, H, W)
    zz = zz.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 0].view(B, 1, 1, 1)  # B x D x H x W
    yy = yy.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 1].view(B, 1, 1, 1)
    xx = xx.unsqueeze(0).expand(B, -1, -1, -1) - centroid[:, 2].view(B, 1, 1, 1)
    grid = torch.stack([xx, yy, zz], dim=-1)  # (B, D, H, W, 3)
    v = grid - half_patch  # shift origin to center

    # Project v onto direction: scalar projection
    projection = (v * direction.view(B, 1, 1, 1, 3)).sum(dim=-1)  # (B, D, H, W)

    # Perpendicular distance squared
    v_squared = (v ** 2).sum(dim=-1)  # (B, D, H, W)
    proj_squared = projection ** 2   # (B, D, H, W)
    perp_dist_sq = v_squared - proj_squared  # (B, D, H, W)

    # Length and radius soft masks
    radius_sq = radius.view(B, 1, 1, 1) ** 2
    length_half = length.view(B, 1, 1, 1) / 2.0

    soft_radial_mask = torch.sigmoid((radius_sq - perp_dist_sq) * sharpness)
    soft_axial_mask = torch.sigmoid((length_half - torch.abs(projection)) * sharpness)

    # Final mask
    mask = soft_radial_mask * soft_axial_mask  # (B, D, H, W)

    return mask.unsqueeze(1)
