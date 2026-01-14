import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_lattice_dmat(h, w, device='cpu', method='euc', max_norm=True, dtype=torch.float32):
    """
    Generates a distance matrix (ground cost matrix) for a 2D grid.
    
    Motivation:
    In spatial transcriptomics, computer vision, or Optimal Transport (OT) tasks, 
    we need to quantify the "cost" of moving mass between grid points. This function
    provides metrics for different underlying geometries:
    1. 'euc': Standard flat Euclidean plane.
    2. 'periodic_euc': Toroidal topology where edges wrap around, useful for 
       eliminating boundary artifacts in periodic data.
    3. 'hyperbolic': Poincaré Disk model, which is effective for capturing 
       hierarchical relationships or tree-like structures in data.
    4. 'radial_diff': Radial distance difference - measures the absolute difference
       between two pixels' distances to the grid center. Useful for radial symmetry
       analysis or ring-based structural detection.

    Args:
        h (int): Grid height.
        w (int): Grid width.
        device (str/torch.device): Computation device.
        method (str): Metric type ('euc', 'periodic_euc', 'hyperbolic').
        max_norm (bool): If True, scales the matrix to [0, 1].
        dtype (torch.dtype): Data type for the output matrix.
    """
    
    if not isinstance(h, int) or not isinstance(w, int):
        raise TypeError("Grid dimensions h and w must be integers.")
        
    eps = 1e-8  # Small constant to prevent division by zero or sqrt(0)
    
    # 1. Coordinate Construction
    # Create a coordinate grid of shape [h*w, 2]
    x = torch.arange(h, device=device, dtype=dtype)
    y = torch.arange(w, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # Shape: [N, 2]
    
    # 2. Distance Computation
    if method == 'euc':
        # Euclidean distance using PyTorch's optimized cdist
        # cdist is numerically stable for p=2
        dist_matrix = torch.cdist(coords, coords, p=2)

    elif method == 'periodic_euc':
        # Implementation: Torus geometry
        # Uses the modulo trick: d_periodic = ((diff + L/2) % L) - L/2
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # [N, N, 2]
        dims = torch.tensor([h, w], device=device, dtype=dtype)
        
        # Map differences to the range [-size/2, size/2]
        diff = (diff + dims / 2) % dims - dims / 2
        dist_matrix = torch.sqrt((diff ** 2).sum(dim=-1) + eps)

    elif method == 'hyperbolic':
        # Implementation: Poincaré Disk Model
        # Hyperbolic distance is highly sensitive to precision; use float64 temporarily.
        orig_dtype = dtype
        coords_64 = coords.to(torch.float64)
        
        # Center coordinates and map into the unit disk
        # Scale to radius 0.85 to avoid the "infinite distance" at the boundary (r=1)
        coords_norm = coords_64 - coords_64.mean(dim=0)
        scale = coords_norm.abs().max() + eps
        coords_p = (coords_norm / scale) * 0.85
        
        # Calculate squared norms and squared distances
        sq_norm = (coords_p ** 2).sum(dim=-1)  # [N]
        sq_dist = ((coords_p.unsqueeze(1) - coords_p.unsqueeze(0)) ** 2).sum(dim=-1)  # [N, N]
        
        # Poincaré distance formula: 
        # d(x,y) = arcosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        denom = (1.0 - sq_norm.unsqueeze(1)) * (1.0 - sq_norm.unsqueeze(0))
        arg = 1.0 + 2.0 * sq_dist / (denom + eps)
        
        # Stability: clamp input to acosh to be strictly >= 1.0
        dist_matrix = torch.acosh(torch.clamp(arg, min=1.0 + eps))
        dist_matrix = dist_matrix.to(orig_dtype)

    elif method == 'radial_diff':
        # Radial distance difference: |r_i - r_j| where r_i is distance from pixel i to center
        # Pixels on the same concentric ring have distance 0
        # Center coordinates (handling even/odd dimensions like the original)
        center_x = (h - 1) / 2.0
        center_y = (w - 1) / 2.0
        center = torch.tensor([center_x, center_y], device=device, dtype=dtype)
        
        # Compute radial distance for each pixel
        radial_dist = torch.sqrt(((coords - center) ** 2).sum(dim=1) + eps)  # [N]
        
        # Pairwise absolute difference of radial distances
        dist_matrix = torch.abs(radial_dist.unsqueeze(1) - radial_dist.unsqueeze(0))  # [N, N]
    else:
        raise ValueError(f"Unknown method: {method}")

    # 3. Normalization
    if max_norm:
        max_val = dist_matrix.max()
        if max_val > 0:
            dist_matrix = dist_matrix / max_val
            
    return dist_matrix.to(torch.float64)


# def plot_grid_dist(grid_dist, h, w, source_pt=(0, 0), method_name="Distance"):
#     """
#     Combines three visualizations:
#     1. Full Distance Matrix Heatmap
#     2. Distance Distribution Histogram
#     3. Spatial Distance Heatmap from a specific source point
#     """
#     # 1. Preparation
#     grid_dist_cpu = grid_dist.detach().cpu()
#     n_total = grid_dist_cpu.shape[0]
#     i, j = source_pt
    
#     # Validation: Ensure source point is within grid bounds
#     if i >= h or j >= w:
#         raise ValueError(f"Source point ({i}, {j}) is out of grid bounds ({h}, {w})")

#     # Set up the figure with three subplots
#     fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
#     sns.set_style("white")

#     # --- Subplot 1: Full Distance Matrix [N x N] ---
#     # This shows the relationship between all pairs of flattened indices
#     im1 = axes[0].imshow(grid_dist_cpu, cmap='bwr')
#     fig.colorbar(im1, ax=axes[0], shrink=0.7)
#     axes[0].set_title(f'Full {method_name} Matrix\n(Shape: {n_total}x{n_total})')
#     axes[0].set_xlabel('Point Index (flattened)')
#     axes[0].set_ylabel('Point Index (flattened)')

#     # --- Subplot 2: Distance Distribution (Histogram) ---
#     # Extracting upper triangle to avoid redundant pairs and the zero-diagonal
#     idx = torch.triu_indices(n_total, n_total, offset=1)
#     vals = grid_dist_cpu[idx[0], idx[1]].numpy()
#     vals = vals[~np.isnan(vals)]
    
#     axes[1].hist(vals, bins=50, density=True, color='#69b3a2', edgecolor='white', alpha=0.8)
#     axes[1].set_title(f'Normalized Distance Distribution\n(Density Plot)')
#     axes[1].set_xlabel('Distance Value')
#     axes[1].set_ylabel('Density')

#     # --- Subplot 3: Spatial Heatmap from Source Point ---
#     # Reshaping a single row of the matrix back to the grid dimensions (h, w)
#     target_idx = i * w + j  # Mapping 2D (i, j) to 1D index
#     distance_grid = grid_dist_cpu[target_idx].reshape(h, w).numpy()
    
#     sns.heatmap(distance_grid, cmap="YlGnBu", ax=axes[2], 
#                 cbar_kws={'label': 'Distance from Source'})
#     # Mark the source point with a red dot
#     axes[2].scatter(j + 0.5, i + 0.5, color='red', s=100, edgecolors='white', label='Source Point')
#     axes[2].set_title(f'Spatial Map: Distances from ({i}, {j})\nGrid Size: {h}x{w}')
#     axes[2].set_xlabel("Grid Column (w)")
#     axes[2].set_ylabel("Grid Row (h)")
#     axes[2].legend()

#     plt.tight_layout()
#     plt.show()

# --- Example Usage ---
# h = 45; w = 45
# grid_dist = get_gridpt_dmat(h, w, method='hyperbolic', device='cuda:0')
# plot_comprehensive_grid_dist(grid_dist, h, w, source_pt=(0, 22), method_name="Hyperbolic")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_grid_dist(grid_dist, h, w, source_pt=(0, 0), 
                         method_name="Distance", 
                         figsize=(20, 5.5), 
                         width_ratios=(1, 1, 1.2)):
    """
    Comprehensive 3-panel visualization for grid distances:
    1. Full NxN Distance Matrix.
    2. Distance Distribution (Histogram).
    3. Spatial Distance Map from a specific source point.

    Args:
        grid_dist (torch.Tensor): [h*w, h*w] distance matrix.
        h, w (int): Grid dimensions.
        source_pt (tuple): (row, col) for the spatial heatmap focus.
        method_name (str): Label for the distance metric used.
        figsize (tuple): Overall figure dimensions (width, height).
        width_ratios (tuple): Relative widths of the three subplots.
    """
    # 1. Preparation and Validation
    grid_dist_cpu = grid_dist.detach().cpu()
    n_total = grid_dist_cpu.shape[0]
    i, j = source_pt
    
    if i >= h or j >= w:
        raise ValueError(f"Source point {source_pt} is outside grid bounds ({h}, {w})")

    # 2. Setup Figure with Configurable Ratios
    fig, axes = plt.subplots(1, 3, figsize=figsize, 
                             gridspec_kw={'width_ratios': width_ratios})
    sns.set_style("white")

    # --- Subplot 1: Full Distance Matrix [N x N] ---
    # Visualizes the raw cost/distance relationship across all flattened indices
    im1 = axes[0].imshow(grid_dist_cpu.numpy(), cmap='coolwarm_r')
    fig.colorbar(im1, ax=axes[0], shrink=0.6, label=method_name)
    axes[0].set_title(f'Full {method_name} Matrix\n(Shape: {n_total}x{n_total})')
    axes[0].set_xlabel('Point Index (flattened)')
    axes[0].set_ylabel('Point Index (flattened)')

    # --- Subplot 2: Distance Distribution (Histogram) ---
    # Using upper triangle to avoid redundant pairs and the zero-diagonal
    idx = torch.triu_indices(n_total, n_total, offset=1)
    vals = grid_dist_cpu[idx[0], idx[1]].numpy()
    vals = vals[~np.isnan(vals)]
    
    axes[1].hist(vals, bins=60, density=True, color='#69b3a2', edgecolor='white', alpha=0.8)
    axes[1].set_title(f'Normalized {method_name} Dist\n(Density Plot)')
    axes[1].set_xlabel('Distance Value')
    axes[1].set_ylabel('Density')
    axes[1].grid(axis='y', linestyle='--', alpha=0.3)

    # --- Subplot 3: Spatial Heatmap from Source Point ---
    # Maps the distance vector back to the physical h x w grid geometry
    target_idx = i * w + j  # Mapping 2D (i, j) to 1D flattened index
    distance_grid = grid_dist_cpu[target_idx].reshape(h, w).numpy()
    
    sns.heatmap(distance_grid, cmap="YlGnBu", ax=axes[2], 
                cbar_kws={'label': f'{method_name} from Source'})
    
    # Overlay the source point for reference
    axes[2].scatter(j + 0.5, i + 0.5, color='red', s=100, 
                    edgecolors='white', linewidth=1.5, label='Source Point')
    
    axes[2].set_title(f'Spatial Map: Distances from ({i}, {j})\nGrid Size: {h}x{w}')
    axes[2].set_xlabel("Grid Column (w)")
    axes[2].set_ylabel("Grid Row (h)")
    axes[2].legend(loc='upper right', frameon=True)

    plt.tight_layout()
    return fig

# --- Example Usage ---
# h, w = 45, 45
# grid_dist = get_gridpt_dmat(h, w, method='hyperbolic', device='cuda')
# fig = plot_grid_dist_panel(grid_dist, h, w, source_pt=(0, 22), 
#                            figsize=(22, 6), width_ratios=(1, 1, 1.3))
# plt.show()
