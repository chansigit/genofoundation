import torch
import ot
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_ot_assignment(
    corr_dist: torch.Tensor,
    grid_dist: torch.Tensor,
    ot_method: str = 'entropic_srgw',
    epsilon: float = 0.001,
    device: str = 'cuda:0',
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute optimal transport assignment between correlation and grid distances.
    
    Parameters
    ----------
    corr_dist : torch.Tensor
        Correlation distance matrix (n_gene x n_gene).
    grid_dist : torch.Tensor
        Grid distance matrix (n_px x n_px).
    ot_method : str, optional
        OT method to use. Options: 'entropic_gw', 'entropic_srgw'.
        Default is 'entropic_srgw'.
    epsilon : float, optional
        Entropic regularization parameter. Default is 0.001.
    device : str, optional
        Device to run computation on. Default is 'cuda:0'.
    verbose : bool, optional
        Whether to print progress. Default is True.
        
    Returns
    -------
    gene_ind : np.ndarray
        Gene indices from the assignment.
    px_ind : np.ndarray
        Pixel indices from the assignment.
    ot_log : dict
        Log dictionary from OT computation.
    """
    dev = torch.device(device)
    corr_dist = corr_dist.to(device=dev, dtype=torch.float64)
    grid_dist = grid_dist.to(device=dev, dtype=torch.float64)
    
    if ot_method == 'entropic_gw':
        ot_plan, ot_log = ot.gromov.entropic_gromov_wasserstein(
            corr_dist, grid_dist, p=None, q=None, epsilon=epsilon,
            loss_fun='square_loss', solver='PGD', log=True, 
            warmstart=True, verbose=verbose
        )
        if verbose:
            print(f"GW Dist = {ot_log['gw_dist'].item():.4e}")
            
    elif ot_method == 'entropic_srgw':
        ot_plan, ot_log = ot.gromov.entropic_semirelaxed_gromov_wasserstein(
            corr_dist, grid_dist, p=None, epsilon=epsilon,
            loss_fun='square_loss', symmetric=True, 
            verbose=verbose, log=True
        )
        if verbose:
            print(f"SRGW Dist = {ot_log['srgw_dist'].item():.4e}")
    else:
        raise NotImplementedError(f'Unknown OT method name: {ot_method}')
    
    # Convert to numpy and compute assignment
    ot_plan = ot_plan.cpu().numpy()
    n_gene, n_px = ot_plan.shape
    assert n_gene <= n_px, f"n_gene ({n_gene}) must be <= n_px ({n_px})"
    
    gene_ind, px_ind = linear_sum_assignment(ot_plan, maximize=False)
    assert len(np.unique(px_ind)) == n_gene, "Assignment should be unique"
    
    return gene_ind, px_ind, ot_log


def create_feature_images(
    vals_to_fillin: torch.Tensor,
    px_ind: np.ndarray,
    img_height: int,
    img_width: int,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create feature images by filling in values at specified pixel indices.
    
    Parameters
    ----------
    vals_to_fillin : torch.Tensor
        Values to fill into the images, shape (n_samples, n_features).
    px_ind : np.ndarray
        Pixel indices where values should be placed.
    img_height : int
        Height of the output images.
    img_width : int
        Width of the output images.
    dtype : torch.dtype, optional
        Data type for the output tensor. Default is torch.float32.
        
    Returns
    -------
    feature_imgs : torch.Tensor
        Feature images with shape (n_samples, img_height, img_width).
    """
    n_samples = vals_to_fillin.shape[0]
    n_pixels = img_height * img_width
    
    # Ensure input is the correct dtype
    vals_to_fillin = vals_to_fillin.to(dtype)
    
    # Create zero tensor and fill in values at pixel indices
    feature_imgs = torch.zeros(n_samples, n_pixels, dtype=dtype)
    feature_imgs[:, px_ind] = vals_to_fillin
    
    # Reshape to image format
    feature_imgs = feature_imgs.reshape(n_samples, img_height, img_width)
    
    return feature_imgs