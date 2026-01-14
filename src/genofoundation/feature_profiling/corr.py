import torch

def correlation_pearson(x):
    """计算 n x g 矩阵中 g 个特征的 Pearson 矩阵"""
    # x: (n, g)
    mean = torch.mean(x, dim=0)
    diffs = x - mean
    # 协方差矩阵 (unnormalized)
    cov = torch.mm(diffs.t(), diffs)
    std = torch.sqrt(torch.diag(cov))
    # 归一化
    std_matrix = torch.outer(std, std)
    corr = cov / (std_matrix + 1e-8) # 加 epsilon 防止除零
    return torch.nan_to_num(corr)
    
# def correlation_pearson(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     x: (n, g)
#     return: (g, g) Pearson correlation
#     """
#     if x.ndim != 2:
#         raise ValueError(f"x must be 2D (n,g), got {x.shape}")
#     n = x.shape[0]
#     if n < 2:
#         raise ValueError("Need n>=2 to compute correlation.")

#     x = x.float()
#     xc = x - x.mean(dim=0, keepdim=True)

#     # std (unbiased, consistent with cov/(n-1))
#     denom = torch.sqrt((xc * xc).sum(dim=0) / (n - 1)).clamp_min(eps)
#     xn = xc / denom

#     corr = (xn.T @ xn) / (n - 1)
#     return corr.clamp(-1.0, 1.0)


# def rank_average_ties(x: torch.Tensor) -> torch.Tensor:
#     """
#     Average ranks for ties (like scipy.stats.rankdata(method='average')),
#     computed column-wise for a 2D tensor.

#     x: (n, g)
#     return ranks: (n, g), ranks are 1..n (float32)
#     """
#     if x.ndim != 2:
#         raise ValueError(f"x must be 2D (n,g), got {x.shape}")

#     n, g = x.shape
#     sorted_x, idx = torch.sort(x, dim=0)  # (n,g), idx are original row indices

#     # boundary_start[i,j] = True if sorted_x[i,j] starts a new tie-group
#     boundary_start = torch.ones_like(sorted_x, dtype=torch.bool)
#     boundary_start[1:, :] = sorted_x[1:, :] != sorted_x[:-1, :]

#     # boundary_end[i,j] = True if sorted_x[i,j] ends a tie-group
#     boundary_end = torch.ones_like(sorted_x, dtype=torch.bool)
#     boundary_end[:-1, :] = sorted_x[:-1, :] != sorted_x[1:, :]

#     # positions 0..n-1 as float, shape (n,1) broadcast to (n,g)
#     pos = torch.arange(n, device=x.device, dtype=torch.float32).view(n, 1)
#     neg_inf = torch.tensor(float("-inf"), device=x.device, dtype=torch.float32)

#     # group_start: for each sorted position, the first index of its tie-group
#     start_marks = torch.where(boundary_start, pos, neg_inf)
#     group_start = torch.cummax(start_marks, dim=0).values

#     # group_end: for each sorted position, the last index of its tie-group
#     end_marks = torch.where(boundary_end, pos, neg_inf)
#     group_end = torch.flip(
#         torch.cummax(torch.flip(end_marks, dims=[0]), dim=0).values,
#         dims=[0]
#     )

#     # average rank in sorted order, convert 0-based indices to 1..n ranks
#     avg_rank_sorted = (group_start + group_end) / 2.0 + 1.0  # (n,g)

#     # scatter back to original row order
#     ranks = torch.empty_like(avg_rank_sorted, dtype=torch.float32)
#     ranks.scatter_(0, idx, avg_rank_sorted)
#     return ranks


# def correlation_spearman(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     """
#     Spearman correlation of columns in x (n,g), tie-aware average ranks.
#     """
#     ranks = rank_average_ties(x)
#     return correlation_pearson(ranks, eps=eps)


def correlation_spearman(x):
    """计算 n x g 矩阵中 g 个特征的 Spearman 矩阵"""
    # x: (n, g)
    # 1. 计算 Rank (处理并列排名使用平均秩)
    # argsort().argsort() 可以得到 0 到 n-1 的排名
    # 我们对每个特征独立计算排名
    ranks = x.argsort(dim=0).argsort(dim=0).float()
    
    # 2. 对 Rank 计算 Pearson 即可
    return correlation_pearson(ranks)

def correlation_kendall(x):
    """
    计算 Kendall's Tau 矩阵 (PyTorch 向量化版)
    警告：复杂度为 O(g^2 * n^2)，当 n 或 g 很大时会消耗巨大内存
    """
    n, g = x.shape
    # 结果矩阵
    kendall_matrix = torch.eye(g, device=x.device)
    
    # 为了避免 O(n^2) 的内存爆炸，我们这里对特征对 (g, g) 
    # 如果 g 很大，建议使用循环或 scipy
    for i in range(g):
        for j in range(i + 1, g):
            xi = x[:, i]
            xj = x[:, j]
            
            # 计算所有对 (i, j) 的方向一致性
            # xi[m] - xi[k] 与 xj[m] - xj[k] 的符号
            diff_x = xi.unsqueeze(0) - xi.unsqueeze(1) # (n, n)
            diff_y = xj.unsqueeze(0) - xj.unsqueeze(1) # (n, n)
            
            # 计算一致对和不一致对
            # sign(diff_x * diff_y) > 0 为一致, < 0 为不一致
            concordance = torch.sign(diff_x * diff_y).sum()
            
            tau = concordance / (n * (n - 1))
            kendall_matrix[i, j] = kendall_matrix[j, i] = tau
            
    return kendall_matrix
# Example usage:
#n, g = 100, 10
#data = torch.randn(n, g)
#corr = correlation_kendall(data)
#print(corr.shape) # Should be (10, 10)

def corr(x, method='spearman'):
    """
    Compute a correlation matrix using the specified method.

    Parameters
    ----------
    x : array-like
        Input data for computing correlations.
    method : str, optional
        The correlation method to use. Options are:
        - 'pearson': Pearson product-moment correlation
        - 'spearman': Spearman rank-order correlation (default)
        - 'kendall': Kendall tau correlation

    Returns
    -------
    Correlation matrix computed using the specified method.

    Raises
    ------
    ValueError
        If an unsupported correlation method is specified.
    """
    if method == 'pearson':
        return correlation_pearson(x)
    elif method == 'spearman':
        return correlation_spearman(x)
    elif method == 'kendall':
        return correlation_kendall(x)
    else:
        raise ValueError('Unsupported method')


def corr2dist(corr, flavor='abscorr', quant_clamp=0.90, max_norm=True):
    """
    Convert a correlation matrix to a distance matrix.

    Parameters
    ----------
    corr : torch.Tensor
        A square correlation matrix of shape [n, n] with values in [-1, 1].
    flavor : str, optional
        The distance transformation to apply. Options are:
        - 'abscorr': D = 1 - |corr| (default). Distance based on absolute
          correlation; ranges from 0 (perfectly correlated or anti-correlated)
          to 1 (uncorrelated).
        - 'negfar': D = (1 - corr) / 2. Treats negative correlations as far
          apart; ranges from 0 (corr=1) to 1 (corr=-1).
        - 'angular': D = arccos(corr) / pi. Angular distance in correlation
          space; ranges from 0 to 1.
    quant_clamp: None or float, default 0.90
        If not None, clamp the values by this quantile.
    maxnorm : bool, optional, default True
        If True, normalize the values by the maximum value.
    Returns
    -------
    torch.Tensor
        A distance matrix of shape [n, n] with zeros on the diagonal.
    """
    if flavor == 'abscorr':
        D = 1 - corr.abs()
        D.fill_diagonal_(0)
    elif flavor == 'negfar':
        D = (1 - corr) / 2
        D.fill_diagonal_(0)
    elif flavor == 'angular':
        eps = 1e-7
        R = corr.clamp(-1 + eps, 1 - eps)
        D = torch.acos(R) / torch.pi
        D.fill_diagonal_(0)
    if quant_clamp is not None:
        D = D.clamp( max= torch.quantile(D, quant_clamp) )
    if max_norm:
        D /= D.max()
    return D.to(torch.float64)



import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering
from scipy.spatial.distance import squareform

def plot_dependency_dist(matrix_tensor, title="Matrix Analysis", 
                         figsize=(12, 5), grid_ratio=(1, 1.2), 
                         cmap='coolwarm', bins=80, use_olo=False):
    """
    Creates a single-figure panel containing both the value distribution 
    and the clustered structural heatmap.
    
    Args:
        matrix_tensor (torch.Tensor): Square distance matrix [n, n].
        title (str): Title prefix for subplots.
        figsize (tuple): Overall figure size.
        grid_ratio (tuple): Width ratio between (histogram, heatmap).
        cmap (str): Colormap for the heatmap.
        bins (int): Number of bins for the histogram.
        use_olo (bool): If True, performs Optimal Leaf Ordering. 
                        Warning: O(N^4) complexity, very slow for N > 1000.
    """
    # 1. Data Preparation
    device = matrix_tensor.device
    n = matrix_tensor.shape[0]
    
    # Extract upper triangle for histogram
    idx = torch.triu_indices(n, n, offset=1, device=device)
    vals = matrix_tensor[idx[0], idx[1]]
    vals_np = vals[~torch.isnan(vals)].detach().cpu().numpy()

    # 2. Hierarchical Clustering
    matrix_np = matrix_tensor.detach().cpu().numpy()
    
    # Convert square matrix to condensed form for scipy
    condensed_dist = squareform(matrix_np, checks=False)
    
    # Standard hierarchical clustering (fast)
    Z = linkage(condensed_dist, method='ward')
    
    # Optional Leaf Ordering (slow but beautiful)
    if use_olo:
        print(f"Computing Optimal Leaf Ordering for N={n}... this may take a while.")
        Z = optimal_leaf_ordering(Z, condensed_dist)
    
    new_order = leaves_list(Z)
    clustered_matrix = matrix_np[new_order, :][:, new_order]

    # 3. Plotting the Panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, 
                                   gridspec_kw={'width_ratios': grid_ratio})
    
    # --- Subplot 1: Distribution Histogram ---
    ax1.hist(vals_np, bins=bins, density=True, color='#2c3e50', alpha=0.8)
    ax1.set_title(f"{title}: Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Subplot 2: Clustered Heatmap ---
    # rasterized=True makes the figure much faster to render in the browser/PDF
    sns.heatmap(clustered_matrix, cmap=cmap, ax=ax2, 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Distance'},
                rasterized=True) 
    
    olo_status = "ON" if use_olo else "OFF"
    ax2.set_title(f"{title}: Clustered Structure (OLO={olo_status})")
    ax2.set_xlabel("Reordered Indices")
    ax2.set_ylabel("Reordered Indices")

    plt.tight_layout()
    return fig

# --- Usage Examples ---

# 1. Fast mode for interactive exploration
# fig = plot_dependency_dist(grid_dist, title="Hyperbolic", use_olo=False)

# 2. High-quality mode for results/papers
# fig = plot_dependency_dist(grid_dist, title="Hyperbolic_Final", use_olo=True, figsize=(14, 6))
