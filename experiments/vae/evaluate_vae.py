"""
VAE Model Evaluation Script

Unified evaluation framework for all VAE models supporting:
- Model loading from checkpoint (best_model.pt)
- Latent variable extraction (z, mu, logvar)
- UMAP visualization
- Latent space quality metrics

Usage:
    # Evaluate a single model
    python evaluate_vae.py --checkpoint outputs/ffnvae_tms/best_model.pt --model_type ffn

    # Evaluate with UMAP visualization
    python evaluate_vae.py --checkpoint outputs/vae_tms/best_model.pt --model_type simple --umap

    # Batch evaluation of multiple models
    python evaluate_vae.py --batch_dir outputs/hparam_search --model_type simple
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LatentAnalysis:
    """Container for latent space analysis results."""
    z: np.ndarray           # Sampled latent vectors [N, latent_dim]
    mu: np.ndarray          # Mean vectors [N, latent_dim]
    logvar: np.ndarray      # Log variance [N, latent_dim]

    # Optional metadata
    labels: Optional[np.ndarray] = None  # Class labels if available

    @property
    def std(self) -> np.ndarray:
        """Standard deviation from logvar."""
        return np.exp(0.5 * self.logvar)

    @property
    def var(self) -> np.ndarray:
        """Variance from logvar."""
        return np.exp(self.logvar)


@dataclass
class EvaluationResult:
    """Container for all evaluation results."""
    checkpoint_path: str
    model_type: str
    config: dict

    # Loss metrics
    val_loss: float
    best_val_loss: float
    recon_loss: float
    kl_loss: float

    # Latent space metrics
    latent_dim: int
    active_dims: int          # Dimensions with meaningful variance
    mean_kl_per_dim: np.ndarray
    posterior_collapse_ratio: float

    # Optional
    epoch: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            'checkpoint_path': self.checkpoint_path,
            'model_type': self.model_type,
            'val_loss': self.val_loss,
            'best_val_loss': self.best_val_loss,
            'recon_loss': self.recon_loss,
            'kl_loss': self.kl_loss,
            'latent_dim': self.latent_dim,
            'active_dims': self.active_dims,
            'posterior_collapse_ratio': self.posterior_collapse_ratio,
            'epoch': self.epoch,
            'config': self.config,
        }


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    device: torch.device = torch.device('cpu'),
    input_dim: Optional[int] = None,
) -> Tuple[nn.Module, dict]:
    """
    Load a VAE model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (best_model.pt)
        model_type: 'simple' for SimpleVAE, 'ffn' for FFN VAE, 'conv' for ConvVAE
        device: Device to load model on
        input_dim: Override input dimension (useful if not in config)

    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Get model dimensions
    state_dict = checkpoint['model_state_dict']

    if model_type == 'simple':
        from genofoundation.models.vae.vanilla_vae import SimpleVAE

        # Infer dimensions from state dict
        encoder_weight = state_dict.get('encoder.0.weight')
        if encoder_weight is not None:
            inferred_input_dim = encoder_weight.shape[1]
        else:
            inferred_input_dim = input_dim or 17099  # Default TMS features

        # Infer latent dim from decoder
        decoder_weight = state_dict.get('decoder.0.weight')
        if decoder_weight is not None:
            latent_dim = decoder_weight.shape[1]
        else:
            latent_dim = config.get('latent_dim', 64)

        model = SimpleVAE(input_dim=inferred_input_dim, latent_dim=latent_dim)

    elif model_type == 'ffn':
        from genofoundation.models.vae.ffn_vae import VAE

        # Infer dimensions from state dict
        encoder_first_layer = None
        for key in state_dict.keys():
            if key.startswith('encoder_module.layers.0'):
                encoder_first_layer = key
                break

        if encoder_first_layer:
            # FFN has complex structure, try to get from config
            pass

        # Build from config if available
        inferred_input_dim = input_dim or 17099
        latent_dim = config.get('latent_dim', 128)

        # Try to get hidden dims from config or infer
        encoder_hidden = config.get('encoder_hidden_dims', [256, 128])
        decoder_hidden = config.get('decoder_hidden_dims', [128, 256])

        model = VAE(
            input_dim=inferred_input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden,
            decoder_hidden_dims=decoder_hidden,
            activation=config.get('activation', 'silu'),
            use_layer_norm=config.get('use_layer_norm', True),
            output_activation=config.get('output_activation', 'softplus'),
        )

    elif model_type == 'conv':
        from genofoundation.models.vae.conv_vae import ConvVAE
        raise NotImplementedError("ConvVAE loading not yet implemented")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def get_model_type_from_path(checkpoint_path: str) -> str:
    """Infer model type from checkpoint path."""
    path_lower = checkpoint_path.lower()
    if 'ffnvae' in path_lower or 'ffn_vae' in path_lower:
        return 'ffn'
    elif 'convvae' in path_lower or 'conv_vae' in path_lower:
        return 'conv'
    else:
        return 'simple'


# ============================================================================
# Latent Variable Extraction
# ============================================================================

def extract_latents(
    model: nn.Module,
    dataloader: DataLoader,
    model_type: str,
    device: torch.device = torch.device('cpu'),
) -> LatentAnalysis:
    """
    Extract latent variables from all data in dataloader.

    Args:
        model: VAE model in eval mode
        dataloader: DataLoader with input data
        model_type: 'simple' or 'ffn'
        device: Device for computation

    Returns:
        LatentAnalysis containing z, mu, logvar
    """
    model.eval()

    all_z = []
    all_mu = []
    all_logvar = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            # Get encoder outputs - handle different model interfaces
            if model_type == 'simple':
                mu, logvar = model.encode(x)
            else:  # ffn or conv
                enc_out = model.encode(x)
                mu = enc_out['mu']
                logvar = enc_out['log_var']

            # Sample z (use mean during eval for determinism)
            z = mu  # or use reparameterize for stochastic

            all_z.append(z.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

    return LatentAnalysis(
        z=np.concatenate(all_z, axis=0),
        mu=np.concatenate(all_mu, axis=0),
        logvar=np.concatenate(all_logvar, axis=0),
    )


# ============================================================================
# Latent Space Metrics
# ============================================================================

def compute_latent_metrics(latents: LatentAnalysis) -> Dict[str, float]:
    """
    Compute various metrics for latent space quality.

    Args:
        latents: LatentAnalysis object

    Returns:
        Dictionary of metrics
    """
    mu = latents.mu
    logvar = latents.logvar
    var = latents.var

    latent_dim = mu.shape[1]

    # KL divergence per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu**2 - np.exp(logvar))
    mean_kl_per_dim = kl_per_dim.mean(axis=0)  # [latent_dim]

    # Active dimensions (KL > threshold indicates information is encoded)
    kl_threshold = 0.1  # Commonly used threshold
    active_dims = (mean_kl_per_dim > kl_threshold).sum()

    # Posterior collapse ratio
    collapse_threshold = 0.01
    collapsed_dims = (mean_kl_per_dim < collapse_threshold).sum()
    posterior_collapse_ratio = collapsed_dims / latent_dim

    # Variance statistics
    mean_var_per_dim = var.mean(axis=0)  # [latent_dim]

    # Mutual information gap (MIG) - simplified version
    # Higher variance in mu indicates more information encoded
    mu_var_per_dim = mu.var(axis=0)

    # AU (Active Units) - units that encode information
    # A unit is active if variance of its mean across data > threshold
    au_threshold = 0.01
    active_units = (mu_var_per_dim > au_threshold).sum()

    return {
        'latent_dim': latent_dim,
        'active_dims': int(active_dims),
        'active_units': int(active_units),
        'collapsed_dims': int(collapsed_dims),
        'posterior_collapse_ratio': float(posterior_collapse_ratio),
        'total_kl': float(mean_kl_per_dim.sum()),
        'mean_kl_per_active_dim': float(mean_kl_per_dim[mean_kl_per_dim > kl_threshold].mean()) if active_dims > 0 else 0.0,
        'mean_posterior_var': float(mean_var_per_dim.mean()),
        'mean_mu_var': float(mu_var_per_dim.mean()),
        'kl_per_dim': mean_kl_per_dim.tolist(),
    }


def compute_reconstruction_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    model_type: str,
    device: torch.device = torch.device('cpu'),
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    Compute reconstruction and KL loss on dataset.

    Returns:
        Dictionary with 'recon_loss', 'kl_loss', 'total_loss'
    """
    model.eval()

    total_recon = 0.0
    total_kl = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            outputs = model(x)

            # Handle different output formats
            if model_type == 'simple':
                mu = outputs['mean']
                logvar = outputs['logvar']
            else:
                mu = outputs['mu']
                logvar = outputs['log_var']

            recon = outputs['reconstruction']

            # Compute losses (normalized by feature dim)
            n_features = x.shape[1]

            # Reconstruction loss
            recon_loss = ((recon - x) ** 2).sum(dim=1).mean() / n_features

            # KL loss
            kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean() / n_features

            batch_size = x.shape[0]
            total_recon += recon_loss.item() * batch_size
            total_kl += kl_loss.item() * batch_size
            total_samples += batch_size

    avg_recon = total_recon / total_samples
    avg_kl = total_kl / total_samples

    return {
        'recon_loss': avg_recon,
        'kl_loss': avg_kl,
        'total_loss': avg_recon + beta * avg_kl,
    }


# ============================================================================
# UMAP Visualization
# ============================================================================

def create_umap_visualization(
    latents: LatentAnalysis,
    output_path: str,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Space UMAP",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    subsample: Optional[int] = 10000,
) -> str:
    """
    Create UMAP visualization of latent space.

    Args:
        latents: LatentAnalysis object
        output_path: Path to save figure
        labels: Optional labels for coloring
        title: Plot title
        n_neighbors: UMAP parameter
        min_dist: UMAP parameter
        metric: Distance metric
        subsample: Max samples to plot (for performance)

    Returns:
        Path to saved figure
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn: pip install umap-learn")

    z = latents.mu  # Use mu for more stable visualization

    # Subsample if needed
    if subsample is not None and z.shape[0] > subsample:
        indices = np.random.choice(z.shape[0], subsample, replace=False)
        z = z[indices]
        if labels is not None:
            labels = labels[indices]

    print(f"Running UMAP on {z.shape[0]} samples with {z.shape[1]} dimensions...")

    # Fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embedding = reducer.fit_transform(z)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1],
            c=labels, cmap='tab20', alpha=0.6, s=5
        )
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(
            embedding[:, 0], embedding[:, 1],
            alpha=0.5, s=5, c='steelblue'
        )

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved UMAP visualization to {output_path}")
    return output_path


def create_latent_analysis_plots(
    latents: LatentAnalysis,
    metrics: Dict[str, float],
    output_dir: str,
    prefix: str = "",
) -> List[str]:
    """
    Create multiple diagnostic plots for latent space analysis.

    Returns:
        List of paths to saved figures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1. KL per dimension plot
    fig, ax = plt.subplots(figsize=(12, 4))
    kl_per_dim = np.array(metrics['kl_per_dim'])
    ax.bar(range(len(kl_per_dim)), kl_per_dim, color='steelblue', alpha=0.7)
    ax.axhline(y=0.1, color='red', linestyle='--', label='Active threshold (0.1)')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean KL Divergence')
    ax.set_title(f'KL Divergence per Dimension (Active: {metrics["active_dims"]}/{metrics["latent_dim"]})')
    ax.legend()
    path = output_dir / f"{prefix}kl_per_dim.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(str(path))

    # 2. Posterior variance distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    mean_var = latents.var.mean(axis=0)
    ax.bar(range(len(mean_var)), mean_var, color='coral', alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle='--', label='Prior variance')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Mean Posterior Variance')
    ax.set_title('Posterior Variance per Dimension')
    ax.legend()
    path = output_dir / f"{prefix}posterior_var.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(str(path))

    # 3. Mu distribution (variance of means)
    fig, ax = plt.subplots(figsize=(10, 4))
    mu_var = latents.mu.var(axis=0)
    ax.bar(range(len(mu_var)), mu_var, color='seagreen', alpha=0.7)
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Variance of Mean')
    ax.set_title('Variance of Posterior Means per Dimension')
    path = output_dir / f"{prefix}mu_variance.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(str(path))

    # 4. Latent space statistics summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram of mu values
    axes[0].hist(latents.mu.flatten(), bins=100, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Posterior Means (μ)')

    # Histogram of logvar values
    axes[1].hist(latents.logvar.flatten(), bins=100, alpha=0.7, color='coral')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Log Variances')

    plt.tight_layout()
    path = output_dir / f"{prefix}latent_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    saved_paths.append(str(path))

    return saved_paths


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_model(
    checkpoint_path: str,
    data_path: str,
    model_type: str = 'auto',
    device: str = 'cuda',
    batch_size: int = 256,
    output_dir: Optional[str] = None,
    create_umap: bool = False,
    umap_subsample: int = 10000,
) -> EvaluationResult:
    """
    Complete evaluation of a VAE model.

    Args:
        checkpoint_path: Path to checkpoint (best_model.pt)
        data_path: Path to data file (.pt with X_train, X_test)
        model_type: 'simple', 'ffn', 'conv', or 'auto' to infer
        device: 'cuda' or 'cpu'
        batch_size: Batch size for evaluation
        output_dir: Directory for saving plots
        create_umap: Whether to create UMAP visualization
        umap_subsample: Max samples for UMAP

    Returns:
        EvaluationResult with all metrics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Auto-detect model type
    if model_type == 'auto':
        model_type = get_model_type_from_path(checkpoint_path)
    print(f"Model type: {model_type}")

    # Load data
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    X_test = data['X_test']
    n_features = data['n_features']

    # Get labels if available
    labels = data.get('y_test', None)
    if labels is not None:
        labels = labels.numpy() if torch.is_tensor(labels) else labels

    print(f"Test samples: {X_test.shape[0]}, Features: {n_features}")

    # Create dataloader
    test_dataset = TensorDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path, model_type, device, input_dim=n_features
    )
    config = checkpoint.get('config', {})

    # Extract latents
    print("Extracting latent representations...")
    latents = extract_latents(model, test_loader, model_type, device)
    latents.labels = labels

    # Compute metrics
    print("Computing latent space metrics...")
    latent_metrics = compute_latent_metrics(latents)

    print("Computing reconstruction metrics...")
    beta = config.get('beta', 1.0)
    recon_metrics = compute_reconstruction_metrics(model, test_loader, model_type, device, beta)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent / 'evaluation'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create plots
    print("Creating diagnostic plots...")
    prefix = Path(checkpoint_path).stem + "_"
    plot_paths = create_latent_analysis_plots(latents, latent_metrics, output_dir, prefix)

    # Create UMAP if requested
    if create_umap:
        umap_path = output_dir / f"{prefix}umap.png"
        title = f"Latent Space UMAP ({model_type.upper()} VAE)"
        create_umap_visualization(
            latents, str(umap_path), labels=labels, title=title,
            subsample=umap_subsample
        )
        plot_paths.append(str(umap_path))

    # Build result
    result = EvaluationResult(
        checkpoint_path=str(checkpoint_path),
        model_type=model_type,
        config=config,
        val_loss=recon_metrics['total_loss'],
        best_val_loss=checkpoint.get('best_val_loss', float('inf')),
        recon_loss=recon_metrics['recon_loss'],
        kl_loss=recon_metrics['kl_loss'],
        latent_dim=latent_metrics['latent_dim'],
        active_dims=latent_metrics['active_dims'],
        mean_kl_per_dim=np.array(latent_metrics['kl_per_dim']),
        posterior_collapse_ratio=latent_metrics['posterior_collapse_ratio'],
        epoch=checkpoint.get('epoch', 0),
    )

    # Save results
    results_path = output_dir / f"{prefix}metrics.json"
    with open(results_path, 'w') as f:
        json.dump({
            **result.to_dict(),
            **latent_metrics,
            **recon_metrics,
        }, f, indent=2)
    print(f"Saved metrics to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model type: {model_type}")
    print(f"Epoch: {result.epoch}")
    print(f"Best val loss: {result.best_val_loss:.4f}")
    print(f"Recon loss: {result.recon_loss:.6f}")
    print(f"KL loss: {result.kl_loss:.6f}")
    print(f"Latent dim: {result.latent_dim}")
    print(f"Active dims: {result.active_dims} ({100*result.active_dims/result.latent_dim:.1f}%)")
    print(f"Posterior collapse ratio: {result.posterior_collapse_ratio:.2%}")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)

    return result


def batch_evaluate(
    checkpoints_dir: str,
    data_path: str,
    model_type: str = 'simple',
    device: str = 'cuda',
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Batch evaluate multiple models in a directory.

    Args:
        checkpoints_dir: Directory containing model subdirectories
        data_path: Path to data file
        model_type: Model type for all models
        device: Device
        output_csv: Path to save results CSV

    Returns:
        DataFrame with results for all models
    """
    checkpoints_dir = Path(checkpoints_dir)

    # Find all best_model.pt files
    checkpoint_paths = list(checkpoints_dir.glob("*/best_model.pt"))
    print(f"Found {len(checkpoint_paths)} models to evaluate")

    results = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n[{i+1}/{len(checkpoint_paths)}] Evaluating {ckpt_path.parent.name}...")
        try:
            result = evaluate_model(
                str(ckpt_path), data_path, model_type, device,
                create_umap=False  # Skip UMAP for batch
            )
            result_dict = result.to_dict()
            result_dict['model_name'] = ckpt_path.parent.name
            results.append(result_dict)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    df = pd.DataFrame(results)

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nSaved batch results to {output_csv}")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE models')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--data', type=str,
                       default='/scratch/users/chensj16/projects/genofoundation/data/tms/tms_preprocessed.pt',
                       help='Path to data file')
    parser.add_argument('--model_type', type=str, default='auto',
                       choices=['auto', 'simple', 'ffn', 'conv'],
                       help='Model type')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--umap', action='store_true',
                       help='Create UMAP visualization')
    parser.add_argument('--umap_subsample', type=int, default=10000,
                       help='Max samples for UMAP')

    # Batch mode
    parser.add_argument('--batch_dir', type=str, default=None,
                       help='Directory for batch evaluation')
    parser.add_argument('--output_csv', type=str, default=None,
                       help='Output CSV for batch results')

    args = parser.parse_args()

    if args.batch_dir:
        batch_evaluate(
            args.batch_dir, args.data, args.model_type,
            args.device, args.output_csv
        )
    elif args.checkpoint:
        evaluate_model(
            args.checkpoint, args.data, args.model_type, args.device,
            args.batch_size, args.output_dir, args.umap, args.umap_subsample
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
