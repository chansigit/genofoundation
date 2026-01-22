"""
Convolutional Variational Autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .conv import ConvEncoder, ConvDecoder


class Tensor2Image(nn.Module):
    """
    Maps a 1D tensor to a 2D image using pixel indices.

    Args:
        px_ind: Target pixel indices for the input vector elements.
        img_h: Height of the output image.
        img_w: Width of the output image.
    """
    def __init__(self, px_ind, img_h: int, img_w: int):
        super().__init__()
        self.register_buffer('px_ind', torch.LongTensor(px_ind))
        self.img_h = img_h
        self.img_w = img_w
        self.n_px = img_h * img_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, n_features]

        Returns:
            Image tensor [B, 1, H, W]
        """
        batch_size = x.size(0)
        img = torch.zeros(batch_size, self.n_px, device=x.device, dtype=x.dtype)
        img[:, self.px_ind] = x
        img = img.reshape(batch_size, 1, self.img_h, self.img_w)
        return img


class Image2Tensor(nn.Module):
    """
    Extracts a 1D tensor from a 2D image using pixel indices.

    Args:
        px_ind: Pixel indices to extract from the flattened image.
    """
    def __init__(self, px_ind):
        super().__init__()
        self.register_buffer('px_ind', torch.LongTensor(px_ind))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Image tensor [B, 1, H, W]

        Returns:
            Extracted tensor [B, n_features]
        """
        batch_size = img.size(0)
        img_flat = img.reshape(batch_size, -1)
        return img_flat[:, self.px_ind]


class ConvVAE(nn.Module):
    """
    Modern Conditional Variational Autoencoder (CVAE).
    Composes Tensor2Image, ConvEncoder, ConvDecoder, and Image2Tensor modules.

    Args:
        px_ind (LongTensor): Target pixel indices for the gene vector elements.
        img_h (int): Height of the internal feature image.
        img_w (int): Width of the internal feature image.
        latent_dim (int): Dimension of the bottleneck latent space.
        condition_dim (int, optional): Dimension of the condition vector.
        base_channels (int): Initial filter count for convolutions.
        num_res_blocks (int): Number of ResBlock stages.
        max_channels (int): Maximum channel cap for ResBlocks.
        groups (int): Number of groups for GroupNorm.
        up_mode (str): Upsampling strategy ('pixelshuffle' or 'nearest').
    """
    def __init__(
        self,
        px_ind,
        img_h: int,
        img_w: int,
        latent_dim: int,
        condition_dim: Optional[int] = None,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        max_channels: int = 512,
        groups: int = 32,
        up_mode: str = 'pixelshuffle'
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Tensor <-> Image transformations
        self.tensor2image = Tensor2Image(px_ind, img_h, img_w)
        self.image2tensor = Image2Tensor(px_ind)

        # Encoder
        self.encoder_module = ConvEncoder(
            img_h=img_h,
            img_w=img_w,
            latent_dim=latent_dim,
            in_channels=1,
            condition_dim=condition_dim,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            max_channels=max_channels,
            groups=groups
        )

        # Decoder
        self.decoder_module = ConvDecoder(
            img_h=img_h,
            img_w=img_w,
            spatial_shape=self.encoder_module.spatial_shape,
            latent_dim=latent_dim,
            out_channels=1,
            condition_dim=condition_dim,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            max_channels=max_channels,
            groups=groups,
            up_mode=up_mode
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x: Input gene vector [B, n_features]
            condition: Optional condition vector [B, condition_dim]

        Returns:
            Dict with 'reconstruction', 'latent', 'mu', 'log_var', 'reconstructed_img'.
        """
        # Transform to image
        img = self.tensor2image(x)

        # Encode
        enc_out = self.encoder_module(img, condition)
        mu, log_var = enc_out['mu'], enc_out['log_var']

        # Sample latent
        z = self.reparameterize(mu, log_var)

        # Decode to image
        recon_img = self.decoder_module(z, condition)

        # Transform back to tensor
        reconstruction = self.image2tensor(recon_img)

        return {
            "reconstruction": reconstruction,
            "latent": z,
            "mu": mu,
            "log_var": log_var,
            "reconstructed_img": recon_img,
        }

    def sample(
        self,
        num_samples: int,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Generate new samples by sampling from the latent prior N(0, 1).

        Args:
            num_samples: Number of samples to generate.
            condition: Optional conditioning tensor.
            device: Device to generate samples on.

        Returns:
            Generated samples as tensor [B, n_features].
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)
        recon_img = self.decoder_module(z, condition)
        return self.image2tensor(recon_img)

    def loss(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + beta * KL.
        Both terms are normalized by feature dimension and averaged over batch.

        Args:
            x: ground-truth input, shape [B, n_features]
            outputs: dict with keys {'reconstruction', 'mu', 'log_var'}
            beta: weight on KL term

        Returns:
            Dict with keys: 'total', 'reconstruction', 'kl'
        """
        recon = outputs["reconstruction"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]

        # Number of features for normalization
        Nf = recon[0].numel()

        # Reconstruction loss: sum over features, mean over batch, normalize by Nf
        recon_loss = F.mse_loss(recon, x, reduction="none")
        recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(1).mean() / Nf

        # KL divergence: sum over latent dims, mean over batch, normalize by Nf
        kl_loss = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(1).mean() / Nf

        total = recon_loss + beta * kl_loss

        return {"total": total, "reconstruction": recon_loss, "kl": kl_loss}
