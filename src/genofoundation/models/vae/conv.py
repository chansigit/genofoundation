"""
Convolutional encoder/decoder modules for VAE.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class ResBlockDown(nn.Module):
    """
    Residual Block for the Encoder path.
    Supports configurable stride to allow for feature extraction without spatial downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=2, groups=32):
        super().__init__()

        # --- Main Path ---
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # --- Shortcut Path ---
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        return self.act(out + identity)


class ResBlockUp(nn.Module):
    """
    Residual Block for the Decoder path.
    Uses PixelShuffle (Sub-pixel convolution) to increase resolution without checkerboard effects.
    """
    def __init__(self, in_channels, out_channels, up_mode='pixelshuffle', groups=32):
        super().__init__()
        self.up_mode = up_mode
        self.act = nn.SiLU(inplace=True)

        # --- Main Path ---
        if up_mode == 'pixelshuffle':
            self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
            self.up = nn.PixelShuffle(2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # --- Shortcut Path ---
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        if self.up_mode == 'pixelshuffle':
            out = self.up(self.conv1(x))
        else:
            out = self.conv1(self.up(x))

        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return self.act(out + identity)


class ConvEncoder(nn.Module):
    """
    Convolutional encoder that processes 2D images and outputs latent distribution parameters.

    Args:
        img_h: Height of the input image.
        img_w: Width of the input image.
        latent_dim: Dimension of the bottleneck latent space.
        in_channels: Number of input image channels. Defaults to 1.
        condition_dim: Dimension of the condition vector (injected as spatial channels).
        base_channels: Initial filter count for convolutions.
        num_res_blocks: Number of ResBlock stages.
        max_channels: Maximum channel cap for ResBlocks.
        groups: Number of groups for GroupNorm.
    """
    def __init__(
        self,
        img_h: int,
        img_w: int,
        latent_dim: int,
        in_channels: int = 1,
        condition_dim: Optional[int] = None,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        max_channels: int = 512,
        groups: int = 32
    ):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.in_channels = in_channels
        self.condition_dim = condition_dim if condition_dim is not None else 0

        # Input channels = image channels + condition channels (if any)
        total_in_channels = in_channels + self.condition_dim
        self.initial_conv = nn.Conv2d(total_in_channels, base_channels, kernel_size=3, padding=1, bias=False)

        # Downsampling ResBlocks
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for _ in range(num_res_blocks):
            stride = 2 if current_channels < max_channels else 1
            next_channels = min(current_channels * 2, max_channels)
            self.down_blocks.append(ResBlockDown(current_channels, next_channels, stride=stride, groups=groups))
            current_channels = next_channels

        # 1x1 Convolutional Bottleneck
        self.bottleneck_1x1 = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, current_channels),
            nn.SiLU(inplace=True)
        )

        # Dynamic Shape Calculation
        self.spatial_shape, self.flat_features = self._get_bottleneck_shape(current_channels, total_in_channels)

        # Final Latent Projections
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_bottleneck_shape(self, channels, in_channels):
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, self.img_h, self.img_w)
            h = self.initial_conv(dummy_x)
            for block in self.down_blocks:
                h = block(h)
            h = self.bottleneck_1x1(h)
            return h.shape[1:], h.numel()

    def forward(self, img: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            img: Input image [B, in_channels, H, W]
            condition: Optional condition vector [B, condition_dim]

        Returns:
            Dict with 'mu' and 'log_var' keys.
        """
        # Inject condition as spatial channels
        if self.condition_dim > 0 and condition is not None:
            cond_spatial = condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.img_h, self.img_w)
            img = torch.cat([img, cond_spatial], dim=1)

        # Feature extraction
        h = self.initial_conv(img)
        for block in self.down_blocks:
            h = block(h)
        h = self.bottleneck_1x1(h)

        # Latent mapping
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)

        return {"mu": mu, "log_var": logvar}


class ConvDecoder(nn.Module):
    """
    Convolutional decoder that reconstructs 2D images from latent vectors.

    Args:
        img_h: Height of the output image.
        img_w: Width of the output image.
        spatial_shape: Shape of the bottleneck feature map (C, H, W).
        latent_dim: Dimension of the latent space.
        out_channels: Number of output image channels. Defaults to 1.
        condition_dim: Dimension of the condition vector (concatenated to latent).
        base_channels: Base filter count for convolutions.
        num_res_blocks: Number of ResBlock stages.
        max_channels: Maximum channel cap (unused, for symmetry with encoder).
        groups: Number of groups for GroupNorm.
        up_mode: Upsampling strategy ('pixelshuffle' or 'nearest').
    """
    def __init__(
        self,
        img_h: int,
        img_w: int,
        spatial_shape: tuple,
        latent_dim: int,
        out_channels: int = 1,
        condition_dim: Optional[int] = None,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        max_channels: int = 512,
        groups: int = 32,
        up_mode: str = 'pixelshuffle'
    ):
        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.spatial_shape = spatial_shape
        self.flat_features = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        self.condition_dim = condition_dim if condition_dim is not None else 0

        # Latent + Condition to Feature Map Projection
        input_dim = latent_dim + self.condition_dim
        self.fc_input = nn.Linear(input_dim, self.flat_features)

        self.bottleneck_1x1 = nn.Sequential(
            nn.Conv2d(spatial_shape[0], spatial_shape[0], kernel_size=1, bias=False),
            nn.GroupNorm(groups, spatial_shape[0]),
            nn.SiLU(inplace=True)
        )

        # Upsampling ResBlocks
        self.up_blocks = nn.ModuleList()
        current_channels = spatial_shape[0]
        for _ in range(num_res_blocks):
            next_channels = max(current_channels // 2, base_channels)
            mode = up_mode if current_channels != next_channels else 'none'
            self.up_blocks.append(ResBlockUp(current_channels, next_channels, up_mode=mode, groups=groups))
            current_channels = next_channels

        self.final_upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Softplus()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: Latent vector [B, latent_dim]
            condition: Optional condition vector [B, condition_dim]

        Returns:
            Reconstructed image [B, out_channels, H, W]
        """
        # Concatenate condition to latent vector
        if self.condition_dim > 0 and condition is not None:
            z = torch.cat([z, condition], dim=1)

        # Project and reshape
        h = self.fc_input(z)
        h = h.view(-1, *self.spatial_shape)
        h = self.bottleneck_1x1(h)

        # Progressive upsampling
        for block in self.up_blocks:
            h = block(h)

        # Final output
        h = self.final_upsample(h)
        img = self.final_conv(h)

        return img
