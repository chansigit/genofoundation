import torch
import torch.nn as nn
from typing import Dict, Optional, List, Union
import torch.nn.functional as F


class ResBlockDown(nn.Module):
    """
    Residual Block for the Encoder path.
    Supports configurable stride to allow for feature extraction without spatial downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=2, groups=32):
        super().__init__()
        
        # --- Main Path ---
        # Stride determines if we downsample (stride=2) or keep resolution (stride=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # --- Shortcut Path ---
        # Match dimensions based on the chosen stride
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
            # Expand channels (out_channels * 4) so PixelShuffle can fold them into spatial dimensions
            self.conv1 = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False)
            self.up = nn.PixelShuffle(2) # Resolution x2, Channels /4
        else:
            # Fallback to Nearest Neighbor + Conv
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        # --- Shortcut Path ---
        # Use simple interpolation + 1x1 Conv to match the main branch's output
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


# class ConvEncoder(nn.Module):
#     """
#     Advanced Encoder that maps 1D gene vectors to a 2D grid, processes them through 
#     configurable ResBlocks with channel capping and spatial preservation logic.
    
#     Args:
#         px_ind (array-like): Target pixel indices for the input vector elements.
#         img_h (int): Height of the initial constructed feature image.
#         img_w (int): Width of the initial constructed feature image.
#         latent_dim (int): Dimension of the bottleneck latent space.
#         base_channels (int): Initial number of convolutional filters.
#         num_res_blocks (int): Total number of ResBlockDown stages.
#         max_channels (int): Maximum allowed number of channels (cap).
#         groups (int): Number of groups for GroupNorm.
#     """
#     def __init__(
#         self, 
#         px_ind, 
#         img_h, 
#         img_w, 
#         latent_dim, 
#         base_channels=64, 
#         num_res_blocks=2, 
#         max_channels=512,
#         groups=32
#     ):
#         super().__init__()
        
#         # 1. Spatial Mapping Config
#         # register_buffer ensures px_ind moves with the model to GPU/CPU
#         self.register_buffer('px_ind', torch.LongTensor(px_ind))
#         self.img_h = img_h
#         self.img_w = img_w
#         self.n_px = img_h * img_w
        
#         # 2. Initial Convolution (1 channel -> base_channels)
#         self.initial_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1, bias=False)
        
#         # 3. Configurable Downsampling ResBlocks with Spatial Preservation
#         self.down_blocks = nn.ModuleList()
#         current_channels = base_channels
        
#         for _ in range(num_res_blocks):
#             if current_channels < max_channels:
#                 # Still doubling: increase channels and downsample (stride=2)
#                 next_channels = min(current_channels * 2, max_channels)
#                 stride = 2
#             else:
#                 # Cap reached: keep channels and spatial resolution constant (stride=1)
#                 next_channels = current_channels
#                 stride = 1
                
#             self.down_blocks.append(
#                 ResBlockDown(current_channels, next_channels, stride=stride, groups=groups)
#             )
#             current_channels = next_channels

#         # 4. 1x1 Convolutional Bottleneck (Pre-latent aggregation)
#         self.bottleneck_1x1 = nn.Sequential(
#             nn.Conv2d(current_channels, current_channels, kernel_size=1, bias=False),
#             nn.GroupNorm(groups, current_channels),
#             nn.SiLU(inplace=True)
#         )
            
#         # 5. Dynamic Shape Calculation
#         # Run a dummy pass to get exact spatial dimensions for the Decoder
#         self.spatial_shape, self.flat_features = self._get_bottleneck_shape(current_channels)
        
#         # 6. Final Latent Projections
#         self.latent_dim = latent_dim
#         self.fc_mu = nn.Linear(self.flat_features, latent_dim)
#         self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

#     def _get_bottleneck_shape(self, channels):
#         """
#         Performs a dry run to determine the exact output shape of the convolutional layers.
#         """
#         with torch.no_grad():
#             dummy_x = torch.zeros(1, 1, self.img_h, self.img_w)
#             h = self.initial_conv(dummy_x)
#             for block in self.down_blocks:
#                 h = block(h)
#             h = self.bottleneck_1x1(h)
            
#             spatial_shape = h.shape[1:] # (Channels, H_final, W_final)
#             flat_features = h.numel()
#             return spatial_shape, flat_features

#     def forward(self, x):
#         """
#         Forward pass converting gene vector to latent distribution parameters.
        
#         Args:
#             x (Tensor): Input feature vector [Batch, n_px]
            
#         Returns:
#             dict: Latent parameters "mu" and "log_var".
#         """
#         batch_size = x.size(0)

#         # Step 1: Mapping to 2D Grid
#         # Create a blank canvas and scatter gene values into pixel positions
#         h = torch.zeros(batch_size, self.n_px, device=x.device)
#         h[:, self.px_ind] = x 
#         h = h.reshape(batch_size, 1, self.img_h, self.img_w)
        
#         # Step 2: Feature Extraction and Downsampling
#         h = self.initial_conv(h)
#         for block in self.down_blocks:
#             h = block(h)
            
#         # Step 3: 1x1 Convolutional Bottleneck
#         h = self.bottleneck_1x1(h)
            
#         # Step 4: Flattening and Latent Projection
#         h = torch.flatten(h, start_dim=1)
        
#         return {
#             "mu": self.fc_mu(h),
#             "log_var": torch.clamp(self.fc_logvar(h), max=10)
#         }

class ConvEncoder(nn.Module):
    """
    Encoder that maps gene vectors to a 2D grid and injects condition vectors 
    as additional spatial channels.
    
    Args:
        px_ind (LongTensor): Target pixel indices for the input vector elements.
        img_h, img_w (int): Feature image dimensions.
        latent_dim (int): Dimension of the bottleneck latent space.
        cond_dim (int): Dimension of the condition vector. Set to None/0 for standard VAE.
        ... [other args same as before]
    """
    def __init__(
        self, px_ind, img_h, img_w, latent_dim, 
        cond_dim=None, base_channels=64, num_res_blocks=2, 
        max_channels=512, groups=32
    ):
        super().__init__()
        
        self.register_buffer('px_ind', torch.LongTensor(px_ind))
        self.img_h, self.img_w = img_h, img_w
        self.n_px = img_h * img_w
        self.cond_dim = cond_dim if cond_dim is not None else 0
        
        # 1. Initial Convolution
        # Input channels = 1 (genes) + cond_dim (condition channels)
        in_channels = 1 + self.cond_dim
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        
        # 2. Configurable Downsampling ResBlocks
        self.down_blocks = nn.ModuleList()
        current_channels = base_channels
        for _ in range(num_res_blocks):
            stride = 2 if current_channels < max_channels else 1
            next_channels = min(current_channels * 2, max_channels)
            self.down_blocks.append(ResBlockDown(current_channels, next_channels, stride=stride, groups=groups))
            current_channels = next_channels

        # 3. 1x1 Convolutional Bottleneck
        self.bottleneck_1x1 = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, current_channels),
            nn.SiLU(inplace=True)
        )
            
        # 4. Dynamic Shape Calculation
        self.spatial_shape, self.flat_features = self._get_bottleneck_shape(current_channels)
        
        # 5. Final Latent Projections
        self.latent_dim = latent_dim
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

    def _get_bottleneck_shape(self, channels):
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1 + self.cond_dim, self.img_h, self.img_w)
            h = self.initial_conv(dummy_x)
            for block in self.down_blocks: h = block(h)
            h = self.bottleneck_1x1(h)
            return h.shape[1:], h.numel()

    def forward(self, x, cond=None):
        batch_size = x.size(0)

        # Step 1: Mapping to 2D Grid [B, 1, H, W]
        h = torch.zeros(batch_size, self.n_px, device=x.device)
        h[:, self.px_ind] = x 
        h = h.reshape(batch_size, 1, self.img_h, self.img_w)
        
        # Step 2: Inject Condition as Spatial Channels [B, cond_dim, H, W]
        if self.cond_dim > 0 and cond is not None:
            cond_spatial = cond.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.img_h, self.img_w)
            h = torch.cat([h, cond_spatial], dim=1) # Concatenate along Channel dimension
        
        # Step 3: Feature Extraction
        h = self.initial_conv(h)
        for block in self.down_blocks: h = block(h)
        h = self.bottleneck_1x1(h)
            
        # Step 4: Latent Mapping
        h = torch.flatten(h, start_dim=1)
        return {
            "mu": self.fc_mu(h),
            "log_var": torch.clamp(self.fc_logvar(h), min=-10, max=10)
        }

# class ConvDecoder(nn.Module):
#     """
#     Decoder that reconstructs a 2D feature map and extracts gene values 
#     using spatial indices.
    
#     Args:
#         px_ind (LongTensor): Target pixel indices for the gene vector elements.
#         img_h (int): Target height of the reconstructed feature image.
#         img_w (int): Target width of the reconstructed feature image.
#         spatial_shape (tuple): The (C, H, W) shape of the bottleneck in the Encoder.
#         latent_dim (int): Dimension of the bottleneck latent space.
#         base_channels (int): Initial number of filters (matching Encoder start).
#         num_res_blocks (int): Total number of ResBlockUp stages.
#         max_channels (int): Maximum channels allowed in the bottleneck.
#         groups (int): Number of groups for GroupNorm.
#         up_mode (str): Upsampling strategy ('pixelshuffle' or 'nearest').
#     """
#     def __init__(
#         self, 
#         px_ind,
#         img_h,
#         img_w,
#         spatial_shape, 
#         latent_dim, 
#         base_channels=64, 
#         num_res_blocks=2, 
#         max_channels=512,
#         groups=32,
#         up_mode='pixelshuffle'
#     ):
#         super().__init__()
        
#         # 1. Spatial and Indexing Config
#         self.register_buffer('px_ind', torch.LongTensor(px_ind))
#         self.img_h = img_h
#         self.img_w = img_w
#         self.n_px = img_h * img_w
#         self.spatial_shape = spatial_shape
#         self.flat_features = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        
#         # 2. Latent to Feature Map Projection
#         self.fc_input = nn.Linear(latent_dim, self.flat_features)
        
#         # 3. 1x1 Convolutional Refiner
#         self.bottleneck_1x1 = nn.Sequential(
#             nn.Conv2d(spatial_shape[0], spatial_shape[0], kernel_size=1, bias=False),
#             nn.GroupNorm(groups, spatial_shape[0]),
#             nn.SiLU(inplace=True)
#         )
        
#         # 4. Configurable Upsampling ResBlocks
#         self.up_blocks = nn.ModuleList()
#         current_channels = spatial_shape[0]
        
#         # We determine the channel transition logic to match the Encoder's downsampling
#         for i in range(num_res_blocks):
#             # Target channels for the next layer (moving toward the output)
#             next_channels = max(current_channels // 2, base_channels)
            
#             # If channels change, we upsample. If they stay the same, we preserve resolution.
#             mode = up_mode if current_channels != next_channels else 'none'
            
#             self.up_blocks.append(
#                 ResBlockUp(current_channels, next_channels, up_mode=mode, groups=groups)
#             )
#             current_channels = next_channels

#         # 5. Final Image Reconstruction
#         # Ensure the final output matches exactly [B, 1, img_h, img_w]
#         # We use a final Interpolation layer to catch any rounding discrepancies
#         self.final_upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear', align_corners=False)
#         self.final_conv = nn.Sequential(
#             nn.Conv2d(current_channels, 1, kernel_size=3, padding=1, bias=False),
#             nn.Softplus() # Use Sigmoid for [0, 1] normalized gene data
#         )

#     def forward(self, z):
#         """
#         Forward pass converting latent vector back to gene vector.
        
#         Args:
#             z (Tensor): Latent vector [Batch, latent_dim]
            
#         Returns:
#             dict: Containing "reconstructed_img" and "reconstructed_genes".
#         """
#         batch_size = z.size(0)

#         # Step 1: Project and Reshape
#         h = self.fc_input(z)
#         h = h.view(-1, *self.spatial_shape)
        
#         # Step 2: Initial Refinement
#         h = self.bottleneck_1x1(h)
        
#         # Step 3: Progressive Upsampling
#         for block in self.up_blocks:
#             h = block(h)
            
#         # Step 4: Spatial Correction and Convolution
#         # Ensures output is exactly (img_h, img_w)
#         h = self.final_upsample(h)
#         img = self.final_conv(h) # Shape: [B, 1, img_h, img_w]
        
#         # Step 5: Gene Extraction via Indexing
#         # Flatten the image spatially and gather values at px_ind
#         img_flat = img.reshape(batch_size, -1)
#         genes = img_flat[:, self.px_ind]
        
#         return {
#             "reconstructed_img": img,
#             "reconstructed_genes": genes
#         }


class ConvDecoder(nn.Module):
    """
    Decoder that reconstructs gene vectors by concatenating condition 
    to the latent vector z.
    """
    def __init__(
        self, px_ind, img_h, img_w, spatial_shape, latent_dim,
        cond_dim=None, base_channels=64, num_res_blocks=2,
        max_channels=512, groups=32, up_mode='pixelshuffle', return_img= False
    ):
        super().__init__()
        
        self.register_buffer('px_ind', torch.LongTensor(px_ind))
        self.img_h, self.img_w = img_h, img_w
        self.spatial_shape = spatial_shape
        self.flat_features = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        self.cond_dim = cond_dim if cond_dim is not None else 0
        
        # Latent + Condition to Feature Map Projection
        # Input size is z + condition
        input_dim = latent_dim + self.cond_dim
        self.fc_input = nn.Linear(input_dim, self.flat_features)
        
        self.bottleneck_1x1 = nn.Sequential(
            nn.Conv2d(spatial_shape[0], spatial_shape[0], kernel_size=1, bias=False),
            nn.GroupNorm(groups, spatial_shape[0]),
            nn.SiLU(inplace=True)
        )
        
        # Up-blocks logic (mirrors Encoder)
        self.up_blocks = nn.ModuleList()
        current_channels = spatial_shape[0]
        for i in range(num_res_blocks):
            next_channels = max(current_channels // 2, base_channels)
            mode = up_mode if current_channels != next_channels else 'none'
            self.up_blocks.append(ResBlockUp(current_channels, next_channels, up_mode=mode, groups=groups))
            current_channels = next_channels

        self.final_upsample = nn.Upsample(size=(img_h, img_w), mode='bilinear', align_corners=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Softplus() 
        )
        self.return_img = return_img

    def forward(self, z, cond=None):
        batch_size = z.size(0)

        # Step 1: Concatenate condition to the latent vector [B, latent_dim + cond_dim]
        if self.cond_dim > 0 and cond is not None:
            z = torch.cat([z, cond], dim=1)
        
        # Step 2: Project and Reshape
        h = self.fc_input(z)
        h = h.view(-1, *self.spatial_shape)
        h = self.bottleneck_1x1(h)
        
        # Step 3: Progressive Upsampling
        for block in self.up_blocks: h = block(h)
            
        # Step 4: Final output
        h = self.final_upsample(h)
        img = self.final_conv(h)
        
        img_flat = img.reshape(batch_size, -1)
        genes = img_flat[:, self.px_ind]
        
        return {
            "reconstructed_img": img if self.return_img else None,
            "reconstructed_genes": genes
        }


class ConvVAE(nn.Module):
    """
    Modern Conditional Variational Autoencoder (CVAE).
    Integrates specialized ResBlocks, spatial conditioning, and 
    automated shape alignment between Encoder and Decoder.
    
    Args:
        px_ind (LongTensor): Target pixel indices for the gene vector elements.
        img_h (int): Height of the internal feature image.
        img_w (int): Width of the internal feature image.
        latent_dim (int): Dimension of the bottleneck latent space.
        cond_dim (int, optional): Dimension of the condition vector. Defaults to None.
        base_channels (int): Initial filter count for convolutions. Defaults to 64.
        num_res_blocks (int): Number of ResBlock stages. Defaults to 2.
        max_channels (int): Maximum channel cap for ResBlocks. Defaults to 512.
        groups (int): Number of groups for GroupNorm. Defaults to 32.
        up_mode (str): Upsampling strategy ('pixelshuffle' or 'nearest').
    """
    def __init__(
        self, 
        px_ind, 
        img_h, 
        img_w, 
        latent_dim, 
        cond_dim=None,
        base_channels=64, 
        num_res_blocks=2, 
        max_channels=512,
        groups=32,
        up_mode='pixelshuffle'
    ):
        super().__init__()
        
        # 1. Initialize Encoder
        # The encoder calculates its own 'spatial_shape' during init
        self.encoder = ConvEncoder(
            px_ind=px_ind,
            img_h=img_h,
            img_w=img_w,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            max_channels=max_channels,
            groups=groups
        )
        
        # 2. Initialize Decoder
        # We pass the encoder's 'spatial_shape' to ensure perfect reconstruction symmetry
        self.decoder = ConvDecoder(
            px_ind=px_ind,
            img_h=img_h,
            img_w=img_w,
            spatial_shape=self.encoder.spatial_shape,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            max_channels=max_channels,
            groups=groups,
            up_mode=up_mode
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + std * epsilon.
        Allows gradients to flow through stochastic sampling.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, condition=None):
        """
        Forward pass for training.
        
        Args:
            x (Tensor): Input gene vector [B, n_px]
            cond (Tensor, optional): Condition vector [B, cond_dim].
            
        Returns:
            dict: Reconstructed data and latent distribution parameters.
        """
        # Step 1: Encode to latent distribution
        enc_out = self.encoder(x, condition)
        mu, log_var = enc_out['mu'], enc_out['log_var']
        
        # Step 2: Sample latent vector z
        z = self.reparameterize(mu, log_var)
        
        # Step 3: Decode z back to feature space and genes
        dec_out = self.decoder(z, condition)
        
        return {
            "reconstructed_img": dec_out['reconstructed_img'],
            "reconstructed_genes": dec_out['reconstructed_genes'],
            "z": z,
            "mu": mu,
            "log_var": log_var
        }

    def sample(self, num_samples, device, cond=None):
        """
        Generate new gene samples by sampling from the latent prior N(0, 1).
        
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device to run the generation on.
            cond (Tensor, optional): Condition vector for generation.
        """
        z = torch.randn(num_samples, self.encoder.latent_dim).to(device)
        return self.decoder(z, cond=cond)


    def loss(
            self,
            x: torch.Tensor,
            outputs: Dict[str, torch.Tensor],
            reduction: str = "mean",
            beta: float = 1.0,
            logvar_clamp: tuple = (-10.0, 10.0),
        ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction (MSE) + beta * KL Divergence.
        """
        recon = outputs["reconstructed_genes"]
        mu = outputs["mu"]
        log_var = outputs["log_var"]
    
        # ---- Reconstruction term (per-sample sum) ----
        # Using elementwise MSE on the extracted gene vectors
        recon_elwise = F.mse_loss(recon, x, reduction="none")           # [B, n_px]
        recon_per = recon_elwise.view(recon_elwise.size(0), -1).sum(1)  # [B]
    
        # ---- KL term (per-sample sum) ----
        # Clamping log_var for numerical stability as requested
        log_var = torch.clamp(log_var, min=logvar_clamp[0], max=logvar_clamp[1])
        var = torch.exp(log_var)
        
        # KL(N(mu, sigma^2) || N(0, I))
        kl_elwise = -0.5 * (1.0 + log_var - mu.pow(2) - var)            # [B, latent_dim]
        kl_per = kl_elwise.view(kl_elwise.size(0), -1).sum(1)           # [B]
    
        # ---- Aggregate across batch ----
        if reduction == "mean":
            recon_loss = recon_per.mean()
            kl_loss = kl_per.mean()
        elif reduction == "sum":
            recon_loss = recon_per.sum()
            kl_loss = kl_per.sum()
        elif reduction == "none":
            recon_loss = recon_per
            kl_loss = kl_per
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
            
        total = recon_loss + beta * kl_loss
    
        return {
            "total": total, 
            "reconstruction": recon_loss, 
            "kl": kl_loss
        }