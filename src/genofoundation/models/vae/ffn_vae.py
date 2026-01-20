"""
Variational Autoencoder implementation based on abstract base classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union
from .ffn import FFN
from .abstract_ae import BaseEmbeddingModel, BaseEncoder, BaseDecoder


class VAEEncoder(BaseEncoder):
    """VAE Encoder that outputs distribution parameters (mu, log_var)."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = True,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
        use_layer_norm: bool = True,
    ):
        super().__init__(input_dim, latent_dim, condition_dim)
        
        self.use_layer_norm = use_layer_norm
        
        # Adjust input dimension if conditioning is used
        actual_input_dim = input_dim
        if condition_dim is not None:
            actual_input_dim += condition_dim
        
        if hidden_dims is None:
            hidden_dims = [512]  # Default single hidden layer
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        prev_dim = actual_input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(FFN(
                d_model=prev_dim,
                hidden_dim=None,  # Auto-infer
                out_dim=hidden_dim,
                kind=activation,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
            ))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(prev_dim, latent_dim, bias=bias)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Concatenate conditioning if provided
        if condition is not None and self.condition_dim is not None:
            x = torch.cat([x, condition], dim=-1)
        
        # Forward through layers
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.norms is not None:
                h = self.norms[i](h)
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)  # Clamp both sides for stability

        return {
            'mu': mu,
            'log_var': logvar
        }


class VAEDecoder(BaseDecoder):
    """VAE Decoder that reconstructs from latent representation."""
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = True,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        output_activation: Optional[str] = "softplus",
    ):
        super().__init__(latent_dim, output_dim, condition_dim)
        
        self.use_layer_norm = use_layer_norm
        self.output_activation = output_activation
        
        # Adjust input dimension if conditioning is used
        actual_input_dim = latent_dim
        if condition_dim is not None:
            actual_input_dim += condition_dim
        
        if hidden_dims is None:
            hidden_dims = [512]  # Default single hidden layer
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_layer_norm else None
        
        prev_dim = actual_input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(FFN(
                d_model=prev_dim,
                hidden_dim=None,  # Auto-infer
                out_dim=hidden_dim,
                kind=activation,
                dropout=dropout,
                bias=bias,
                spectral_norm=spectral_norm,
            ))
            if use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer
        self.fc_output = nn.Linear(prev_dim, output_dim, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Concatenate conditioning if provided
        if condition is not None and self.condition_dim is not None:
            z = torch.cat([z, condition], dim=-1)
        
        # Forward through layers
        h = z
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if self.norms is not None:
                h = self.norms[i](h)
        
        # Output projection
        output = self.fc_output(h)
        
        # Apply output activation if specified
        if self.output_activation == "softplus":
            output = F.softplus(output)
        elif self.output_activation == "relu":
            output = F.relu(output)
        elif self.output_activation == "sigmoid":
            output = torch.sigmoid(output)
        
        return output


class VAE(BaseEmbeddingModel):
    """
    Variational Autoencoder implementation.
    
    Args:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        output_dim: Output dimension (defaults to input_dim)
        encoder_hidden_dims: List of hidden dimensions for encoder
        decoder_hidden_dims: List of hidden dimensions for decoder
        activation: Activation type {"swiglu", "geglu", "gelu", "silu"}
        dropout: Dropout probability
        bias: Whether to use bias in linear layers
        spectral_norm: Whether to apply spectral normalization
        condition_dim: Conditioning vector dimension
        condition_on_encoder: Whether to use conditioning in encoder
        condition_on_decoder: Whether to use conditioning in decoder
        use_layer_norm: Whether to use layer normalization
        output_activation: Output activation {"softplus", "sigmoid", None}
        beta: Weight for KL divergence in loss (beta-VAE)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: Optional[int] = None,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = True,
        spectral_norm: bool = False,
        condition_dim: Optional[int] = None,
        condition_on_encoder: bool = True,
        condition_on_decoder: bool = True,
        use_layer_norm: bool = True,
        output_activation: Optional[str] = "softplus",
        beta: float = 1.0,
    ):
        super().__init__(input_dim, latent_dim, output_dim, condition_dim)
        
        self.condition_on_encoder = condition_on_encoder
        self.condition_on_decoder = condition_on_decoder
        self.beta = beta
        
        # Create encoder
        encoder_condition_dim = condition_dim if condition_on_encoder else None
        self.encoder_module = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            condition_dim=encoder_condition_dim,
            use_layer_norm=use_layer_norm,
        )
        
        # Create decoder
        decoder_condition_dim = condition_dim if condition_on_decoder else None
        self.decoder_module = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=self.output_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
            spectral_norm=spectral_norm,
            condition_dim=decoder_condition_dim,
            use_layer_norm=use_layer_norm,
            output_activation=output_activation,
        )
    
    def encode(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Returns dict with 'mu' and 'log_var' keys.
        """
        encoder_condition = condition if self.condition_on_encoder else None
        return self.encoder_module(x, encoder_condition)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, return the mean
            return mu
    
    def decode(
        self, 
        z: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        decoder_condition = condition if self.condition_on_decoder else None
        return self.decoder_module(z, decoder_condition)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Returns dictionary with:
            - 'reconstruction': Reconstructed output
            - 'latent': Sampled latent representation
            - 'mu': Mean of latent distribution
            - 'log_var': Log variance of latent distribution
        """
        # Encode
        encoder_output = self.encode(x, condition)
        mu = encoder_output['mu']
        log_var = encoder_output['log_var']
        
        # Sample
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decode(z, condition)
        
        return {
            'reconstruction': reconstruction,
            'latent': z,
            'mu': mu,
            'log_var': log_var,
        }

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
            x: ground-truth input, shape [B, ...]
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
        # Note: log_var is already clamped in encoder
        kl_loss = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(1).mean() / Nf

        total = recon_loss + beta * kl_loss

        return {"total": total, "reconstruction": recon_loss, "kl": kl_loss} 
    
    
    def sample(
        self, 
        num_samples: int, 
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Sample from the prior and decode.
        
        Args:
            num_samples: Number of samples to generate
            condition: Optional conditioning tensor
            device: Device to generate samples on
        
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode
        return self.decode(z, condition)

