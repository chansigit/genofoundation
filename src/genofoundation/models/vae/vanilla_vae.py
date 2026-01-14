import torch.nn as nn

# Simple VAE model for testing
class SimpleVAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean and logvar
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(), 
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = h.chunk(2, dim=-1)
        return mean, logvar
        
    def decode(self, z):
        return self.decoder(z)
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x, condition=None):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return {
            'reconstruction': recon,
            'mean': mean,
            'logvar': logvar,
            'z': z
        }
        
    def loss(self, x, outputs, beta=1.0):
        recon = outputs['reconstruction']
        mean = outputs['mean']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss
        }