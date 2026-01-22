"""
Tests for SimpleVAE (vanilla VAE implementation).
"""

import pytest
import torch
import torch.nn as nn

import os
import sys

# Add the src directory to path for imports
src_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "src"
)
sys.path.insert(0, os.path.abspath(src_path))

from genofoundation.models.vae.vanilla_vae import SimpleVAE


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def input_dim():
    return 128


@pytest.fixture
def latent_dim():
    return 32


@pytest.fixture
def batch_size():
    return 8


# =============================================================================
# Tests for SimpleVAE
# =============================================================================

class TestSimpleVAE:
    """Tests for SimpleVAE module."""

    def test_forward_output_keys(self, input_dim, latent_dim, batch_size):
        """Test that forward returns expected keys."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        expected_keys = {"reconstruction", "mean", "logvar", "z"}
        assert set(out.keys()) == expected_keys

    def test_forward_output_shapes(self, input_dim, latent_dim, batch_size):
        """Test that forward outputs have correct shapes."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, input_dim)
        assert out["mean"].shape == (batch_size, latent_dim)
        assert out["logvar"].shape == (batch_size, latent_dim)
        assert out["z"].shape == (batch_size, latent_dim)

    def test_encode(self, input_dim, latent_dim, batch_size):
        """Test encode method."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        mean, logvar = vae.encode(x)

        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)

    def test_logvar_clamped(self, input_dim, latent_dim):
        """Test that logvar is clamped."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        # Large input to potentially produce extreme values
        x = torch.randn(4, input_dim) * 100
        mean, logvar = vae.encode(x)

        assert logvar.min() >= -10
        assert logvar.max() <= 10

    def test_decode(self, input_dim, latent_dim, batch_size):
        """Test decode method."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        z = torch.randn(batch_size, latent_dim)
        recon = vae.decode(z)

        assert recon.shape == (batch_size, input_dim)

    def test_decode_positive(self, input_dim, latent_dim, batch_size):
        """Test that decode output is positive (due to Softplus)."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        z = torch.randn(batch_size, latent_dim)
        recon = vae.decode(z)

        assert (recon > 0).all()

    def test_reparameterize(self, input_dim, latent_dim, batch_size):
        """Test reparameterization trick."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        mean = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)

        # With mean=0 and logvar=0, z should be sampled from N(0, 1)
        z = vae.reparameterize(mean, logvar)

        assert z.shape == (batch_size, latent_dim)

    def test_reparameterize_with_specific_values(self, latent_dim):
        """Test reparameterization with specific mean and variance."""
        vae = SimpleVAE(input_dim=100, latent_dim=latent_dim)

        batch_size = 1000
        mean = torch.ones(batch_size, latent_dim) * 5.0
        logvar = torch.zeros(batch_size, latent_dim)  # var = 1

        z = vae.reparameterize(mean, logvar)

        # Mean of z should be close to 5.0
        assert torch.abs(z.mean() - 5.0) < 0.2

    def test_forward_with_condition_ignored(self, input_dim, latent_dim, batch_size):
        """Test that condition parameter is accepted but ignored."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, 10)

        # Should not raise error, condition is ignored
        out = vae(x, condition=condition)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    def test_loss_computation(self, input_dim, latent_dim, batch_size):
        """Test loss computation."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out, beta=1.0)

        assert "total" in loss
        assert "reconstruction" in loss
        assert "kl" in loss

        # All losses should be scalars
        assert loss["total"].ndim == 0
        assert loss["reconstruction"].ndim == 0
        assert loss["kl"].ndim == 0

    def test_loss_total_equals_sum(self, input_dim, latent_dim, batch_size):
        """Test that total loss equals reconstruction + beta * kl."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out, beta=1.0)

        torch.testing.assert_close(
            loss["total"],
            loss["reconstruction"] + loss["kl"]
        )

    def test_loss_with_beta_zero(self, input_dim, latent_dim, batch_size):
        """Test loss with beta=0."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out, beta=0.0)

        # With beta=0, total should equal reconstruction
        torch.testing.assert_close(loss["total"], loss["reconstruction"])

    def test_loss_with_different_beta(self, input_dim, latent_dim, batch_size):
        """Test loss computation with different beta values."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        loss_beta1 = vae.loss(x, out, beta=1.0)
        loss_beta2 = vae.loss(x, out, beta=2.0)

        # Total should increase with beta (assuming KL > 0)
        assert loss_beta2["total"] >= loss_beta1["total"]

    def test_loss_normalized_by_features(self, latent_dim, batch_size):
        """Test that loss is normalized by number of features."""
        vae_small = SimpleVAE(input_dim=100, latent_dim=latent_dim)
        vae_large = SimpleVAE(input_dim=1000, latent_dim=latent_dim)

        torch.manual_seed(42)
        x_small = torch.randn(batch_size, 100)
        x_large = torch.randn(batch_size, 1000)

        out_small = vae_small(x_small)
        out_large = vae_large(x_large)

        loss_small = vae_small.loss(x_small, out_small)
        loss_large = vae_large.loss(x_large, out_large)

        # Losses should be on similar scale due to normalization
        # (not exactly equal due to different architectures, but same order of magnitude)
        assert loss_small["total"].item() < 100
        assert loss_large["total"].item() < 100

    def test_gradient_flow(self, input_dim, latent_dim, batch_size):
        """Test that gradients flow properly through the model."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        loss["total"].backward()

        # Check that gradients exist for all parameters
        for name, param in vae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_reconstruction_positive(self, input_dim, latent_dim, batch_size):
        """Test that reconstruction is positive (due to Softplus in decoder)."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert (out["reconstruction"] > 0).all()


# =============================================================================
# Integration Tests
# =============================================================================

class TestSimpleVAEIntegration:
    """Integration tests for SimpleVAE."""

    def test_training_step(self, input_dim, latent_dim):
        """Test a complete training step."""
        batch_size = 8

        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training step
        vae.train()
        x = torch.randn(batch_size, input_dim)

        optimizer.zero_grad()
        out = vae(x)
        loss = vae.loss(x, out)
        loss["total"].backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss["total"])

    def test_multiple_training_steps(self, input_dim, latent_dim):
        """Test that loss decreases over multiple training steps."""
        batch_size = 16

        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Fixed data for consistent training
        torch.manual_seed(42)
        x = torch.rand(batch_size, input_dim) + 0.1  # Positive data

        losses = []
        vae.train()

        for _ in range(20):
            optimizer.zero_grad()
            out = vae(x)
            loss = vae.loss(x, out, beta=0.1)
            loss["total"].backward()
            optimizer.step()
            losses.append(loss["total"].item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_overfitting_single_sample(self, input_dim, latent_dim):
        """Test that model can overfit to a single sample."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Single positive sample
        torch.manual_seed(42)
        x = torch.rand(1, input_dim) + 0.5

        vae.train()
        for _ in range(100):
            optimizer.zero_grad()
            out = vae(x)
            loss = vae.loss(x, out, beta=0.01)
            loss["total"].backward()
            optimizer.step()

        # Reconstruction should be close to input
        vae.eval()
        with torch.no_grad():
            out = vae(x)

        # Check reconstruction error is small
        recon_error = (out["reconstruction"] - x).abs().mean()
        assert recon_error < 0.5  # Reasonably close

    def test_latent_interpolation(self, input_dim, latent_dim):
        """Test interpolation in latent space."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        # Two random latent vectors
        z1 = torch.randn(1, latent_dim)
        z2 = torch.randn(1, latent_dim)

        # Interpolate
        alphas = torch.linspace(0, 1, 5)
        reconstructions = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            recon = vae.decode(z_interp)
            reconstructions.append(recon)

        # All reconstructions should be valid (positive due to Softplus)
        for recon in reconstructions:
            assert (recon > 0).all()
            assert recon.shape == (1, input_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, input_dim, latent_dim):
        """Test that model works on CUDA."""
        batch_size = 4

        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim).cuda()
        x = torch.randn(batch_size, input_dim).cuda()

        out = vae(x)
        loss = vae.loss(x, out)

        assert out["reconstruction"].device.type == "cuda"
        assert loss["total"].device.type == "cuda"

    def test_deterministic_eval(self, input_dim, latent_dim, batch_size):
        """Test that evaluation is deterministic with fixed seed."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
        vae.eval()

        x = torch.randn(batch_size, input_dim)

        # Note: SimpleVAE always uses reparameterization (stochastic)
        # So we need to set seed for reproducibility
        torch.manual_seed(42)
        with torch.no_grad():
            out1 = vae(x)

        torch.manual_seed(42)
        with torch.no_grad():
            out2 = vae(x)

        torch.testing.assert_close(out1["reconstruction"], out2["reconstruction"])


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestSimpleVAEEdgeCases:
    """Edge case tests for SimpleVAE."""

    def test_single_sample_batch(self, input_dim, latent_dim):
        """Test with batch size of 1."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(1, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        assert out["reconstruction"].shape == (1, input_dim)
        assert torch.isfinite(loss["total"])

    def test_large_batch(self, input_dim, latent_dim):
        """Test with large batch size."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(256, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        assert out["reconstruction"].shape == (256, input_dim)
        assert torch.isfinite(loss["total"])

    def test_zero_input(self, input_dim, latent_dim, batch_size):
        """Test with zero input."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.zeros(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        assert torch.isfinite(loss["total"])

    def test_large_input_values(self, input_dim, latent_dim, batch_size):
        """Test with large input values."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim) * 100
        out = vae(x)
        loss = vae.loss(x, out)

        # Should still produce finite loss
        assert torch.isfinite(loss["total"])

    def test_small_latent_dim(self, input_dim, batch_size):
        """Test with very small latent dimension."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=2)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        assert out["z"].shape == (batch_size, 2)
        assert torch.isfinite(loss["total"])

    def test_large_latent_dim(self, input_dim, batch_size):
        """Test with large latent dimension."""
        vae = SimpleVAE(input_dim=input_dim, latent_dim=256)

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        assert out["z"].shape == (batch_size, 256)
        assert torch.isfinite(loss["total"])
