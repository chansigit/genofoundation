"""
Tests for FFN-based VAE and its components.
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

from genofoundation.models.vae.ffn_vae import VAE, VAEEncoder, VAEDecoder


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


@pytest.fixture
def hidden_dims():
    return [64, 64]


# =============================================================================
# Tests for VAEEncoder
# =============================================================================

class TestVAEEncoder:
    """Tests for VAEEncoder module."""

    def test_output_shape(self, input_dim, latent_dim, batch_size):
        """Test encoder output shape."""
        encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = encoder(x)

        assert "mu" in out
        assert "log_var" in out
        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)

    def test_with_condition(self, input_dim, latent_dim, batch_size):
        """Test encoder with conditioning."""
        condition_dim = 10
        encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[64],
            condition_dim=condition_dim,
        )

        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, condition_dim)

        out = encoder(x, condition)

        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)

    def test_logvar_clamped(self, input_dim, latent_dim):
        """Test that log_var is clamped."""
        encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[64],
        )

        # Large input to potentially produce extreme values
        x = torch.randn(4, input_dim) * 100
        out = encoder(x)

        assert out["log_var"].min() >= -10
        assert out["log_var"].max() <= 10

    def test_different_activations(self, input_dim, latent_dim, batch_size):
        """Test encoder with different activation functions."""
        for activation in ["silu", "gelu"]:
            encoder = VAEEncoder(
                input_dim=input_dim,
                latent_dim=latent_dim,
                hidden_dims=[64],
                activation=activation,
            )

            x = torch.randn(batch_size, input_dim)
            out = encoder(x)

            assert out["mu"].shape == (batch_size, latent_dim)

    def test_with_dropout(self, input_dim, latent_dim, batch_size):
        """Test encoder with dropout."""
        encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[64],
            dropout=0.1,
        )

        x = torch.randn(batch_size, input_dim)

        # Train mode should apply dropout
        encoder.train()
        out1 = encoder(x)

        # Eval mode should not apply dropout
        encoder.eval()
        out2 = encoder(x)

        # Both should produce valid outputs
        assert out1["mu"].shape == (batch_size, latent_dim)
        assert out2["mu"].shape == (batch_size, latent_dim)

    def test_without_layer_norm(self, input_dim, latent_dim, batch_size):
        """Test encoder without layer normalization."""
        encoder = VAEEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[64],
            use_layer_norm=False,
        )

        assert encoder.norms is None

        x = torch.randn(batch_size, input_dim)
        out = encoder(x)

        assert out["mu"].shape == (batch_size, latent_dim)


# =============================================================================
# Tests for VAEDecoder
# =============================================================================

class TestVAEDecoder:
    """Tests for VAEDecoder module."""

    def test_output_shape(self, input_dim, latent_dim, batch_size):
        """Test decoder output shape."""
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
        )

        z = torch.randn(batch_size, latent_dim)
        out = decoder(z)

        assert out.shape == (batch_size, input_dim)

    def test_with_condition(self, input_dim, latent_dim, batch_size):
        """Test decoder with conditioning."""
        condition_dim = 10
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
            condition_dim=condition_dim,
        )

        z = torch.randn(batch_size, latent_dim)
        condition = torch.randn(batch_size, condition_dim)

        out = decoder(z, condition)

        assert out.shape == (batch_size, input_dim)

    def test_output_activation_softplus(self, input_dim, latent_dim, batch_size):
        """Test decoder with softplus output activation."""
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
            output_activation="softplus",
        )

        z = torch.randn(batch_size, latent_dim)
        out = decoder(z)

        assert (out > 0).all()

    def test_output_activation_sigmoid(self, input_dim, latent_dim, batch_size):
        """Test decoder with sigmoid output activation."""
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
            output_activation="sigmoid",
        )

        z = torch.randn(batch_size, latent_dim)
        out = decoder(z)

        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_output_activation_relu(self, input_dim, latent_dim, batch_size):
        """Test decoder with relu output activation."""
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
            output_activation="relu",
        )

        z = torch.randn(batch_size, latent_dim)
        out = decoder(z)

        assert (out >= 0).all()

    def test_output_activation_none(self, input_dim, latent_dim, batch_size):
        """Test decoder with no output activation."""
        decoder = VAEDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=[64],
            output_activation=None,
        )

        z = torch.randn(batch_size, latent_dim)
        out = decoder(z)

        # Can have negative values
        assert out.shape == (batch_size, input_dim)


# =============================================================================
# Tests for VAE
# =============================================================================

class TestVAE:
    """Tests for VAE module."""

    def test_forward_output_keys(self, input_dim, latent_dim, batch_size):
        """Test that forward returns expected keys."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        expected_keys = {"reconstruction", "latent", "mu", "log_var"}
        assert set(out.keys()) == expected_keys

    def test_forward_output_shapes(self, input_dim, latent_dim, batch_size):
        """Test that forward outputs have correct shapes."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, input_dim)
        assert out["latent"].shape == (batch_size, latent_dim)
        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)

    def test_different_input_output_dims(self, latent_dim, batch_size):
        """Test VAE with different input and output dimensions."""
        input_dim = 100
        output_dim = 50

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, output_dim)

    def test_forward_with_condition(self, input_dim, latent_dim, batch_size):
        """Test forward pass with conditioning."""
        condition_dim = 10

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            condition_dim=condition_dim,
        )

        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, condition_dim)

        out = vae(x, condition)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    def test_condition_on_encoder_only(self, input_dim, latent_dim, batch_size):
        """Test conditioning on encoder only."""
        condition_dim = 10

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            condition_dim=condition_dim,
            condition_on_encoder=True,
            condition_on_decoder=False,
        )

        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, condition_dim)

        out = vae(x, condition)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    def test_condition_on_decoder_only(self, input_dim, latent_dim, batch_size):
        """Test conditioning on decoder only."""
        condition_dim = 10

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            condition_dim=condition_dim,
            condition_on_encoder=False,
            condition_on_decoder=True,
        )

        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, condition_dim)

        out = vae(x, condition)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    def test_training_vs_eval_mode(self, input_dim, latent_dim, batch_size):
        """Test that reparameterization differs in train vs eval mode."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)

        # In eval mode, latent should equal mu
        vae.eval()
        with torch.no_grad():
            out_eval = vae(x)

        torch.testing.assert_close(out_eval["latent"], out_eval["mu"])

    def test_encode_method(self, input_dim, latent_dim, batch_size):
        """Test encode method."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        enc_out = vae.encode(x)

        assert "mu" in enc_out
        assert "log_var" in enc_out
        assert enc_out["mu"].shape == (batch_size, latent_dim)

    def test_decode_method(self, input_dim, latent_dim, batch_size):
        """Test decode method."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        z = torch.randn(batch_size, latent_dim)
        recon = vae.decode(z)

        assert recon.shape == (batch_size, input_dim)

    def test_sample(self, input_dim, latent_dim):
        """Test sample generation."""
        num_samples = 4

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        samples = vae.sample(num_samples)

        assert samples.shape == (num_samples, input_dim)

    def test_sample_with_condition(self, input_dim, latent_dim):
        """Test sample generation with conditioning."""
        num_samples = 4
        condition_dim = 10

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            condition_dim=condition_dim,
        )

        condition = torch.randn(num_samples, condition_dim)
        samples = vae.sample(num_samples, condition=condition)

        assert samples.shape == (num_samples, input_dim)

    def test_loss_computation(self, input_dim, latent_dim, batch_size):
        """Test loss computation."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

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

        # Total should equal reconstruction + kl
        torch.testing.assert_close(
            loss["total"],
            loss["reconstruction"] + loss["kl"]
        )

    def test_loss_with_beta(self, input_dim, latent_dim, batch_size):
        """Test loss computation with different beta values."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        loss_beta0 = vae.loss(x, out, beta=0.0)
        loss_beta1 = vae.loss(x, out, beta=1.0)
        loss_beta2 = vae.loss(x, out, beta=2.0)

        # With beta=0, total should equal reconstruction
        torch.testing.assert_close(loss_beta0["total"], loss_beta0["reconstruction"])

        # Total should increase with beta (assuming KL > 0)
        assert loss_beta2["total"] >= loss_beta1["total"]

    def test_gradient_flow(self, input_dim, latent_dim, batch_size):
        """Test that gradients flow properly through the model."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)
        loss = vae.loss(x, out)

        loss["total"].backward()

        # Check that gradients exist for all parameters
        for name, param in vae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_reconstruction_positive_with_softplus(self, input_dim, latent_dim, batch_size):
        """Test that reconstruction is positive with softplus activation."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            output_activation="softplus",
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert (out["reconstruction"] > 0).all()


# =============================================================================
# Integration Tests
# =============================================================================

class TestVAEIntegration:
    """Integration tests for VAE."""

    def test_training_step(self, input_dim, latent_dim):
        """Test a complete training step."""
        batch_size = 8

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

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

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        )

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

    def test_with_spectral_norm(self, input_dim, latent_dim, batch_size):
        """Test VAE with spectral normalization."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
            spectral_norm=True,
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    def test_deep_architecture(self, input_dim, latent_dim, batch_size):
        """Test VAE with deeper architecture."""
        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[128, 64, 32],
            decoder_hidden_dims=[32, 64, 128],
        )

        x = torch.randn(batch_size, input_dim)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, input_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, input_dim, latent_dim):
        """Test that model works on CUDA."""
        batch_size = 4

        vae = VAE(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_hidden_dims=[64],
            decoder_hidden_dims=[64],
        ).cuda()

        x = torch.randn(batch_size, input_dim).cuda()

        out = vae(x)
        loss = vae.loss(x, out)

        assert out["reconstruction"].device.type == "cuda"
        assert loss["total"].device.type == "cuda"
